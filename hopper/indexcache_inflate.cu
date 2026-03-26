// indexcache_inflate.cu — v2
// GPU kernel to convert compact IndexCache metadata → FA3 sparse packed arrays.
// Processes ALL layers in a single kernel launch.
//
// Input (compact, stacked across layers):
//   kvcolidx:         (num_layers, H_q, total_kv_indices) int64, -1=invalid
//   la_hot_tile_code: (num_layers, H_q, total_la_bytes)   uint8, bit-packed MSB-first
//   doc_page_offsets: (num_docs+1,)  int32, zero-prefixed cumulative pages
//   la_byte_offsets:  (num_docs+1,)  int32, zero-prefixed cumulative bytes
//   kvcol_offsets:    (num_docs,)    int32, cumulative kvcolidx entry counts per non-last doc
//
// Output (FA3 sparse format, all layers packed):
//   sparse_n_indices:     (total_entries,) int32
//   sparse_n_offsets:     (num_layers * H_q * M + 1,) int32
//   sparse_n_mask_counts: (num_layers * H_q * M,) int32

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cstdint>

// Max n-blocks we can collect per m-block in shared memory.
// 2048 covers 256K context at 128-token pages.
#define MAX_N_PER_MBLOCK 2048
#define BLOCK_THREADS 128

// --------------------------------------------------------------------------
// Device helpers
// --------------------------------------------------------------------------

__device__ __forceinline__
bool read_bit(const uint8_t* __restrict__ bytes, int bit_idx) {
    int byte_idx = bit_idx >> 3;
    int bit_in_byte = 7 - (bit_idx & 7);
    return (bytes[byte_idx] >> bit_in_byte) & 1;
}

// Binary search for doc_idx: find d such that doc_page_offsets[d] <= page < doc_page_offsets[d+1]
__device__ __forceinline__
int find_doc(const int32_t* doc_page_offsets, int num_docs, int page) {
    int lo = 0, hi = num_docs - 1;
    while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if (doc_page_offsets[mid] <= page)
            lo = mid;
        else
            hi = mid - 1;
    }
    return lo;
}

// --------------------------------------------------------------------------
// Pass 1: Count n-blocks per (layer, head, m_block)
//   One threadblock per (layer, head, m_block).
//   Threads cooperatively scan CA entries and LA bits.
// --------------------------------------------------------------------------

__global__ void indexcache_count_v2(
    const int64_t* __restrict__ kvcolidx,       // (num_layers, H_q, total_kv_indices)
    const uint8_t* __restrict__ la_code,        // (num_layers, H_q, total_la_bytes)
    const int32_t* __restrict__ doc_page_offsets,
    const int32_t* __restrict__ la_byte_offsets,
    const int32_t* __restrict__ kvcol_offsets,
    int num_layers, int H_q, int M, int M_doc, int num_docs,
    int total_kv_indices, int total_la_bytes, int num_n_blocks,
    int32_t* __restrict__ counts               // (num_layers * H_q * M,)
) {
    int bid = blockIdx.x;
    if (bid >= num_layers * H_q * M) return;
    int tid = threadIdx.x;

    int layer = bid / (H_q * M);
    int rem   = bid % (H_q * M);
    int h_q   = rem / M;
    int m     = rem % M;

    if (m >= M_doc) {
        // Question m-block: full causal
        if (tid == 0)
            counts[bid] = min(m + 1, num_n_blocks);
        return;
    }

    // Find which document this m-block belongs to
    int doc_idx = find_doc(doc_page_offsets, num_docs, m);
    int doc_start_page = doc_page_offsets[doc_idx];
    int local_m = m - doc_start_page;

    // --- CA: parallel scan of kvcolidx ---
    int ca_end = (doc_idx > 0) ? kvcol_offsets[doc_idx] : 0;
    int kv_base = (layer * H_q + h_q) * total_kv_indices;

    int local_ca = 0;
    for (int ci = tid; ci < ca_end; ci += BLOCK_THREADS) {
        if (kvcolidx[kv_base + ci] >= 0) local_ca++;
    }

    // --- LA: parallel bit decode ---
    int la_base_bytes = (layer * H_q + h_q) * total_la_bytes;
    int doc_bit_start = la_byte_offsets[doc_idx] * 8;
    int row_bit_start = doc_bit_start + local_m * (local_m + 1) / 2;
    int num_bits = local_m + 1;

    int local_la = 0;
    for (int col = tid; col < num_bits; col += BLOCK_THREADS) {
        int bit_idx = row_bit_start + col;
        int byte_idx = la_base_bytes + (bit_idx >> 3);
        int bit_in_byte = 7 - (bit_idx & 7);
        if ((la_code[byte_idx] >> bit_in_byte) & 1) local_la++;
    }

    // Block reduction via warp shuffle
    int local_total = local_ca + local_la;

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        local_total += __shfl_down_sync(0xFFFFFFFF, local_total, offset);

    // Cross-warp reduction via shared memory
    __shared__ int warp_sums[BLOCK_THREADS / 32];
    int warp_id = tid / 32;
    int lane = tid % 32;

    if (lane == 0) warp_sums[warp_id] = local_total;
    __syncthreads();

    if (tid == 0) {
        int total = 0;
        for (int w = 0; w < BLOCK_THREADS / 32; w++)
            total += warp_sums[w];
        counts[bid] = total;
    }
}


// --------------------------------------------------------------------------
// Pass 2: Write n-block indices and mask counts
//   One threadblock per (layer, head, m_block).
//   Threads cooperatively collect indices into shared memory,
//   perform parallel bitonic sort, then write to global memory.
// --------------------------------------------------------------------------

__global__ void indexcache_write_v2(
    const int64_t* __restrict__ kvcolidx,
    const uint8_t* __restrict__ la_code,
    const int32_t* __restrict__ doc_page_offsets,
    const int32_t* __restrict__ la_byte_offsets,
    const int32_t* __restrict__ kvcol_offsets,
    int num_layers, int H_q, int M, int M_doc, int num_docs,
    int total_kv_indices, int total_la_bytes, int num_n_blocks,
    int32_t* __restrict__ sparse_n_indices,
    const int32_t* __restrict__ sparse_n_offsets,
    int32_t* __restrict__ sparse_n_mask_counts
) {
    int bid = blockIdx.x;
    if (bid >= num_layers * H_q * M) return;
    int tid = threadIdx.x;

    int layer = bid / (H_q * M);
    int rem   = bid % (H_q * M);
    int h_q   = rem / M;
    int m     = rem % M;

    int write_base = sparse_n_offsets[bid];
    int n_count    = sparse_n_offsets[bid + 1] - write_base;

    if (m >= M_doc) {
        // Question m-block: write 0..n_count-1 in descending order
        for (int i = tid; i < n_count; i += BLOCK_THREADS) {
            sparse_n_indices[write_base + i] = n_count - 1 - i;
        }
        if (tid == 0) {
            // mask_count = number of blocks >= threshold (= m)
            // In descending order, first block is n_count-1, threshold = m
            // n_count = min(m+1, num_n_blocks), so all blocks < m+1
            // blocks >= m: only block index m itself (if present)
            // Actually: threshold = m. Blocks are [n_count-1, n_count-2, ..., 0].
            // Blocks >= m: {m, m+1, ..., n_count-1} but n_count <= m+1, so only block m if n_count > m.
            // n_count = min(m+1, num_n_blocks). If num_n_blocks > m, n_count = m+1, so block m exists → mask_count=1.
            // If num_n_blocks <= m, n_count = num_n_blocks <= m, highest block = n_count-1 < m → mask_count=0.
            sparse_n_mask_counts[bid] = (n_count > m) ? 1 : 0;
        }
        return;
    }

    // Document m-block: collect into shared memory, sort, write
    extern __shared__ char smem_raw[];
    int32_t* s_buf = reinterpret_cast<int32_t*>(smem_raw);
    // s_buf[0] = atomic write counter
    // s_buf[1 .. MAX_N_PER_MBLOCK] = index workspace

    int32_t* s_count = s_buf;
    int32_t* s_idx   = s_buf + 1;

    if (tid == 0) *s_count = 0;
    __syncthreads();

    int doc_idx = find_doc(doc_page_offsets, num_docs, m);
    int doc_start_page = doc_page_offsets[doc_idx];
    int local_m = m - doc_start_page;

    // --- CA: parallel compact into shared memory ---
    int ca_end = (doc_idx > 0) ? kvcol_offsets[doc_idx] : 0;
    int kv_base = (layer * H_q + h_q) * total_kv_indices;

    for (int ci = tid; ci < ca_end; ci += BLOCK_THREADS) {
        int64_t page = kvcolidx[kv_base + ci];
        if (page >= 0) {
            int pos = atomicAdd(s_count, 1);
            s_idx[pos] = (int32_t)page;
        }
    }
    __syncthreads();

    // --- LA: parallel compact into shared memory ---
    int la_base_bytes = (layer * H_q + h_q) * total_la_bytes;
    int doc_bit_start = la_byte_offsets[doc_idx] * 8;
    int row_bit_start = doc_bit_start + local_m * (local_m + 1) / 2;
    int num_bits = local_m + 1;

    for (int col = tid; col < num_bits; col += BLOCK_THREADS) {
        int bit_idx = row_bit_start + col;
        int byte_idx = la_base_bytes + (bit_idx >> 3);
        int bit_in_byte = 7 - (bit_idx & 7);
        if ((la_code[byte_idx] >> bit_in_byte) & 1) {
            int pos = atomicAdd(s_count, 1);
            s_idx[pos] = doc_start_page + col;
        }
    }
    __syncthreads();

    int total = *s_count;  // should equal n_count

    // --- Bitonic sort descending in shared memory ---
    // Pad to next power of 2
    int n_padded = 1;
    while (n_padded < total) n_padded <<= 1;

    // Fill padding with -1 (will sort to end in descending order)
    for (int i = total + tid; i < n_padded; i += BLOCK_THREADS)
        s_idx[i] = -1;
    __syncthreads();

    // Bitonic sort (descending: larger values first)
    for (int k = 2; k <= n_padded; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = tid; i < (n_padded >> 1); i += BLOCK_THREADS) {
                int l = (i / j) * (j << 1) + (i % j);
                int r = l + j;
                // Descending: swap if s_idx[l] < s_idx[r]
                bool ascending = ((l & k) == 0);
                if (ascending) {
                    if (s_idx[l] < s_idx[r]) {
                        int32_t tmp = s_idx[l];
                        s_idx[l] = s_idx[r];
                        s_idx[r] = tmp;
                    }
                } else {
                    if (s_idx[l] > s_idx[r]) {
                        int32_t tmp = s_idx[l];
                        s_idx[l] = s_idx[r];
                        s_idx[r] = tmp;
                    }
                }
            }
            __syncthreads();
        }
    }

    // --- Write sorted indices to global memory ---
    for (int i = tid; i < total; i += BLOCK_THREADS)
        sparse_n_indices[write_base + i] = s_idx[i];

    // --- Compute mask_count (thread 0) ---
    if (tid == 0) {
        int threshold = m;
        int mask_count = 0;
        for (int i = 0; i < total; i++) {
            if (s_idx[i] >= threshold)
                mask_count++;
            else
                break;
        }
        sparse_n_mask_counts[bid] = mask_count;
    }
}


// --------------------------------------------------------------------------
// Host launcher — ALL layers in one call
// --------------------------------------------------------------------------

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
build_indexcache_metadata_cuda(
    torch::Tensor kvcolidx,            // (num_layers, H_q, total_kv_indices) int64
    torch::Tensor la_hot_tile_code,    // (num_layers, H_q, total_la_bytes)   uint8
    torch::Tensor doc_page_offsets,    // (num_docs+1,) int32
    torch::Tensor la_byte_offsets,     // (num_docs+1,) int32
    torch::Tensor kvcol_offsets,       // (num_docs,)   int32
    int num_layers,
    int seqlen_total,
    int question_len,
    int kBlockN
) {
    TORCH_CHECK(kvcolidx.is_cuda(), "kvcolidx must be CUDA");
    TORCH_CHECK(la_hot_tile_code.is_cuda(), "la_hot_tile_code must be CUDA");

    int H_q = kvcolidx.size(1);
    int total_kv_indices = kvcolidx.size(2);
    int total_la_bytes = la_hot_tile_code.size(2);
    int num_n_blocks = (seqlen_total + kBlockN - 1) / kBlockN;
    int M = num_n_blocks;
    int context_len = seqlen_total - question_len;
    int M_doc = (context_len + kBlockN - 1) / kBlockN;
    int num_docs = doc_page_offsets.size(0) - 1;

    auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(kvcolidx.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int total_entries = num_layers * H_q * M;
    int smem_bytes = (1 + MAX_N_PER_MBLOCK) * sizeof(int32_t);

    // --- Pass 1: count (one threadblock per entry) ---
    torch::Tensor counts = torch::empty({total_entries}, opts_i32);

    indexcache_count_v2<<<total_entries, BLOCK_THREADS, 0, stream>>>(
        kvcolidx.data_ptr<int64_t>(),
        la_hot_tile_code.data_ptr<uint8_t>(),
        doc_page_offsets.data_ptr<int32_t>(),
        la_byte_offsets.data_ptr<int32_t>(),
        kvcol_offsets.data_ptr<int32_t>(),
        num_layers, H_q, M, M_doc, num_docs,
        total_kv_indices, total_la_bytes, num_n_blocks,
        counts.data_ptr<int32_t>()
    );

    // --- Prefix sum → offsets (single CUB call) ---
    torch::Tensor sparse_n_offsets = torch::empty({total_entries + 1}, opts_i32);
    cudaMemsetAsync(sparse_n_offsets.data_ptr<int32_t>(), 0, sizeof(int32_t), stream);

    size_t temp_bytes = 0;
    cub::DeviceScan::InclusiveSum(
        nullptr, temp_bytes,
        counts.data_ptr<int32_t>(),
        sparse_n_offsets.data_ptr<int32_t>() + 1,
        total_entries, stream
    );
    torch::Tensor temp_storage = torch::empty(
        {(int64_t)temp_bytes},
        torch::TensorOptions().dtype(torch::kUInt8).device(kvcolidx.device())
    );
    cub::DeviceScan::InclusiveSum(
        static_cast<void*>(temp_storage.data_ptr<uint8_t>()), temp_bytes,
        counts.data_ptr<int32_t>(),
        sparse_n_offsets.data_ptr<int32_t>() + 1,
        total_entries, stream
    );

    // Read total (single sync for all layers)
    int32_t total_indices_host;
    cudaMemcpyAsync(&total_indices_host,
                    sparse_n_offsets.data_ptr<int32_t>() + total_entries,
                    sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // --- Pass 2: write (one threadblock per entry, with shared memory) ---
    torch::Tensor sparse_n_indices = torch::empty({total_indices_host}, opts_i32);
    torch::Tensor sparse_n_mask_counts = torch::empty({total_entries}, opts_i32);

    indexcache_write_v2<<<total_entries, BLOCK_THREADS, smem_bytes, stream>>>(
        kvcolidx.data_ptr<int64_t>(),
        la_hot_tile_code.data_ptr<uint8_t>(),
        doc_page_offsets.data_ptr<int32_t>(),
        la_byte_offsets.data_ptr<int32_t>(),
        kvcol_offsets.data_ptr<int32_t>(),
        num_layers, H_q, M, M_doc, num_docs,
        total_kv_indices, total_la_bytes, num_n_blocks,
        sparse_n_indices.data_ptr<int32_t>(),
        sparse_n_offsets.data_ptr<int32_t>(),
        sparse_n_mask_counts.data_ptr<int32_t>()
    );

    return {sparse_n_indices, sparse_n_offsets, sparse_n_mask_counts};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("build_indexcache_metadata_cuda", &build_indexcache_metadata_cuda,
          "Convert compact IndexCache metadata to FA3 sparse packed arrays (CUDA, all layers)");
}
