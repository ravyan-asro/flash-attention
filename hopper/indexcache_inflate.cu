// indexcache_inflate.cu — v3
// GPU kernel to convert compact IndexCache metadata → FA3 sparse packed arrays.
// Processes ALL layers in a single kernel launch.
//
// v3: kvcolidx is now uint8 bit-packed (MSB-first, byte-aligned per document),
//     same format as la_hot_tile_code.  Each bit corresponds to one page (= one
//     FA3 n-block since page_size == kBlockN == 128).
//
// Input (compact, stacked across layers):
//   kvcolidx:         (num_layers, H_q, total_kv_bytes) uint8, bit-packed MSB-first
//   la_hot_tile_code: (num_layers, H_q, total_la_bytes) uint8, bit-packed MSB-first
//   doc_page_offsets: (num_docs+1,) int32, zero-prefixed cumulative pages
//   la_byte_offsets:  (num_docs+1,) int32, zero-prefixed cumulative LA bytes
//   kvcol_byte_offsets: (num_docs,) int32, cumulative kvcolidx byte counts per non-last doc
//   kvcol_page_counts:  (num_docs-1,) int32, number of real pages per non-last doc
//                        (needed to mask off padding bits at byte boundaries)
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
//   Threads cooperatively scan CA bits and LA bits.
// --------------------------------------------------------------------------

__global__ void indexcache_count_v3(
    const uint8_t* __restrict__ kvcolidx,       // (num_layers, H_q, total_kv_bytes) uint8
    const uint8_t* __restrict__ la_code,         // (num_layers, H_q, total_la_bytes) uint8
    const int32_t* __restrict__ doc_page_offsets,
    const int32_t* __restrict__ la_byte_offsets,
    const int32_t* __restrict__ kvcol_byte_offsets,
    const int32_t* __restrict__ kvcol_page_counts,
    int num_layers, int H_q, int M, int M_doc, int num_docs,
    int total_kv_bytes, int total_la_bytes, int num_n_blocks,
    int num_ca_docs,  // = num_docs - 1
    int32_t* __restrict__ counts
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

    // --- CA: parallel scan of kvcolidx bit vector ---
    // Only scan docs 0..doc_idx-1 (pages BEFORE the m-block's document).
    // Pages from later documents have positions > m and violate causality.
    // This matches the old kernel's behavior: ca_end = kvcol_offsets[doc_idx].
    int kv_base = (layer * H_q + h_q) * total_kv_bytes;

    // --- CA: byte-level popcount scan of kvcolidx bit vector ---
    int local_ca = 0;
    int ca_docs = min(doc_idx, num_ca_docs);  // docs 0..doc_idx-1
    for (int d = 0; d < ca_docs; d++) {
        int n_pages_d = kvcol_page_counts[d];
        int byte_start = (d > 0) ? kvcol_byte_offsets[d] : 0;
        int full_bytes = n_pages_d >> 3;
        int tail_bits = n_pages_d & 7;
        // Count set bits in full bytes using popcount
        for (int b = tid; b < full_bytes; b += BLOCK_THREADS) {
            local_ca += __popc(kvcolidx[kv_base + byte_start + b]);
        }
        // Handle tail bits in last partial byte (mask off padding)
        if (tail_bits > 0 && tid == 0) {
            uint8_t last = kvcolidx[kv_base + byte_start + full_bytes];
            uint8_t mask = (uint8_t)(0xFF << (8 - tail_bits));
            local_ca += __popc(last & mask);
        }
    }

    // --- LA: byte-level popcount scan ---
    int la_base_bytes = (layer * H_q + h_q) * total_la_bytes;
    int doc_bit_start = la_byte_offsets[doc_idx] * 8;
    int row_bit_start = doc_bit_start + local_m * (local_m + 1) / 2;
    int num_bits = local_m + 1;

    int local_la = 0;
    // Process full bytes with popcount
    int la_first_bit = row_bit_start;
    int la_last_bit = row_bit_start + num_bits - 1;
    int la_first_byte = la_first_bit >> 3;
    int la_last_byte = la_last_bit >> 3;
    int la_start_offset = la_first_bit & 7;

    if (la_first_byte == la_last_byte) {
        // All bits in one byte
        if (tid == 0) {
            uint8_t b = la_code[la_base_bytes + la_first_byte];
            // Mask: clear bits before start and after end
            int end_offset = la_last_bit & 7;
            uint8_t mask = (uint8_t)((0xFF >> la_start_offset) &
                                     (0xFF << (7 - end_offset)));
            local_la = __popc(b & mask);
        }
    } else {
        // First partial byte
        if (tid == 0 && la_start_offset > 0) {
            uint8_t b = la_code[la_base_bytes + la_first_byte];
            local_la += __popc(b & (uint8_t)(0xFF >> la_start_offset));
        }
        // Full bytes in the middle
        int mid_start = la_first_byte + (la_start_offset > 0 ? 1 : 0);
        int mid_end = la_last_byte;
        for (int b = mid_start + tid; b < mid_end; b += BLOCK_THREADS) {
            local_la += __popc(la_code[la_base_bytes + b]);
        }
        // Last partial byte
        int end_offset = la_last_bit & 7;
        if (tid == 0) {
            uint8_t b = la_code[la_base_bytes + la_last_byte];
            local_la += __popc(b & (uint8_t)(0xFF << (7 - end_offset)));
        }
    }

    // Block reduction via warp shuffle
    int local_total = local_ca + local_la;

    for (int offset = 16; offset > 0; offset >>= 1)
        local_total += __shfl_down_sync(0xFFFFFFFF, local_total, offset);

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

__global__ void indexcache_write_v3(
    const uint8_t* __restrict__ kvcolidx,
    const uint8_t* __restrict__ la_code,
    const int32_t* __restrict__ doc_page_offsets,
    const int32_t* __restrict__ la_byte_offsets,
    const int32_t* __restrict__ kvcol_byte_offsets,
    const int32_t* __restrict__ kvcol_page_counts,
    int num_layers, int H_q, int M, int M_doc, int num_docs,
    int total_kv_bytes, int total_la_bytes, int num_n_blocks,
    int num_ca_docs,
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
            sparse_n_mask_counts[bid] = (n_count > m) ? 1 : 0;
        }
        return;
    }

    // Document m-block: collect into shared memory, sort, write
    extern __shared__ char smem_raw[];
    int32_t* s_buf = reinterpret_cast<int32_t*>(smem_raw);
    int32_t* s_count = s_buf;
    int32_t* s_idx   = s_buf + 1;

    if (tid == 0) *s_count = 0;
    __syncthreads();

    int doc_idx = find_doc(doc_page_offsets, num_docs, m);
    int doc_start_page = doc_page_offsets[doc_idx];
    int local_m = m - doc_start_page;

    // --- CA: parallel compact into shared memory from bit vector ---
    // Only docs 0..doc_idx-1 (causal: pages before m-block's document)
    int kv_base = (layer * H_q + h_q) * total_kv_bytes;

    int ca_docs = min(doc_idx, num_ca_docs);
    for (int d = 0; d < ca_docs; d++) {
        int n_pages_d = kvcol_page_counts[d];
        int byte_start = (d > 0) ? kvcol_byte_offsets[d] : 0;

        for (int p = tid; p < n_pages_d; p += BLOCK_THREADS) {
            int bit_idx = byte_start * 8 + p;
            if (read_bit(kvcolidx + kv_base, bit_idx)) {
                int global_page = doc_page_offsets[d] + p;
                int pos = atomicAdd(s_count, 1);
                s_idx[pos] = global_page;
            }
        }
    }
    __syncthreads();

    // --- LA: parallel compact into shared memory (same as before) ---
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

    int total = *s_count;

    // --- Bitonic sort descending in shared memory ---
    int n_padded = 1;
    while (n_padded < total) n_padded <<= 1;

    for (int i = total + tid; i < n_padded; i += BLOCK_THREADS)
        s_idx[i] = -1;
    __syncthreads();

    for (int k = 2; k <= n_padded; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = tid; i < (n_padded >> 1); i += BLOCK_THREADS) {
                int l = (i / j) * (j << 1) + (i % j);
                int r = l + j;
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
    torch::Tensor kvcolidx,            // (num_layers, H_q, total_kv_bytes) uint8
    torch::Tensor la_hot_tile_code,    // (num_layers, H_q, total_la_bytes) uint8
    torch::Tensor doc_page_offsets,    // (num_docs+1,) int32
    torch::Tensor la_byte_offsets,     // (num_docs+1,) int32
    torch::Tensor kvcol_byte_offsets,  // (num_docs,)   int32
    torch::Tensor kvcol_page_counts,   // (num_docs-1,) int32
    int num_layers,
    int seqlen_total,
    int question_len,
    int kBlockN
) {
    TORCH_CHECK(kvcolidx.is_cuda(), "kvcolidx must be CUDA");
    TORCH_CHECK(la_hot_tile_code.is_cuda(), "la_hot_tile_code must be CUDA");
    TORCH_CHECK(kvcolidx.dtype() == torch::kUInt8, "kvcolidx must be uint8");

    int H_q = kvcolidx.size(1);
    int total_kv_bytes = kvcolidx.size(2);
    int total_la_bytes = la_hot_tile_code.size(2);
    int num_n_blocks = (seqlen_total + kBlockN - 1) / kBlockN;
    int M = num_n_blocks;
    int context_len = seqlen_total - question_len;
    int M_doc = (context_len + kBlockN - 1) / kBlockN;
    int num_docs = doc_page_offsets.size(0) - 1;
    int num_ca_docs = kvcol_page_counts.size(0);

    auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(kvcolidx.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int total_entries = num_layers * H_q * M;
    int smem_bytes = (1 + MAX_N_PER_MBLOCK) * sizeof(int32_t);

    // --- Pass 1: count ---
    torch::Tensor counts = torch::empty({total_entries}, opts_i32);

    indexcache_count_v3<<<total_entries, BLOCK_THREADS, 0, stream>>>(
        kvcolidx.data_ptr<uint8_t>(),
        la_hot_tile_code.data_ptr<uint8_t>(),
        doc_page_offsets.data_ptr<int32_t>(),
        la_byte_offsets.data_ptr<int32_t>(),
        kvcol_byte_offsets.data_ptr<int32_t>(),
        kvcol_page_counts.data_ptr<int32_t>(),
        num_layers, H_q, M, M_doc, num_docs,
        total_kv_bytes, total_la_bytes, num_n_blocks,
        num_ca_docs,
        counts.data_ptr<int32_t>()
    );

    // --- Prefix sum → offsets ---
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

    // Read total
    int32_t total_indices_host;
    cudaMemcpyAsync(&total_indices_host,
                    sparse_n_offsets.data_ptr<int32_t>() + total_entries,
                    sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // --- Pass 2: write ---
    torch::Tensor sparse_n_indices = torch::empty({total_indices_host}, opts_i32);
    torch::Tensor sparse_n_mask_counts = torch::empty({total_entries}, opts_i32);

    indexcache_write_v3<<<total_entries, BLOCK_THREADS, smem_bytes, stream>>>(
        kvcolidx.data_ptr<uint8_t>(),
        la_hot_tile_code.data_ptr<uint8_t>(),
        doc_page_offsets.data_ptr<int32_t>(),
        la_byte_offsets.data_ptr<int32_t>(),
        kvcol_byte_offsets.data_ptr<int32_t>(),
        kvcol_page_counts.data_ptr<int32_t>(),
        num_layers, H_q, M, M_doc, num_docs,
        total_kv_bytes, total_la_bytes, num_n_blocks,
        num_ca_docs,
        sparse_n_indices.data_ptr<int32_t>(),
        sparse_n_offsets.data_ptr<int32_t>(),
        sparse_n_mask_counts.data_ptr<int32_t>()
    );

    return {sparse_n_indices, sparse_n_offsets, sparse_n_mask_counts};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("build_indexcache_metadata_cuda", &build_indexcache_metadata_cuda,
          "Convert compact IndexCache metadata to FA3 sparse packed arrays (CUDA, all layers, bitvec kvcolidx)");
}
