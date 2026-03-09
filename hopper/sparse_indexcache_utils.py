"""
sparse_indexcache_utils.py

Utilities for building sparse N-block metadata consumed by FA3's sparse attention kernel.

Three functions:
  1. build_dense_sparse_metadata  — full-density lists for correctness testing
  2. build_block_mask_metadata    — convert a (B, H_kv, M, N) block mask to packed arrays
  3. build_indexcache_metadata    — convert IndexCache metadata (kvcolidx, LA_hot_tile_code)
                                   to packed arrays

All return (sparse_n_indices, sparse_n_offsets, sparse_n_mask_counts) as int32 GPU tensors,
matching the format expected by flash.h / flash_api.cpp.
"""

import math
from typing import Tuple, List, Optional

import torch


def _causal_n_block_max(m_block: int, kBlockM: int, kBlockN: int,
                        seqlen_q: int, seqlen_k: int, num_n_blocks: int) -> int:
    """FA3's n_block_max for causal attention (from block.h get_n_block_min_max)."""
    return min(
        math.ceil(((m_block + 1) * kBlockM + seqlen_k - seqlen_q) / kBlockN),
        num_n_blocks,
    )


def _causal_threshold(m_block: int, kBlockM: int, kBlockN: int,
                      seqlen_q: int, seqlen_k: int) -> int:
    """FA3's n_block_min_causal_local_mask (from block.h).
    N-blocks >= this threshold need causal masking."""
    return max(0, (m_block * kBlockM + seqlen_k - seqlen_q) // kBlockN)


# --------------------------------------------------------------------------- #
#  1.  Dense metadata (every valid N-block) — for correctness baseline
# --------------------------------------------------------------------------- #

def build_dense_sparse_metadata(
    batch_size: int,
    num_heads_kv: int,
    seqlen_q: int,
    seqlen_k: int,
    kBlockM: int = 128,
    kBlockN: int = 128,
    causal: bool = True,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build full-density sparse metadata.  Output of sparse FA3 with this
    metadata should be bit-for-bit identical to standard (non-sparse) FA3."""
    num_m_blocks = math.ceil(seqlen_q / kBlockM)
    num_n_blocks = math.ceil(seqlen_k / kBlockN)

    all_indices: List[int] = []
    all_offsets: List[int] = [0]
    all_mask_counts: List[int] = []

    for _b in range(batch_size):
        for _h in range(num_heads_kv):
            for m in range(num_m_blocks):
                if causal:
                    n_max = _causal_n_block_max(m, kBlockM, kBlockN,
                                                seqlen_q, seqlen_k, num_n_blocks)
                    threshold = _causal_threshold(m, kBlockM, kBlockN,
                                                  seqlen_q, seqlen_k)
                else:
                    n_max = num_n_blocks
                    threshold = 0

                # Descending order — matches FA3's original iteration direction
                n_blocks = list(range(n_max - 1, -1, -1))
                mask_count = sum(1 for nb in n_blocks if nb >= threshold)

                all_indices.extend(n_blocks)
                all_offsets.append(len(all_indices))
                all_mask_counts.append(mask_count)

    return (
        torch.tensor(all_indices, dtype=torch.int32, device=device),
        torch.tensor(all_offsets, dtype=torch.int32, device=device),
        torch.tensor(all_mask_counts, dtype=torch.int32, device=device),
    )


# --------------------------------------------------------------------------- #
#  2.  Block-mask → packed metadata
# --------------------------------------------------------------------------- #

def build_block_mask_metadata(
    block_mask: torch.Tensor,
    seqlen_q: int,
    seqlen_k: int,
    kBlockM: int = 128,
    kBlockN: int = 128,
    causal: bool = True,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert an explicit block-level boolean mask to packed sparse metadata.

    Args:
        block_mask: (B, H_kv, num_m_blocks, num_n_blocks) bool.
                    True = attend to this N-block, False = skip.
        seqlen_q, seqlen_k: actual sequence lengths (for causal threshold).
        kBlockM, kBlockN: tile sizes.
        causal: whether causal masking is in effect.
        device: target device.

    Returns:
        (sparse_n_indices, sparse_n_offsets, sparse_n_mask_counts)
    """
    B, H, M, N = block_mask.shape
    assert M == math.ceil(seqlen_q / kBlockM)
    assert N == math.ceil(seqlen_k / kBlockN)

    all_indices: List[int] = []
    all_offsets: List[int] = [0]
    all_mask_counts: List[int] = []

    mask_cpu = block_mask.cpu().bool()

    for b in range(B):
        for h in range(H):
            for m in range(M):
                # Gather active N-blocks in descending order
                active = [n for n in range(N - 1, -1, -1) if mask_cpu[b, h, m, n]]

                if causal:
                    threshold = _causal_threshold(m, kBlockM, kBlockN,
                                                  seqlen_q, seqlen_k)
                    mask_count = sum(1 for nb in active if nb >= threshold)
                else:
                    mask_count = 0

                all_indices.extend(active)
                all_offsets.append(len(all_indices))
                all_mask_counts.append(mask_count)

    return (
        torch.tensor(all_indices, dtype=torch.int32, device=device),
        torch.tensor(all_offsets, dtype=torch.int32, device=device),
        torch.tensor(all_mask_counts, dtype=torch.int32, device=device),
    )


# --------------------------------------------------------------------------- #
#  3.  IndexCache metadata → packed metadata
# --------------------------------------------------------------------------- #

def _decode_la_hot_tile_code(hot_bytes: torch.Tensor, num_pages: int) -> List[Tuple[int, int]]:
    """Decode bit-packed lower-triangular LA mask into list of (row_page, col_page) pairs.

    hot_bytes: 1-D uint8 tensor of length num_bytes for one (head, document).
    num_pages: number of pages in this document.

    Returns list of (row_page, col_page) where row_page >= col_page and bit=1.
    """
    if num_pages == 0:
        return []

    hot = hot_bytes.cpu().numpy()
    pairs = []
    tile_idx = 0
    # Lower-triangular enumeration: row-major, (0,0), (1,0), (1,1), (2,0), ...
    for row in range(num_pages):
        for col in range(row + 1):
            byte_idx = tile_idx // 8
            bit_pos = 7 - (tile_idx % 8)  # MSB-first
            if byte_idx < len(hot) and (hot[byte_idx] >> bit_pos) & 1:
                pairs.append((row, col))
            tile_idx += 1

    return pairs


def build_indexcache_metadata(
    seqlen_total: int,
    offset_list: List[int],
    question_len: int,
    kvcolidx_cache: torch.Tensor,
    la_hot_tile_code_cache: torch.Tensor,
    num_heads_kv: int,
    kBlockM: int = 128,
    kBlockN: int = 128,
    page_size: int = 128,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert IndexCache metadata to FA3 sparse packed arrays (per-Q-head).

    Args:
        seqlen_total: total sequence length (context + question) = seqlen_q = seqlen_k.
        offset_list: cumulative token lengths per document (NOT zero-prefixed).
                     E.g., [512, 1024, 1536] for 3 docs of 512 tokens each.
        question_len: number of question tokens appended after context.
        kvcolidx_cache: (1, H_q, total_kv_indices) int64 — global page indices,
                        -1=invalid.  H_q = number of Q heads.
                        Pages from non-last documents only.
        la_hot_tile_code_cache: (1, H_q, total_bytes) uint8 — bit-packed LA mask,
                                all docs.
        num_heads_kv: number of KV heads (unused — kept for API compat).
                      Metadata is emitted per-Q-head; the FA3 kernel indexes
                      by bidh (Q head) directly.
        kBlockM, kBlockN: FA3 tile sizes.
        page_size: IndexCache page size (should match kBlockN for 1:1 mapping).
        device: target device.

    Returns:
        (sparse_n_indices, sparse_n_offsets, sparse_n_mask_counts)
        Sized for B * H_q * num_m_blocks (per-Q-head, NOT per-KV-head).
    """
    assert page_size > 0
    seqlen_q = seqlen_total
    seqlen_k = seqlen_total
    context_len = seqlen_total - question_len

    num_heads_q = kvcolidx_cache.shape[1]

    # Document boundaries (in tokens, zero-prefixed)
    doc_token_starts = [0]
    for off in offset_list:
        doc_token_starts.append(off)
    num_docs = len(offset_list)

    # Document page boundaries
    doc_page_starts = [0]
    for d in range(num_docs):
        doc_len = doc_token_starts[d + 1] - doc_token_starts[d]
        doc_page_starts.append(doc_page_starts[-1] + math.ceil(doc_len / page_size))

    # LA byte boundaries per doc (for decoding hot_tile_code)
    la_byte_starts = [0]
    for d in range(num_docs):
        doc_len = doc_token_starts[d + 1] - doc_token_starts[d]
        np = math.ceil(doc_len / page_size)
        num_tiles = np * (np + 1) // 2
        num_bytes = (num_tiles + 7) // 8
        la_byte_starts.append(la_byte_starts[-1] + num_bytes)

    # kvcolidx boundaries per doc (non-last docs only)
    doc_lengths = []
    prev = 0
    for off in offset_list:
        doc_lengths.append(off - prev)
        prev = off

    kvcol_starts = [0]
    for d in range(num_docs - 1):
        np = math.ceil(doc_lengths[d] / page_size)
        kvcol_starts.append(kvcol_starts[-1] + np)

    num_m_blocks = math.ceil(seqlen_q / kBlockM)
    pages_per_n_block = max(1, kBlockN // page_size)
    n_blocks_per_page = max(1, page_size // kBlockN)

    all_indices: List[int] = []
    all_offsets: List[int] = [0]
    all_mask_counts: List[int] = []

    kvcolidx_cpu = kvcolidx_cache.cpu().long()  # (1, H_q, total_kv_indices)
    la_code_cpu = la_hot_tile_code_cache.cpu()    # (1, H_q, total_bytes)

    # Per-Q-head: iterate over all Q heads directly
    for h_q in range(num_heads_q):
        for m in range(num_m_blocks):
            q_start = m * kBlockM
            q_end = min((m + 1) * kBlockM, seqlen_q)
            q_mid = (q_start + q_end) // 2

            in_question = q_mid >= context_len

            doc_idx = -1
            if not in_question:
                for d in range(num_docs):
                    if q_mid < doc_token_starts[d + 1]:
                        doc_idx = d
                        break

            n_block_set = set()

            if in_question:
                causal_max = _causal_n_block_max(m, kBlockM, kBlockN,
                                                 seqlen_q, seqlen_k,
                                                 math.ceil(seqlen_k / kBlockN))
                for nb in range(causal_max):
                    n_block_set.add(nb)
            else:
                # --- CA: selected KV pages from docs before doc_idx ---
                if doc_idx > 0:
                    ca_start = kvcol_starts[0]
                    ca_end = kvcol_starts[min(doc_idx, len(kvcol_starts) - 1)]
                    for ci in range(ca_start, ca_end):
                        if ci < kvcolidx_cpu.shape[2]:
                            page_idx = kvcolidx_cpu[0, h_q, ci].item()
                            if page_idx >= 0:
                                if page_size == kBlockN:
                                    n_block_set.add(page_idx)
                                elif page_size < kBlockN:
                                    n_block_set.add(page_idx // pages_per_n_block)
                                else:
                                    for sub in range(n_blocks_per_page):
                                        nb = page_idx * n_blocks_per_page + sub
                                        n_block_set.add(nb)

                # --- LA: hot tiles within doc_idx ---
                doc_start_page = doc_page_starts[doc_idx]
                doc_num_pages = doc_page_starts[doc_idx + 1] - doc_page_starts[doc_idx]
                la_byte_start = la_byte_starts[doc_idx]
                la_byte_end = la_byte_starts[doc_idx + 1]
                la_bytes = la_code_cpu[0, h_q, la_byte_start:la_byte_end]

                hot_pairs = _decode_la_hot_tile_code(la_bytes, doc_num_pages)

                m_block_page_start = (q_start - doc_token_starts[doc_idx]) // page_size
                m_block_page_end = min(
                    (q_end - 1 - doc_token_starts[doc_idx]) // page_size + 1,
                    doc_num_pages,
                )

                for row_page, col_page in hot_pairs:
                    if m_block_page_start <= row_page < m_block_page_end:
                        global_page = doc_start_page + col_page
                        if page_size == kBlockN:
                            n_block_set.add(global_page)
                        elif page_size < kBlockN:
                            n_block_set.add(global_page // pages_per_n_block)
                        else:
                            for sub in range(n_blocks_per_page):
                                nb = global_page * n_blocks_per_page + sub
                                n_block_set.add(nb)

            # Filter to valid range and sort descending
            num_n_blocks = math.ceil(seqlen_k / kBlockN)
            n_blocks_desc = sorted(
                [nb for nb in n_block_set if 0 <= nb < num_n_blocks],
                reverse=True,
            )

            # Compute mask_count
            threshold = _causal_threshold(m, kBlockM, kBlockN, seqlen_q, seqlen_k)
            mask_count = 0
            for nb in n_blocks_desc:
                if nb >= threshold:
                    mask_count += 1
                else:
                    break

            all_indices.extend(n_blocks_desc)
            all_offsets.append(len(all_indices))
            all_mask_counts.append(mask_count)

    return (
        torch.tensor(all_indices, dtype=torch.int32, device=device),
        torch.tensor(all_offsets, dtype=torch.int32, device=device),
        torch.tensor(all_mask_counts, dtype=torch.int32, device=device),
    )


def build_causal_lower_half_metadata(
    batch_size: int,
    num_heads_kv: int,
    seqlen_q: int,
    seqlen_k: int,
    kBlockM: int = 128,
    kBlockN: int = 128,
    density: float = 0.5,
    device: str = "cuda",
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build sparse metadata that keeps random blocks in the causal lower triangle.

    Always keeps the diagonal N-block (the one containing the causal boundary)
    to ensure the output is non-zero for every query row.

    Returns:
        (sparse_n_indices, sparse_n_offsets, sparse_n_mask_counts, block_mask)

    block_mask: (B, H_kv, num_m_blocks, num_n_blocks) bool — for reference computation.
    """
    torch.manual_seed(seed)
    num_m_blocks = math.ceil(seqlen_q / kBlockM)
    num_n_blocks = math.ceil(seqlen_k / kBlockN)

    all_indices: List[int] = []
    all_offsets: List[int] = [0]
    all_mask_counts: List[int] = []
    block_mask = torch.zeros(batch_size, num_heads_kv, num_m_blocks, num_n_blocks, dtype=torch.bool)

    for b in range(batch_size):
        for h in range(num_heads_kv):
            for m in range(num_m_blocks):
                n_max = _causal_n_block_max(m, kBlockM, kBlockN,
                                            seqlen_q, seqlen_k, num_n_blocks)
                threshold = _causal_threshold(m, kBlockM, kBlockN,
                                              seqlen_q, seqlen_k)

                # Always include the causal-boundary block (n_max - 1)
                selected = set()
                if n_max > 0:
                    selected.add(n_max - 1)

                # Randomly include other blocks at the given density
                for nb in range(n_max):
                    if torch.rand(1).item() < density:
                        selected.add(nb)

                n_blocks_desc = sorted(selected, reverse=True)
                mask_count = sum(1 for nb in n_blocks_desc if nb >= threshold)

                for nb in n_blocks_desc:
                    block_mask[b, h, m, nb] = True

                all_indices.extend(n_blocks_desc)
                all_offsets.append(len(all_indices))
                all_mask_counts.append(mask_count)

    return (
        torch.tensor(all_indices, dtype=torch.int32, device=device),
        torch.tensor(all_offsets, dtype=torch.int32, device=device),
        torch.tensor(all_mask_counts, dtype=torch.int32, device=device),
        block_mask.to(device),
    )
