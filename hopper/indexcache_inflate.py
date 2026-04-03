"""
indexcache_inflate.py — v3

Converts compact IndexCache metadata → FA3 sparse packed arrays on GPU.
Processes ALL layers in a single kernel launch.

v3: kvcolidx is now uint8 bit-packed (MSB-first, byte-aligned per document),
    same format as la_hot_tile_code.

Public API:
    build_indexcache_metadata_gpu_all_layers(
        seqlen_total, offset_list, question_len,
        kvcolidx_per_layer,          # list[Tensor] or stacked Tensor (uint8)
        la_hot_tile_code_per_layer,  # list[Tensor] or stacked Tensor (uint8)
        device="cuda",
    ) -> list[(sparse_n_indices, sparse_n_offsets, sparse_n_mask_counts)]

Also exposes the single-layer drop-in:
    build_indexcache_metadata_gpu(...)  # same signature as old Python version
"""

import math
import os
from typing import List, Tuple, Union

import torch

# ---------------------------------------------------------------------------
_module = None

def _get_module():
    global _module
    if _module is not None:
        return _module
    from torch.utils.cpp_extension import load
    this_dir = os.path.dirname(os.path.abspath(__file__))
    _module = load(
        name="indexcache_inflate_cuda",
        sources=[os.path.join(this_dir, "indexcache_inflate.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17"],
        verbose=False,
    )
    return _module


# ---------------------------------------------------------------------------
# Helpers to build the small auxiliary tensors from offset_list
# ---------------------------------------------------------------------------

def _build_aux_tensors(offset_list, page_size, device):
    num_docs = len(offset_list)

    # Document token boundaries (cumulative → per-doc)
    doc_token_starts = [0]
    for off in offset_list:
        doc_token_starts.append(off)

    # doc_page_offsets: cumulative page counts (all docs)
    doc_page_list = [0]
    for d in range(num_docs):
        doc_len = doc_token_starts[d + 1] - doc_token_starts[d]
        doc_page_list.append(doc_page_list[-1] + math.ceil(doc_len / page_size))

    # la_byte_offsets: cumulative LA bytes per doc
    la_byte_list = [0]
    for d in range(num_docs):
        doc_len = doc_token_starts[d + 1] - doc_token_starts[d]
        np_ = math.ceil(doc_len / page_size)
        num_tiles = np_ * (np_ + 1) // 2
        la_byte_list.append(la_byte_list[-1] + (num_tiles + 7) // 8)

    # Per-doc page counts and byte offsets for kvcolidx (docs 0..N-2 only)
    doc_lengths = []
    prev = 0
    for off in offset_list:
        doc_lengths.append(off - prev)
        prev = off

    # kvcol_page_counts: real page count per CA doc (for masking padding bits)
    kvcol_page_counts_list = []
    for d in range(num_docs - 1):
        kvcol_page_counts_list.append(math.ceil(doc_lengths[d] / page_size))

    # kvcol_byte_offsets: cumulative byte offsets per CA doc
    kvcol_byte_list = [0]
    for d in range(num_docs - 1):
        n_pages = kvcol_page_counts_list[d]
        kvcol_byte_list.append(kvcol_byte_list[-1] + (n_pages + 7) // 8)

    doc_page_offsets    = torch.tensor(doc_page_list, dtype=torch.int32, device=device)
    la_byte_offsets     = torch.tensor(la_byte_list,  dtype=torch.int32, device=device)
    kvcol_byte_offsets  = torch.tensor(kvcol_byte_list, dtype=torch.int32, device=device)
    kvcol_page_counts   = torch.tensor(kvcol_page_counts_list, dtype=torch.int32, device=device)

    return doc_page_offsets, la_byte_offsets, kvcol_byte_offsets, kvcol_page_counts


# ---------------------------------------------------------------------------
# All-layers API (preferred)
# ---------------------------------------------------------------------------

def build_indexcache_metadata_gpu_all_layers(
    seqlen_total: int,
    offset_list: List[int],
    question_len: int,
    kvcolidx_per_layer: Union[List[torch.Tensor], torch.Tensor],
    la_hot_tile_code_per_layer: Union[List[torch.Tensor], torch.Tensor],
    device: str = "cuda",
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Process all layers in a single GPU kernel launch.

    Args:
        kvcolidx_per_layer: list of (1, H_q, kv_bytes) uint8 per layer, OR
                            pre-stacked (num_layers, H_q, kv_bytes) tensor.
        la_hot_tile_code_per_layer: list of (1, H_q, la_bytes) uint8, OR
                                    pre-stacked (num_layers, H_q, la_bytes).

    Returns:
        List of (sparse_n_indices, sparse_n_offsets, sparse_n_mask_counts)
        per layer, each sized for H_q * M.
    """
    PAGE_SIZE = 128
    kBlockN = 128

    # Stack layers into (num_layers, H_q, ...) if given as list
    if isinstance(kvcolidx_per_layer, list):
        kvcolidx_stacked = torch.stack(
            [t.squeeze(0) for t in kvcolidx_per_layer], dim=0
        ).to(device)
    else:
        kvcolidx_stacked = kvcolidx_per_layer.to(device)

    if isinstance(la_hot_tile_code_per_layer, list):
        la_stacked = torch.stack(
            [t.squeeze(0) for t in la_hot_tile_code_per_layer], dim=0
        ).to(device)
    else:
        la_stacked = la_hot_tile_code_per_layer.to(device)

    num_layers = kvcolidx_stacked.shape[0]
    H_q = kvcolidx_stacked.shape[1]
    M = math.ceil(seqlen_total / kBlockN)

    doc_page_offsets, la_byte_offsets, kvcol_byte_offsets, kvcol_page_counts = \
        _build_aux_tensors(offset_list, PAGE_SIZE, device)

    mod = _get_module()
    flat_indices, flat_offsets, flat_mask_counts = mod.build_indexcache_metadata_cuda(
        kvcolidx_stacked, la_stacked,
        doc_page_offsets, la_byte_offsets, kvcol_byte_offsets, kvcol_page_counts,
        num_layers, seqlen_total, question_len, kBlockN,
    )

    # Split flat outputs back into per-layer tuples.
    per_layer_size = H_q * M
    offsets_cpu = flat_offsets.cpu()

    result = []
    for li in range(num_layers):
        off_start = li * per_layer_size
        off_end   = (li + 1) * per_layer_size
        idx_start = offsets_cpu[off_start].item()
        idx_end   = offsets_cpu[off_end].item()
        layer_indices = flat_indices[idx_start:idx_end]
        layer_offsets = flat_offsets[off_start : off_end + 1] - idx_start
        layer_mask    = flat_mask_counts[off_start:off_end]
        result.append((layer_indices, layer_offsets, layer_mask))

    return result


# ---------------------------------------------------------------------------
# Single-layer drop-in (backward compat — calls all-layers with num_layers=1)
# ---------------------------------------------------------------------------

def build_indexcache_metadata_gpu(
    seqlen_total: int,
    offset_list: List[int],
    question_len: int,
    kvcolidx_cache: torch.Tensor,
    la_hot_tile_code_cache: torch.Tensor,
    device: str = "cuda",
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single-layer drop-in replacement for build_indexcache_metadata()."""
    results = build_indexcache_metadata_gpu_all_layers(
        seqlen_total, offset_list, question_len,
        [kvcolidx_cache], [la_hot_tile_code_cache],
        device=device,
    )
    return results[0]
