"""
Benchmark: Dense FA3 vs Sparse IndexCache FA3 on Llama3-8B config.

Simulates a realistic IndexCache sparsity pattern:
  - N document chunks with dense local attention (lower-triangular within each doc)
  - Cross-attention (CA) tiles: ~15% of pages from earlier docs selected per doc
  - A final question region with full causal attention
  - Target ~60% overall sparsity (i.e., ~40% of blocks are attended)

Llama3-8B config: B=1, H=32, H_kv=8, d=128, seqlen=32000

Usage:
    CUDA_VISIBLE_DEVICES=2 python bench_sparse_fa3.py
"""

import sys
import os
import math
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flash_attn_interface import _flash_attn_forward
from sparse_indexcache_utils import _causal_n_block_max, _causal_threshold


def build_indexcache_style_metadata(
    batch_size: int,
    num_heads_kv: int,
    seqlen: int,
    doc_lengths: list,
    question_len: int,
    ca_density: float = 0.15,
    la_density: float = 0.50,
    kBlockM: int = 128,
    kBlockN: int = 128,
    device: str = "cuda",
    seed: int = 42,
):
    """
    Build a realistic IndexCache-style sparsity pattern.

    Layout: [doc0 | doc1 | ... | docN-1 | question]
    - Within each doc: lower-triangular local attention with la_density
      (always includes diagonal blocks + first-column blocks for streaming)
    - Between docs: cross-attention from later docs to earlier docs at ca_density
    - Question region: full causal attention to everything before

    Returns:
        (sparse_n_indices, sparse_n_offsets, sparse_n_mask_counts,
         total_blocks, total_possible_blocks)
    """
    torch.manual_seed(seed)
    assert sum(doc_lengths) + question_len == seqlen

    num_m_blocks = math.ceil(seqlen / kBlockM)
    num_n_blocks = math.ceil(seqlen / kBlockN)

    # Document boundaries in tokens
    doc_starts = [0]
    for dl in doc_lengths:
        doc_starts.append(doc_starts[-1] + dl)
    num_docs = len(doc_lengths)
    context_len = doc_starts[-1]
    # question starts at context_len

    # Document boundaries in blocks
    doc_block_starts = [ds // kBlockN for ds in doc_starts]
    doc_block_ends = [math.ceil(ds / kBlockN) for ds in doc_starts[1:]]
    question_block_start = context_len // kBlockN

    all_indices = []
    all_offsets = [0]
    all_mask_counts = []
    total_blocks = 0
    total_possible = 0

    for b in range(batch_size):
        for h in range(num_heads_kv):
            for m in range(num_m_blocks):
                q_start = m * kBlockM
                q_mid = q_start + kBlockM // 2
                if q_mid >= seqlen:
                    q_mid = seqlen - 1

                # Causal n_block_max
                n_max = _causal_n_block_max(m, kBlockM, kBlockN, seqlen, seqlen, num_n_blocks)
                threshold = _causal_threshold(m, kBlockM, kBlockN, seqlen, seqlen)
                total_possible += n_max

                selected = set()

                if q_mid >= context_len:
                    # --- Question region: full causal attention ---
                    for nb in range(n_max):
                        selected.add(nb)
                else:
                    # --- Context region ---
                    # Find which document this m_block belongs to
                    doc_idx = -1
                    for d in range(num_docs):
                        if q_mid < doc_starts[d + 1]:
                            doc_idx = d
                            break

                    # Local attention within this document (lower-triangular)
                    doc_n_start = doc_block_starts[doc_idx]
                    doc_n_end = min(doc_block_ends[doc_idx], n_max)

                    for nb in range(doc_n_start, doc_n_end):
                        if nb > m:
                            continue  # causal: can't attend to future blocks
                        if nb == m:
                            # Diagonal block: always include
                            selected.add(nb)
                        elif nb == doc_n_start:
                            # First column of this doc: always include (streaming sink)
                            selected.add(nb)
                        elif torch.rand(1).item() < la_density:
                            # Random LA tile
                            selected.add(nb)

                    # Cross-attention to earlier documents
                    for d_prev in range(doc_idx):
                        prev_start = doc_block_starts[d_prev]
                        prev_end = doc_block_ends[d_prev]
                        for nb in range(prev_start, prev_end):
                            if torch.rand(1).item() < ca_density:
                                selected.add(nb)

                # Filter to valid causal range and sort descending
                n_blocks_desc = sorted(
                    [nb for nb in selected if 0 <= nb < n_max],
                    reverse=True,
                )

                mask_count = sum(1 for nb in n_blocks_desc if nb >= threshold)
                total_blocks += len(n_blocks_desc)

                all_indices.extend(n_blocks_desc)
                all_offsets.append(len(all_indices))
                all_mask_counts.append(mask_count)

    density = total_blocks / total_possible if total_possible > 0 else 0

    return (
        torch.tensor(all_indices, dtype=torch.int32, device=device),
        torch.tensor(all_offsets, dtype=torch.int32, device=device),
        torch.tensor(all_mask_counts, dtype=torch.int32, device=device),
        total_blocks,
        total_possible,
        density,
    )


def benchmark_kernel(fn, warmup=5, repeats=20):
    """Benchmark a kernel function with CUDA events."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Timed runs
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]

    for i in range(repeats):
        start_events[i].record()
        fn()
        end_events[i].record()
    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    times.sort()
    # Drop top and bottom 2 for stability
    if len(times) > 4:
        times = times[2:-2]
    avg_ms = sum(times) / len(times)
    return avg_ms


def main():
    # Llama3-8B config
    B = 1
    seqlen = 32000
    H = 32       # query heads
    H_kv = 8     # KV heads (GQA 4:1)
    d = 128      # head dim
    kBlockM = kBlockN = 128

    # Document layout: 8 docs of 3750 tokens each + 2000 token question = 32000
    num_docs = 8
    doc_len = 3750
    question_len = seqlen - num_docs * doc_len
    doc_lengths = [doc_len] * num_docs

    print(f"Llama3-8B config: B={B}, seqlen={seqlen}, H={H}, H_kv={H_kv}, d={d}")
    print(f"Layout: {num_docs} docs x {doc_len} tokens + {question_len} token question")
    print(f"kBlockM={kBlockM}, kBlockN={kBlockN}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    # Create tensors
    torch.manual_seed(42)
    q = torch.randn(B, seqlen, H, d, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B, seqlen, H_kv, d, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B, seqlen, H_kv, d, dtype=torch.bfloat16, device="cuda")
    softmax_scale = d ** (-0.5)

    # Build sparse metadata
    print("Building IndexCache-style sparse metadata...")
    (indices, offsets, mask_counts,
     total_blocks, total_possible, density) = build_indexcache_style_metadata(
        B, H_kv, seqlen, doc_lengths, question_len,
        ca_density=0.15,
        la_density=0.50,
        kBlockM=kBlockM, kBlockN=kBlockN,
        device="cuda",
    )

    sparsity = 1.0 - density
    print(f"  Total blocks: {total_blocks} / {total_possible} "
          f"({density:.1%} density, {sparsity:.1%} sparsity)")
    print(f"  Sparse metadata: {indices.shape[0]} indices, "
          f"{offsets.shape[0] - 1} (B*H_kv*M) entries")
    print()

    # --- Benchmark dense FA3 ---
    print("Benchmarking dense FA3 (standard)...")
    def run_dense():
        return _flash_attn_forward(
            q, k, v,
            softmax_scale=softmax_scale,
            causal=True,
        )

    dense_ms = benchmark_kernel(run_dense)

    # Compute TFLOPS for dense causal attention
    # FLOPs = 2 * B * H * seqlen_q * seqlen_k * d (for QK) + 2 * B * H * seqlen_q * seqlen_k * d (for PV)
    # But causal → approximately half the FLOPs
    flops_dense = 2 * 2 * B * H * seqlen * seqlen * d * 0.5  # causal factor
    tflops_dense = flops_dense / (dense_ms * 1e-3) / 1e12

    print(f"  Dense FA3:  {dense_ms:.2f} ms  ({tflops_dense:.1f} TFLOPS)")
    print()

    # --- Benchmark sparse FA3 ---
    print("Benchmarking sparse FA3 (IndexCache-style)...")
    def run_sparse():
        return _flash_attn_forward(
            q, k, v,
            softmax_scale=softmax_scale,
            causal=True,
            sparse_n_indices=indices,
            sparse_n_offsets=offsets,
            sparse_n_mask_counts=mask_counts,
        )

    sparse_ms = benchmark_kernel(run_sparse)

    # Approximate FLOPS for sparse: scale by density
    flops_sparse = flops_dense * density
    tflops_sparse = flops_sparse / (sparse_ms * 1e-3) / 1e12

    print(f"  Sparse FA3: {sparse_ms:.2f} ms  ({tflops_sparse:.1f} TFLOPS)")
    print()

    # --- Results ---
    speedup = dense_ms / sparse_ms
    print("=" * 60)
    print(f"Results (seqlen={seqlen}, {sparsity:.0%} sparsity):")
    print(f"  Dense:  {dense_ms:.2f} ms")
    print(f"  Sparse: {sparse_ms:.2f} ms")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Dense TFLOPS:  {tflops_dense:.1f}")
    print(f"  Sparse TFLOPS: {tflops_sparse:.1f}")
    print("=" * 60)

    # --- Correctness sanity check ---
    print("\nSanity check: sparse output is finite...")
    out_sparse, _, _, _ = run_sparse()
    assert torch.isfinite(out_sparse).all(), "Sparse output contains non-finite values!"
    print("  OK - all outputs finite")

    # Check that sparse != all zeros (meaningful computation happened)
    print(f"  Output norm: {out_sparse.float().norm().item():.2f}")
    print(f"  Output abs max: {out_sparse.float().abs().max().item():.4f}")


if __name__ == "__main__":
    torch.cuda.set_device(0)
    main()
