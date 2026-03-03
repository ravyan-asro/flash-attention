"""
Test script for FA3 sparse attention.

Test 1: Dense sparse metadata (all N-blocks) must match standard FA3.
Test 2: Sparse attention with a known block mask must match naive PyTorch reference.

Usage:
    CUDA_VISIBLE_DEVICES=2 python test_sparse_attn_fa3.py
"""

import sys
import os
import math

import torch
import torch.nn.functional as F

# Add the hopper directory to the path so we can import FA3
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flash_attn_interface import _flash_attn_forward
from sparse_indexcache_utils import (
    build_dense_sparse_metadata,
    build_causal_lower_half_metadata,
)


def naive_attention_with_block_mask(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask: torch.Tensor,
    kBlockM: int,
    kBlockN: int,
    causal: bool = True,
    softmax_scale: float = None,
) -> torch.Tensor:
    """
    Naive PyTorch attention with a block-level mask.

    q: (B, seqlen_q, H, d)
    k: (B, seqlen_k, H_kv, d)
    v: (B, seqlen_k, H_kv, d)
    block_mask: (B, H_kv, num_m_blocks, num_n_blocks) bool

    Returns: (B, seqlen_q, H, d)
    """
    B, seqlen_q, H, d = q.shape
    _, seqlen_k, H_kv, _ = k.shape
    assert H % H_kv == 0
    gqa_ratio = H // H_kv

    if softmax_scale is None:
        softmax_scale = d ** (-0.5)

    num_m_blocks = math.ceil(seqlen_q / kBlockM)
    num_n_blocks = math.ceil(seqlen_k / kBlockN)

    # Expand KV heads to match Q heads
    k_expanded = k.unsqueeze(3).expand(B, seqlen_k, H_kv, gqa_ratio, d).reshape(B, seqlen_k, H, d)
    v_expanded = v.unsqueeze(3).expand(B, seqlen_k, H_kv, gqa_ratio, d).reshape(B, seqlen_k, H, d)

    # Build token-level mask from block mask
    # block_mask is per KV-head; expand to per Q-head
    # (B, H_kv, M, N) -> (B, H, seqlen_q, seqlen_k)
    token_mask = torch.zeros(B, H, seqlen_q, seqlen_k, dtype=torch.bool, device=q.device)

    for b in range(B):
        for h_kv in range(H_kv):
            for m in range(num_m_blocks):
                for n in range(num_n_blocks):
                    if block_mask[b, h_kv, m, n]:
                        q_start = m * kBlockM
                        q_end = min((m + 1) * kBlockM, seqlen_q)
                        k_start = n * kBlockN
                        k_end = min((n + 1) * kBlockN, seqlen_k)
                        for h_q in range(h_kv * gqa_ratio, (h_kv + 1) * gqa_ratio):
                            token_mask[b, h_q, q_start:q_end, k_start:k_end] = True

    # Apply causal mask on top of block mask
    if causal:
        causal_mask = torch.ones(seqlen_q, seqlen_k, dtype=torch.bool, device=q.device)
        for i in range(seqlen_q):
            for j in range(seqlen_k):
                if j > i + (seqlen_k - seqlen_q):
                    causal_mask[i, j] = False
        token_mask = token_mask & causal_mask.unsqueeze(0).unsqueeze(0)

    # Compute attention
    # q: (B, seqlen_q, H, d) -> (B, H, seqlen_q, d)
    q_t = q.transpose(1, 2).float()
    k_t = k_expanded.transpose(1, 2).float()
    v_t = v_expanded.transpose(1, 2).float()

    scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * softmax_scale  # (B, H, seqlen_q, seqlen_k)

    # Apply mask: -inf where mask is False
    scores = scores.masked_fill(~token_mask, float('-inf'))

    attn = torch.softmax(scores, dim=-1)
    attn = attn.masked_fill(torch.isnan(attn), 0.0)  # Handle all-masked rows

    out = torch.matmul(attn, v_t)  # (B, H, seqlen_q, d)
    out = out.transpose(1, 2)  # (B, seqlen_q, H, d)

    return out.to(q.dtype)


def test_dense_sparse_matches_standard():
    """Test 1: Sparse FA3 with full-density metadata matches standard FA3."""
    print("=" * 60)
    print("Test 1: Dense sparse metadata matches standard FA3")
    print("=" * 60)

    B, seqlen, H, H_kv, d = 1, 512, 32, 8, 128
    kBlockM = kBlockN = 128

    torch.manual_seed(42)
    q = torch.randn(B, seqlen, H, d, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B, seqlen, H_kv, d, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B, seqlen, H_kv, d, dtype=torch.bfloat16, device="cuda")

    softmax_scale = d ** (-0.5)

    # Standard FA3 (no sparse)
    out_ref, lse_ref, _, _ = _flash_attn_forward(
        q, k, v,
        softmax_scale=softmax_scale,
        causal=True,
    )

    # Sparse FA3 with full density
    indices, offsets, mask_counts = build_dense_sparse_metadata(
        B, H_kv, seqlen, seqlen,
        kBlockM=kBlockM, kBlockN=kBlockN,
        causal=True, device="cuda",
    )

    print(f"  Sparse metadata: {indices.shape[0]} total indices, "
          f"{offsets.shape[0] - 1} entries, "
          f"mask_counts range [{mask_counts.min().item()}, {mask_counts.max().item()}]")

    out_sparse, lse_sparse, _, _ = _flash_attn_forward(
        q, k, v,
        softmax_scale=softmax_scale,
        causal=True,
        sparse_n_indices=indices,
        sparse_n_offsets=offsets,
        sparse_n_mask_counts=mask_counts,
    )

    # Compare
    max_diff_out = (out_ref.float() - out_sparse.float()).abs().max().item()
    max_diff_lse = (lse_ref.float() - lse_sparse.float()).abs().max().item()

    print(f"  Output max diff: {max_diff_out:.6e}")
    print(f"  LSE max diff:    {max_diff_lse:.6e}")

    # BF16 has ~1e-2 precision, but these should be bit-for-bit since
    # the sparse path iterates the exact same blocks in the same order
    passed = max_diff_out < 1e-5 and max_diff_lse < 1e-5
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_dense_sparse_larger():
    """Test 1b: Larger sequence length."""
    print("\n" + "=" * 60)
    print("Test 1b: Dense sparse, seqlen=1024, GQA")
    print("=" * 60)

    B, seqlen, H, H_kv, d = 2, 1024, 32, 8, 128
    kBlockM = kBlockN = 128

    torch.manual_seed(123)
    q = torch.randn(B, seqlen, H, d, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B, seqlen, H_kv, d, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B, seqlen, H_kv, d, dtype=torch.bfloat16, device="cuda")

    softmax_scale = d ** (-0.5)

    out_ref, lse_ref, _, _ = _flash_attn_forward(
        q, k, v,
        softmax_scale=softmax_scale,
        causal=True,
    )

    indices, offsets, mask_counts = build_dense_sparse_metadata(
        B, H_kv, seqlen, seqlen,
        kBlockM=kBlockM, kBlockN=kBlockN,
        causal=True, device="cuda",
    )

    out_sparse, lse_sparse, _, _ = _flash_attn_forward(
        q, k, v,
        softmax_scale=softmax_scale,
        causal=True,
        sparse_n_indices=indices,
        sparse_n_offsets=offsets,
        sparse_n_mask_counts=mask_counts,
    )

    max_diff_out = (out_ref.float() - out_sparse.float()).abs().max().item()
    max_diff_lse = (lse_ref.float() - lse_sparse.float()).abs().max().item()

    print(f"  Output max diff: {max_diff_out:.6e}")
    print(f"  LSE max diff:    {max_diff_lse:.6e}")

    passed = max_diff_out < 1e-5 and max_diff_lse < 1e-5
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_sparse_vs_naive():
    """Test 2: Sparse FA3 with random block mask matches naive PyTorch reference."""
    print("\n" + "=" * 60)
    print("Test 2: Sparse FA3 vs naive PyTorch reference")
    print("=" * 60)

    B, seqlen, H, H_kv, d = 1, 512, 8, 8, 128
    kBlockM = kBlockN = 128
    density = 0.7

    torch.manual_seed(42)
    q = torch.randn(B, seqlen, H, d, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B, seqlen, H_kv, d, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B, seqlen, H_kv, d, dtype=torch.bfloat16, device="cuda")

    softmax_scale = d ** (-0.5)

    # Build random sparse metadata
    indices, offsets, mask_counts, block_mask = build_causal_lower_half_metadata(
        B, H_kv, seqlen, seqlen,
        kBlockM=kBlockM, kBlockN=kBlockN,
        density=density, device="cuda", seed=42,
    )

    num_total_blocks = block_mask.sum().item()
    num_possible = sum(
        math.ceil(((m + 1) * kBlockM) / kBlockN)
        for m in range(math.ceil(seqlen / kBlockM))
    ) * B * H_kv
    actual_density = num_total_blocks / num_possible if num_possible > 0 else 0
    print(f"  Block mask: {num_total_blocks}/{num_possible} blocks "
          f"({actual_density:.1%} density)")

    # Sparse FA3
    out_sparse, lse_sparse, _, _ = _flash_attn_forward(
        q, k, v,
        softmax_scale=softmax_scale,
        causal=True,
        sparse_n_indices=indices,
        sparse_n_offsets=offsets,
        sparse_n_mask_counts=mask_counts,
    )

    # Naive reference
    out_naive = naive_attention_with_block_mask(
        q, k, v, block_mask,
        kBlockM=kBlockM, kBlockN=kBlockN,
        causal=True, softmax_scale=softmax_scale,
    )

    max_diff = (out_sparse.float() - out_naive.float()).abs().max().item()
    mean_diff = (out_sparse.float() - out_naive.float()).abs().mean().item()

    print(f"  Output max diff:  {max_diff:.6e}")
    print(f"  Output mean diff: {mean_diff:.6e}")

    # BF16 matmul accumulation differences — tolerance needs to be higher
    passed = max_diff < 0.05 and mean_diff < 0.01
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_sparse_single_block():
    """Test 3: Sparse attention with only diagonal blocks (each m_block attends to only its own n_block)."""
    print("\n" + "=" * 60)
    print("Test 3: Diagonal-only sparse attention")
    print("=" * 60)

    B, seqlen, H, H_kv, d = 1, 512, 8, 8, 128
    kBlockM = kBlockN = 128

    torch.manual_seed(42)
    q = torch.randn(B, seqlen, H, d, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B, seqlen, H_kv, d, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B, seqlen, H_kv, d, dtype=torch.bfloat16, device="cuda")

    softmax_scale = d ** (-0.5)

    num_m_blocks = math.ceil(seqlen / kBlockM)
    num_n_blocks = math.ceil(seqlen / kBlockN)

    # Build diagonal-only block mask
    block_mask = torch.zeros(B, H_kv, num_m_blocks, num_n_blocks, dtype=torch.bool, device="cuda")
    for m in range(num_m_blocks):
        block_mask[:, :, m, m] = True

    # Build sparse metadata manually
    all_indices = []
    all_offsets = [0]
    all_mask_counts = []

    for b in range(B):
        for h in range(H_kv):
            for m in range(num_m_blocks):
                all_indices.append(m)  # only the diagonal block
                all_offsets.append(len(all_indices))
                all_mask_counts.append(1)  # the diagonal block always needs masking

    indices = torch.tensor(all_indices, dtype=torch.int32, device="cuda")
    offsets = torch.tensor(all_offsets, dtype=torch.int32, device="cuda")
    mask_counts = torch.tensor(all_mask_counts, dtype=torch.int32, device="cuda")

    # Sparse FA3
    out_sparse, _, _, _ = _flash_attn_forward(
        q, k, v,
        softmax_scale=softmax_scale,
        causal=True,
        sparse_n_indices=indices,
        sparse_n_offsets=offsets,
        sparse_n_mask_counts=mask_counts,
    )

    # Naive reference
    out_naive = naive_attention_with_block_mask(
        q, k, v, block_mask,
        kBlockM=kBlockM, kBlockN=kBlockN,
        causal=True, softmax_scale=softmax_scale,
    )

    max_diff = (out_sparse.float() - out_naive.float()).abs().max().item()
    mean_diff = (out_sparse.float() - out_naive.float()).abs().mean().item()

    print(f"  Output max diff:  {max_diff:.6e}")
    print(f"  Output mean diff: {mean_diff:.6e}")

    passed = max_diff < 0.05 and mean_diff < 0.01
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed


if __name__ == "__main__":
    torch.cuda.set_device(0)  # Use first visible GPU
    print(f"Running on GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA compute capability: {torch.cuda.get_device_capability()}")
    print()

    results = []

    try:
        results.append(("Dense sparse == standard FA3", test_dense_sparse_matches_standard()))
    except Exception as e:
        print(f"  FAILED with exception: {e}")
        import traceback; traceback.print_exc()
        results.append(("Dense sparse == standard FA3", False))

    try:
        results.append(("Dense sparse larger", test_dense_sparse_larger()))
    except Exception as e:
        print(f"  FAILED with exception: {e}")
        import traceback; traceback.print_exc()
        results.append(("Dense sparse larger", False))

    try:
        results.append(("Sparse vs naive reference", test_sparse_vs_naive()))
    except Exception as e:
        print(f"  FAILED with exception: {e}")
        import traceback; traceback.print_exc()
        results.append(("Sparse vs naive reference", False))

    try:
        results.append(("Diagonal-only sparse", test_sparse_single_block()))
    except Exception as e:
        print(f"  FAILED with exception: {e}")
        import traceback; traceback.print_exc()
        results.append(("Diagonal-only sparse", False))

    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("All tests passed!")
    else:
        print("Some tests FAILED!")
        sys.exit(1)
