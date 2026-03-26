"""
test_indexcache_inflate.py — v2

Validates the CUDA indexcache_inflate kernel (all-layers) against the Python reference.

Usage:
    cd /var/tmp/rsanovar3/flash-attention/hopper
    python test_indexcache_inflate.py
"""

import math
import time
import torch

from sparse_indexcache_utils import build_indexcache_metadata
from indexcache_inflate import build_indexcache_metadata_gpu_all_layers


def make_synthetic_indexcache(
    num_docs, doc_len, question_len, H_q, num_layers=4,
    page_size=128, ca_selectivity=0.1, la_selectivity=0.4, seed=42,
):
    torch.manual_seed(seed)
    offset_list = [doc_len * (d + 1) for d in range(num_docs)]
    seqlen_total = offset_list[-1] + question_len
    pages_per_doc = math.ceil(doc_len / page_size)
    total_kv_entries = (num_docs - 1) * pages_per_doc

    # Create per-layer metadata (different random patterns per layer)
    kv_per_layer = []
    la_per_layer = []

    total_la_bytes = 0
    for d in range(num_docs):
        np_ = pages_per_doc
        num_tiles = np_ * (np_ + 1) // 2
        total_la_bytes += (num_tiles + 7) // 8

    for layer in range(num_layers):
        kvcolidx = torch.full((1, H_q, total_kv_entries), -1, dtype=torch.long, device="cuda")
        for d in range(num_docs - 1):
            doc_page_start = d * pages_per_doc
            for h in range(H_q):
                for p in range(pages_per_doc):
                    if torch.rand(1).item() < ca_selectivity:
                        kvcolidx[0, h, d * pages_per_doc + p] = doc_page_start + p
        kv_per_layer.append(kvcolidx)

        la_code = torch.zeros((1, H_q, total_la_bytes), dtype=torch.uint8, device="cuda")
        byte_offset = 0
        for d in range(num_docs):
            np_ = pages_per_doc
            num_tiles = np_ * (np_ + 1) // 2
            num_bytes = (num_tiles + 7) // 8
            for h in range(H_q):
                tile_idx = 0
                for row in range(np_):
                    for col in range(row + 1):
                        is_hot = (row == col) or (col == 0) or (torch.rand(1).item() < la_selectivity)
                        if is_hot:
                            bi = tile_idx >> 3
                            bit_pos = 7 - (tile_idx & 7)
                            la_code[0, h, byte_offset + bi] |= (1 << bit_pos)
                        tile_idx += 1
            byte_offset += num_bytes
        la_per_layer.append(la_code)

    return seqlen_total, offset_list, question_len, kv_per_layer, la_per_layer


def run_test(num_docs, doc_len, question_len, H_q, num_layers=4, page_size=128, label=None):
    if label is None:
        seqlen = num_docs * doc_len + question_len
        label = f"docs={num_docs}, doc_len={doc_len}, H_q={H_q}, L={num_layers}, total={seqlen}"

    print(f"\nTest: {label}")

    seqlen_total, offset_list, q_len, kv_per_layer, la_per_layer = \
        make_synthetic_indexcache(num_docs, doc_len, question_len, H_q, num_layers, page_size)

    # Python reference (per-layer)
    t0 = time.time()
    ref_results = []
    for li in range(num_layers):
        ref = build_indexcache_metadata(
            seqlen_total, offset_list, q_len,
            kv_per_layer[li], la_per_layer[li],
            num_heads_kv=H_q, kBlockM=page_size, kBlockN=page_size, page_size=page_size,
        )
        ref_results.append(ref)
    t_py = time.time() - t0

    # GPU all-layers kernel
    torch.cuda.synchronize()
    t0 = time.time()
    gpu_results = build_indexcache_metadata_gpu_all_layers(
        seqlen_total, offset_list, q_len,
        kv_per_layer, la_per_layer,
    )
    torch.cuda.synchronize()
    t_gpu = time.time() - t0

    # Compare
    all_match = True
    for li in range(num_layers):
        ref_idx, ref_off, ref_mc = ref_results[li]
        gpu_idx, gpu_off, gpu_mc = gpu_results[li]

        ok_off = torch.equal(ref_off.cpu(), gpu_off.cpu())
        ok_mc  = torch.equal(ref_mc.cpu(), gpu_mc.cpu())
        ok_idx = torch.equal(ref_idx.cpu(), gpu_idx.cpu())

        if not (ok_off and ok_mc and ok_idx):
            all_match = False
            if li < 2:  # only verbose for first 2 failing layers
                print(f"  Layer {li}: FAIL (off={ok_off}, mc={ok_mc}, idx={ok_idx})")
                if not ok_off:
                    # Find which m-blocks have different counts (diff consecutive offsets)
                    ref_counts = ref_off[1:].cpu() - ref_off[:-1].cpu()
                    gpu_counts = gpu_off[1:].cpu() - gpu_off[:-1].cpu()
                    diff_m = (ref_counts != gpu_counts).nonzero(as_tuple=True)[0]
                    for dm in diff_m[:5]:
                        m_idx = dm.item()
                        print(f"    m_block={m_idx}: ref_count={ref_counts[m_idx].item()}, gpu_count={gpu_counts[m_idx].item()}")
            if not ok_idx and ref_idx.shape == gpu_idx.shape:
                d = (ref_idx.cpu() != gpu_idx.cpu()).nonzero(as_tuple=True)[0]
                i = d[0].item()
                print(f"    indices[{i}]: ref={ref_idx[i].item()}, gpu={gpu_idx[i].item()}")

    status = "PASS" if all_match else "FAIL"
    print(f"  [{status}] {label}")
    print(f"  Python: {t_py*1000:.1f}ms, CUDA: {t_gpu*1000:.3f}ms, speedup: {t_py/max(t_gpu,1e-9):.0f}x")
    return all_match


if __name__ == "__main__":
    print("=" * 60)
    print("IndexCache Inflate v2: All-Layers CUDA vs Python Reference")
    print("=" * 60)

    ok = True
    ok &= run_test(3, 256, 128, H_q=4, num_layers=4)
    ok &= run_test(5, 1024, 256, H_q=8, num_layers=8)
    ok &= run_test(10, 3200, 2048, H_q=32, num_layers=32, label="32K ctx, 32 heads, 32 layers")
    ok &= run_test(1, 1024, 128, H_q=4, num_layers=4, label="single doc")
    ok &= run_test(2, 512, 128, H_q=4, num_layers=4, label="2 docs")
    # MiniMax-like: aligned doc_len (must be multiple of 128)
    ok &= run_test(26, 1280, 69, H_q=40, num_layers=62, label="MiniMax-like aligned: 26 docs, 40h, 62L")
    # MiniMax-like: non-aligned doc_len
    ok &= run_test(26, 1228, 69, H_q=40, num_layers=62, label="MiniMax-like unaligned: 26 docs, 40h, 62L")

    print("\n" + "=" * 60)
    print(f"Overall: {'ALL PASSED' if ok else 'SOME FAILED'}")
    print("=" * 60)
