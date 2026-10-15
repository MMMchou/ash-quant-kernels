"""Grid-search v2 launch configs per (layer, M); prints best config and speed vs fp16.

Usage: python bench/tune_v2.py [--ms 1,16,64,256] [--top 3]
"""

from __future__ import annotations

import argparse
import itertools

import torch
import triton

from ashqk.kernels.w4a16_gemm_v2 import w4a16_gemm_v2
from ashqk.ref import quantize_awq

SHAPES = [
    ("qkv_proj", 6144, 4096),
    ("o_proj", 4096, 4096),
    ("gate_up_proj", 24576, 4096),
    ("down_proj", 4096, 12288),
]


def space(m):
    bms = [16] if m <= 16 else [16, 32, 64, 128]
    for bm, bn, bk, sk, nw, ns in itertools.product(
        bms, [32, 64, 128, 256], [32, 64, 128], [1, 2, 4, 8], [4, 8], [2, 3, 4]
    ):
        if m > 64 and sk > 2:
            continue
        yield dict(block_m=bm, block_n=bn, block_k=bk, split_k=sk, num_warps=nw, num_stages=ns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", default="1,16,64,256")
    ap.add_argument("--top", type=int, default=3)
    args = ap.parse_args()
    for name, n, k in SHAPES:
        torch.manual_seed(0)
        qw, qz, s = quantize_awq(torch.randn(k, n, device="cuda") * 0.02, 128)
        w16 = torch.randn(k, n, device="cuda", dtype=torch.float16)
        for m in map(int, args.ms.split(",")):
            x = torch.randn(m, k, device="cuda", dtype=torch.float16)
            t16 = triton.testing.do_bench(lambda: x @ w16, warmup=10, rep=50) * 1e3
            res = []
            for cfg in space(m):
                try:
                    t = triton.testing.do_bench(
                        lambda: w4a16_gemm_v2(x, qw, qz, s, cfg), warmup=5, rep=20
                    ) * 1e3
                    res.append((t, cfg))
                except Exception:  # out of shared memory etc.
                    pass
            res.sort(key=lambda r: r[0])
            print(f"{name} M={m}: fp16 {t16:.1f} us")
            for t, cfg in res[: args.top]:
                print(f"    {t:7.1f} us  x{t16 / t:.2f} vs fp16  {cfg}")


if __name__ == "__main__":
    main()
