"""Grid-search launch configs per (layer, M); prints best config and speed vs fp16.

Usage: python bench/tune.py --model qwen3.5-9b --impl gemv --timer cudagraph [--ms 1,16] [--top 3]
"""

from __future__ import annotations

import argparse
import itertools

import os
import sys

import torch
import triton

sys.path.insert(0, os.path.dirname(__file__))
from shapes import DEFAULT_MODEL, MODELS  # noqa: E402

from ashqk.kernels.w4a16_gemm_v2 import w4a16_gemm_v2
from ashqk.kernels.w4a16_gemm_v3 import w4a16_gemm_v3
from ashqk.kernels.w4a16_gemv import w4a16_gemv

IMPLS = {"v2": w4a16_gemm_v2, "v3": w4a16_gemm_v3, "gemv": w4a16_gemv}
from ashqk.ref import quantize_awq



def space(m, impl):
    if impl == "gemv":
        for bm, bn, bk, sk, nw in itertools.product(
            sorted({1, 2, 4} & set(range(1, m + 1))), [32, 64, 128], [32, 64, 128], [4, 8, 16, 32], [2, 4]
        ):
            yield dict(block_m=bm, block_n=bn, block_k=bk, split_k=sk, num_warps=nw, num_stages=2)
        return
    bms = [16] if m <= 16 else [16, 32, 64, 128]
    bks = [32, 64, 128] if impl == "v2" else [64, 128, 256, 512]
    for bm, bn, bk, sk, nw, ns in itertools.product(
        bms, [32, 64, 128, 256], bks, [1, 2, 4, 8], [4, 8], [2, 3, 4]
    ):
        if m > 64 and sk > 2:
            continue
        yield dict(block_m=bm, block_n=bn, block_k=bk, split_k=sk, num_warps=nw, num_stages=ns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", default="1,16,64,256")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--impl", default="v3", choices=list(IMPLS))
    ap.add_argument("--timer", default="cudagraph", choices=["eager", "cudagraph"])
    ap.add_argument("--layers", default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS))
    args = ap.parse_args()
    fn = IMPLS[args.impl]

    def bench(f, rep):
        if args.timer == "cudagraph":
            return triton.testing.do_bench_cudagraph(f, rep=rep) * 1e3
        return triton.testing.do_bench(f, warmup=5, rep=rep) * 1e3

    for name, n, k, _ in MODELS[args.model]:
        if args.layers and name not in args.layers.split(","):
            continue
        torch.manual_seed(0)
        qw, qz, s = quantize_awq(torch.randn(k, n, device="cuda") * 0.02, 128)
        w16 = torch.randn(k, n, device="cuda", dtype=torch.float16)
        for m in map(int, args.ms.split(",")):
            x = torch.randn(m, k, device="cuda", dtype=torch.float16)
            t16 = bench(lambda: x @ w16, 50)
            res = []
            for cfg in space(m, args.impl):
                try:
                    t = bench(lambda: fn(x, qw, qz, s, cfg), 20)
                    res.append((t, cfg))
                except Exception:  # out of shared memory etc.
                    pass
            res.sort(key=lambda r: r[0])
            print(f"{name} M={m} [{args.impl}, {args.timer}]: fp16 {t16:.1f} us", flush=True)
            for t, cfg in res[: args.top]:
                print(f"    {t:7.1f} us  x{t16 / t:.2f} vs fp16  {cfg}")


if __name__ == "__main__":
    main()
