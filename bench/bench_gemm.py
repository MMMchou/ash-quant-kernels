"""Benchmark W4A16 GEMM on real LLM layer shapes (bench/shapes.py; default Qwen3.5-9B).

Providers:
  fp16       cuBLAS fp16 GEMM with an already-dequantized W (the fp16 model; upper bound
             for "no quantization" and the bar W4A16 must beat in decode)
  dequant+mm dequantize to HBM then cuBLAS (the naive unfused W4A16 path)
  ashqk-v1   naive fused Triton kernel
  ashqk-v2   load-once unpack (+ split-K)
  ashqk-v3   multi-group K tiles + single-launch split-K (default configs)
  ashqk      public w4a16_linear dispatcher (GEMV for M <= 2, else v3)

Metrics: latency (us), TFLOPS, effective bandwidth = bytes actually required / time,
where bytes = x + packed weights + scales + zeros + y (W4A16) or x + fp16 W + y (fp16).

Usage: python bench/bench_gemm.py [--model qwen3.5-9b|qwen3-8b] [--timer cudagraph|eager] [--out results/<name>.md]
"""

from __future__ import annotations

import argparse
import datetime as dt

import os
import sys

import torch
import triton

sys.path.insert(0, os.path.dirname(__file__))
from shapes import DEFAULT_MODEL, MODELS  # noqa: E402

from ashqk.kernels.w4a16_gemm import w4a16_gemm
from ashqk.kernels.w4a16_gemm_v2 import w4a16_gemm_v2
from ashqk.kernels.w4a16_gemm_v3 import w4a16_gemm_v3
from ashqk.ops import w4a16_linear
from ashqk.ref import dequantize_awq, quantize_awq, w4a16_gemm_ref

MS = [1, 16, 64, 256, 1024, 4096]
GROUP_SIZE = 128

H200_HBM_TBPS = 4.8      # spec sheet
H200_FP16_TFLOPS = 989   # dense, spec sheet


def w4a16_bytes(m, n, k, g):
    return m * k * 2 + k * n // 2 + (k // g) * n * 2 + (k // g) * n // 2 + m * n * 2


def fp16_bytes(m, n, k):
    return m * k * 2 + k * n * 2 + m * n * 2


def bench_ms(fn, timer):
    if timer == "cudagraph":  # how vLLM / SGLang run decode: no per-launch CPU overhead
        return triton.testing.do_bench_cudagraph(fn, rep=100)
    return triton.testing.do_bench(fn, warmup=25, rep=100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--providers", default="fp16,dequant+mm,ashqk-v3,ashqk")
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS))
    ap.add_argument("--timer", default="cudagraph", choices=["eager", "cudagraph"])
    ap.add_argument("--ms", default=",".join(map(str, MS)))
    args = ap.parse_args()
    providers = args.providers.split(",")

    dev = torch.cuda.get_device_name()
    header = (
        f"# W4A16 GEMM benchmark\n\n"
        f"- model shapes: {args.model}\n"
        f"- device: {dev}; torch {torch.__version__}; triton {triton.__version__}\n"
        f"- date: {dt.date.today()}; group_size {GROUP_SIZE}; timer: {args.timer}\n"
        f"- BW% = effective bandwidth / {H200_HBM_TBPS} TB/s\n\n"
        "| layer | N | K | M | provider | us | TFLOPS | GB/s | BW% |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for name, n, k, _ in MODELS[args.model]:
        torch.manual_seed(0)
        w = torch.randn(k, n, device="cuda") * 0.02
        qw, qz, s = quantize_awq(w, GROUP_SIZE)
        w16 = dequantize_awq(qw, qz, s, dtype=torch.float16)
        for m in map(int, args.ms.split(",")):
            x = torch.randn(m, k, device="cuda", dtype=torch.float16)
            fns = {
                "fp16": (lambda: x @ w16, fp16_bytes(m, n, k)),
                "dequant+mm": (lambda: w4a16_gemm_ref(x, qw, qz, s), w4a16_bytes(m, n, k, GROUP_SIZE)),
                "ashqk-v1": (lambda: w4a16_gemm(x, qw, qz, s), w4a16_bytes(m, n, k, GROUP_SIZE)),
                "ashqk-v2": (lambda: w4a16_gemm_v2(x, qw, qz, s), w4a16_bytes(m, n, k, GROUP_SIZE)),
                "ashqk-v3": (lambda: w4a16_gemm_v3(x, qw, qz, s), w4a16_bytes(m, n, k, GROUP_SIZE)),
                "ashqk": (lambda: w4a16_linear(x, qw, qz, s), w4a16_bytes(m, n, k, GROUP_SIZE)),
            }
            for p in providers:
                fn, nbytes = fns[p]
                ms = bench_ms(fn, args.timer)
                tflops = 2 * m * n * k / ms / 1e9
                gbps = nbytes / ms / 1e6
                row = (
                    f"| {name} | {n} | {k} | {m} | {p} | {ms * 1e3:.1f} | {tflops:.1f} "
                    f"| {gbps:.0f} | {gbps / (H200_HBM_TBPS * 1e3) * 100:.0f}% |"
                )
                rows.append(row)
                print(row, flush=True)
        del w, w16, qw, qz, s
        torch.cuda.empty_cache()

    report = header + "\n".join(rows) + "\n"
    if args.out:
        with open(args.out, "w") as f:
            f.write(report)
        print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
