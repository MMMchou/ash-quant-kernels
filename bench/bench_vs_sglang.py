"""Head-to-head vs production kernels used by SGLang 0.5.9 on Hopper for AWQ checkpoints.

Providers (same AWQ tensors in, same fp16 activations):
  fp16          cuBLAS on the dequantized weight
  awq_marlin    sgl_kernel.gptq_marlin_gemm after SGLang's own AWQ->Marlin repack
                (exactly AWQMarlinLinearMethod.process_weights_after_loading + apply)
  sgl_awq       sgl_kernel.awq_dequantize + matmul (SGLang `--quantization awq` on CUDA)
  ashqk         ashqk.w4a16_linear

Needs an environment with sglang + sgl_kernel installed (separate venv on our H200).
Usage: python bench/bench_vs_sglang.py [--model qwen3.5-9b] [--ms 1,2,4,8,16] [--out results/x.md]
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

from ashqk import w4a16_linear  # noqa: E402
from ashqk.ref import dequantize_awq, quantize_awq, w4a16_gemm_fp32  # noqa: E402

GROUP_SIZE = 128


def make_marlin(qw, qz, s, k, n):
    from sgl_kernel import awq_marlin_repack
    from sglang.srt.layers.quantization.marlin_utils import (
        apply_awq_marlin_linear,
        awq_to_marlin_zero_points,
        marlin_make_empty_g_idx,
        marlin_make_workspace,
        marlin_permute_scales,
    )
    from sglang.srt.layers.quantization.utils import get_scalar_types

    _, scalar_types = get_scalar_types()
    dev = qw.device
    mq = awq_marlin_repack(qw, size_k=k, size_n=n, num_bits=4)
    ms = marlin_permute_scales(s, size_k=k, size_n=n, group_size=GROUP_SIZE)
    mz = awq_to_marlin_zero_points(qz, size_k=k // GROUP_SIZE, size_n=n, num_bits=4)
    ws = marlin_make_workspace(dev)
    g = marlin_make_empty_g_idx(dev)
    gs = marlin_make_empty_g_idx(dev)

    def run(x):
        return apply_awq_marlin_linear(
            x, mq, ms, mz, g, gs, ws, scalar_types.uint4,
            output_size_per_partition=n, input_size_per_partition=k,
        )

    return run


def make_sgl_awq(qw, qz, s):
    from sgl_kernel import awq_dequantize

    def run(x):
        return x @ awq_dequantize(qw, s, qz)

    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS))
    ap.add_argument("--ms", default="1,2,4,8,16")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    import sglang
    import sgl_kernel

    header = (
        f"# vs SGLang production kernels\n\n"
        f"- model shapes: {args.model}; device: {torch.cuda.get_device_name()}\n"
        f"- sglang {sglang.__version__}, sgl_kernel {sgl_kernel.__version__}, torch {torch.__version__}, "
        f"triton {triton.__version__}; date {dt.date.today()}\n"
        f"- timer: triton.testing.do_bench_cudagraph (L2 flushed); group_size {GROUP_SIZE}\n"
        f"- rel_err: vs fp32 ground truth on the same dequantized weights (cuBLAS fp16 shown for scale)\n\n"
        "| layer | N | K | M | fp16 | awq_marlin | sgl_awq | ashqk | ashqk / marlin | err marlin | err ashqk | err fp16 |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    totals = {}
    for name, n, k, nlayers in MODELS[args.model]:
        torch.manual_seed(0)
        qw, qz, s = quantize_awq(torch.randn(k, n, device="cuda") * 0.02, GROUP_SIZE)
        w16 = dequantize_awq(qw, qz, s, dtype=torch.float16)
        marlin = make_marlin(qw, qz, s, k, n)
        sgl_awq = make_sgl_awq(qw, qz, s)
        for m in map(int, args.ms.split(",")):
            x = torch.randn(m, k, device="cuda", dtype=torch.float16)
            yt = w4a16_gemm_fp32(x, qw, qz, s)

            def err(y):
                return ((y.float() - yt).norm() / yt.norm()).item()

            fns = {
                "fp16": lambda: x @ w16,
                "awq_marlin": lambda: marlin(x),
                "sgl_awq": lambda: sgl_awq(x),
                "ashqk": lambda: w4a16_linear(x, qw, qz, s),
            }
            t = {p: triton.testing.do_bench_cudagraph(f, rep=100) * 1e3 for p, f in fns.items()}
            e = {p: err(fns[p]()) for p in ("awq_marlin", "ashqk", "fp16")}
            for p in t:
                totals[(m, p)] = totals.get((m, p), 0.0) + t[p] * nlayers
            row = (
                f"| {name} | {n} | {k} | {m} | {t['fp16']:.1f} | {t['awq_marlin']:.1f} | {t['sgl_awq']:.1f} "
                f"| {t['ashqk']:.1f} | {t['awq_marlin'] / t['ashqk']:.2f}× | {e['awq_marlin']:.1e} "
                f"| {e['ashqk']:.1e} | {e['fp16']:.1e} |"
            )
            rows.append(row)
            print(row, flush=True)

    tot = ["", "## All linear layers per token (sum over layers), us", "",
           "| M | fp16 | awq_marlin | sgl_awq | ashqk | ashqk / marlin |", "|---|---|---|---|---|---|"]
    for m in map(int, args.ms.split(",")):
        v = {p: totals[(m, p)] for p in ("fp16", "awq_marlin", "sgl_awq", "ashqk")}
        tot.append(f"| {m} | {v['fp16']:.0f} | {v['awq_marlin']:.0f} | {v['sgl_awq']:.0f} | {v['ashqk']:.0f} "
                   f"| {v['awq_marlin'] / v['ashqk']:.2f}× |")
    print("\n".join(tot))
    if args.out:
        with open(args.out, "w") as f:
            f.write(header + "\n".join(rows) + "\n" + "\n".join(tot) + "\n")


if __name__ == "__main__":
    main()
