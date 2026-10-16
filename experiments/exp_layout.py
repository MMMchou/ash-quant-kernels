"""Experiment: is the AWQ checkpoint layout itself the bandwidth limiter?

Load-only kernels (no unpack, no dot) stream the packed int4 weight once, either
  rowmajor : AWQ layout [K, N/8]; a [BK, BN/8] tile = BK rows of BN/2 bytes each
  tiled    : repacked [N/BN, K/BK, BK*BN/8]; each tile is one contiguous chunk
             (the idea behind Marlin's offline repack in vLLM's awq_marlin)
and report achieved bandwidth under CUDA graphs.

Usage: python experiments/exp_layout.py
"""

from __future__ import annotations

import itertools

import torch
import triton
import triton.language as tl


@triton.jit
def _load_rowmajor(qw_ptr, out_ptr, K, NP, BK: tl.constexpr, BNP: tl.constexpr, SPLIT_K: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_k = tl.arange(0, BK)
    offs_pn = pid_n * BNP + tl.arange(0, BNP)
    acc = tl.zeros((BNP,), dtype=tl.int32)
    for k0 in range(pid_k * BK, K, BK * SPLIT_K):
        acc ^= tl.reduce(tl.load(qw_ptr + (k0 + offs_k)[:, None] * NP + offs_pn[None, :]), 0, _xor)
    tl.store(out_ptr + (pid_k * NP + offs_pn), acc)


@triton.jit
def _load_tiled(qw_ptr, out_ptr, K, NP, BK: tl.constexpr, BNP: tl.constexpr, SPLIT_K: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    kt_total = K // BK
    tile = tl.arange(0, BK)[:, None] * BNP + tl.arange(0, BNP)[None, :]
    acc = tl.zeros((BNP,), dtype=tl.int32)
    for kt in range(pid_k, kt_total, SPLIT_K):
        base = (pid_n * kt_total + kt) * (BK * BNP)
        acc ^= tl.reduce(tl.load(qw_ptr + base + tile), 0, _xor)
    tl.store(out_ptr + (pid_k * NP + pid_n * BNP + tl.arange(0, BNP)), acc)


@triton.jit
def _xor(a, b):
    return a ^ b


def repack_tiled(qw, bk, bnp):
    k, np_ = qw.shape
    return (
        qw.view(k // bk, bk, np_ // bnp, bnp).permute(2, 0, 1, 3).contiguous().view(-1)
    )


def main():
    print(f"device: {torch.cuda.get_device_name()}  (CUDA graph timing, load-only)")
    for name, n, k in [("qkv_proj", 6144, 4096), ("down_proj", 4096, 12288)]:
        np_ = n // 8
        qw = torch.randint(-2**31, 2**31 - 1, (k, np_), dtype=torch.int32, device="cuda")
        nbytes = qw.numel() * 4
        print(f"{name}: {nbytes / 1e6:.1f} MB")
        for bn, bk, sk, nw in itertools.product([64, 128, 256], [64, 128, 256], [1, 4, 8], [4, 8]):
            bnp = bn // 8
            out = torch.empty((sk, np_), dtype=torch.int32, device="cuda")
            grid = (np_ // bnp, sk)
            qt = repack_tiled(qw, bk, bnp)
            row = lambda: _load_rowmajor[grid](qw, out, k, np_, BK=bk, BNP=bnp, SPLIT_K=sk, num_warps=nw)
            til = lambda: _load_tiled[grid](qt, out, k, np_, BK=bk, BNP=bnp, SPLIT_K=sk, num_warps=nw)
            t_r = triton.testing.do_bench_cudagraph(row, rep=50) * 1e3
            t_t = triton.testing.do_bench_cudagraph(til, rep=50) * 1e3
            print(
                f"  BN={bn:3d} BK={bk:3d} SK={sk} W={nw}: rowmajor {t_r:5.1f} us "
                f"{nbytes / t_r / 1e3:5.0f} GB/s | tiled {t_t:5.1f} us {nbytes / t_t / 1e3:5.0f} GB/s"
            )


if __name__ == "__main__":
    main()
