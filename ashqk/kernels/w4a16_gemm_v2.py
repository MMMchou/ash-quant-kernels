"""W4A16 fused dequant GEMM in Triton -- v2 (load-once unpack + split-K).

Changes vs v1 (see results/v0.1-naive-findings.md for why):
  1. Load-once: the B tile is loaded as [BLOCK_K, BLOCK_N // 8] int32 (0.5 B/weight)
     and expanded to [BLOCK_K, BLOCK_N] in registers with 3x tl.interleave
     (w0 w1 ... -> w0 w0 w0 w0 w0 w0 w0 w0 w1 ...). v1 gathered each word 8x.
  2. Split-K: grid axis 1 splits the K loop. For decode (M <= 16) there are only
     N / BLOCK_N output tiles (e.g. 96 for N=6144), fewer than H200's 132 SMs;
     split-K multiplies the parallelism. Partial sums are atomically added in fp32.
  3. Dequant in fp16: (q - z) is an exact small integer in fp16 and fp16 multiply is
     correctly rounded, so (q - z) * s in fp16 is bit-identical to fp32-then-round.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _w4a16_gemm_v2_kernel(
    x_ptr, qw_ptr, qz_ptr, s_ptr, y_ptr,
    M, N, K, group_size,
    stride_xm, stride_xk,
    stride_qwk, stride_qwn,
    stride_qzg, stride_qzn,
    stride_sg, stride_sn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_pn = pid_n * (BLOCK_N // 8) + tl.arange(0, BLOCK_N // 8)  # packed columns
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask_pn = offs_pn < N // 8

    # After 3x interleave, tile column c holds word c // 8, nibble for column c % 8.
    # AWQ: column j of an 8-block lives in nibble (j % 2) * 4 + j // 2.
    j = tl.arange(0, BLOCK_N) % 8
    shifts = ((j % 2) * 4 + j // 2) * 4

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(pid_k * BLOCK_K, K, BLOCK_K * SPLIT_K):
        ks = k0 + offs_k
        mask_k = ks < K

        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + ks[None, :] * stride_xk,
            mask=mask_m[:, None] & mask_k[None, :], other=0.0,
        )

        qw = tl.load(
            qw_ptr + ks[:, None] * stride_qwk + offs_pn[None, :] * stride_qwn,
            mask=mask_k[:, None] & mask_pn[None, :], other=0,
        )  # [BLOCK_K, BLOCK_N // 8]
        qw = tl.interleave(qw, qw)
        qw = tl.interleave(qw, qw)
        qw = tl.interleave(qw, qw)  # [BLOCK_K, BLOCK_N]
        q = (qw >> shifts[None, :]) & 0xF

        g = k0 // group_size
        qz = tl.load(qz_ptr + g * stride_qzg + offs_pn * stride_qzn, mask=mask_pn, other=0)
        qz = tl.interleave(qz, qz)
        qz = tl.interleave(qz, qz)
        qz = tl.interleave(qz, qz)
        z = (qz >> shifts) & 0xF
        s = tl.load(s_ptr + g * stride_sg + offs_n * stride_sn, mask=mask_n, other=0.0)

        w = (q - z[None, :]).to(tl.float16) * s[None, :]
        acc = tl.dot(x, w, acc)

    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    mask_y = mask_m[:, None] & mask_n[None, :]
    if SPLIT_K == 1:
        tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=mask_y)
    else:
        tl.atomic_add(y_ptrs, acc, mask=mask_y)


def _default_config(M: int) -> dict:
    # From bench/tune_v2.py on H200 (do_bench timing). Split-K wins under CUDA graphs
    # but not yet with eager launches (extra zero-init + cast kernels); see results/.
    if M <= 16:
        return dict(block_m=16, block_n=64, block_k=128, split_k=1, num_warps=4, num_stages=4)
    if M <= 64:
        return dict(block_m=64, block_n=64, block_k=128, split_k=1, num_warps=4, num_stages=3)
    return dict(block_m=128, block_n=128, block_k=64, split_k=1, num_warps=8, num_stages=3)


def w4a16_gemm_v2(x, qweight, qzeros, scales, config: dict | None = None) -> torch.Tensor:
    """x: [M, K] fp16; AWQ tensors as in ashqk.ref. Returns y: [M, N] fp16."""
    assert x.dtype == torch.float16 and x.dim() == 2
    M, K = x.shape
    N = qweight.shape[1] * 8
    group_size = K // scales.shape[0]
    assert qweight.shape == (K, N // 8)
    assert qzeros.shape == (K // group_size, N // 8)
    assert scales.shape == (K // group_size, N)

    cfg = dict(_default_config(M), **(config or {}))
    block_k = min(cfg["block_k"], group_size)  # one scale row per K-tile
    assert group_size % block_k == 0 and K % group_size == 0
    assert cfg["block_n"] % 8 == 0
    split_k = cfg["split_k"]

    if split_k == 1:
        y = torch.empty((M, N), dtype=x.dtype, device=x.device)
    else:
        y = torch.zeros((M, N), dtype=torch.float32, device=x.device)
    grid = (triton.cdiv(M, cfg["block_m"]) * triton.cdiv(N, cfg["block_n"]), split_k)
    _w4a16_gemm_v2_kernel[grid](
        x, qweight, qzeros, scales, y,
        M, N, K, group_size,
        x.stride(0), x.stride(1),
        qweight.stride(0), qweight.stride(1),
        qzeros.stride(0), qzeros.stride(1),
        scales.stride(0), scales.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=cfg["block_m"], BLOCK_N=cfg["block_n"], BLOCK_K=block_k,
        SPLIT_K=split_k,
        num_warps=cfg["num_warps"], num_stages=cfg["num_stages"],
    )
    return y if split_k == 1 else y.to(x.dtype)
