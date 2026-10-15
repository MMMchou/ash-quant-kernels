"""W4A16 fused dequant GEMM in Triton -- v1 (naive, correctness first).

y[M, N] = x[M, K] @ dequant(qweight[K, N/8], qzeros[K/G, N/8], scales[K/G, N])

v1 design (deliberately simple; optimizations are tracked for v0.2):
  * One program computes a [BLOCK_M, BLOCK_N] tile of y, looping over K.
  * BLOCK_K divides group_size, so each K-tile uses exactly one row of scales/zeros.
  * Each of the BLOCK_N columns gathers its own int32 (`offs_n // 8`), i.e. the same
    packed word is loaded 8 times (served from L1). v0.2: load once + tl.interleave.
  * Dequant in registers: (q - z) * s -> fp16, then tl.dot with fp32 accumulator.
    (vLLM's awq_gemm_triton accumulates in fp16; we use fp32 for accuracy.)
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _w4a16_gemm_kernel(
    x_ptr, qw_ptr, qz_ptr, s_ptr, y_ptr,
    M, N, K, group_size,
    stride_xm, stride_xk,
    stride_qwk, stride_qwn,
    stride_qzg, stride_qzn,
    stride_sg, stride_sn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_n = offs_n < N

    # AWQ: column j of an 8-column block lives in nibble (j % 2) * 4 + j // 2.
    j = offs_n % 8
    shifts = ((j % 2) * 4 + j // 2) * 4          # [BLOCK_N]
    offs_pn = offs_n // 8                        # packed column index

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        ks = k0 + offs_k
        mask_k = ks < K

        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + ks[None, :] * stride_xk,
            mask=mask_m[:, None] & mask_k[None, :], other=0.0,
        )  # [BLOCK_M, BLOCK_K] fp16

        qw = tl.load(
            qw_ptr + ks[:, None] * stride_qwk + offs_pn[None, :] * stride_qwn,
            mask=mask_k[:, None] & mask_n[None, :], other=0,
        )  # [BLOCK_K, BLOCK_N] int32 (each word repeated 8x across N)
        q = (qw >> shifts[None, :]) & 0xF

        g = k0 // group_size
        qz = tl.load(qz_ptr + g * stride_qzg + offs_pn * stride_qzn, mask=mask_n, other=0)
        z = (qz >> shifts) & 0xF                                     # [BLOCK_N]
        s = tl.load(s_ptr + g * stride_sg + offs_n * stride_sn, mask=mask_n, other=0.0)

        w = ((q - z[None, :]).to(tl.float32) * s[None, :].to(tl.float32)).to(tl.float16)
        acc = tl.dot(x, w, acc)

    tl.store(
        y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
        acc.to(y_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_n[None, :],
    )


def w4a16_gemm(
    x: torch.Tensor,
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    block_m: int = 16,
    block_n: int = 64,
    block_k: int = 64,
    num_warps: int = 4,
    num_stages: int = 3,
) -> torch.Tensor:
    """x: [M, K] fp16; AWQ tensors as in ashqk.ref. Returns y: [M, N] fp16."""
    assert x.dtype == torch.float16 and x.dim() == 2
    M, K = x.shape
    N = qweight.shape[1] * 8
    group_size = K // scales.shape[0]
    assert qweight.shape == (K, N // 8)
    assert qzeros.shape == (K // group_size, N // 8)
    assert scales.shape == (K // group_size, N)
    block_k = min(block_k, group_size)  # one scale row per K-tile
    assert group_size % block_k == 0 and K % group_size == 0

    y = torch.empty((M, N), dtype=x.dtype, device=x.device)
    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)
    _w4a16_gemm_kernel[grid](
        x, qweight, qzeros, scales, y,
        M, N, K, group_size,
        x.stride(0), x.stride(1),
        qweight.stride(0), qweight.stride(1),
        qzeros.stride(0), qzeros.stride(1),
        scales.stride(0), scales.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        num_warps=num_warps, num_stages=num_stages,
    )
    return y
