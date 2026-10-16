"""W4A16 decode GEMV in Triton (small M, no tensor cores).

Why a separate path: at M = 1 a tl.dot tile with BLOCK_M = 16 wastes 15/16 of the MMA, and
results/v0.2-step2-findings.md showed the cost is converting unpacked weights into the
dot-operand layout -- not HBM bandwidth.

Two ideas:
  1. No interleave. The packed tile P[BK, BN/8] is expanded by *broadcasting* to
     [BK, BN/8, 8] with per-nibble shifts (pure elementwise, no cross-thread movement).
     Element (p, i) is output column 8p + AWQ_ORDER[i]; for fixed p the 8 columns are
     8p..8p+7 in permuted order, so scales/output loads stay contiguous.
  2. Factor the dequant out of the K reduction, per quantization group g:
         sum_{k in g} x_k (q_kn - z_gn) s_gn = s_gn * (sum_k x_k q_kn  -  z_gn * sum_k x_k)
     so the inner loop is one multiply-add per weight; zero/scale are applied once per group.

One program handles BLOCK_M rows (weights loaded once, reused across rows -- matters for
speculative-decoding verification, M = 4..8) and BLOCK_N output columns; split-K uses the
same single-launch fixup as v3.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _w4a16_gemv_kernel(
    x_ptr, qw_ptr, qz_ptr, s_ptr, y_ptr, ws_ptr, cnt_ptr,
    M, N, K, group_size,
    stride_xm, stride_xk,
    stride_qwk, stride_qwn,
    stride_qzg, stride_qzn,
    stride_sg, stride_sn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_ROWS: tl.constexpr, SPLIT_K: tl.constexpr,
):
    NUM_G: tl.constexpr = BLOCK_K // GROUP_ROWS
    BNP: tl.constexpr = BLOCK_N // 8
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    num_pid_n = tl.cdiv(N, BLOCK_N)

    offs_pn = pid_n * BNP + tl.arange(0, BNP)
    mask_pn = offs_pn < N // 8
    nib = tl.arange(0, 8)
    shifts = nib * 4                                   # nibble i
    col_in_blk = (nib % 4) * 2 + nib // 4              # AWQ_ORDER[i]
    cols = offs_pn[:, None] * 8 + col_in_blk[None, :]  # [BNP, 8]
    mask_c = mask_pn[:, None] & (cols < N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_g = tl.arange(0, NUM_G)

    acc = tl.zeros((BLOCK_M, BNP, 8), dtype=tl.float32)
    for k0 in range(pid_k * BLOCK_K, K, BLOCK_K * SPLIT_K):
        ks = k0 + offs_k
        mask_k = ks < K
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + ks[None, :] * stride_xk,
            mask=mask_m[:, None] & mask_k[None, :], other=0.0,
        ).to(tl.float32)                                               # [BM, BK]
        qw = tl.load(
            qw_ptr + ks[:, None] * stride_qwk + offs_pn[None, :] * stride_qwn,
            mask=mask_k[:, None] & mask_pn[None, :], other=0,
        )                                                              # [BK, BNP]
        q = ((qw[:, :, None] >> shifts[None, None, :]) & 0xF).to(tl.float32)  # [BK, BNP, 8]

        # [BM, BK] x [BK, BNP*8] as broadcast-multiply + reduce within each group.
        xg = tl.reshape(x, (BLOCK_M, NUM_G, GROUP_ROWS))
        xsum = tl.sum(xg, axis=2)                                      # [BM, NUM_G]
        q4 = tl.reshape(q, (NUM_G, GROUP_ROWS, BNP * 8))
        qx = tl.sum(q4[None, :, :, :] * xg[:, :, :, None], axis=2)     # [BM, NUM_G, BNP*8]
        qx = tl.reshape(qx, (BLOCK_M, NUM_G, BNP, 8))

        gs = k0 // group_size + offs_g
        mask_g = gs < K // group_size
        qz = tl.load(
            qz_ptr + gs[:, None] * stride_qzg + offs_pn[None, :] * stride_qzn,
            mask=mask_g[:, None] & mask_pn[None, :], other=0,
        )
        z = ((qz[:, :, None] >> shifts[None, None, :]) & 0xF).to(tl.float32)   # [NUM_G, BNP, 8]
        s = tl.load(
            s_ptr + gs[:, None, None] * stride_sg + cols[None, :, :] * stride_sn,
            mask=mask_g[:, None, None] & mask_c[None, :, :], other=0.0,
        ).to(tl.float32)                                               # [NUM_G, BNP, 8]
        acc += tl.sum((qx - z[None] * xsum[:, :, None, None]) * s[None], axis=1)

    mask_y = mask_m[:, None, None] & mask_c[None, :, :]
    y_ptrs = y_ptr + offs_m[:, None, None] * stride_ym + cols[None, :, :] * stride_yn
    if SPLIT_K == 1:
        tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=mask_y)
    else:
        ws_ptrs = ws_ptr + offs_m[:, None, None] * N + cols[None, :, :]
        tl.atomic_add(ws_ptrs, acc, mask=mask_y, sem="release")
        tl.debug_barrier()
        tile = pid_m * num_pid_n + pid_n
        ticket = tl.atomic_add(cnt_ptr + tile, 1, sem="acq_rel")
        if ticket == SPLIT_K - 1:
            tot = tl.load(ws_ptrs, mask=mask_y, other=0.0, cache_modifier=".cg")
            tl.store(y_ptrs, tot.to(y_ptr.dtype.element_ty), mask=mask_y)
            tl.store(ws_ptrs, tl.zeros((BLOCK_M, BNP, 8), dtype=tl.float32), mask=mask_y)
            tl.store(cnt_ptr + tile, 0)


_WORKSPACE: dict = {}


def _workspace(m, n, num_tiles, device):
    key = (device, m, n, num_tiles)
    if key not in _WORKSPACE:
        _WORKSPACE[key] = (
            torch.zeros((m, n), dtype=torch.float32, device=device),
            torch.zeros((num_tiles,), dtype=torch.int32, device=device),
        )
    return _WORKSPACE[key]


def _default_config(M: int) -> dict:
    # Tuned on H200, CUDA-graph timing (results/v0.2-step3-findings.md).
    if M == 1:
        return dict(block_m=1, block_n=128, block_k=128, split_k=16, num_warps=2, num_stages=2)
    return dict(block_m=2, block_n=128, block_k=64, split_k=16, num_warps=2, num_stages=2)


def w4a16_gemv(x, qweight, qzeros, scales, config: dict | None = None) -> torch.Tensor:
    """x: [M, K] fp16 with small M (decode); AWQ tensors as in ashqk.ref. Returns [M, N] fp16."""
    assert x.dtype == torch.float16 and x.dim() == 2
    M, K = x.shape
    N = qweight.shape[1] * 8
    group_size = K // scales.shape[0]
    assert qweight.shape == (K, N // 8)
    assert qzeros.shape == (K // group_size, N // 8)
    assert scales.shape == (K // group_size, N)
    assert K % group_size == 0

    cfg = dict(_default_config(M), **(config or {}))
    block_k = cfg["block_k"]
    group_rows = min(block_k, group_size)
    assert group_size % group_rows == 0 and block_k % group_rows == 0, (block_k, group_size)
    assert cfg["block_n"] % 8 == 0
    split_k = cfg["split_k"]

    block_m = cfg["block_m"]
    num_pid_n = triton.cdiv(N, cfg["block_n"])
    num_pid_m = triton.cdiv(M, block_m)
    y = torch.empty((M, N), dtype=x.dtype, device=x.device)
    ws, cnt = _workspace(M, N, num_pid_m * num_pid_n, x.device) if split_k > 1 else (y, y)
    _w4a16_gemv_kernel[(num_pid_n, split_k, num_pid_m)](
        x, qweight, qzeros, scales, y, ws, cnt,
        M, N, K, group_size,
        x.stride(0), x.stride(1),
        qweight.stride(0), qweight.stride(1),
        qzeros.stride(0), qzeros.stride(1),
        scales.stride(0), scales.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=block_m, BLOCK_N=cfg["block_n"], BLOCK_K=block_k,
        GROUP_ROWS=group_rows, SPLIT_K=split_k,
        num_warps=cfg["num_warps"], num_stages=cfg["num_stages"],
    )
    return y
