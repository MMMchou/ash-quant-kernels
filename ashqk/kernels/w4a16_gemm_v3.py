"""W4A16 fused dequant GEMM in Triton -- v3 (bigger K tiles + single-launch split-K).

Changes vs v2 (see results/v0.2-step1-findings.md for why):
  1. BLOCK_K may span several quantization groups (BLOCK_K = NUM_G * group_size), so each
     CTA keeps more bytes in flight per iteration. Scales/zeros are loaded as [NUM_G, BN]
     and broadcast over the group's rows via a [NUM_G, GROUP_ROWS, BN] view.
  2. Split-K in a single launch ("serial split-K fixup", as in CUTLASS): partial sums are
     atomically added into a persistent fp32 workspace; a per-tile arrival counter tells
     the last CTA to cast the tile to fp16, write y, and reset workspace + counter to zero.
     v2 needed two extra kernels (zero-init, fp32->fp16 cast) per call.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _w4a16_gemm_v3_kernel(
    x_ptr, qw_ptr, qz_ptr, s_ptr, y_ptr, ws_ptr, cnt_ptr,
    M, N, K, group_size,
    stride_xm, stride_xk,
    stride_qwk, stride_qwn,
    stride_qzg, stride_qzn,
    stride_sg, stride_sn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_ROWS: tl.constexpr,  # rows of a K-tile that share one scale row = min(BLOCK_K, G)
    SPLIT_K: tl.constexpr,
):
    NUM_G: tl.constexpr = BLOCK_K // GROUP_ROWS
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_pn = pid_n * (BLOCK_N // 8) + tl.arange(0, BLOCK_N // 8)
    offs_k = tl.arange(0, BLOCK_K)
    offs_g = tl.arange(0, NUM_G)
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask_pn = offs_pn < N // 8

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
        )
        qw = tl.interleave(qw, qw)
        qw = tl.interleave(qw, qw)
        qw = tl.interleave(qw, qw)
        q = (qw >> shifts[None, :]) & 0xF                            # [BK, BN]

        gs = k0 // group_size + offs_g                               # [NUM_G]
        mask_g = gs < K // group_size
        qz = tl.load(
            qz_ptr + gs[:, None] * stride_qzg + offs_pn[None, :] * stride_qzn,
            mask=mask_g[:, None] & mask_pn[None, :], other=0,
        )
        qz = tl.interleave(qz, qz)
        qz = tl.interleave(qz, qz)
        qz = tl.interleave(qz, qz)
        z = (qz >> shifts[None, :]) & 0xF                            # [NUM_G, BN]
        s = tl.load(
            s_ptr + gs[:, None] * stride_sg + offs_n[None, :] * stride_sn,
            mask=mask_g[:, None] & mask_n[None, :], other=0.0,
        )                                                            # [NUM_G, BN]

        q3 = tl.reshape(q, (NUM_G, GROUP_ROWS, BLOCK_N))
        w3 = (q3 - z[:, None, :]).to(tl.float16) * s[:, None, :]
        w = tl.reshape(w3, (BLOCK_K, BLOCK_N))
        acc = tl.dot(x, w, acc)

    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    mask_y = mask_m[:, None] & mask_n[None, :]
    if SPLIT_K == 1:
        tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=mask_y)
    else:
        # Workspace is [M, N] fp32, contiguous.
        ws_ptrs = ws_ptr + offs_m[:, None] * N + offs_n[None, :]
        tl.atomic_add(ws_ptrs, acc, mask=mask_y, sem="release")
        tl.debug_barrier()  # all threads' partials issued before the arrival ticket
        ticket = tl.atomic_add(cnt_ptr + pid, 1, sem="acq_rel")
        if ticket == SPLIT_K - 1:
            # Last arriver: read the reduced tile from L2 (bypass L1), emit fp16, reset.
            tot = tl.load(ws_ptrs, mask=mask_y, other=0.0, cache_modifier=".cg")
            tl.store(y_ptrs, tot.to(y_ptr.dtype.element_ty), mask=mask_y)
            tl.store(ws_ptrs, tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32), mask=mask_y)
            tl.store(cnt_ptr + pid, 0)


_WORKSPACE: dict = {}


def _workspace(m, n, num_tiles, device):
    """Persistent zeroed fp32 workspace + counters; kernels leave them zeroed on exit."""
    key = (device, m, n, num_tiles)
    if key not in _WORKSPACE:
        _WORKSPACE[key] = (
            torch.zeros((m, n), dtype=torch.float32, device=device),
            torch.zeros((num_tiles,), dtype=torch.int32, device=device),
        )
    return _WORKSPACE[key]


def _default_config(M: int) -> dict:
    # Tuned on H200 with CUDA-graph timing (bench/tune.py); see results/v0.2-step2-findings.md.
    if M <= 16:
        return dict(block_m=16, block_n=128, block_k=64, split_k=8, num_warps=4, num_stages=2)
    if M <= 64:
        return dict(block_m=64, block_n=64, block_k=128, split_k=1, num_warps=4, num_stages=3)
    return dict(block_m=128, block_n=128, block_k=64, split_k=1, num_warps=8, num_stages=3)


def w4a16_gemm_v3(x, qweight, qzeros, scales, config: dict | None = None) -> torch.Tensor:
    """x: [M, K] fp16; AWQ tensors as in ashqk.ref. Returns y: [M, N] fp16.

    Note: split-K > 1 shares a cached workspace per (M, N); calls on the same device must
    be stream-ordered (true for normal PyTorch usage and CUDA graphs).
    """
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

    num_tiles = triton.cdiv(M, cfg["block_m"]) * triton.cdiv(N, cfg["block_n"])
    y = torch.empty((M, N), dtype=x.dtype, device=x.device)
    if split_k > 1:
        ws, cnt = _workspace(M, N, num_tiles, x.device)
    else:
        ws, cnt = y, y  # unused
    _w4a16_gemm_v3_kernel[(num_tiles, split_k)](
        x, qweight, qzeros, scales, y, ws, cnt,
        M, N, K, group_size,
        x.stride(0), x.stride(1),
        qweight.stride(0), qweight.stride(1),
        qzeros.stride(0), qzeros.stride(1),
        scales.stride(0), scales.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=cfg["block_m"], BLOCK_N=cfg["block_n"], BLOCK_K=block_k,
        GROUP_ROWS=group_rows, SPLIT_K=split_k,
        num_warps=cfg["num_warps"], num_stages=cfg["num_stages"],
    )
    return y
