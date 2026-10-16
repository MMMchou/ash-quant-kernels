"""W4A16 fused dequant GEMM in Triton -- v4 (no-interleave "nibble-sliced" dot).

Finding from v3 (results/v0.2-step2-findings.md): a load-only kernel on the raw AWQ layout
reaches ~3.4 TB/s, but expanding packed words with tl.interleave forces register-layout
conversions (through shared memory) that dominate the K-loop.

v4 never expands. A packed tile P = qweight[BK, BN/8] holds 8 column-slices: nibble i of
P[:, p] is column 8p + AWQ_ORDER[i]. So

    y[:, 8p + AWQ_ORDER[i]] = x @ dequant(nibble_i(P))[:, p]      for i = 0..7

and each slice is just (P >> 4i) & 0xF, elementwise, in the layout P was loaded in.
The kernel keeps 8 accumulators [BM, BN/8] and issues 8 small tl.dot per K-tile.
Scales / zeros for slice i are the strided columns 8p + AWQ_ORDER[i].
Split-K uses the same single-launch fixup as v3.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _slice_dot(acc, x, qw, qz, s_ptr_row, offs_pn, mask_pn, NIB: tl.constexpr, COL: tl.constexpr,
               NUM_G: tl.constexpr, GROUP_ROWS: tl.constexpr, BLOCK_K: tl.constexpr,
               BNP: tl.constexpr, stride_sn, mask_g):
    q = (qw >> (4 * NIB)) & 0xF                                      # [BK, BNP]
    z = (qz >> (4 * NIB)) & 0xF                                      # [NUM_G, BNP]
    s = tl.load(
        s_ptr_row + (offs_pn * 8 + COL)[None, :] * stride_sn,
        mask=mask_g[:, None] & mask_pn[None, :], other=0.0,
    )                                                                # [NUM_G, BNP]
    q3 = tl.reshape(q, (NUM_G, GROUP_ROWS, BNP))
    w = tl.reshape((q3 - z[:, None, :]).to(tl.float16) * s[:, None, :], (BLOCK_K, BNP))
    return tl.dot(x, w, acc)


@triton.jit
def _store_slice(acc, y_ptr, ws_ptr, offs_m, offs_pn, mask_m, mask_pn, N, stride_ym, stride_yn,
                 COL: tl.constexpr, SPLIT_K: tl.constexpr):
    cols = offs_pn * 8 + COL
    mask = mask_m[:, None] & mask_pn[None, :]
    if SPLIT_K == 1:
        tl.store(y_ptr + offs_m[:, None] * stride_ym + cols[None, :] * stride_yn,
                 acc.to(y_ptr.dtype.element_ty), mask=mask)
    else:
        tl.atomic_add(ws_ptr + offs_m[:, None] * N + cols[None, :], acc, mask=mask, sem="release")


@triton.jit
def _finish_slice(y_ptr, ws_ptr, offs_m, offs_pn, mask_m, mask_pn, N, stride_ym, stride_yn,
                  COL: tl.constexpr, BLOCK_M: tl.constexpr, BNP: tl.constexpr):
    cols = offs_pn * 8 + COL
    mask = mask_m[:, None] & mask_pn[None, :]
    ws = ws_ptr + offs_m[:, None] * N + cols[None, :]
    tot = tl.load(ws, mask=mask, other=0.0, cache_modifier=".cg")
    tl.store(y_ptr + offs_m[:, None] * stride_ym + cols[None, :] * stride_yn,
             tot.to(y_ptr.dtype.element_ty), mask=mask)
    tl.store(ws, tl.zeros((BLOCK_M, BNP), dtype=tl.float32), mask=mask)


@triton.jit
def _w4a16_gemm_v4_kernel(
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
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_pn = pid_n * BNP + tl.arange(0, BNP)
    offs_k = tl.arange(0, BLOCK_K)
    offs_g = tl.arange(0, NUM_G)
    mask_m = offs_m < M
    mask_pn = offs_pn < N // 8

    # AWQ_ORDER = (0, 2, 4, 6, 1, 3, 5, 7): nibble i holds column AWQ_ORDER[i].
    a0 = tl.zeros((BLOCK_M, BNP), dtype=tl.float32)
    a1 = tl.zeros((BLOCK_M, BNP), dtype=tl.float32)
    a2 = tl.zeros((BLOCK_M, BNP), dtype=tl.float32)
    a3 = tl.zeros((BLOCK_M, BNP), dtype=tl.float32)
    a4 = tl.zeros((BLOCK_M, BNP), dtype=tl.float32)
    a5 = tl.zeros((BLOCK_M, BNP), dtype=tl.float32)
    a6 = tl.zeros((BLOCK_M, BNP), dtype=tl.float32)
    a7 = tl.zeros((BLOCK_M, BNP), dtype=tl.float32)
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
        gs = k0 // group_size + offs_g
        mask_g = gs < K // group_size
        qz = tl.load(
            qz_ptr + gs[:, None] * stride_qzg + offs_pn[None, :] * stride_qzn,
            mask=mask_g[:, None] & mask_pn[None, :], other=0,
        )
        s_row = s_ptr + gs[:, None] * stride_sg
        a0 = _slice_dot(a0, x, qw, qz, s_row, offs_pn, mask_pn, 0, 0, NUM_G, GROUP_ROWS, BLOCK_K, BNP, stride_sn, mask_g)
        a1 = _slice_dot(a1, x, qw, qz, s_row, offs_pn, mask_pn, 1, 2, NUM_G, GROUP_ROWS, BLOCK_K, BNP, stride_sn, mask_g)
        a2 = _slice_dot(a2, x, qw, qz, s_row, offs_pn, mask_pn, 2, 4, NUM_G, GROUP_ROWS, BLOCK_K, BNP, stride_sn, mask_g)
        a3 = _slice_dot(a3, x, qw, qz, s_row, offs_pn, mask_pn, 3, 6, NUM_G, GROUP_ROWS, BLOCK_K, BNP, stride_sn, mask_g)
        a4 = _slice_dot(a4, x, qw, qz, s_row, offs_pn, mask_pn, 4, 1, NUM_G, GROUP_ROWS, BLOCK_K, BNP, stride_sn, mask_g)
        a5 = _slice_dot(a5, x, qw, qz, s_row, offs_pn, mask_pn, 5, 3, NUM_G, GROUP_ROWS, BLOCK_K, BNP, stride_sn, mask_g)
        a6 = _slice_dot(a6, x, qw, qz, s_row, offs_pn, mask_pn, 6, 5, NUM_G, GROUP_ROWS, BLOCK_K, BNP, stride_sn, mask_g)
        a7 = _slice_dot(a7, x, qw, qz, s_row, offs_pn, mask_pn, 7, 7, NUM_G, GROUP_ROWS, BLOCK_K, BNP, stride_sn, mask_g)

    args = (y_ptr, ws_ptr, offs_m, offs_pn, mask_m, mask_pn, N, stride_ym, stride_yn)
    _store_slice(a0, *args, 0, SPLIT_K)
    _store_slice(a1, *args, 2, SPLIT_K)
    _store_slice(a2, *args, 4, SPLIT_K)
    _store_slice(a3, *args, 6, SPLIT_K)
    _store_slice(a4, *args, 1, SPLIT_K)
    _store_slice(a5, *args, 3, SPLIT_K)
    _store_slice(a6, *args, 5, SPLIT_K)
    _store_slice(a7, *args, 7, SPLIT_K)
    if SPLIT_K > 1:
        tl.debug_barrier()
        ticket = tl.atomic_add(cnt_ptr + pid, 1, sem="acq_rel")
        if ticket == SPLIT_K - 1:
            for c in tl.static_range(8):
                _finish_slice(*args, c, BLOCK_M, BNP)
            tl.store(cnt_ptr + pid, 0)


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
    if M <= 16:
        return dict(block_m=16, block_n=128, block_k=256, split_k=8, num_warps=4, num_stages=3)
    if M <= 64:
        return dict(block_m=64, block_n=128, block_k=128, split_k=1, num_warps=4, num_stages=3)
    return dict(block_m=128, block_n=256, block_k=64, split_k=1, num_warps=8, num_stages=3)


def w4a16_gemm_v4(x, qweight, qzeros, scales, config: dict | None = None) -> torch.Tensor:
    """x: [M, K] fp16; AWQ tensors as in ashqk.ref. Returns y: [M, N] fp16.

    Requires BLOCK_N >= 128 (each of the 8 slices is BLOCK_N / 8 >= 16 wide for tl.dot).
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
    assert cfg["block_n"] >= 128 and cfg["block_n"] % 8 == 0
    split_k = cfg["split_k"]

    num_tiles = triton.cdiv(M, cfg["block_m"]) * triton.cdiv(N, cfg["block_n"])
    y = torch.empty((M, N), dtype=x.dtype, device=x.device)
    ws, cnt = _workspace(M, N, num_tiles, x.device) if split_k > 1 else (y, y)
    _w4a16_gemm_v4_kernel[(num_tiles, split_k)](
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
