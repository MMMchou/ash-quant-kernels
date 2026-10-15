"""PyTorch reference implementation of AWQ-format W4A16 (int4 weight, fp16 activation).

Layout follows the AutoAWQ "GEMM" checkpoint format, which is what vLLM's
`awq_gemm_triton` / `awq_marlin` and SGLang's `awq` / `awq_marlin` load:

    qweight : [K, N // 8]       int32   8 x uint4 per int32, packed along N
    qzeros  : [K // G, N // 8]  int32   same packing as qweight
    scales  : [K // G, N]       fp16
    y = x @ W,  x: [M, K] fp16,  W[k, n] = (q[k, n] - z[k // G, n]) * s[k // G, n]

Packing order: inside one int32, nibble i (bits 4i..4i+3) stores column
AWQ_ORDER[i] of its 8-column block. Equivalently, column j lives in nibble
AWQ_REVERSE_ORDER[j] = (j % 2) * 4 + j // 2.
"""

from __future__ import annotations

import torch

AWQ_ORDER = (0, 2, 4, 6, 1, 3, 5, 7)          # nibble i -> column
AWQ_REVERSE_ORDER = (0, 4, 1, 5, 2, 6, 3, 7)  # column j -> nibble
PACK_FACTOR = 8  # 32 bits / 4 bits


def pack_awq(x: torch.Tensor) -> torch.Tensor:
    """Pack uint4 values [R, C] (C % 8 == 0, values in 0..15) into int32 [R, C // 8]."""
    assert x.dim() == 2 and x.shape[1] % PACK_FACTOR == 0
    assert int(x.min()) >= 0 and int(x.max()) <= 15, "values must be uint4"
    r, c = x.shape
    x = x.to(torch.int64).view(r, c // PACK_FACTOR, PACK_FACTOR)
    packed = torch.zeros(r, c // PACK_FACTOR, dtype=torch.int64, device=x.device)
    for nibble, col in enumerate(AWQ_ORDER):
        packed |= x[..., col] << (4 * nibble)
    # Reinterpret the low 32 bits as signed int32 (nibble 7 can set the sign bit).
    packed = torch.where(packed >= 2**31, packed - 2**32, packed)
    return packed.to(torch.int32)


def unpack_awq(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of pack_awq: int32 [R, C] -> int32 [R, C * 8] with values in 0..15."""
    assert packed.dtype == torch.int32 and packed.dim() == 2
    shifts = torch.tensor(AWQ_REVERSE_ORDER, dtype=torch.int32, device=packed.device) * 4
    # Arithmetic right shift on negative int32 is fine because we mask with 0xF.
    vals = (packed[..., None] >> shifts) & 0xF  # [R, C, 8]
    return vals.reshape(packed.shape[0], -1)


def quantize_awq(w: torch.Tensor, group_size: int = 128):
    """Asymmetric round-to-nearest int4 quantization of W [K, N] (fp16/fp32).

    Not the AWQ *algorithm* (no activation-aware scaling search) -- only the AWQ
    *storage format*. Kernels don't care how scales were chosen.
    Returns (qweight, qzeros, scales) in AWQ layout.
    """
    k, n = w.shape
    assert k % group_size == 0 and n % PACK_FACTOR == 0
    wg = w.float().view(k // group_size, group_size, n)
    w_max = wg.amax(dim=1)
    w_min = wg.amin(dim=1)
    scales = ((w_max - w_min).clamp(min=1e-5) / 15.0)            # [K/G, N]
    zeros = torch.round(-w_min / scales).clamp(0, 15)            # [K/G, N]
    q = torch.round(wg / scales[:, None, :]) + zeros[:, None, :]
    q = q.clamp(0, 15).to(torch.int32).view(k, n)
    scales = scales.to(torch.float16)
    return pack_awq(q), pack_awq(zeros.to(torch.int32)), scales


def dequantize_awq(qweight, qzeros, scales, dtype=torch.float32) -> torch.Tensor:
    """W[k, n] = (q - z) * s, computed in fp32 then cast to `dtype`. Returns [K, N]."""
    q = unpack_awq(qweight)                      # [K, N]
    z = unpack_awq(qzeros)                       # [K/G, N]
    k, n = q.shape
    group_size = k // scales.shape[0]
    z = z.repeat_interleave(group_size, dim=0)
    s = scales.float().repeat_interleave(group_size, dim=0)
    return ((q - z).float() * s).to(dtype)


def w4a16_gemm_ref(x, qweight, qzeros, scales) -> torch.Tensor:
    """Unfused baseline: dequantize to fp16 in HBM, then cuBLAS fp16 GEMM."""
    w = dequantize_awq(qweight, qzeros, scales, dtype=x.dtype)
    return x @ w


def w4a16_gemm_fp32(x, qweight, qzeros, scales) -> torch.Tensor:
    """Ground truth for correctness tests: everything in fp32 (TF32 off)."""
    w = dequantize_awq(qweight, qzeros, scales, dtype=torch.float32)
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        return x.float() @ w
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
