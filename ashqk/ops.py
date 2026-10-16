"""Public entry point: picks the best kernel for the problem size.

M <= 2  -> w4a16_gemv (CUDA-core FMA, dequant factored per group; beats fp16 cuBLAS at M = 1)
M >  2  -> w4a16_gemm_v3 (tensor cores)
Threshold measured on H200 (results/v0.2-step3-findings.md).
"""

from __future__ import annotations

import torch

from .kernels.w4a16_gemm_v3 import w4a16_gemm_v3
from .kernels.w4a16_gemv import w4a16_gemv

GEMV_MAX_M = 2


def w4a16_linear(x: torch.Tensor, qweight, qzeros, scales) -> torch.Tensor:
    """y = x @ dequant(AWQ weights). x: [..., K] fp16 -> [..., N] fp16."""
    lead = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1])
    if x2.shape[0] <= GEMV_MAX_M:
        y = w4a16_gemv(x2, qweight, qzeros, scales)
    else:
        y = w4a16_gemm_v3(x2, qweight, qzeros, scales)
    return y.reshape(*lead, y.shape[-1])
