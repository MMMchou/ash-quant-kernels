"""Correctness of the Triton kernel against an fp32 ground truth.

Criterion: rel_err(kernel) <= 2 * rel_err(cuBLAS fp16) + 1e-5, where
rel_err(y) = ||y - y_fp32||_F / ||y_fp32||_F and both use the same dequantized W.
A fixed atol would either be too loose for small K or flaky for large K, because
fp16 output rounding / accumulation error grows with K; cuBLAS is the yardstick.
"""

import pytest
import torch

from ashqk.ref import dequantize_awq, quantize_awq, w4a16_gemm_fp32

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def rel_err(y, y_true):
    return ((y.float() - y_true).norm() / y_true.norm()).item()


def make_problem(m, n, k, group_size, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(m, k, device="cuda", dtype=torch.float16)
    w = torch.randn(k, n, device="cuda") * 0.02
    qw, qz, s = quantize_awq(w, group_size)
    return x, qw, qz, s


@pytest.mark.parametrize("group_size", [32, 64, 128])
@pytest.mark.parametrize(
    "m,n,k",
    [
        (1, 4096, 4096),     # decode, single token
        (7, 512, 256),       # M not a multiple of BLOCK_M
        (16, 6144, 4096),    # qkv_proj
        (33, 1000, 512),     # N not a multiple of BLOCK_N (but of 8)
        (256, 4096, 12288),  # down_proj, large K
        (1024, 2048, 1024),
    ],
)
def test_w4a16_gemm_matches_fp32(m, n, k, group_size):
    from ashqk.kernels.w4a16_gemm import w4a16_gemm

    if n % 8:
        pytest.skip("AWQ requires N % 8 == 0")
    x, qw, qz, s = make_problem(m, n, k, group_size)
    y_true = w4a16_gemm_fp32(x, qw, qz, s)
    y_cublas = x @ dequantize_awq(qw, qz, s, dtype=torch.float16)
    y = w4a16_gemm(x, qw, qz, s)

    assert y.shape == (m, n) and y.dtype == torch.float16
    e_kernel, e_cublas = rel_err(y, y_true), rel_err(y_cublas, y_true)
    assert e_kernel <= 2 * e_cublas + 1e-5, f"kernel {e_kernel:.2e} vs cuBLAS {e_cublas:.2e}"
