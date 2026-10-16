"""Correctness of the Triton kernel against an fp32 ground truth.

Criterion: rel_err(kernel) <= 2 * rel_err(cuBLAS fp16) + 1e-5, where
rel_err(y) = ||y - y_fp32||_F / ||y_fp32||_F and both use the same dequantized W.
A fixed atol would either be too loose for small K or flaky for large K, because
fp16 output rounding / accumulation error grows with K; cuBLAS is the yardstick.
"""

import os

import pytest
import torch

from ashqk.ref import dequantize_awq, quantize_awq, w4a16_gemm_fp32

# GPU: full shapes. CPU CI: TRITON_INTERPRET=1 runs kernels in the Triton interpreter on
# small shapes (same code paths: masking, AWQ unpack, split-K atomics).
ON_GPU = torch.cuda.is_available()
INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
DEVICE = "cuda" if ON_GPU else "cpu"
pytestmark = pytest.mark.skipif(not (ON_GPU or INTERPRET), reason="needs CUDA or TRITON_INTERPRET=1")

GPU_SHAPES = [
    (1, 4096, 4096),     # decode, single token
    (7, 512, 256),       # M not a multiple of BLOCK_M
    (16, 6144, 4096),    # qkv_proj
    (33, 1000, 512),     # N not a multiple of BLOCK_N (but of 8)
    (256, 4096, 12288),  # down_proj, large K
    (1024, 2048, 1024),
]
CPU_SHAPES = [
    (1, 64, 256),
    (7, 72, 256),        # N not a multiple of BLOCK_N
    (17, 128, 384),      # M > BLOCK_M, K not a power of 2
]


def rel_err(y, y_true):
    return ((y.float() - y_true).norm() / y_true.norm()).item()


def make_problem(m, n, k, group_size, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(m, k, device=DEVICE, dtype=torch.float16)
    w = torch.randn(k, n, device=DEVICE) * 0.02
    qw, qz, s = quantize_awq(w, group_size)
    return x, qw, qz, s


def _v1(x, qw, qz, s):
    from ashqk.kernels.w4a16_gemm import w4a16_gemm
    return w4a16_gemm(x, qw, qz, s)


def _v2(split_k):
    def f(x, qw, qz, s):
        from ashqk.kernels.w4a16_gemm_v2 import w4a16_gemm_v2
        return w4a16_gemm_v2(x, qw, qz, s, {"split_k": split_k})
    return f


def _v3(split_k, block_k):
    def f(x, qw, qz, s):
        from ashqk.kernels.w4a16_gemm_v3 import w4a16_gemm_v3
        return w4a16_gemm_v3(x, qw, qz, s, {"split_k": split_k, "block_k": block_k})
    return f


IMPLS = {
    "v1": _v1,
    "v2-sk1": _v2(1),
    "v2-sk4": _v2(4),
    "v3-sk1-bk64": _v3(1, 64),      # BLOCK_K <= group_size for every G tested
    "v3-sk1-bk128": _v3(1, 128),    # spans 4 / 2 / 1 groups for G = 32 / 64 / 128
    "v3-sk4-bk128": _v3(4, 128),    # single-launch split-K fixup
}


@pytest.mark.parametrize("impl", list(IMPLS))
@pytest.mark.parametrize("group_size", [32, 64, 128])
@pytest.mark.parametrize("m,n,k", GPU_SHAPES if ON_GPU else CPU_SHAPES)
def test_w4a16_gemm_matches_fp32(m, n, k, group_size, impl):
    if n % 8:
        pytest.skip("AWQ requires N % 8 == 0")
    x, qw, qz, s = make_problem(m, n, k, group_size)
    y_true = w4a16_gemm_fp32(x, qw, qz, s)
    y_cublas = x @ dequantize_awq(qw, qz, s, dtype=torch.float16)
    y = IMPLS[impl](x, qw, qz, s)

    assert y.shape == (m, n) and y.dtype == torch.float16
    e_kernel, e_cublas = rel_err(y, y_true), rel_err(y_cublas, y_true)
    assert e_kernel <= 2 * e_cublas + 1e-5, f"kernel {e_kernel:.2e} vs cuBLAS {e_cublas:.2e}"


@pytest.mark.parametrize("impl", ["v3-sk4-bk128"])
def test_split_k_workspace_is_reset(impl):
    """Repeated calls reuse the cached workspace; results must not accumulate."""
    x, qw, qz, s = make_problem(16, 256, 1024, 128)
    y1 = IMPLS[impl](x, qw, qz, s)
    y2 = IMPLS[impl](x, qw, qz, s)
    assert torch.equal(y1, y2)
