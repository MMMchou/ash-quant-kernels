"""Tests for the PyTorch reference (format correctness). Run on CPU or GPU."""

import pytest
import torch

from ashqk.ref import (
    AWQ_ORDER,
    dequantize_awq,
    pack_awq,
    quantize_awq,
    unpack_awq,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_pack_layout_matches_awq_order():
    # One row, 8 columns with distinct values: column j holds value j + 1.
    x = torch.arange(1, 9, dtype=torch.int32).view(1, 8)
    packed = int(pack_awq(x)[0, 0]) & 0xFFFFFFFF
    nibbles = [(packed >> (4 * i)) & 0xF for i in range(8)]
    # Nibble i must hold column AWQ_ORDER[i].
    assert nibbles == [AWQ_ORDER[i] + 1 for i in range(8)]


def test_pack_sign_bit():
    # Value 15 in nibble 7 sets bit 31 -> negative int32; must round-trip.
    x = torch.full((2, 8), 15, dtype=torch.int32)
    p = pack_awq(x)
    assert p.dtype == torch.int32 and int(p[0, 0]) == -1
    assert torch.equal(unpack_awq(p), x)


@pytest.mark.parametrize("shape", [(1, 8), (64, 128), (4096, 512)])
def test_pack_unpack_roundtrip(shape):
    g = torch.Generator().manual_seed(0)
    x = torch.randint(0, 16, shape, dtype=torch.int32, generator=g)
    assert torch.equal(unpack_awq(pack_awq(x)), x)


@pytest.mark.parametrize("group_size", [32, 64, 128])
def test_quantize_error_bound(group_size):
    torch.manual_seed(0)
    k, n = 512, 256
    w = torch.randn(k, n, device=DEVICE)
    qw, qz, s = quantize_awq(w, group_size)
    assert qw.shape == (k, n // 8) and qz.shape == (k // group_size, n // 8)
    assert s.shape == (k // group_size, n) and s.dtype == torch.float16
    w_hat = dequantize_awq(qw, qz, s)
    # RTN error per element <= ~scale (half step + zero-point rounding + fp16 scale).
    step = s.float().repeat_interleave(group_size, dim=0)
    assert ((w - w_hat).abs() <= step * 1.01).all()
