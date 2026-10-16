# ash-quant-kernels

[![tests](https://github.com/MMMchou/ash-quant-kernels/actions/workflows/tests.yml/badge.svg)](https://github.com/MMMchou/ash-quant-kernels/actions/workflows/tests.yml)

Triton kernels for weight-only quantized LLM inference, starting with an
**AWQ-format W4A16 fused dequant-GEMM**, benchmarked on NVIDIA H200 against
cuBLAS fp16 and the kernels used by vLLM / SGLang.

## Why

In decode (M = 1–16 tokens per step) a linear layer is memory-bound: time is
dominated by reading the weights from HBM. INT4 weights are 4× smaller than
fp16, but only if the kernel dequantizes **in registers** instead of writing an
fp16 copy back to HBM. This repo builds that kernel step by step and measures
how close it gets to the bandwidth roofline.

## Format

Drop-in compatible with AutoAWQ "GEMM" checkpoints, i.e. the same tensors that
vLLM (`awq`, `awq_marlin`) and SGLang (`awq`, `awq_marlin`) load:

| tensor | shape | dtype |
|---|---|---|
| `qweight` | `[K, N/8]` | int32, 8 × uint4, AWQ interleaved order `[0,2,4,6,1,3,5,7]` |
| `qzeros` | `[K/G, N/8]` | int32, same packing |
| `scales` | `[K/G, N]` | fp16 |

`W[k, n] = (q[k, n] - z[k // G, n]) * s[k // G, n]`, `y = x @ W`.

## Roadmap

| version | content | status |
|---|---|---|
| v0.1 | PyTorch reference, correctness tests, naive fused Triton kernel, benchmark harness | done |
| v0.2 | load-once unpacking ✅, single-launch split-K ✅, multi-group K tiles ✅, CUDA-graph timing ✅, CI ✅; decode GEMV path ✅ (beats fp16 at M=1); profiling, 4 ≤ M ≤ 16 | in progress |
| v0.3 | head-to-head with vLLM `awq_gemm_triton` and Marlin; compressed-tensors W4A16 layout; write-up | planned |
| v0.4 | tree-attention verification kernel for speculative decoding (EAGLE draft trees) | planned |

## Correctness criterion

Error is measured against an fp32 ground truth using the same dequantized
weights. A kernel passes if its relative Frobenius error is at most 2× that of
cuBLAS fp16 on the same problem (see `tests/test_gemm.py`).

## Usage

```bash
pip install -e ".[dev]"
pytest -q
python bench/bench_gemm.py --out results/h200-v0.1.md
```

## Results

Benchmarks on H200 with Qwen3-8B layer shapes, group size 128: see `results/`.

Current status (M=1 decode, CUDA-graph timing, us; details in `results/`):

| layer | fp16 cuBLAS | **ashqk** (`w4a16_linear`) | speedup |
|---|---|---|---|
| qkv_proj 6144×4096 | 13.9 | **13.5** | 1.03× |
| o_proj 4096×4096 | 10.0 | **9.2** | 1.09× |
| gate_up_proj 24576×4096 | 50.2 | **48.6** | 1.03× |
| down_proj 4096×12288 | 29.8 | **21.5** | 1.39× |

Optimization path on qkv_proj, M=1: naive v1 230.6 us (eager) → load-once v2 22.9 →
single-launch split-K v3 19.8 → decode GEMV **13.5**. M ≥ 4 and prefill are not yet
faster than fp16 — see `results/v0.2-step3-findings.md` for limits and next steps.

```python
from ashqk import w4a16_linear  # AutoAWQ qweight / qzeros / scales
y = w4a16_linear(x, qweight, qzeros, scales)
```

## License

Apache-2.0
