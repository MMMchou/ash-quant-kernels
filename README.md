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
| v0.2 | load-once unpacking ✅, single-launch split-K ✅, multi-group K tiles ✅, CUDA-graph timing ✅, CI ✅; decode GEMV path, profiling, beat fp16 at M ≤ 16 | in progress |
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

| layer | fp16 cuBLAS | vLLM awq_triton* | ashqk v1 | ashqk v2 | **ashqk v3** |
|---|---|---|---|---|---|
| qkv_proj 6144×4096 | 14.1 | 68.0 | 230.6* | 22.9 | **19.8** |
| down_proj 4096×12288 | 30.4 | — | — | 39.1 | **34.6** |

\* eager timing. v3 reaches 0.71–0.88× of fp16 cuBLAS. Weight streaming alone runs at
3.4 TB/s, so the remaining gap is in-register unpack / layout conversion, not HBM
(see `results/v0.2-step2-findings.md`, `experiments/`).

## License

Apache-2.0
