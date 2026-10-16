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

## Changelog

Newest first. Every entry links to the findings note with full numbers.
Timings: H200, Qwen3-8B qkv_proj (6144×4096), M = 1, CUDA graph unless noted.

| date | change | qkv_proj M=1 | notes |
|---|---|---|---|
| 2026-10-16 | decode GEMV (no `tl.dot`, per-group factored dequant) + `w4a16_linear` dispatcher | **13.5 us** (fp16 13.9) | first win over fp16 on all 4 layers; down_proj 1.39× — [step 3](results/v0.2-step3-findings.md) |
| 2026-10-16 | v3: multi-group K tiles, single-launch split-K; CUDA-graph timer; CPU CI | 19.8 us | HBM not the limit (load-only 3.4 TB/s); nibble-sliced v4 rejected — [step 2](results/v0.2-step2-findings.md) |
| 2026-10-15 | v2: load-once unpack via `tl.interleave`, split-K | 22.9 us | 5.6× over v1, faster than vLLM `awq_triton` — [step 1](results/v0.2-step1-findings.md) |
| 2026-10-15 | v0.1: AWQ-format reference, tests, naive fused kernel, bench harness | 230.6 us (eager) | each packed word loaded 8× — [naive](results/v0.1-naive-findings.md) |

## License

Apache-2.0
