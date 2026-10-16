# ash-quant-kernels

[![tests](https://github.com/MMMchou/ash-quant-kernels/actions/workflows/tests.yml/badge.svg)](https://github.com/MMMchou/ash-quant-kernels/actions/workflows/tests.yml)

Triton kernels for weight-only quantized LLM inference, starting with an
**AWQ-format W4A16 fused dequant-GEMM**, benchmarked on NVIDIA H200 against
cuBLAS fp16 and the kernels used by vLLM / SGLang, on **Qwen3.5-9B** layer shapes
(Gated DeltaNet + gated attention hybrid).

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
| v0.3 | head-to-head with Marlin ✅ (Marlin 1.67× faster at M=1); Marlin-style offline repack + lop3 unpack in Triton; SGLang integration + e2e; write-up | in progress |
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

Benchmarks on H200, group size 128, shapes from `bench/shapes.py`: see `results/`.

Current status: Qwen3.5-9B, all linear layers per decoded token, CUDA-graph timing, us
([vs SGLang kernels](results/v0.2-vs-marlin-findings.md), [per layer](results/v0.2-qwen3.5-findings.md)):

| M | fp16 cuBLAS | SGLang `awq_marlin` (production) | SGLang `awq` | **ashqk** |
|---|---|---|---|---|
| 1 | 3886 | **1990** | 9445 | 3325 |
| 4 | 3899 | **2002** | 9431 | 5580 |

Honest summary: ashqk beats fp16 cuBLAS (1.16×) and SGLang's non-Marlin AWQ path (2.8×) at M=1,
but **Marlin is still 1.67× faster**. Closing that gap (offline repack into tensor-core operand order,
lop3 int4→fp16) is the v0.3 goal.

```python
from ashqk import w4a16_linear  # AutoAWQ qweight / qzeros / scales
y = w4a16_linear(x, qweight, qzeros, scales)
```

## Changelog

Newest first. Every entry links to the findings note with full numbers.
Timings: H200, M = 1, CUDA graph unless noted. Column "qkv_proj M=1" is Qwen3-8B qkv_proj (6144×4096) for comparability.

| date | change | qkv_proj M=1 | notes |
|---|---|---|---|
| 2026-10-16 | head-to-head vs SGLang 0.5.9 `awq_marlin` / `awq` (`bench/bench_vs_sglang.py`) | 13.5 us | Marlin 1990 vs ashqk 3325 us/token at M=1 → Marlin 1.67× faster; ashqk 2.8× faster than SGLang `awq` — [vs marlin](results/v0.2-vs-marlin-findings.md) |
| 2026-10-16 | benchmark target → Qwen3.5-9B shapes; skinny-N GEMV config; flaky split-K test fixed | 13.5 us | Qwen3.5-9B all linear layers 1.16× fp16 per token — [qwen3.5](results/v0.2-qwen3.5-findings.md) |
| 2026-10-16 | decode GEMV (no `tl.dot`, per-group factored dequant) + `w4a16_linear` dispatcher | **13.5 us** (fp16 13.9) | first win over fp16 on all 4 layers; down_proj 1.39× — [step 3](results/v0.2-step3-findings.md) |
| 2026-10-16 | v3: multi-group K tiles, single-launch split-K; CUDA-graph timer; CPU CI | 19.8 us | HBM not the limit (load-only 3.4 TB/s); nibble-sliced v4 rejected — [step 2](results/v0.2-step2-findings.md) |
| 2026-10-15 | v2: load-once unpack via `tl.interleave`, split-K | 22.9 us | 5.6× over v1, faster than vLLM `awq_triton` — [step 1](results/v0.2-step1-findings.md) |
| 2026-10-15 | v0.1: AWQ-format reference, tests, naive fused kernel, bench harness | 230.6 us (eager) | each packed word loaded 8× — [naive](results/v0.1-naive-findings.md) |

## License

Apache-2.0
