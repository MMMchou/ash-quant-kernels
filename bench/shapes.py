"""Linear-layer shapes (name, N, K, layers_per_model) for benchmark models.

Qwen3.5-9B (default): hidden 4096, intermediate 12288, 32 layers in a 3:1 pattern of
Gated DeltaNet (linear attention) and gated full attention. Fused projections follow
SGLang `models/qwen3_5.py`:
  in_proj_qkvz = [key_dim, key_dim, value_dim, value_dim] = [2048, 2048, 4096, 4096]
  in_proj_ba   = [num_v_heads, num_v_heads] = [32, 32]
  qkv_proj     = 16 heads x 256 x (1 + output gate) + 2 x 4 kv heads x 256
"""

MODELS = {
    "qwen3.5-9b": [
        ("in_proj_qkvz", 12288, 4096, 24),   # linear-attention layers
        ("in_proj_ba", 64, 4096, 24),
        ("out_proj", 4096, 4096, 24),
        ("qkv_proj", 10240, 4096, 8),        # full-attention layers (with output gate)
        ("o_proj", 4096, 4096, 8),
        ("gate_up_proj", 24576, 4096, 32),
        ("down_proj", 4096, 12288, 32),
    ],
    "qwen3-8b": [
        ("qkv_proj", 6144, 4096, 36),
        ("o_proj", 4096, 4096, 36),
        ("gate_up_proj", 24576, 4096, 36),
        ("down_proj", 4096, 12288, 36),
    ],
}
DEFAULT_MODEL = "qwen3.5-9b"
