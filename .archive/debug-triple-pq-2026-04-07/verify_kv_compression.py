#!/usr/bin/env python3
"""验证KV cache本身的压缩率"""

import sys
sys.path.insert(0, "mlx-lm-source")

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

MODEL_PATH = "/Volumes/toshiba/models/qwen3-8b-mlx"

print("="*70)
print("验证KV cache压缩率")
print("="*70)

model, tokenizer = load(MODEL_PATH)

# 计算KV cache理论大小 (Qwen3-8B: 8 KV heads, 128 dim, 36 layers)
n_kv_heads = 8
head_dim = 128
n_layers = 36
seq_len = 16384

# BF16 KV cache大小
kv_bf16_bytes = n_kv_heads * head_dim * n_layers * 2 * 2 * seq_len  # 2 (K+V) × 2 bytes (bf16)
kv_bf16_mb = kv_bf16_bytes / (1024**2)

# Q8 KV cache大小 (int8 + scales)
kv_q8_data = n_kv_heads * head_dim * n_layers * 2 * 1 * seq_len  # 1 byte (int8)
kv_q8_scales = n_kv_heads * 1 * n_layers * 2 * 2 * seq_len  # per-token scales (bf16)
kv_q8_mb = (kv_q8_data + kv_q8_scales) / (1024**2)

print(f"\n理论计算 ({seq_len} tokens):")
print(f"  KV cache (bf16): {kv_bf16_mb:.0f} MB")
print(f"  KV cache (Q8):   {kv_q8_mb:.0f} MB")
print(f"  理论压缩: {(kv_bf16_mb - kv_q8_mb) / kv_bf16_mb * 100:.1f}%")

# 实际测试
text = "The quick brown fox jumps over the lazy dog. " * 2000
tokens_list = tokenizer.encode(text)[:16384]
tokens = mx.array([tokens_list])

print(f"\n实际测试:")

# Standard
mx.clear_cache()
cache_std = make_prompt_cache(model)  # No kv_cache param = standard
logits = model(tokens, cache=cache_std)
mx.eval(logits)
for _ in range(10):
    next_token = mx.argmax(logits[:, -1:, :], axis=-1)
    logits = model(next_token, cache=cache_std)
    mx.eval(logits)
tg_std = mx.get_active_memory() / (1024**2)
del cache_std, logits
mx.clear_cache()

# triple_pq + Q8
mx.clear_cache()
cache_tpq = make_prompt_cache(model, kv_cache="triple_pq", kv_flat_quant="q8_0")
logits = model(tokens, cache=cache_tpq)
mx.eval(logits)
for _ in range(10):
    next_token = mx.argmax(logits[:, -1:, :], axis=-1)
    logits = model(next_token, cache=cache_tpq)
    mx.eval(logits)
tg_tpq = mx.get_active_memory() / (1024**2)
del cache_tpq, logits
mx.clear_cache()

print(f"  Standard TG: {tg_std:.0f} MB")
print(f"  triple_pq TG: {tg_tpq:.0f} MB")

# KV cache实际压缩
kv_saved_mb = tg_std - tg_tpq
kv_compression_pct = kv_saved_mb / kv_bf16_mb * 100

print(f"\n  实际节省: {kv_saved_mb:.0f} MB")
print(f"  相对理论KV cache: {kv_compression_pct:.1f}% 压缩")

if abs(kv_compression_pct - 47) < 10:
    print(f"\n✅ 压缩率正常！实际{kv_compression_pct:.1f}% ≈ 理论49.9%")
else:
    print(f"\n⚠️  压缩率偏离理论值")
