#!/usr/bin/env python3
"""最终验证：32K context, 匹配官方benchmark"""

import sys
sys.path.insert(0, "mlx-lm-source")

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

MODEL_PATH = "/Volumes/toshiba/models/qwen3-8b-mlx"

print("="*70)
print("最终验证：32K context")
print("="*70)

model, tokenizer = load(MODEL_PATH)

text = "The quick brown fox jumps over the lazy dog. " * 4000
tokens_list = tokenizer.encode(text)[:32768]
tokens = mx.array([tokens_list])
print(f"Context: {len(tokens_list)} tokens\n")

def test(name, **kwargs):
    mx.clear_cache()
    mx.reset_peak_memory()

    cache = make_prompt_cache(model, **kwargs)

    # Prefill
    logits = model(tokens, cache=cache)
    mx.eval(logits)
    pp_peak = mx.get_peak_memory() / (1024**2)

    # TG
    for _ in range(10):
        next_token = mx.argmax(logits[:, -1:, :], axis=-1)
        logits = model(next_token, cache=cache)
        mx.eval(logits)

    tg_mem = mx.get_active_memory() / (1024**2)

    print(f"{name:20s}: PP Peak {pp_peak:5.0f} MB, TG Mem {tg_mem:5.0f} MB")

    del cache, logits
    mx.clear_cache()
    return pp_peak, tg_mem

# Standard
pp_std, tg_std = test("Standard")

# triple_pq + Q8 (修复后)
pp_tpq, tg_tpq = test("triple_pq + Q8", kv_cache="triple_pq", kv_flat_quant="q8_0")

# 分析
print("\n" + "="*70)
print("对比官方benchmark (model_cards/qwen3-8b-mlx-4bit.json)")
print("="*70)

pp_improvement = (pp_std - pp_tpq) / pp_std * 100
tg_improvement = (tg_std - tg_tpq) / tg_std * 100

print(f"\n修复后结果:")
print(f"  PP Peak: {pp_std:.0f} MB → {pp_tpq:.0f} MB ({pp_improvement:+.1f}%)")
print(f"  TG Mem:  {tg_std:.0f} MB → {tg_tpq:.0f} MB ({tg_improvement:+.1f}%)")

print(f"\n官方benchmark期望:")
print(f"  PP Peak: 4840 MB → 526 MB (-89%)")
print(f"  TG Mem:  4647 MB → 529 MB (-89%)")

# KV cache理论压缩
n_kv_heads = 8
head_dim = 128
n_layers = 36
seq_len = 32768
kv_bf16_mb = n_kv_heads * head_dim * n_layers * 2 * 2 * seq_len / (1024**2)
kv_saved_mb = tg_std - tg_tpq
kv_compression = kv_saved_mb / kv_bf16_mb * 100

print(f"\nKV cache压缩率:")
print(f"  实际节省: {kv_saved_mb:.0f} MB")
print(f"  理论KV cache (bf16): {kv_bf16_mb:.0f} MB")
print(f"  压缩率: {kv_compression:.1f}% (理论~49%)")

if abs(kv_compression - 49) < 10 and pp_improvement >= 0:
    print(f"\n✅ 修复成功！")
    print(f"   - Prefill bug: ✅ 无内存爆炸 ({pp_improvement:+.1f}%)")
    print(f"   - KV压缩: ✅ {kv_compression:.1f}% (期望~49%)")
else:
    print(f"\n⚠️  仍有问题")
