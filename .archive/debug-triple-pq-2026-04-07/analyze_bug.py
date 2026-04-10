#!/usr/bin/env python3
"""系统分析triple_pq的bug"""

import sys
sys.path.insert(0, "mlx-lm-source")

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

MODEL_PATH = "/Volumes/toshiba/models/qwen3-8b-mlx"

print("="*70)
print("Bug分析: triple_pq vs standard")
print("="*70)

model, tokenizer = load(MODEL_PATH)

# 用16K测试（足够看到bug，但不会OOM）
text = "The quick brown fox jumps over the lazy dog. " * 2000
tokens_list = tokenizer.encode(text)[:16384]
tokens = mx.array([tokens_list])
print(f"\n测试: {len(tokens_list)} tokens (单次加载)\n")

def detailed_test(name, cache_kwargs):
    print(f"{name}")
    print("-"*70)

    mx.clear_cache()
    mx.reset_peak_memory()

    cache = make_prompt_cache(model, **cache_kwargs)

    # 只做prefill，看内存
    print("Prefill中...")
    logits = model(tokens, cache=cache)
    mx.eval(logits)

    pp_peak = mx.get_peak_memory() / (1024**2)
    pp_active = mx.get_active_memory() / (1024**2)

    print(f"Prefill后: Peak {pp_peak:.0f} MB, Active {pp_active:.0f} MB")

    # 第一个TG token（触发flat mode）
    print("First TG token...")
    next_token = mx.argmax(logits[:, -1:, :], axis=-1)
    logits = model(next_token, cache=cache)
    mx.eval(logits)

    tg1_peak = mx.get_peak_memory() / (1024**2)
    tg1_active = mx.get_active_memory() / (1024**2)

    print(f"First TG后: Peak {tg1_peak:.0f} MB, Active {tg1_active:.0f} MB")

    # 多生成几个token
    for _ in range(9):
        next_token = mx.argmax(logits[:, -1:, :], axis=-1)
        logits = model(next_token, cache=cache)
        mx.eval(logits)

    tg_final_active = mx.get_active_memory() / (1024**2)
    print(f"10 TG tokens后: Active {tg_final_active:.0f} MB")

    del cache, logits
    mx.clear_cache()

    return {
        'pp_peak': pp_peak,
        'pp_active': pp_active,
        'tg1_peak': tg1_peak,
        'tg1_active': tg1_active,
        'tg_final': tg_final_active
    }

# Test 1: Standard
std = detailed_test("1. Standard", {})

# Test 2: triple_pq + Q8
print()
tpq = detailed_test("2. triple_pq + Q8", {
    "kv_cache": "triple_pq",
    "kv_warm_quantizer": "polarquant",
    "kv_warm_bits": 4,
    "kv_flat_quant": "q8_0"
})

# 分析
print("\n" + "="*70)
print("分析")
print("="*70)

pp_diff = ((tpq['pp_peak'] - std['pp_peak']) / std['pp_peak'] * 100)
tg_compression = ((std['tg_final'] - tpq['tg_final']) / std['tg_final'] * 100)

print(f"\nPrefill Peak: triple_pq {tpq['pp_peak']:.0f} vs std {std['pp_peak']:.0f}")
print(f"  差异: {pp_diff:+.1f}%")

if pp_diff > 20:
    print(f"  ❌ BUG: triple_pq在prefill时比standard多用{pp_diff:.1f}%内存")
    print(f"     原因: 量化层+反量化层同时在内存（双重存储）")
elif abs(pp_diff) < 10:
    print(f"  ✅ OK: 内存使用相近")
else:
    print(f"  ⚠️  轻微差异")

print(f"\nTG Final Active: triple_pq {tpq['tg_final']:.0f} vs std {std['tg_final']:.0f}")
print(f"  压缩率: {tg_compression:.1f}%")

if tg_compression > 40:
    print(f"  ✅ GOOD: 达到预期压缩（~50%）")
elif tg_compression > 20:
    print(f"  ⚠️  压缩偏低（期望~50%）")
else:
    print(f"  ❌ BAD: 压缩严重不足（期望~50%）")

print(f"\n理论Q8压缩: KV cache应该减少~50%")
print(f"实际压缩: {tg_compression:.1f}%")
