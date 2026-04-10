#!/usr/bin/env python3
"""正确的测试方式：使用model card配置"""

import sys
sys.path.insert(0, "mlx-lm-source")

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from flashmlx import load_card

MODEL_PATH = "/Volumes/toshiba/models/qwen3-8b-mlx"

print("="*80)
print("使用Model Card配置测试（正确的方式）")
print("="*80)

# 加载model和card
model, tokenizer = load(MODEL_PATH)
card = load_card(MODEL_PATH)

print(f"Model: {card.model_name}")
print(f"Card ID: {card.model_id}")
print(f"Optimal: {card.optimal_strategy}")
print(f"Available modes: {list(card.modes.keys())}")

# 32K context
text = "The quick brown fox jumps over the lazy dog. " * 4000
tokens_list = tokenizer.encode(text)[:32768]
tokens = mx.array([tokens_list])
print(f"\nContext: {len(tokens_list)} tokens\n")

def test(mode_name):
    """使用model card配置测试"""
    mx.clear_cache()
    mx.reset_peak_memory()

    # 从card获取配置（正确的方式！）
    kwargs = card.to_cache_kwargs(mode=mode_name)
    print(f"{mode_name} mode:")
    print(f"  Config: {kwargs}")

    cache = make_prompt_cache(model, **kwargs)

    # Prefill
    logits = model(tokens, cache=cache)
    mx.eval(logits)
    pp_peak = mx.get_peak_memory() / (1024**2)

    # TG
    for _ in range(100):
        next_token = mx.argmax(logits[:, -1:, :], axis=-1)
        logits = model(next_token, cache=cache)
        mx.eval(logits)

    tg_mem = mx.get_active_memory() / (1024**2)

    print(f"  Result: PP Peak {pp_peak:.0f} MB, TG Mem {tg_mem:.0f} MB\n")

    del cache, logits
    mx.clear_cache()
    return pp_peak, tg_mem

# Test 1: Standard baseline
print("Baseline (no compression):")
mx.clear_cache()
cache_std = make_prompt_cache(model)
logits = model(tokens, cache=cache_std)
mx.eval(logits)
pp_std = mx.get_peak_memory() / (1024**2)
for _ in range(100):
    next_token = mx.argmax(logits[:, -1:, :], axis=-1)
    logits = model(next_token, cache=cache_std)
    mx.eval(logits)
tg_std = mx.get_active_memory() / (1024**2)
print(f"  Result: PP Peak {pp_std:.0f} MB, TG Mem {tg_std:.0f} MB\n")
del cache_std, logits
mx.clear_cache()

# Test 2: Optimal (scored_pq)
try:
    pp_opt, tg_opt = test(None)  # None = use optimal
    opt_success = True
except Exception as e:
    print(f"  ❌ Optimal failed: {e}\n")
    pp_opt, tg_opt = 0, 0
    opt_success = False

# Test 3: no_calibration mode (triple_pq)
pp_nc, tg_nc = test("no_calibration")

# 分析
print("="*80)
print("结果对比")
print("="*80)

print(f"\nBaseline: PP {pp_std:.0f} MB, TG {tg_std:.0f} MB")

if opt_success:
    pp_opt_save = (pp_std - pp_opt) / pp_std * 100
    tg_opt_save = (tg_std - tg_opt) / tg_std * 100
    print(f"Optimal (scored_pq): PP {pp_opt:.0f} MB ({pp_opt_save:+.1f}%), TG {tg_opt:.0f} MB ({tg_opt_save:+.1f}%)")

pp_nc_save = (pp_std - pp_nc) / pp_std * 100
tg_nc_save = (tg_std - tg_nc) / tg_std * 100
print(f"no_calibration (triple_pq): PP {pp_nc:.0f} MB ({pp_nc_save:+.1f}%), TG {tg_nc:.0f} MB ({tg_nc_save:+.1f}%)")

print(f"\n✅ 正确的测试方式:")
print(f"   1. 用 load_card() 加载配置")
print(f"   2. 用 card.to_cache_kwargs(mode) 获取参数")
print(f"   3. 不要手写参数！")
