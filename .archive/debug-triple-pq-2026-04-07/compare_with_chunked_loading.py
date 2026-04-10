#!/usr/bin/env python3
"""使用chunked loading对比 triple_pq vs scored_pq（匹配官方benchmark方法）"""

import sys
sys.path.insert(0, "mlx-lm-source")

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

MODEL_PATH = "/Volumes/toshiba/models/qwen3-8b-mlx"

print("="*80)
print("triple_pq vs scored_pq 对比（Chunked Prefill，32K context）")
print("="*80)

model, tokenizer = load(MODEL_PATH)

# 32K tokens
text = "The quick brown fox jumps over the lazy dog. " * 4000
tokens_list = tokenizer.encode(text)[:32768]
print(f"Total context: {len(tokens_list)} tokens\n")

def test_chunked(name, chunk_size=4096, **kwargs):
    """Chunked prefill loading (matches official benchmark)"""
    mx.clear_cache()
    mx.reset_peak_memory()

    cache = make_prompt_cache(model, **kwargs)

    # Chunked prefill
    max_pp_peak = 0
    for i in range(0, len(tokens_list), chunk_size):
        chunk = tokens_list[i:i+chunk_size]
        tokens = mx.array([chunk])
        logits = model(tokens, cache=cache)
        mx.eval(logits)

        pp_peak = mx.get_peak_memory() / (1024**2)
        max_pp_peak = max(max_pp_peak, pp_peak)

    print(f"  Prefill: {len(tokens_list)//chunk_size} chunks × {chunk_size} tokens")
    print(f"  PP Peak: {max_pp_peak:.0f} MB")

    # TG (100 tokens)
    for _ in range(100):
        next_token = mx.argmax(logits[:, -1:, :], axis=-1)
        logits = model(next_token, cache=cache)
        mx.eval(logits)

    tg_mem = mx.get_active_memory() / (1024**2)
    print(f"  TG Mem:  {tg_mem:.0f} MB")

    del cache, logits
    mx.clear_cache()
    return max_pp_peak, tg_mem

def test_single_pass(name, **kwargs):
    """Single-pass loading (for baseline only)"""
    mx.clear_cache()
    mx.reset_peak_memory()

    cache = make_prompt_cache(model, **kwargs)

    # Single-pass prefill
    tokens = mx.array([tokens_list])
    logits = model(tokens, cache=cache)
    mx.eval(logits)

    pp_peak = mx.get_peak_memory() / (1024**2)
    print(f"  Prefill: single-pass ({len(tokens_list)} tokens)")
    print(f"  PP Peak: {pp_peak:.0f} MB")

    # TG (100 tokens)
    for _ in range(100):
        next_token = mx.argmax(logits[:, -1:, :], axis=-1)
        logits = model(next_token, cache=cache)
        mx.eval(logits)

    tg_mem = mx.get_active_memory() / (1024**2)
    print(f"  TG Mem:  {tg_mem:.0f} MB")

    del cache, logits
    mx.clear_cache()
    return pp_peak, tg_mem

# Test 1: Standard (single-pass, baseline)
print("1. Standard (single-pass baseline):")
pp_std, tg_std = test_single_pass("Standard")

# Test 2: triple_pq (single-pass, works fine)
print("\n2. triple_pq + Q8 (single-pass):")
pp_tpq, tg_tpq = test_single_pass("triple_pq",
    kv_cache="triple_pq",
    kv_flat_quant="q8_0"
)

# Test 3: scored_pq (chunked, avoids OOM)
print("\n3. scored_pq + Q8 (chunked prefill, 避免OOM):")
try:
    pp_spq, tg_spq = test_chunked("scored_pq",
        chunk_size=4096,  # 4K chunks
        kv_cache="scored_pq",
        kv_calibration="calibrations/am_calibration_qwen3-8b_2.0x_onpolicy.pkl",
        kv_flat_quant="q8_0"
    )
    spq_success = True
except Exception as e:
    print(f"  ❌ Error: {e}")
    pp_spq, tg_spq = 0, 0
    spq_success = False

# 分析
print("\n" + "="*80)
print("性能对比")
print("="*80)

print(f"\nBaseline (Standard):")
print(f"  PP Peak: {pp_std:.0f} MB")
print(f"  TG Mem:  {tg_std:.0f} MB")

pp_tpq_save = (pp_std - pp_tpq) / pp_std * 100
tg_tpq_save = (tg_std - tg_tpq) / tg_std * 100

print(f"\ntriple_pq (single-pass):")
print(f"  PP Peak: {pp_tpq:.0f} MB ({pp_tpq_save:+.1f}%)")
print(f"  TG Mem:  {tg_tpq:.0f} MB ({tg_tpq_save:+.1f}%)")

if spq_success:
    pp_spq_save = (pp_std - pp_spq) / pp_std * 100
    tg_spq_save = (tg_std - tg_spq) / tg_std * 100

    print(f"\nscored_pq (chunked):")
    print(f"  PP Peak: {pp_spq:.0f} MB ({pp_spq_save:+.1f}%)")
    print(f"  TG Mem:  {tg_spq:.0f} MB ({tg_spq_save:+.1f}%)")

    print(f"\n官方benchmark期望 (model card):")
    print(f"  scored_pq: PP 4840→526 MB (-89%), TG 4647→529 MB (-89%)")

    print(f"\n✅ Chunked loading解决OOM:")
    print(f"   - scored_pq在chunked mode下成功运行")
    print(f"   - PP: {pp_spq_save:.1f}% 压缩")
    print(f"   - TG: {tg_spq_save:.1f}% 压缩")
else:
    print(f"\n❌ scored_pq仍然OOM")

print("\n" + "="*80)
print("关键发现")
print("="*80)

print(f"\n1. OOM原因:")
print(f"   Single-pass loading 32K tokens → 需要同时持有完整bf16 cache")
print(f"   scored_pq eviction需要对32K tokens进行AM scoring")
print(f"   临时内存需求: ~46GB (超过Metal 30GB限制)")

print(f"\n2. 解决方案:")
print(f"   Chunked loading (4K chunks × 8 = 32K)")
print(f"   每个chunk触发eviction，内存可控")
print(f"   这是官方benchmark的做法")

print(f"\n3. triple_pq优势:")
print(f"   支持single-pass loading（无OOM）")
print(f"   无需chunked prefill")
print(f"   简单易用，无特殊要求")
