#!/usr/bin/env python3
"""
测试 MAC 在重复内容下的性能
"""

import sys
sys.path.insert(0, "src")
sys.path.insert(0, "mlx-lm-source")

from mlx_lm import load, generate
import flashmlx
import time

model, tokenizer = load("/Volumes/toshiba/models/qwen3-8b-mlx")

# 高重复内容 - 应该有高 hit rate
contexts = [
    ("20K重复", "今天天气真好，" * 5000),
    ("30K重复", "今天天气真好，" * 7500),
    ("40K重复", "今天天气真好，" * 10000),
]

results = []

for name, prompt in contexts:
    actual_len = len(tokenizer.encode(prompt))

    print(f"{'='*80}")
    print(f"{name} ({actual_len:,} tokens)")
    print(f"{'='*80}")

    # Standard
    flashmlx.unpatch_mlx_lm()
    t0 = time.time()
    _ = generate(model, tokenizer, prompt=prompt, max_tokens=5, verbose=False)
    t_std = time.time() - t0
    tps_std = 5 / t_std
    print(f"  Standard: {t_std:.1f}s ({tps_std:.2f} tok/s)")

    # MAC
    flashmlx.patch_mlx_lm()
    t0 = time.time()
    _ = generate(model, tokenizer, prompt=prompt, max_tokens=5, verbose=False)
    t_mac = time.time() - t0
    tps_mac = 5 / t_mac
    print(f"  MAC:      {t_mac:.1f}s ({tps_mac:.2f} tok/s)")

    speedup = t_std / t_mac
    status = "🔥" if speedup >= 1.2 else ("✅" if speedup > 1.0 else "❌")
    print(f"  加速比:   {speedup:.2f}× {status}\n")

    results.append({'name': name, 'tokens': actual_len, 'speedup': speedup})

# 总结
print("=" * 80)
print("重复内容场景性能总结")
print("=" * 80)
print(f"{'Context':<15} {'Tokens':>8} {'Speedup':>10}")
print("-" * 80)

for r in results:
    status = "🔥" if r['speedup'] >= 1.2 else ("✅" if r['speedup'] > 1.0 else "❌")
    print(f"{r['name']:<15} {r['tokens']:>8,} {r['speedup']:>9.2f}× {status}")

best = max(results, key=lambda x: x['speedup'])
print(f"\n最佳加速: {best['name']} - {best['speedup']:.2f}×")
