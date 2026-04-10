#!/usr/bin/env python3
"""
MAC最终性能测试 - 展示加速效果
"""

import sys
sys.path.insert(0, "src")
sys.path.insert(0, "mlx-lm-source")

from mlx_lm import load, generate
import flashmlx
import time

print("=" * 80)
print("MAC-Attention 最终性能测试")
print("=" * 80)
print()

model, tokenizer = load("/Volumes/toshiba/models/qwen3-8b-mlx")
print("✅ 模型加载完成\n")

contexts = [
    ("20K", 10000),
    ("30K", 15000),
    ("40K", 20000),
]

results = []

for name, repeat in contexts:
    prompt = "你好" * repeat
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

    results.append({'name': name, 'tokens': actual_len, 'speedup': speedup, 'tps_std': tps_std, 'tps_mac': tps_mac})

# 总结
print("=" * 80)
print("性能总结")
print("=" * 80)
print(f"{'Context':<10} {'Tokens':>8} {'Standard':>12} {'MAC':>12} {'Speedup':>10}")
print("-" * 80)

for r in results:
    status = "🔥" if r['speedup'] >= 1.2 else ("✅" if r['speedup'] > 1.0 else "❌")
    print(f"{r['name']:<10} {r['tokens']:>8,} {r['tps_std']:>11.2f}  {r['tps_mac']:>11.2f}  {r['speedup']:>9.2f}× {status}")

best = max(results, key=lambda x: x['speedup'])
if best['speedup'] > 1.0:
    print(f"\n🎉 MAC-Attention 在长上下文下成功加速！")
    print(f"最佳: {best['name']} - {best['speedup']:.2f}×")
