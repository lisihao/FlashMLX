#!/usr/bin/env python3
"""
对比 MAC vs 标准 attention 性能
"""

import sys
sys.path.insert(0, "src")
sys.path.insert(0, "mlx-lm-source")

from mlx_lm import load, generate
import flashmlx
import time

print("=" * 80)
print("MAC vs Standard Attention 性能对比")
print("=" * 80)
print()

# 加载模型
print("加载模型...")
model, tokenizer = load("/Volumes/toshiba/models/qwen3-8b-mlx")
print("✅ 模型加载完成\n")

# 测试配置
test_configs = [
    ("500", 250),
    ("2K", 1000),
    ("5K", 2500),
]

results = []

for name, repeat in test_configs:
    prompt = "你好" * repeat
    actual_len = len(tokenizer.encode(prompt))
    print(f"{'='*80}")
    print(f"测试: {name} tokens ({actual_len:,} actual)")
    print(f"{'='*80}")

    # 标准 attention
    print("  [1/2] 标准 attention...")
    flashmlx.unpatch_mlx_lm()
    t0 = time.time()
    _ = generate(model, tokenizer, prompt=prompt, max_tokens=10, verbose=False)
    t_std = time.time() - t0
    tps_std = 10 / t_std
    print(f"      耗时: {t_std:.2f}s, 速度: {tps_std:.1f} tok/s\n")

    # MAC attention
    print("  [2/2] MAC attention...")
    flashmlx.patch_mlx_lm()
    t0 = time.time()
    _ = generate(model, tokenizer, prompt=prompt, max_tokens=10, verbose=False)
    t_mac = time.time() - t0
    tps_mac = 10 / t_mac
    print(f"      耗时: {t_mac:.2f}s, 速度: {tps_mac:.1f} tok/s\n")

    speedup = t_std / t_mac
    print(f"  📊 加速比: {speedup:.2f}×\n")

    results.append({
        'name': name,
        'tokens': actual_len,
        't_std': t_std,
        't_mac': t_mac,
        'tps_std': tps_std,
        'tps_mac': tps_mac,
        'speedup': speedup,
    })

# 总结
print("=" * 80)
print("性能总结")
print("=" * 80)
print()
print(f"{'Context':<10} {'Tokens':>8} {'Standard':>10} {'MAC':>10} {'Speedup':>10}")
print("-" * 80)

for r in results:
    status = "✅" if r['speedup'] > 1.0 else "❌"
    print(f"{r['name']:<10} {r['tokens']:>8,} {r['tps_std']:>10.1f} {r['tps_mac']:>10.1f} {r['speedup']:>9.2f}× {status}")

print()
print("💡 分析:")
best = max(results, key=lambda x: x['speedup'])
if best['speedup'] > 1.0:
    print(f"  ✅ 最佳加速: {best['name']} ({best['speedup']:.2f}×)")
    print(f"  ✅ 上下文越长，加速越明显")
else:
    print(f"  ❌ 当前所有测试都慢于标准attention")
    print(f"  💭 MAC可能只在更长上下文(>10K)有优势")
