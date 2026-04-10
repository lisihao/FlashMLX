#!/usr/bin/env python3
"""
测试MAC在超长上下文下的表现
"""

import sys
sys.path.insert(0, "src")
sys.path.insert(0, "mlx-lm-source")

from mlx_lm import load, generate
import flashmlx
import time

print("=" * 80)
print("MAC 超长上下文测试")
print("=" * 80)
print()

# 加载模型
print("加载模型...")
model, tokenizer = load("/Volumes/toshiba/models/qwen3-8b-mlx")
print("✅ 模型加载完成\n")

# 测试超长上下文
test_configs = [
    ("10K", 5000),
    ("20K", 10000),
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
    try:
        _ = generate(model, tokenizer, prompt=prompt, max_tokens=5, verbose=False)
        t_std = time.time() - t0
        tps_std = 5 / t_std
        print(f"      耗时: {t_std:.2f}s, 速度: {tps_std:.2f} tok/s\n")
    except Exception as e:
        print(f"      ❌ 失败: {e}\n")
        t_std = None
        tps_std = 0

    # MAC attention
    print("  [2/2] MAC attention...")
    flashmlx.patch_mlx_lm()
    t0 = time.time()
    try:
        _ = generate(model, tokenizer, prompt=prompt, max_tokens=5, verbose=False)
        t_mac = time.time() - t0
        tps_mac = 5 / t_mac
        print(f"      耗时: {t_mac:.2f}s, 速度: {tps_mac:.2f} tok/s\n")
    except Exception as e:
        print(f"      ❌ 失败: {e}\n")
        t_mac = None
        tps_mac = 0

    if t_std and t_mac:
        speedup = t_std / t_mac
        print(f"  📊 加速比: {speedup:.2f}×\n")
        results.append({
            'name': name,
            'tokens': actual_len,
            'speedup': speedup,
            'tps_std': tps_std,
            'tps_mac': tps_mac,
        })

# 总结
if results:
    print("=" * 80)
    print("结果")
    print("=" * 80)
    for r in results:
        status = "✅" if r['speedup'] > 1.0 else "❌"
        print(f"{r['name']}: {r['speedup']:.2f}× {status}")

    best = max(results, key=lambda x: x['speedup'])
    if best['speedup'] > 1.0:
        print(f"\n🎉 找到加速点！{best['name']} 达到 {best['speedup']:.2f}×")
    else:
        print(f"\n😞 最高只有 {best['speedup']:.2f}×，仍未达到加速")
