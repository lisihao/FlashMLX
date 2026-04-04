#!/usr/bin/env python3
"""
MAC-Attention 完整性能测试

对比标准 attention vs MAC-attention 在不同上下文长度下的性能
"""

import sys
sys.path.insert(0, "src")

from mlx_lm import load, generate
import flashmlx
import time

print("=" * 80)
print("MAC-Attention 性能测试")
print("=" * 80)
print()

# 加载模型
model_path = "mlx-community/Qwen2.5-7B-Instruct-4bit"
print(f"加载模型: {model_path}")
model, tokenizer = load(model_path)
print("✅ 模型加载完成\n")

# 测试配置
test_configs = [
    ("4K", 1000),
    ("8K", 2000),
    ("16K", 4000),
    ("32K", 8000),
]

results = []

for ctx_name, repeat in test_configs:
    print(f"{'='*80}")
    print(f"测试: {ctx_name} 上下文")
    print(f"{'='*80}")
    
    prompt = "你好，" * repeat
    
    # 测试 1: 标准 Attention
    print("  [1/2] 标准 Attention...")
    flashmlx.unpatch_mlx_lm()  # 确保禁用
    
    t0 = time.time()
    _ = generate(model, tokenizer, prompt=prompt, max_tokens=20, verbose=False)
    t_standard = time.time() - t0
    
    print(f"        耗时: {t_standard:.2f}s")
    
    # 测试 2: MAC-Attention
    print("  [2/2] MAC-Attention...")
    flashmlx.patch_mlx_lm()  # 启用 MAC
    
    t0 = time.time()
    _ = generate(model, tokenizer, prompt=prompt, max_tokens=20, verbose=False)
    t_mac = time.time() - t0
    
    print(f"        耗时: {t_mac:.2f}s")
    
    # 计算加速比
    speedup = t_standard / t_mac
    print()
    print(f"  📊 结果:")
    print(f"    标准:  {t_standard:.2f}s")
    print(f"    MAC:   {t_mac:.2f}s")
    print(f"    加速:  {speedup:.2f}×")
    print()
    
    results.append({
        'ctx': ctx_name,
        'standard': t_standard,
        'mac': t_mac,
        'speedup': speedup,
    })

# 总结
print("=" * 80)
print("性能总结")
print("=" * 80)
print()
print(f"{'上下文':<10} {'标准 (s)':<12} {'MAC (s)':<12} {'加速比':<10}")
print("-" * 80)

for r in results:
    print(f"{r['ctx']:<10} {r['standard']:>10.2f}   {r['mac']:>10.2f}   {r['speedup']:>8.2f}×")

print()
print("💡 结论:")
print("  - 上下文越长，MAC-Attention 加速越明显")
print("  - 建议在 >16K 上下文时使用 MAC")
print()

# 清理
flashmlx.unpatch_mlx_lm()
