#!/usr/bin/env python3
"""
Qwen3 8B + MAC-Attention 实战测试

对比标准 vs MAC 在真实模型上的性能
"""

import sys
sys.path.insert(0, "src")
sys.path.insert(0, "mlx-lm-source")

import time
import mlx.core as mx

print("=" * 80)
print("Qwen3 8B + MAC-Attention 实战测试")
print("=" * 80)
print()

# 加载模型
print("🔄 加载模型: /Volumes/toshiba/models/qwen3-8b-mlx")
print()

from mlx_lm import load, generate

model_path = "/Volumes/toshiba/models/qwen3-8b-mlx"
model, tokenizer = load(model_path)

print("✅ 模型加载完成")
print()

# 测试配置
test_configs = [
    ("8K", "你好，请介绍一下你自己。" * 2000, 20),
    ("16K", "你好，请介绍一下你自己。" * 4000, 20),
    ("32K", "你好，请介绍一下你自己。" * 8000, 20),
]

results = []

for ctx_name, prompt, max_tokens in test_configs:
    print(f"{'='*80}")
    print(f"测试: {ctx_name} 上下文 (生成 {max_tokens} tokens)")
    print(f"{'='*80}")
    
    # 计算实际 token 数
    prompt_tokens = tokenizer.encode(prompt)
    actual_len = len(prompt_tokens)
    print(f"  实际上下文长度: {actual_len:,} tokens")
    print()
    
    # 测试 1: 标准 Attention
    print("  [1/2] 标准 Attention (baseline)...")
    
    import flashmlx
    flashmlx.unpatch_mlx_lm()  # 确保禁用 MAC
    
    # Warmup
    _ = generate(model, tokenizer, prompt=prompt[:100], max_tokens=1, verbose=False)
    mx.eval(mx.array(0))
    
    # 实际测试
    t0 = time.time()
    response_std = generate(
        model, tokenizer, 
        prompt=prompt, 
        max_tokens=max_tokens, 
        verbose=False
    )
    t_standard = time.time() - t0
    
    # 计算生成速度
    generated_tokens = len(tokenizer.encode(response_std)) - actual_len
    std_tps = generated_tokens / t_standard if t_standard > 0 else 0
    
    print(f"      耗时: {t_standard:.2f}s")
    print(f"      生成: {generated_tokens} tokens")
    print(f"      速度: {std_tps:.1f} tok/s")
    print()
    
    # 测试 2: MAC-Attention
    print("  [2/2] MAC-Attention (optimized)...")
    
    flashmlx.patch_mlx_lm()  # 启用 MAC
    
    # Warmup
    _ = generate(model, tokenizer, prompt=prompt[:100], max_tokens=1, verbose=False)
    mx.eval(mx.array(0))
    
    # 实际测试
    t0 = time.time()
    response_mac = generate(
        model, tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        verbose=False
    )
    t_mac = time.time() - t0
    
    # 计算生成速度
    generated_tokens = len(tokenizer.encode(response_mac)) - actual_len
    mac_tps = generated_tokens / t_mac if t_mac > 0 else 0
    
    print(f"      耗时: {t_mac:.2f}s")
    print(f"      生成: {generated_tokens} tokens")
    print(f"      速度: {mac_tps:.1f} tok/s")
    print()
    
    # 计算加速比
    speedup = t_standard / t_mac if t_mac > 0 else 0
    tps_speedup = mac_tps / std_tps if std_tps > 0 else 0
    
    print(f"  📊 结果:")
    print(f"    标准:  {t_standard:>6.2f}s  ({std_tps:>6.1f} tok/s)")
    print(f"    MAC:   {t_mac:>6.2f}s  ({mac_tps:>6.1f} tok/s)")
    print(f"    加速:  {speedup:>6.2f}×  ({tps_speedup:>6.2f}× tok/s)")
    
    # 验证输出一致性
    if response_std[:50] == response_mac[:50]:
        print(f"    ✅ 输出一致")
    else:
        print(f"    ⚠️  输出有差异")
    
    print()
    
    results.append({
        'ctx': ctx_name,
        'tokens': actual_len,
        't_std': t_standard,
        't_mac': t_mac,
        'tps_std': std_tps,
        'tps_mac': mac_tps,
        'speedup': speedup,
    })

# 总结
print("=" * 80)
print("📈 性能总结")
print("=" * 80)
print()
print(f"{'Context':<10} {'Tokens':<10} {'标准 (s)':<12} {'MAC (s)':<12} {'加速比':<10}")
print("-" * 80)

for r in results:
    print(f"{r['ctx']:<10} {r['tokens']:<10,} {r['t_std']:>10.2f}   {r['t_mac']:>10.2f}   {r['speedup']:>8.2f}×")

print()
print(f"{'Context':<10} {'标准 tok/s':<15} {'MAC tok/s':<15} {'提升':<10}")
print("-" * 80)

for r in results:
    print(f"{r['ctx']:<10} {r['tps_std']:>12.1f}    {r['tps_mac']:>12.1f}    {r['tps_mac']/r['tps_std']:>7.2f}×")

print()
print("=" * 80)
print("🎯 结论")
print("=" * 80)
print()

# 找出最佳加速比
best = max(results, key=lambda x: x['speedup'])
print(f"✨ 最佳加速: {best['ctx']} 上下文 ({best['speedup']:.2f}×)")
print()
print("💡 建议:")
if any(r['speedup'] > 2.0 for r in results):
    print("  ✅ MAC-Attention 在长上下文场景下效果显著")
    print("  ✅ 建议在 >16K 上下文时启用")
else:
    print("  ⚠️  当前测试未达到预期加速比")
    print("  💭 可能原因：模型较小 / 上下文不够长 / 系统状态")

print()

# 清理
flashmlx.unpatch_mlx_lm()
print("🔄 已恢复标准 attention")
