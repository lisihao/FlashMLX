#!/usr/bin/env python3
"""
Benchmark: Simplified SSM Cache Impact on PP/TG/TTFT

⚠️  DEPRECATED: 2026-03-22 ⚠️
This benchmark is DISABLED. SSM cache has been deprecated.
See: SSM_CACHE_DEPRECATION.md

使用简化的 SSM 缓存（单层 dict，无 Hot/Warm/Cold）测试性能影响。

对比：
1. Baseline: 无 SSM 缓存
2. Simplified SSM Cache: 启用简化缓存（11x 开销）

测量指标：
- PP (Prompt Processing): tokens/second
- TG (Token Generation): tokens/second
- TTFT (Time To First Token): seconds
- Memory: MB
"""

import sys
print("⚠️  This benchmark is deprecated and disabled.")
print("SSM cache has been sealed. See SSM_CACHE_DEPRECATION.md")
print("\nUse ThunderLLAMA prefix caching instead:")
print("  - thunderllama.conf: THUNDER_LMCACHE=1")
print("  - See: ThunderLLAMA/.solar/benchmark-checklist.md")
sys.exit(0)

# Original benchmark code preserved below (not executed)
# ============================================================================

import time
import gc
import mlx.core as mx
from mlx_lm import load, generate
from flashmlx.cache import inject_simplified_ssm_cache


def get_memory_mb():
    """Get current memory usage in MB."""
    return mx.get_active_memory() / (1024 ** 2)


# Note: inject_simplified_ssm_cache is now imported from flashmlx.cache


def benchmark_generation(
    model_path: str,
    prompt: str,
    max_tokens: int,
    use_simplified_cache: bool,
    scenario_name: str
):
    """
    测试生成性能。

    Args:
        model_path: 模型路径
        prompt: 输入 prompt
        max_tokens: 最大生成 token 数
        use_simplified_cache: 是否使用简化 SSM 缓存
        scenario_name: 场景名称

    Returns:
        性能指标字典
    """
    print(f"\n{'='*70}")
    print(f"场景: {scenario_name}")
    print(f"{'='*70}")

    # 清理内存
    gc.collect()
    mx.clear_cache()

    # 加载模型
    print("\n加载模型...")
    model, tokenizer = load(model_path)

    # 注入缓存（如果启用）
    manager = None
    if use_simplified_cache:
        print("\n注入简化 SSM 缓存...")
        cache_list, manager = inject_simplified_ssm_cache(
            model,
            max_size_bytes=100 * 1024 * 1024,
            auto_inject=True  # 关键：自动注入到 model.cache
        )

    # Tokenize
    prompt_tokens = tokenizer.encode(prompt)
    prompt_length = len(prompt_tokens)

    print(f"\nPrompt tokens: {prompt_length}")
    print(f"Max tokens: {max_tokens}")

    # 测量内存
    mem_before = get_memory_mb()

    # 生成
    print("\n开始生成...")
    start_time = time.time()

    response = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        verbose=False
    )

    end_time = time.time()

    # 计算指标
    output_tokens = len(tokenizer.encode(response))
    generated_tokens = output_tokens - prompt_length
    total_time = end_time - start_time

    # 粗略估算 PP 和 TG
    # 假设 TTFT 占总时间的 20-30%
    estimated_ttft = total_time * 0.25
    estimated_tg_time = total_time - estimated_ttft

    pp_speed = prompt_length / estimated_ttft if estimated_ttft > 0 else 0
    tg_speed = generated_tokens / estimated_tg_time if estimated_tg_time > 0 else 0

    # 测量内存
    mem_after = get_memory_mb()
    mem_delta = mem_after - mem_before

    # 获取缓存统计
    cache_stats = None
    if manager is not None:
        cache_stats = manager.get_statistics()

    # 打印结果
    print(f"\n{'='*70}")
    print("结果")
    print(f"{'='*70}")

    print(f"\nTokens:")
    print(f"  Prompt:        {prompt_length:>8}")
    print(f"  Generated:     {generated_tokens:>8}")
    print(f"  Total:         {output_tokens:>8}")

    print(f"\nTiming:")
    print(f"  Total time:    {total_time:>8.2f} s")
    print(f"  Est. TTFT:     {estimated_ttft:>8.2f} s")
    print(f"  Est. TG time:  {estimated_tg_time:>8.2f} s")

    print(f"\nThroughput:")
    print(f"  PP (est.):     {pp_speed:>8.1f} tok/s")
    print(f"  TG (est.):     {tg_speed:>8.1f} tok/s")
    print(f"  Overall:       {output_tokens / total_time:>8.1f} tok/s")

    print(f"\nMemory:")
    print(f"  Before:        {mem_before:>8.1f} MB")
    print(f"  After:         {mem_after:>8.1f} MB")
    print(f"  Delta:         {mem_delta:>8.1f} MB")

    if cache_stats:
        print(f"\nSSM Cache Statistics:")
        print(f"  Entry count:   {cache_stats['entry_count']:>8}")
        print(f"  Size:          {cache_stats['size_bytes'] / 1024:>8.1f} KB")
        print(f"  Hits:          {cache_stats['hits']:>8}")
        print(f"  Misses:        {cache_stats['misses']:>8}")
        print(f"  Hit rate:      {cache_stats['hit_rate']:>8.1%}")

    return {
        'scenario': scenario_name,
        'prompt_tokens': prompt_length,
        'generated_tokens': generated_tokens,
        'total_time': total_time,
        'estimated_ttft': estimated_ttft,
        'pp_speed': pp_speed,
        'tg_speed': tg_speed,
        'overall_speed': output_tokens / total_time,
        'mem_delta': mem_delta,
        'cache_stats': cache_stats
    }


def main():
    print("="*70)
    print("简化 SSM 缓存性能测试")
    print("="*70)

    # 配置
    model_path = "/Volumes/toshiba/models/qwen3.5-35b-mlx/"

    # 中等长度 prompt
    prompt = """Write a concise explanation of how transformers work in machine learning.
Include the key concepts of attention mechanism and self-attention."""

    max_tokens = 100

    print(f"\n配置:")
    print(f"  Model: {model_path}")
    print(f"  Max tokens: {max_tokens}")

    # Baseline
    print(f"\n{'='*70}")
    print("测试 1/2: Baseline (无 SSM 缓存)")
    print(f"{'='*70}")

    result_baseline = benchmark_generation(
        model_path=model_path,
        prompt=prompt,
        max_tokens=max_tokens,
        use_simplified_cache=False,
        scenario_name="Baseline (No SSM Cache)"
    )

    # 清理内存
    print("\n清理内存...")
    gc.collect()
    mx.clear_cache()
    time.sleep(2)

    # Simplified SSM Cache
    print(f"\n{'='*70}")
    print("测试 2/2: Simplified SSM Cache")
    print(f"{'='*70}")

    result_cached = benchmark_generation(
        model_path=model_path,
        prompt=prompt,
        max_tokens=max_tokens,
        use_simplified_cache=True,
        scenario_name="Simplified SSM Cache"
    )

    # 对比
    print(f"\n{'='*70}")
    print("性能对比")
    print(f"{'='*70}")

    print(f"\n{'Metric':<25} {'Baseline':<15} {'Simplified':<15} {'Impact':<15}")
    print("-"*70)

    # PP
    pp_baseline = result_baseline['pp_speed']
    pp_cached = result_cached['pp_speed']
    pp_impact = ((pp_cached - pp_baseline) / pp_baseline * 100) if pp_baseline > 0 else 0
    print(f"{'PP (tok/s)':<25} {pp_baseline:>12.1f}    {pp_cached:>12.1f}    {pp_impact:>+12.1f}%")

    # TG
    tg_baseline = result_baseline['tg_speed']
    tg_cached = result_cached['tg_speed']
    tg_impact = ((tg_cached - tg_baseline) / tg_baseline * 100) if tg_baseline > 0 else 0
    print(f"{'TG (tok/s)':<25} {tg_baseline:>12.1f}    {tg_cached:>12.1f}    {tg_impact:>+12.1f}%")

    # TTFT
    ttft_baseline = result_baseline['estimated_ttft']
    ttft_cached = result_cached['estimated_ttft']
    ttft_impact = ((ttft_cached - ttft_baseline) / ttft_baseline * 100) if ttft_baseline > 0 else 0
    print(f"{'TTFT (s)':<25} {ttft_baseline:>12.2f}    {ttft_cached:>12.2f}    {ttft_impact:>+12.1f}%")

    # Overall
    overall_baseline = result_baseline['overall_speed']
    overall_cached = result_cached['overall_speed']
    overall_impact = ((overall_cached - overall_baseline) / overall_baseline * 100) if overall_baseline > 0 else 0
    print(f"{'Overall (tok/s)':<25} {overall_baseline:>12.1f}    {overall_cached:>12.1f}    {overall_impact:>+12.1f}%")

    # Memory
    mem_baseline = result_baseline['mem_delta']
    mem_cached = result_cached['mem_delta']
    mem_impact = ((mem_cached - mem_baseline) / mem_baseline * 100) if mem_baseline > 0 else 0
    print(f"{'Memory (MB)':<25} {mem_baseline:>12.1f}    {mem_cached:>12.1f}    {mem_impact:>+12.1f}%")

    print(f"\n{'='*70}")
    print("✓ 测试完成！")
    print(f"{'='*70}")

    # 总结
    print(f"\n关键发现:")
    if abs(tg_impact) < 1.0:
        print(f"  ✅ TG 影响极小 ({tg_impact:+.1f}%)")
        print(f"  → 简化 SSM 缓存开销可忽略")
    elif tg_impact < -1.0:
        print(f"  ⚠️  TG 略有下降 ({tg_impact:+.1f}%)")
        print(f"  → 缓存管理开销仍存在")
    else:
        print(f"  ✅ TG 略有提升 ({tg_impact:+.1f}%)")

    if result_cached['cache_stats']:
        hit_rate = result_cached['cache_stats']['hit_rate']
        if hit_rate > 0.7:
            print(f"  ✅ SSM 缓存命中率 {hit_rate:.1%}")
        else:
            print(f"  ⚠️  SSM 缓存命中率较低 {hit_rate:.1%}")


if __name__ == "__main__":
    main()
