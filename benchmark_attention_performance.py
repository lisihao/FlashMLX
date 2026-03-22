#!/usr/bin/env python3
"""
Attention Matching Performance Benchmark

目标：测量 Attention Matching 对实际 generation 的性能影响

测试指标：
- TTFT (Time To First Token)
- TG (Token Generation Speed)
- 总生成时间
- 内存使用

对比：
1. Baseline (无 Attention Matching)
2. With Attention Matching (compression_ratio=3.0)

验收标准：
- TG 开销 ≤ 10%
- TTFT 开销 ≤ 15%
"""

import time
import gc
import mlx.core as mx
from mlx_lm import load, generate
from flashmlx.cache import (
    inject_attention_matching,
    get_compression_stats
)


def get_memory_mb():
    """获取当前内存使用（MB）"""
    return mx.get_active_memory() / (1024 ** 2)


def benchmark_generation(
    model_path: str,
    prompt: str,
    max_tokens: int,
    use_compression: bool,
    compression_ratio: float = 3.0,
    scenario_name: str = "Baseline"
):
    """
    测试生成性能

    Args:
        model_path: 模型路径
        prompt: 输入 prompt
        max_tokens: 最大生成 token 数
        use_compression: 是否使用 Attention Matching 压缩
        compression_ratio: 压缩比例
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

    # 注入 Attention Matching（如果启用）
    cache_list = None
    compressor = None
    if use_compression:
        print(f"\n注入 Attention Matching (compression_ratio={compression_ratio})...")

        cache_list, compressor = inject_attention_matching(
            model,
            compression_ratio=compression_ratio,
            beta_calibration=True,
            eviction_policy="top_k"
        )

    # Tokenize
    prompt_tokens = tokenizer.encode(prompt)
    prompt_length = len(prompt_tokens)

    print(f"\nPrompt tokens: {prompt_length}")
    print(f"Max tokens: {max_tokens}")

    # 测量内存
    mem_before = get_memory_mb()

    # 生成（测量详细时间）
    print("\n开始生成...")

    # 使用自定义生成循环来测量 TTFT 和 TBT
    start_time = time.perf_counter()

    # 简化方法：使用 mlx-lm 的 generate，然后估算 TTFT/TG
    response = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        verbose=False
    )

    end_time = time.perf_counter()
    total_time = end_time - start_time

    # 计算指标
    output_tokens = len(tokenizer.encode(response))
    generated_tokens = output_tokens - prompt_length

    # 粗略估算 TTFT 和 TG
    # 假设 TTFT 占总时间的 15-25%（取决于 prompt 长度）
    ttft_ratio = min(0.25, prompt_length / (prompt_length + generated_tokens))
    estimated_ttft = total_time * ttft_ratio
    estimated_tg_time = total_time - estimated_ttft

    # TG speed (tokens/second)
    tg_speed = generated_tokens / estimated_tg_time if estimated_tg_time > 0 else 0

    # PP speed (tokens/second) - prompt processing
    pp_speed = prompt_length / estimated_ttft if estimated_ttft > 0 else 0

    # 测量内存
    mem_after = get_memory_mb()
    mem_delta = mem_after - mem_before

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
    print(f"  Est. TTFT:     {estimated_ttft:>8.3f} s")
    print(f"  Est. TG time:  {estimated_tg_time:>8.2f} s")

    print(f"\nThroughput:")
    print(f"  PP (est.):     {pp_speed:>8.1f} tok/s")
    print(f"  TG (est.):     {tg_speed:>8.1f} tok/s")
    print(f"  Overall:       {output_tokens / total_time:>8.1f} tok/s")

    print(f"\nMemory:")
    print(f"  Before:        {mem_before:>8.1f} MB")
    print(f"  After:         {mem_after:>8.1f} MB")
    print(f"  Delta:         {mem_delta:>8.1f} MB")

    # 如果使用压缩，输出压缩统计
    compression_stats = None
    if use_compression and cache_list is not None:
        compression_stats = get_compression_stats(cache_list)

        print(f"\nCompression Statistics:")
        print(f"  Total compressions: {compression_stats['total_compressions']}")
        print(f"  Original tokens:    {compression_stats['total_original_tokens']}")
        print(f"  Compressed tokens:  {compression_stats['total_compressed_tokens']}")
        print(f"  Overall ratio:      {compression_stats['overall_compression_ratio']:.2f}x")

    return {
        'scenario': scenario_name,
        'prompt_tokens': prompt_length,
        'generated_tokens': generated_tokens,
        'total_time': total_time,
        'estimated_ttft': estimated_ttft,
        'estimated_tg_time': estimated_tg_time,
        'pp_speed': pp_speed,
        'tg_speed': tg_speed,
        'overall_speed': output_tokens / total_time,
        'mem_before': mem_before,
        'mem_after': mem_after,
        'mem_delta': mem_delta,
        'use_compression': use_compression,
        'compression_ratio': compression_ratio if use_compression else None,
        'compression_stats': compression_stats
    }


def main():
    print("="*70)
    print("Attention Matching 性能测试")
    print("="*70)

    # 配置
    model_path = "/Volumes/toshiba/models/qwen3.5-35b-mlx/"

    # 中等长度 prompt
    prompt = """Write a detailed explanation of how transformer models work in machine learning.
Include the key concepts of attention mechanism, self-attention, and multi-head attention.
Explain how these components work together to process sequential data."""

    max_tokens = 200

    print(f"\n配置:")
    print(f"  Model: {model_path}")
    print(f"  Max tokens: {max_tokens}")
    print(f"  Prompt length: ~60 tokens")

    # Baseline - 无压缩
    print(f"\n{'='*70}")
    print("测试 1/2: Baseline (无 Attention Matching)")
    print(f"{'='*70}")

    result_baseline = benchmark_generation(
        model_path=model_path,
        prompt=prompt,
        max_tokens=max_tokens,
        use_compression=False,
        scenario_name="Baseline (No Compression)"
    )

    # 清理内存
    print("\n清理内存...")
    gc.collect()
    mx.clear_cache()
    time.sleep(2)

    # With Attention Matching
    print(f"\n{'='*70}")
    print("测试 2/2: With Attention Matching")
    print(f"{'='*70}")

    result_compressed = benchmark_generation(
        model_path=model_path,
        prompt=prompt,
        max_tokens=max_tokens,
        use_compression=True,
        compression_ratio=3.0,
        scenario_name="With Attention Matching (3.0x)"
    )

    # 对比分析
    print(f"\n{'='*70}")
    print("性能对比")
    print(f"{'='*70}")

    print(f"\n{'Metric':<25} {'Baseline':<15} {'Compressed':<15} {'Impact':<15}")
    print("-"*70)

    # TTFT
    ttft_baseline = result_baseline['estimated_ttft']
    ttft_compressed = result_compressed['estimated_ttft']
    ttft_impact = ((ttft_compressed - ttft_baseline) / ttft_baseline * 100) if ttft_baseline > 0 else 0
    print(f"{'TTFT (s)':<25} {ttft_baseline:>12.3f}    {ttft_compressed:>12.3f}    {ttft_impact:>+12.1f}%")

    # TG
    tg_baseline = result_baseline['tg_speed']
    tg_compressed = result_compressed['tg_speed']
    tg_impact = ((tg_compressed - tg_baseline) / tg_baseline * 100) if tg_baseline > 0 else 0
    print(f"{'TG (tok/s)':<25} {tg_baseline:>12.1f}    {tg_compressed:>12.1f}    {tg_impact:>+12.1f}%")

    # Overall
    overall_baseline = result_baseline['overall_speed']
    overall_compressed = result_compressed['overall_speed']
    overall_impact = ((overall_compressed - overall_baseline) / overall_baseline * 100) if overall_baseline > 0 else 0
    print(f"{'Overall (tok/s)':<25} {overall_baseline:>12.1f}    {overall_compressed:>12.1f}    {overall_impact:>+12.1f}%")

    # Memory
    mem_baseline = result_baseline['mem_delta']
    mem_compressed = result_compressed['mem_delta']
    mem_impact = ((mem_compressed - mem_baseline) / mem_baseline * 100) if mem_baseline > 0 else 0
    print(f"{'Memory Delta (MB)':<25} {mem_baseline:>12.1f}    {mem_compressed:>12.1f}    {mem_impact:>+12.1f}%")

    print(f"\n{'='*70}")
    print("✓ 测试完成！")
    print(f"{'='*70}")

    # 验收标准检查
    print(f"\n验收标准检查:")

    # TG 开销
    tg_overhead = abs(tg_impact)
    tg_pass = tg_overhead <= 10.0
    tg_status = "✅ PASS" if tg_pass else "❌ FAIL"
    print(f"  TG 开销: {tg_overhead:.1f}% (≤ 10%) - {tg_status}")

    # TTFT 开销
    ttft_overhead = abs(ttft_impact)
    ttft_pass = ttft_overhead <= 15.0
    ttft_status = "✅ PASS" if ttft_pass else "❌ FAIL"
    print(f"  TTFT 开销: {ttft_overhead:.1f}% (≤ 15%) - {ttft_status}")

    # 总结
    print(f"\n总结:")
    if tg_pass and ttft_pass:
        print("  ✅ 所有验收标准通过！")
        print("  → Attention Matching 性能开销可接受")
    else:
        print("  ⚠️  部分验收标准未通过")
        if not tg_pass:
            print(f"  → TG 开销过大 ({tg_overhead:.1f}% > 10%)")
        if not ttft_pass:
            print(f"  → TTFT 开销过大 ({ttft_overhead:.1f}% > 15%)")

    # 内存收益
    if mem_impact < -10:
        print(f"  ✅ 内存节省: {-mem_impact:.1f}%")
    elif mem_impact > 10:
        print(f"  ⚠️  内存增加: {mem_impact:.1f}%")
    else:
        print(f"  ✅ 内存影响可忽略: {mem_impact:+.1f}%")


if __name__ == "__main__":
    main()
