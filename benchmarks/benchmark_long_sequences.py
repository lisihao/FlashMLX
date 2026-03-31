#!/usr/bin/env python3
"""
Long Sequence Attention Matching Benchmark

测试不同序列长度下 Attention Matching 的性能影响：
- 1K tokens
- 2K tokens
- 4K tokens
- 8K tokens

关键指标：
- TG 速度 (tokens/second)
- TTFT (Time To First Token)
- 内存使用
- 实际压缩率
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


def generate_long_prompt(target_tokens: int) -> str:
    """
    生成指定长度的 prompt

    Args:
        target_tokens: 目标 token 数量

    Returns:
        生成的 prompt 文本
    """
    # 基础文本（约每句 20-30 tokens）
    sentences = [
        "Machine learning is a subset of artificial intelligence that focuses on developing algorithms and statistical models.",
        "Deep learning, a subset of machine learning, uses neural networks with multiple layers to learn hierarchical representations.",
        "Natural language processing enables computers to understand, interpret, and generate human language in a valuable way.",
        "Computer vision allows machines to interpret and understand visual information from the world, similar to human vision.",
        "Reinforcement learning is a type of machine learning where agents learn to make decisions by interacting with an environment.",
        "Transfer learning leverages knowledge gained from solving one problem and applies it to a different but related problem.",
        "Generative models can create new data instances that resemble the training data, such as images, text, or audio.",
        "Attention mechanisms allow models to focus on specific parts of the input when making predictions or generating outputs.",
        "Transformer architecture has revolutionized natural language processing by using self-attention mechanisms effectively.",
        "Large language models like GPT and BERT have achieved remarkable performance on various NLP tasks and benchmarks.",
    ]

    # 重复句子直到达到目标长度
    prompt = ""
    while len(prompt.split()) < target_tokens * 0.75:  # 粗略估算（单词数 ≈ token数 × 0.75）
        for sentence in sentences:
            prompt += sentence + " "
            if len(prompt.split()) >= target_tokens * 0.75:
                break

    # 添加问题
    prompt += "\n\nBased on the above information, please provide a detailed summary of the key concepts in machine learning and deep learning."

    return prompt


def benchmark_sequence_length(
    model,
    tokenizer,
    prompt_tokens: int,
    compression_ratio: float = 3.0,
    generation_tokens: int = 100
):
    """
    测试特定序列长度的性能

    Args:
        model: 模型实例
        tokenizer: Tokenizer 实例
        prompt_tokens: Prompt token 数量
        compression_ratio: 压缩比例
        generation_tokens: 生成 token 数量

    Returns:
        性能指标字典
    """
    print(f"\n{'='*70}")
    print(f"测试序列长度: {prompt_tokens} tokens")
    print(f"{'='*70}")

    # 生成长 prompt
    prompt = generate_long_prompt(prompt_tokens)

    # Tokenize 并截断到目标长度
    prompt_token_ids = tokenizer.encode(prompt)
    if len(prompt_token_ids) > prompt_tokens:
        prompt_token_ids = prompt_token_ids[:prompt_tokens]
    actual_prompt_tokens = len(prompt_token_ids)

    # Decode 回文本
    prompt = tokenizer.decode(prompt_token_ids)

    print(f"\nPrompt tokens: {actual_prompt_tokens}")
    print(f"Generation tokens: {generation_tokens}")

    results = {}

    # ====================================================================
    # 测试 1: Baseline (无压缩)
    # ====================================================================
    print(f"\n--- Baseline (无压缩) ---")

    gc.collect()
    mx.clear_cache()

    mem_before = get_memory_mb()

    start = time.perf_counter()
    response = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=generation_tokens,
        verbose=False
    )
    end = time.perf_counter()

    mem_after = get_memory_mb()

    baseline_time = end - start
    baseline_tokens = len(tokenizer.encode(response))
    baseline_gen_tokens = baseline_tokens - actual_prompt_tokens
    baseline_mem = mem_after - mem_before

    # 估算 TTFT 和 TG
    ttft_ratio = min(0.25, actual_prompt_tokens / (actual_prompt_tokens + baseline_gen_tokens))
    baseline_ttft = baseline_time * ttft_ratio
    baseline_tg_time = baseline_time - baseline_ttft
    baseline_tg_speed = baseline_gen_tokens / baseline_tg_time if baseline_tg_time > 0 else 0

    print(f"  Time: {baseline_time:.2f}s")
    print(f"  Est. TTFT: {baseline_ttft:.3f}s")
    print(f"  TG: {baseline_tg_speed:.1f} tok/s")
    print(f"  Memory: {baseline_mem:.1f} MB")

    results['baseline'] = {
        'total_time': baseline_time,
        'ttft': baseline_ttft,
        'tg_speed': baseline_tg_speed,
        'memory': baseline_mem,
        'generated_tokens': baseline_gen_tokens
    }

    # ====================================================================
    # 测试 2: With Compression
    # ====================================================================
    print(f"\n--- With Compression ({compression_ratio}x) ---")

    gc.collect()
    mx.clear_cache()

    # 重新加载模型（清除旧 cache）
    mem_before = get_memory_mb()

    # 注入压缩
    cache_list, compressor = inject_attention_matching(
        model,
        compression_ratio=compression_ratio,
        beta_calibration=True,
        eviction_policy="top_k"
    )

    start = time.perf_counter()
    response = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=generation_tokens,
        verbose=False
    )
    end = time.perf_counter()

    mem_after = get_memory_mb()

    compressed_time = end - start
    compressed_tokens = len(tokenizer.encode(response))
    compressed_gen_tokens = compressed_tokens - actual_prompt_tokens
    compressed_mem = mem_after - mem_before

    # 估算 TTFT 和 TG
    ttft_ratio = min(0.25, actual_prompt_tokens / (actual_prompt_tokens + compressed_gen_tokens))
    compressed_ttft = compressed_time * ttft_ratio
    compressed_tg_time = compressed_time - compressed_ttft
    compressed_tg_speed = compressed_gen_tokens / compressed_tg_time if compressed_tg_time > 0 else 0

    # 获取压缩统计
    compression_stats = get_compression_stats(cache_list)

    print(f"  Time: {compressed_time:.2f}s")
    print(f"  Est. TTFT: {compressed_ttft:.3f}s")
    print(f"  TG: {compressed_tg_speed:.1f} tok/s")
    print(f"  Memory: {compressed_mem:.1f} MB")
    print(f"  Compression ratio: {compression_stats['overall_compression_ratio']:.2f}x")
    print(f"  Total compressions: {compression_stats['total_compressions']}")

    results['compressed'] = {
        'total_time': compressed_time,
        'ttft': compressed_ttft,
        'tg_speed': compressed_tg_speed,
        'memory': compressed_mem,
        'generated_tokens': compressed_gen_tokens,
        'compression_ratio': compression_stats['overall_compression_ratio'],
        'total_compressions': compression_stats['total_compressions']
    }

    # ====================================================================
    # 对比
    # ====================================================================
    print(f"\n--- 对比 ---")

    tg_impact = ((compressed_tg_speed - baseline_tg_speed) / baseline_tg_speed * 100) if baseline_tg_speed > 0 else 0
    ttft_impact = ((compressed_ttft - baseline_ttft) / baseline_ttft * 100) if baseline_ttft > 0 else 0
    mem_impact = ((compressed_mem - baseline_mem) / baseline_mem * 100) if baseline_mem > 0 else 0

    print(f"  TG: {baseline_tg_speed:.1f} → {compressed_tg_speed:.1f} tok/s ({tg_impact:+.1f}%)")
    print(f"  TTFT: {baseline_ttft:.3f} → {compressed_ttft:.3f}s ({ttft_impact:+.1f}%)")
    print(f"  Memory: {baseline_mem:.1f} → {compressed_mem:.1f} MB ({mem_impact:+.1f}%)")

    results['impact'] = {
        'tg_impact': tg_impact,
        'ttft_impact': ttft_impact,
        'mem_impact': mem_impact
    }

    return results


def main():
    print("="*70)
    print("Long Sequence Attention Matching Benchmark")
    print("="*70)

    model_path = "/Volumes/toshiba/models/qwen3.5-35b-mlx/"

    print(f"\n加载模型: {model_path}")
    model, tokenizer = load(model_path)

    # 测试不同序列长度
    sequence_lengths = [1000, 2000, 4000, 8000]
    compression_ratio = 3.0
    generation_tokens = 100

    all_results = {}

    for seq_len in sequence_lengths:
        results = benchmark_sequence_length(
            model=model,
            tokenizer=tokenizer,
            prompt_tokens=seq_len,
            compression_ratio=compression_ratio,
            generation_tokens=generation_tokens
        )
        all_results[seq_len] = results

        # 清理
        gc.collect()
        mx.clear_cache()
        time.sleep(2)

    # ====================================================================
    # 汇总报告
    # ====================================================================
    print(f"\n{'='*70}")
    print("汇总报告")
    print(f"{'='*70}")

    print(f"\n{'Seq Len':<10} {'TG Impact':<15} {'TTFT Impact':<15} {'Mem Impact':<15} {'Compression':<15}")
    print("-"*70)

    for seq_len in sequence_lengths:
        results = all_results[seq_len]
        impact = results['impact']
        compressed = results['compressed']

        print(f"{seq_len:<10} {impact['tg_impact']:>+12.1f}%   {impact['ttft_impact']:>+12.1f}%   {impact['mem_impact']:>+12.1f}%   {compressed['compression_ratio']:>12.2f}x")

    # 验收标准
    print(f"\n{'='*70}")
    print("验收标准检查 (TG ≤ -10%, TTFT ≤ +15%)")
    print(f"{'='*70}")

    for seq_len in sequence_lengths:
        results = all_results[seq_len]
        impact = results['impact']

        tg_pass = impact['tg_impact'] >= -10.0
        ttft_pass = impact['ttft_impact'] <= 15.0

        tg_status = "✅" if tg_pass else "❌"
        ttft_status = "✅" if ttft_pass else "❌"

        print(f"\n{seq_len} tokens:")
        print(f"  TG: {impact['tg_impact']:+.1f}% {tg_status}")
        print(f"  TTFT: {impact['ttft_impact']:+.1f}% {ttft_status}")

        if tg_pass and ttft_pass:
            print(f"  → ✅ PASS")
        else:
            print(f"  → ❌ FAIL")

    print(f"\n{'='*70}")
    print("✓ 测试完成")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
