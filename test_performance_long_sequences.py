#!/usr/bin/env python3
"""
长序列性能测试 - 对比有/无 Attention Matching 的性能影响

测试指标：
1. TG (Tokens/Second) - 生成速度
2. TTFT (Time To First Token) - 首个 token 延迟
3. Total Time - 总耗时

测试序列长度：
- 1K tokens (1024)
- 2K tokens (2048)
- 4K tokens (4096)
"""

import mlx.core as mx
from mlx_lm import load, generate
from flashmlx.cache.attention_matching_compressor_v2 import AttentionMatchingCompressorV2
import time
from typing import Dict, List
import statistics


def generate_long_context(length_tokens: int) -> str:
    """生成指定长度的长文本（约每个单词 1.3 tokens）"""

    # 基础段落
    base_text = """
    Deep learning is a subset of machine learning that uses neural networks with multiple layers
    to progressively extract higher-level features from raw input. The key advantage of deep learning
    is that features are learned automatically from data, rather than being manually engineered.
    This approach has achieved remarkable success in various domains including computer vision,
    natural language processing, and speech recognition. The architecture of deep neural networks
    typically consists of an input layer, multiple hidden layers, and an output layer. Each layer
    transforms its input data to a more abstract representation. Training these networks requires
    large amounts of data and significant computational resources, but the results can be
    extraordinarily powerful. Modern deep learning frameworks like PyTorch and TensorFlow have
    made it easier to build and train complex models. The field continues to evolve rapidly with
    new architectures and techniques being developed constantly.
    """

    # 重复生成足够长的文本
    words_needed = int(length_tokens / 1.3)  # 约 1.3 tokens/word
    base_words = base_text.split()
    words = (base_words * (words_needed // len(base_words) + 1))[:words_needed]

    return " ".join(words)


def measure_generation_performance(
    model,
    tokenizer,
    prompt: str,
    use_compression: bool = False,
    compression_ratio: float = 2.0,
    max_tokens: int = 100,
    num_runs: int = 3
) -> Dict:
    """测量生成性能（多次运行取平均）"""

    ttfts = []
    tgs = []
    total_times = []

    for run in range(num_runs):
        if use_compression:
            # 注入压缩
            compressor = AttentionMatchingCompressorV2(
                compression_ratio=compression_ratio,
                score_method='max',
                beta_method='nnls',
                c2_method='lsq',
                num_queries=100
            )

            # 计数器
            first_token_time = None
            start_time = None

            original_forward = model.model.layers[0].__class__.forward

            def compressed_forward(self, x, mask=None, cache=None):
                nonlocal first_token_time, start_time

                if start_time is None:
                    start_time = time.time()

                output = original_forward(self, x, mask, cache)

                # 记录第一个 token
                if first_token_time is None and cache is not None:
                    first_token_time = time.time() - start_time

                # 压缩 cache
                if cache is not None and hasattr(cache, 'keys') and hasattr(cache, 'values'):
                    layer_idx = self.layer_idx if hasattr(self, 'layer_idx') else 0
                    compressed_keys, compressed_values = compressor.compress_kv_cache(
                        layer_idx=layer_idx,
                        kv_cache=(cache.keys, cache.values)
                    )
                    cache.keys = compressed_keys
                    cache.values = compressed_values

                return output

            # 注入到所有层
            for layer_idx, layer in enumerate(model.model.layers):
                layer.layer_idx = layer_idx
                layer.__class__.forward = compressed_forward

        else:
            # 无压缩
            first_token_time = None
            start_time = None

            original_forward = model.model.layers[0].__class__.forward

            def uncompressed_forward(self, x, mask=None, cache=None):
                nonlocal first_token_time, start_time

                if start_time is None:
                    start_time = time.time()

                output = original_forward(self, x, mask, cache)

                # 记录第一个 token
                if first_token_time is None and cache is not None:
                    first_token_time = time.time() - start_time

                return output

            # 注入到所有层
            for layer_idx, layer in enumerate(model.model.layers):
                layer.layer_idx = layer_idx
                layer.__class__.forward = uncompressed_forward

        # 生成
        start = time.time()
        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False
        )
        total_time = time.time() - start

        # 计算指标
        tokens_generated = len(tokenizer.encode(response)) - len(tokenizer.encode(prompt))
        tg = tokens_generated / total_time if total_time > 0 else 0
        ttft = first_token_time if first_token_time is not None else 0

        ttfts.append(ttft)
        tgs.append(tg)
        total_times.append(total_time)

        # 重置 model
        for layer in model.model.layers:
            layer.__class__.forward = original_forward

    # 取平均
    return {
        "ttft_avg": statistics.mean(ttfts),
        "ttft_std": statistics.stdev(ttfts) if len(ttfts) > 1 else 0,
        "tg_avg": statistics.mean(tgs),
        "tg_std": statistics.stdev(tgs) if len(tgs) > 1 else 0,
        "total_time_avg": statistics.mean(total_times),
        "total_time_std": statistics.stdev(total_times) if len(total_times) > 1 else 0,
        "num_runs": num_runs
    }


def test_sequence_length(
    model,
    tokenizer,
    seq_length: int,
    compression_ratio: float = 2.0,
    max_tokens: int = 100
) -> Dict:
    """测试指定序列长度"""

    print(f"\n{'='*70}")
    print(f"测试序列长度: {seq_length} tokens")
    print(f"{'='*70}")

    # 生成长文本
    long_text = generate_long_context(seq_length)
    prompt = f"{long_text}\n\nSummarize the above text in 2-3 sentences:"

    # 验证长度
    actual_tokens = len(tokenizer.encode(prompt))
    print(f"实际 Prompt tokens: {actual_tokens}")

    # 无压缩
    print(f"\n测试无压缩（3 runs）...")
    baseline = measure_generation_performance(
        model, tokenizer, prompt,
        use_compression=False,
        max_tokens=max_tokens,
        num_runs=3
    )
    print(f"✓ 完成")

    # 有压缩
    print(f"\n测试有压缩 ({compression_ratio}x, 3 runs)...")
    compressed = measure_generation_performance(
        model, tokenizer, prompt,
        use_compression=True,
        compression_ratio=compression_ratio,
        max_tokens=max_tokens,
        num_runs=3
    )
    print(f"✓ 完成")

    # 计算差异
    ttft_diff = ((compressed["ttft_avg"] - baseline["ttft_avg"]) / baseline["ttft_avg"]) * 100
    tg_diff = ((compressed["tg_avg"] - baseline["tg_avg"]) / baseline["tg_avg"]) * 100
    total_diff = ((compressed["total_time_avg"] - baseline["total_time_avg"]) / baseline["total_time_avg"]) * 100

    print(f"\n性能对比:")
    print(f"  {'指标':<20} {'无压缩':<20} {'有压缩':<20} {'差异'}")
    print(f"  {'-'*75}")
    print(f"  {'TTFT (s)':<20} {baseline['ttft_avg']:>8.3f} ± {baseline['ttft_std']:>5.3f}  {compressed['ttft_avg']:>8.3f} ± {compressed['ttft_std']:>5.3f}  {ttft_diff:>+6.1f}%")
    print(f"  {'TG (tok/s)':<20} {baseline['tg_avg']:>8.2f} ± {baseline['tg_std']:>5.2f}  {compressed['tg_avg']:>8.2f} ± {compressed['tg_std']:>5.2f}  {tg_diff:>+6.1f}%")
    print(f"  {'Total Time (s)':<20} {baseline['total_time_avg']:>8.3f} ± {baseline['total_time_std']:>5.3f}  {compressed['total_time_avg']:>8.3f} ± {compressed['total_time_std']:>5.3f}  {total_diff:>+6.1f}%")

    # 判断
    if abs(tg_diff) <= 10:
        perf_rating = "✅ EXCELLENT (<10% impact)"
    elif abs(tg_diff) <= 20:
        perf_rating = "⚠️  ACCEPTABLE (10-20% impact)"
    else:
        perf_rating = "❌ POOR (>20% impact)"

    print(f"\n性能评级: {perf_rating}")

    return {
        "seq_length": seq_length,
        "actual_tokens": actual_tokens,
        "baseline": baseline,
        "compressed": compressed,
        "ttft_diff": ttft_diff,
        "tg_diff": tg_diff,
        "total_diff": total_diff,
        "perf_rating": perf_rating
    }


def main():
    print("="*70)
    print("长序列性能测试 - Attention Matching")
    print("="*70)

    # 加载模型
    print("\n加载模型...")
    model_path = "mlx-community/Qwen2.5-3B-Instruct-4bit"
    model, tokenizer = load(model_path)
    print(f"✓ 模型加载完成: {model_path}")

    # 测试不同序列长度
    seq_lengths = [1024, 2048, 4096]
    compression_ratio = 2.0

    results = []
    for seq_len in seq_lengths:
        result = test_sequence_length(
            model, tokenizer,
            seq_length=seq_len,
            compression_ratio=compression_ratio,
            max_tokens=100
        )
        results.append(result)
        time.sleep(2)  # 冷却

    # 总结
    print(f"\n\n{'='*70}")
    print("总结")
    print(f"{'='*70}")

    print(f"\n{'Seq Length':<15} {'TTFT Δ%':<12} {'TG Δ%':<12} {'Total Δ%':<12} {'评级'}")
    print("-" * 70)

    for result in results:
        print(f"{result['actual_tokens']:<15} {result['ttft_diff']:>+7.1f}%     {result['tg_diff']:>+7.1f}%     {result['total_diff']:>+7.1f}%     {result['perf_rating']}")

    # 最终判断
    avg_tg_diff = sum(r['tg_diff'] for r in results) / len(results)
    print(f"\n平均 TG 影响: {avg_tg_diff:+.1f}%")

    print(f"\n{'='*70}")
    if abs(avg_tg_diff) <= 10:
        print("✅ 最终结果: EXCELLENT - 平均性能影响 <10%")
    elif abs(avg_tg_diff) <= 20:
        print("⚠️  最终结果: ACCEPTABLE - 平均性能影响 10-20%")
    else:
        print("❌ 最终结果: POOR - 平均性能影响 >20%")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
