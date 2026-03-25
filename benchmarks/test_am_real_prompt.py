#!/usr/bin/env python3
"""
AM 真实 Prompt 测试 - 使用正常的、非重复的 prompt
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "mlx-lm-source"))

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import KVCache, ArraysCache
from mlx_lm.models.compacted_cache import CompactedKVCache


# 一个真实的长 prompt（来自机器学习教程）
REAL_PROMPT = """Machine learning is a branch of artificial intelligence that focuses on teaching computers to learn from data. Instead of being explicitly programmed to perform a task, machine learning algorithms use statistical techniques to enable computers to improve their performance on a specific task through experience.

The core idea behind machine learning is to create algorithms that can automatically identify patterns in data and make decisions or predictions based on those patterns. This approach has proven highly effective in many domains, from image recognition and natural language processing to recommendation systems and autonomous vehicles.

There are three main categories of machine learning: supervised learning, unsupervised learning, and reinforcement learning. Supervised learning involves training a model on labeled data where the desired output is known. The algorithm learns to map inputs to outputs by analyzing the examples provided. Common applications include spam detection, image classification, and predictive modeling.

Unsupervised learning, on the other hand, works with unlabeled data. The algorithm must find patterns and structures in the data without being told what to look for. Clustering and dimensionality reduction are typical unsupervised learning tasks. For example, customer segmentation in marketing often uses unsupervised learning to group customers with similar characteristics.

Reinforcement learning is a type of machine learning where an agent learns to make decisions by interacting with an environment. The agent receives rewards or penalties based on its actions and learns to maximize cumulative rewards over time. This approach has been particularly successful in game playing, robotics, and autonomous systems.

Deep learning, a subset of machine learning, uses artificial neural networks with multiple layers to learn hierarchical representations of data. These deep neural networks have achieved remarkable success in tasks such as computer vision, speech recognition, and natural language understanding. The success of deep learning can be attributed to the availability of large datasets, powerful computing resources, and algorithmic innovations.

Now, given this context about machine learning, please answer the following question:"""


def test_with_cache(model, tokenizer, prompt, cache_name, cache, max_gen=100):
    """Test generation with given cache"""
    print(f"\n{'='*70}")
    print(f"{cache_name}")
    print(f"{'='*70}")

    tokens = mx.array([tokenizer.encode(prompt)])
    prompt_tokens = tokens.shape[1]
    print(f"Prompt tokens: {prompt_tokens}")

    # PP phase
    pp_start = time.time()
    logits = model(tokens, cache=cache)
    mx.eval(logits)
    pp_time = time.time() - pp_start

    print(f"PP time: {pp_time:.3f}s ({prompt_tokens/pp_time:.2f} tok/s)")

    # Check compression stats
    if hasattr(cache[0], 'get_stats'):
        stats = cache[0].get_stats()
        print(f"Compression: {stats['num_compressions']} times, "
              f"current size: {stats['current_size']}")

    # TG phase
    generated_tokens = []
    next_token = mx.argmax(logits[:, -1, :], axis=-1).item()

    tg_start = time.time()

    for _ in range(max_gen):
        if next_token == tokenizer.eos_token_id:
            break

        generated_tokens.append(next_token)
        tokens = mx.array([[next_token]])
        logits = model(tokens, cache=cache)
        mx.eval(logits)
        next_token = mx.argmax(logits[:, -1, :], axis=-1).item()

    tg_time = time.time() - tg_start
    tg_speed = len(generated_tokens) / tg_time if tg_time > 0 else 0

    print(f"TG time: {tg_time:.3f}s ({tg_speed:.2f} tok/s)")
    print(f"Generated: {len(generated_tokens)} tokens")

    output = tokenizer.decode(generated_tokens)

    print(f"\nOutput:")
    print(f"{'-'*70}")
    print(output)
    print(f"{'-'*70}")

    return output, len(generated_tokens), tg_speed


def main():
    print("="*70)
    print("AM 真实 Prompt 测试")
    print("="*70)

    model_path = "/Volumes/toshiba/models/qwen3-8b-mlx"

    print("\nLoading model...")
    model, tokenizer = load(model_path)
    num_layers = len(model.layers)

    # Test 1: Baseline (no compression)
    baseline_cache = ArraysCache(size=num_layers)
    for i in range(num_layers):
        baseline_cache[i] = KVCache()

    baseline_output, baseline_len, baseline_speed = test_with_cache(
        model, tokenizer, REAL_PROMPT, "Baseline (无压缩)", baseline_cache
    )

    # Test 2: AM (Fast Path) - 中等压缩
    am_cache = ArraysCache(size=num_layers)
    for i in range(num_layers):
        am_cache[i] = CompactedKVCache(
            max_size=512,
            compression_ratio=2.0,
            use_quality_path=False
        )

    am_output, am_len, am_speed = test_with_cache(
        model, tokenizer, REAL_PROMPT, "AM Fast Path (2x 压缩)", am_cache
    )

    # Test 3: AM (Quality Path) - 中等压缩
    am_quality_cache = ArraysCache(size=num_layers)
    for i in range(num_layers):
        am_quality_cache[i] = CompactedKVCache(
            max_size=512,
            compression_ratio=2.0,
            use_quality_path=True
        )

    am_quality_output, am_quality_len, am_quality_speed = test_with_cache(
        model, tokenizer, REAL_PROMPT, "AM Quality Path (2x 压缩)", am_quality_cache
    )

    # 对比分析
    print(f"\n{'='*70}")
    print("对比分析")
    print(f"{'='*70}")

    print(f"\n1. 生成 Token 数量:")
    print(f"   Baseline: {baseline_len} tokens")
    print(f"   AM Fast: {am_len} tokens")
    print(f"   AM Quality: {am_quality_len} tokens")

    print(f"\n2. TG 速度:")
    print(f"   Baseline: {baseline_speed:.2f} tok/s")
    print(f"   AM Fast: {am_speed:.2f} tok/s ({(am_speed/baseline_speed-1)*100:+.1f}%)")
    print(f"   AM Quality: {am_quality_speed:.2f} tok/s ({(am_quality_speed/baseline_speed-1)*100:+.1f}%)")

    print(f"\n3. 输出是否相同:")
    print(f"   Baseline vs AM Fast: {'✅ 相同' if baseline_output == am_output else '❌ 不同'}")
    print(f"   Baseline vs AM Quality: {'✅ 相同' if baseline_output == am_quality_output else '❌ 不同'}")

    # 简单质量检查
    def check_quality(text):
        """简单的质量检查"""
        words = text.split()
        if len(words) < 10:
            return "❌ 太短"
        # 检查是否有大量重复短语
        phrases = [" ".join(words[i:i+3]) for i in range(len(words)-2)]
        unique_phrases = len(set(phrases))
        diversity = unique_phrases / len(phrases) if len(phrases) > 0 else 0
        if diversity < 0.5:
            return f"❌ 重复度高 (多样性: {diversity:.2f})"
        return f"✅ 正常 (多样性: {diversity:.2f})"

    print(f"\n4. 质量评估:")
    print(f"   Baseline: {check_quality(baseline_output)}")
    print(f"   AM Fast: {check_quality(am_output)}")
    print(f"   AM Quality: {check_quality(am_quality_output)}")


if __name__ == "__main__":
    main()
