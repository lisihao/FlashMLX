#!/usr/bin/env python3
"""
AM 最终测试 - 强制触发压缩并验证质量
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "mlx-lm-source"))

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import KVCache, ArraysCache
from mlx_lm.models.compacted_cache import CompactedKVCache


# 真实的长 prompt（扩展版）
LONG_PROMPT = """Machine learning is a branch of artificial intelligence that focuses on teaching computers to learn from data. Instead of being explicitly programmed to perform a task, machine learning algorithms use statistical techniques to enable computers to improve their performance on a specific task through experience.

The core idea behind machine learning is to create algorithms that can automatically identify patterns in data and make decisions or predictions based on those patterns. This approach has proven highly effective in many domains, from image recognition and natural language processing to recommendation systems and autonomous vehicles.

There are three main categories of machine learning: supervised learning, unsupervised learning, and reinforcement learning. Supervised learning involves training a model on labeled data where the desired output is known. The algorithm learns to map inputs to outputs by analyzing the examples provided. Common applications include spam detection, image classification, and predictive modeling.

Unsupervised learning, on the other hand, works with unlabeled data. The algorithm must find patterns and structures in the data without being told what to look for. Clustering and dimensionality reduction are typical unsupervised learning tasks. For example, customer segmentation in marketing often uses unsupervised learning to group customers with similar characteristics.

Reinforcement learning is a type of machine learning where an agent learns to make decisions by interacting with an environment. The agent receives rewards or penalties based on its actions and learns to maximize cumulative rewards over time. This approach has been particularly successful in game playing, robotics, and autonomous systems.

Deep learning, a subset of machine learning, uses artificial neural networks with multiple layers to learn hierarchical representations of data. These deep neural networks have achieved remarkable success in tasks such as computer vision, speech recognition, and natural language understanding. The success of deep learning can be attributed to the availability of large datasets, powerful computing resources, and algorithmic innovations.

Neural networks are inspired by the structure and function of the human brain. They consist of interconnected nodes, or neurons, organized in layers. Each neuron receives input from other neurons, processes this information, and passes the result to the next layer. The connections between neurons have weights that are adjusted during training to minimize prediction errors.

The training process involves feeding the neural network with examples and adjusting the weights through a process called backpropagation. This algorithm calculates the gradient of the loss function with respect to each weight and updates the weights in the direction that reduces the error. The learning rate, a hyperparameter, controls how much the weights are adjusted at each step.

Convolutional neural networks (CNNs) are specifically designed for processing grid-like data, such as images. They use convolutional layers to automatically learn spatial hierarchies of features, from simple edges to complex objects. Recurrent neural networks (RNNs), on the other hand, are designed for sequential data and have connections that form directed cycles, allowing them to maintain information about previous inputs.

Now, given this comprehensive context about machine learning and deep learning, please answer the following question:"""


def test_with_cache(model, tokenizer, prompt, cache_name, cache, max_gen=50):
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
    compressed = False
    if hasattr(cache[0], 'get_stats'):
        stats = cache[0].get_stats()
        compressed = stats['num_compressions'] > 0
        print(f"Compression: {stats['num_compressions']} times, "
              f"current size: {stats['current_size']}")
        if compressed:
            print(f"  ✅ 压缩已触发")
        else:
            print(f"  ⚠️  压缩未触发")

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

    return output, len(generated_tokens), tg_speed, compressed


def main():
    print("="*70)
    print("AM 最终测试 - 强制触发压缩")
    print("="*70)

    model_path = "/Volumes/toshiba/models/qwen3-8b-mlx"

    print("\nLoading model...")
    model, tokenizer = load(model_path)
    num_layers = len(model.layers)

    # Test 1: Baseline (no compression)
    baseline_cache = ArraysCache(size=num_layers)
    for i in range(num_layers):
        baseline_cache[i] = CompactedKVCache(
            max_size=10000,  # Very large to avoid compression
            enable_compression=False
        )

    baseline_output, baseline_len, baseline_speed, _ = test_with_cache(
        model, tokenizer, LONG_PROMPT, "Baseline (无压缩)", baseline_cache
    )

    # Test 2: AM (Fast Path) - 必定触发压缩
    am_cache = ArraysCache(size=num_layers)
    for i in range(num_layers):
        am_cache[i] = CompactedKVCache(
            max_size=256,  # Small to force compression
            compression_ratio=2.0,
            use_quality_path=False
        )

    am_output, am_len, am_speed, am_compressed = test_with_cache(
        model, tokenizer, LONG_PROMPT, "AM Fast Path (256 max_size)", am_cache
    )

    # 对比分析
    print(f"\n{'='*70}")
    print("对比分析")
    print(f"{'='*70}")

    print(f"\n1. 压缩状态:")
    print(f"   Baseline: 压缩未触发 ✅")
    print(f"   AM Fast: {'压缩已触发 ✅' if am_compressed else '压缩未触发 ❌'}")

    print(f"\n2. 生成 Token 数量:")
    print(f"   Baseline: {baseline_len} tokens")
    print(f"   AM Fast: {am_len} tokens")

    print(f"\n3. TG 速度:")
    print(f"   Baseline: {baseline_speed:.2f} tok/s")
    print(f"   AM Fast: {am_speed:.2f} tok/s ({(am_speed/baseline_speed-1)*100:+.1f}%)")

    print(f"\n4. 输出是否相同:")
    if baseline_output == am_output:
        print(f"   ✅ 完全相同 - AM 压缩后输出质量保持")
    else:
        print(f"   ❌ 不同 - AM 压缩影响了输出")

        # 检查相似度
        baseline_words = baseline_output.split()
        am_words = am_output.split()
        common_words = set(baseline_words) & set(am_words)
        similarity = len(common_words) / max(len(baseline_words), len(am_words))
        print(f"      词汇相似度: {similarity:.2%}")

        # 检查是否有重复
        def has_repetition(text):
            words = text.split()
            if len(words) < 10:
                return False
            for i in range(len(words) - 6):
                phrase = " ".join(words[i:i+3])
                if text.count(phrase) >= 3:
                    return True
            return False

        baseline_rep = has_repetition(baseline_output)
        am_rep = has_repetition(am_output)

        print(f"      Baseline 有重复: {'是' if baseline_rep else '否'}")
        print(f"      AM Fast 有重复: {'是' if am_rep else '否'}")

    print(f"\n5. 结论:")
    if am_compressed:
        if baseline_output == am_output:
            print(f"   ✅ AM 压缩成功，输出质量无损！")
        else:
            print(f"   ❌ AM 压缩导致输出变化，需要进一步分析")
    else:
        print(f"   ⚠️  AM 压缩未触发，测试无效")


if __name__ == "__main__":
    main()
