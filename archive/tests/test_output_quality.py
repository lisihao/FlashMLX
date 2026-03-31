#!/usr/bin/env python3
"""
Output Quality Comparison Test

对比 Baseline 和 Attention Matching 的输出质量。

测试场景：
1. 技术解释（需要准确性）
2. 创意写作（需要连贯性）
3. 逻辑推理（需要正确性）
4. 长文本生成（需要一致性）
"""

import gc
import mlx.core as mx
from mlx_lm import load, generate
from flashmlx.cache import inject_attention_matching


def test_scenario(
    model,
    tokenizer,
    prompt: str,
    scenario_name: str,
    max_tokens: int = 200,
    compression_ratio: float = 3.0
):
    """
    测试单个场景的输出质量

    Args:
        model: 模型实例
        tokenizer: Tokenizer
        prompt: 输入 prompt
        scenario_name: 场景名称
        max_tokens: 最大生成 tokens
        compression_ratio: 压缩比例
    """
    print(f"\n{'='*70}")
    print(f"场景: {scenario_name}")
    print(f"{'='*70}")

    print(f"\nPrompt:\n{prompt[:200]}...")

    # ====================================================================
    # Baseline
    # ====================================================================
    print(f"\n--- Baseline 输出 ---")

    gc.collect()
    mx.clear_cache()

    baseline_output = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        verbose=False
    )

    print(f"\n{baseline_output}")

    # ====================================================================
    # With Compression
    # ====================================================================
    print(f"\n--- Compressed 输出 (压缩率 {compression_ratio}x) ---")

    gc.collect()
    mx.clear_cache()

    # 注入压缩
    cache_list, compressor = inject_attention_matching(
        model,
        compression_ratio=compression_ratio,
        beta_calibration=True,
        eviction_policy="top_k"
    )

    compressed_output = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        verbose=False
    )

    print(f"\n{compressed_output}")

    # ====================================================================
    # 对比
    # ====================================================================
    print(f"\n--- 质量对比 ---")

    # 简单相似度检查
    baseline_tokens = set(baseline_output.split())
    compressed_tokens = set(compressed_output.split())

    overlap = len(baseline_tokens & compressed_tokens)
    union = len(baseline_tokens | compressed_tokens)
    similarity = overlap / union if union > 0 else 0

    print(f"  Token overlap: {overlap}/{union} ({similarity:.1%})")

    # 长度对比
    baseline_len = len(tokenizer.encode(baseline_output))
    compressed_len = len(tokenizer.encode(compressed_output))

    print(f"  Length: Baseline {baseline_len} tokens, Compressed {compressed_len} tokens")

    # 人工检查提示
    print(f"\n  请人工检查：")
    print(f"    1. 是否都回答了问题？")
    print(f"    2. 内容是否连贯？")
    print(f"    3. 是否有明显错误？")
    print(f"    4. 是否有重复或胡言乱语？")

    return {
        'baseline': baseline_output,
        'compressed': compressed_output,
        'similarity': similarity,
        'baseline_len': baseline_len,
        'compressed_len': compressed_len
    }


def main():
    print("="*70)
    print("Output Quality Comparison Test")
    print("="*70)

    model_path = "/Volumes/toshiba/models/qwen3.5-35b-mlx/"

    print(f"\n加载模型: {model_path}")
    model, tokenizer = load(model_path)

    # ====================================================================
    # 场景 1: 技术解释（需要准确性）
    # ====================================================================
    scenario1 = test_scenario(
        model=model,
        tokenizer=tokenizer,
        prompt="Explain how the attention mechanism works in transformer models. Include key concepts like queries, keys, values, and self-attention.",
        scenario_name="技术解释（准确性）",
        max_tokens=200,
        compression_ratio=3.0
    )

    # ====================================================================
    # 场景 2: 创意写作（需要连贯性）
    # ====================================================================
    scenario2 = test_scenario(
        model=model,
        tokenizer=tokenizer,
        prompt="Write a short story about a robot who learns to paint. The story should have a beginning, middle, and end.",
        scenario_name="创意写作（连贯性）",
        max_tokens=200,
        compression_ratio=3.0
    )

    # ====================================================================
    # 场景 3: 逻辑推理（需要正确性）
    # ====================================================================
    scenario3 = test_scenario(
        model=model,
        tokenizer=tokenizer,
        prompt="""Solve this logic puzzle:
- Alice is taller than Bob.
- Bob is taller than Charlie.
- David is shorter than Charlie.

Who is the tallest? Who is the shortest? Explain your reasoning step by step.""",
        scenario_name="逻辑推理（正确性）",
        max_tokens=200,
        compression_ratio=3.0
    )

    # ====================================================================
    # 场景 4: 长 Context（一致性）
    # ====================================================================
    long_context = """Machine learning is a subset of artificial intelligence. It focuses on developing algorithms that can learn from data. Deep learning is a subset of machine learning that uses neural networks with multiple layers. These networks can learn hierarchical representations of data. Convolutional neural networks are specialized for processing grid-like data such as images. Recurrent neural networks are designed for sequential data like text or time series. Transformer models have revolutionized natural language processing by using attention mechanisms. They can process entire sequences in parallel, unlike RNNs which process sequentially.

Based on the above information, please:
1. Summarize the relationship between AI, ML, and DL.
2. Explain the key difference between CNNs and RNNs.
3. What makes transformers different from RNNs?"""

    scenario4 = test_scenario(
        model=model,
        tokenizer=tokenizer,
        prompt=long_context,
        scenario_name="长 Context（一致性）",
        max_tokens=200,
        compression_ratio=3.0
    )

    # ====================================================================
    # 汇总
    # ====================================================================
    print(f"\n{'='*70}")
    print("质量对比汇总")
    print(f"{'='*70}")

    scenarios = [
        ("技术解释", scenario1),
        ("创意写作", scenario2),
        ("逻辑推理", scenario3),
        ("长 Context", scenario4)
    ]

    print(f"\n{'场景':<20} {'Token Overlap':<15} {'Length Diff':<15}")
    print("-"*50)

    for name, result in scenarios:
        len_diff = abs(result['compressed_len'] - result['baseline_len'])
        print(f"{name:<20} {result['similarity']:>12.1%}   {len_diff:>12} tokens")

    print(f"\n{'='*70}")
    print("✓ 测试完成")
    print(f"{'='*70}")

    print(f"\n总结:")
    print(f"  1. 检查每个场景的输出是否合理")
    print(f"  2. Token overlap 反映内容相似度")
    print(f"  3. 如果 overlap < 50%，说明输出差异较大")
    print(f"  4. 建议人工评审关键场景的输出质量")


if __name__ == "__main__":
    main()
