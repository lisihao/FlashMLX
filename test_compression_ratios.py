#!/usr/bin/env python3
"""
Compression Ratio Quality Test

测试不同压缩率下的输出质量，找到性能和质量的平衡点。

测试压缩率：
- 1.5x (保守)
- 2.0x (中等)
- 2.5x (激进)
- 3.0x (当前失败)

质量目标：Token overlap ≥ 50%
"""

import gc
import mlx.core as mx
from mlx_lm import load, generate
from flashmlx.cache import inject_attention_matching


def test_compression_ratio(
    model,
    tokenizer,
    prompt: str,
    compression_ratio: float,
    max_tokens: int = 200
):
    """
    测试单个压缩率的输出质量

    Args:
        model: 模型实例
        tokenizer: Tokenizer
        prompt: 输入 prompt
        compression_ratio: 压缩比例
        max_tokens: 最大生成 tokens

    Returns:
        {
            'compression_ratio': float,
            'output': str,
            'length': int
        }
    """
    gc.collect()
    mx.clear_cache()

    # 注入压缩
    cache_list, compressor = inject_attention_matching(
        model,
        compression_ratio=compression_ratio,
        beta_calibration=True,
        eviction_policy="top_k"
    )

    # 生成
    output = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        verbose=False
    )

    # 计算长度
    length = len(tokenizer.encode(output))

    return {
        'compression_ratio': compression_ratio,
        'output': output,
        'length': length
    }


def calculate_similarity(text1: str, text2: str) -> float:
    """计算两个文本的 token overlap"""
    tokens1 = set(text1.split())
    tokens2 = set(text2.split())

    overlap = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)

    return overlap / union if union > 0 else 0.0


def main():
    print("="*70)
    print("Compression Ratio Quality Test")
    print("="*70)

    model_path = "/Volumes/toshiba/models/qwen3.5-35b-mlx/"

    print(f"\n加载模型: {model_path}")
    model, tokenizer = load(model_path)

    # 测试 prompt（技术解释场景）
    prompt = "Explain how the attention mechanism works in transformer models. Include key concepts like queries, keys, values, and self-attention."

    print(f"\n{'='*70}")
    print("测试场景: 技术解释（准确性）")
    print(f"{'='*70}")
    print(f"\nPrompt:\n{prompt}\n")

    # ====================================================================
    # Baseline（无压缩）
    # ====================================================================
    print(f"--- Baseline (无压缩) ---")

    gc.collect()
    mx.clear_cache()

    baseline_output = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=200,
        verbose=False
    )

    baseline_len = len(tokenizer.encode(baseline_output))

    print(f"Length: {baseline_len} tokens")
    print(f"\n输出预览:\n{baseline_output[:200]}...\n")

    # ====================================================================
    # 测试不同压缩率
    # ====================================================================
    compression_ratios = [1.5, 2.0, 2.5, 3.0]
    results = []

    for ratio in compression_ratios:
        print(f"{'='*70}")
        print(f"测试压缩率: {ratio}x")
        print(f"{'='*70}")

        result = test_compression_ratio(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            compression_ratio=ratio,
            max_tokens=200
        )

        # 计算与 baseline 的相似度
        similarity = calculate_similarity(baseline_output, result['output'])

        result['similarity'] = similarity

        print(f"\nLength: {result['length']} tokens")
        print(f"Token overlap with baseline: {similarity:.1%}")
        print(f"\n输出预览:\n{result['output'][:200]}...\n")

        # 质量判断
        if similarity >= 0.5:
            quality = "✅ PASS (≥50% overlap)"
        elif similarity >= 0.3:
            quality = "⚠️ MARGINAL (30-50% overlap)"
        else:
            quality = "❌ FAIL (<30% overlap)"

        print(f"质量: {quality}\n")

        results.append(result)

        # 清理
        gc.collect()
        mx.clear_cache()

    # ====================================================================
    # 汇总报告
    # ====================================================================
    print(f"\n{'='*70}")
    print("质量对比汇总")
    print(f"{'='*70}\n")

    print(f"{'压缩率':<15} {'Token Overlap':<20} {'质量':<15}")
    print("-"*50)

    print(f"{'Baseline':<15} {'100.0%':<20} {'✅ PASS':<15}")

    for result in results:
        ratio_str = f"{result['compression_ratio']}x"
        overlap_str = f"{result['similarity']:.1%}"

        if result['similarity'] >= 0.5:
            quality = "✅ PASS"
        elif result['similarity'] >= 0.3:
            quality = "⚠️ MARGINAL"
        else:
            quality = "❌ FAIL"

        print(f"{ratio_str:<15} {overlap_str:<20} {quality:<15}")

    # ====================================================================
    # 推荐
    # ====================================================================
    print(f"\n{'='*70}")
    print("✓ 测试完成")
    print(f"{'='*70}\n")

    # 找到最高的合格压缩率
    passing_results = [r for r in results if r['similarity'] >= 0.5]

    if passing_results:
        best_ratio = max(passing_results, key=lambda r: r['compression_ratio'])
        print(f"推荐压缩率: {best_ratio['compression_ratio']}x")
        print(f"  - Token overlap: {best_ratio['similarity']:.1%}")
        print(f"  - 质量标准: PASS (≥50%)")
    else:
        marginal_results = [r for r in results if r['similarity'] >= 0.3]
        if marginal_results:
            best_ratio = max(marginal_results, key=lambda r: r['compression_ratio'])
            print(f"⚠️ 没有压缩率达到 50% overlap 标准")
            print(f"最接近的压缩率: {best_ratio['compression_ratio']}x")
            print(f"  - Token overlap: {best_ratio['similarity']:.1%}")
            print(f"  - 质量标准: MARGINAL (30-50%)")
        else:
            print(f"❌ 所有压缩率都未达到质量标准")
            print(f"建议: 尝试更低的压缩率 (< 1.5x) 或改进算法")


if __name__ == "__main__":
    main()
