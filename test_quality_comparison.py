#!/usr/bin/env python3
"""
完整质量对比测试 - 对比有/无 Attention Matching 的输出差异

测试场景：
1. 技术解释
2. 创意写作
3. 逻辑推理
4. 长 Context 总结
"""

import mlx.core as mx
from mlx_lm import load, generate
from flashmlx.cache.attention_matching_compressor_v2 import AttentionMatchingCompressorV2
import time
from typing import Dict, List


def calculate_token_overlap(text1: str, text2: str) -> float:
    """计算两个文本的 token overlap（简化版，用空格分割）"""
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())

    if len(tokens1) == 0 and len(tokens2) == 0:
        return 1.0

    intersection = tokens1 & tokens2
    union = tokens1 | tokens2

    return len(intersection) / len(union) if len(union) > 0 else 0.0


def calculate_exact_match_ratio(text1: str, text2: str) -> float:
    """计算完全匹配的单词比例"""
    words1 = text1.lower().split()
    words2 = text2.lower().split()

    min_len = min(len(words1), len(words2))
    if min_len == 0:
        return 0.0

    matches = sum(1 for w1, w2 in zip(words1, words2) if w1 == w2)
    return matches / min_len


def generate_with_config(
    model,
    tokenizer,
    prompt: str,
    use_compression: bool = False,
    compression_ratio: float = 2.0,
    max_tokens: int = 150
) -> tuple[str, float]:
    """使用指定配置生成文本"""

    if use_compression:
        # 注入压缩
        compressor = AttentionMatchingCompressorV2(
            compression_ratio=compression_ratio,
            score_method='max',
            beta_method='nnls',
            c2_method='lsq',
            num_queries=100
        )

        original_forward = model.model.layers[0].__class__.forward

        def compressed_forward(self, x, mask=None, cache=None):
            output = original_forward(self, x, mask, cache)

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

    # 生成
    start = time.time()
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        verbose=False
    )
    elapsed = time.time() - start

    return response, elapsed


def run_test_scenario(
    model,
    tokenizer,
    scenario_name: str,
    prompt: str,
    compression_ratio: float = 2.0,
    max_tokens: int = 150
) -> Dict:
    """运行单个测试场景"""

    print(f"\n{'='*70}")
    print(f"场景: {scenario_name}")
    print(f"{'='*70}")
    print(f"Prompt: {prompt[:100]}...")
    print(f"Compression ratio: {compression_ratio}x")
    print()

    # 无压缩
    print("生成中（无压缩）...")
    baseline_output, baseline_time = generate_with_config(
        model, tokenizer, prompt,
        use_compression=False,
        max_tokens=max_tokens
    )
    print(f"✓ 完成 ({baseline_time:.2f}s)")

    # 有压缩
    print("生成中（有压缩）...")
    compressed_output, compressed_time = generate_with_config(
        model, tokenizer, prompt,
        use_compression=True,
        compression_ratio=compression_ratio,
        max_tokens=max_tokens
    )
    print(f"✓ 完成 ({compressed_time:.2f}s)")

    # 计算差异
    token_overlap = calculate_token_overlap(baseline_output, compressed_output)
    exact_match = calculate_exact_match_ratio(baseline_output, compressed_output)

    print(f"\n质量指标:")
    print(f"  Token Overlap: {token_overlap*100:.1f}%")
    print(f"  Exact Match Ratio: {exact_match*100:.1f}%")

    print(f"\n输出对比:")
    print(f"\n--- 无压缩 ---")
    print(baseline_output)
    print(f"\n--- 有压缩 ({compression_ratio}x) ---")
    print(compressed_output)

    # 判断质量等级
    if token_overlap >= 0.50:
        quality = "✅ PASS (≥50%)"
    elif token_overlap >= 0.40:
        quality = "⚠️  MARGINAL (40-50%)"
    else:
        quality = "❌ FAIL (<40%)"

    print(f"\n质量评级: {quality}")

    return {
        "scenario": scenario_name,
        "token_overlap": token_overlap,
        "exact_match": exact_match,
        "baseline_output": baseline_output,
        "compressed_output": compressed_output,
        "baseline_time": baseline_time,
        "compressed_time": compressed_time,
        "quality": quality
    }


def main():
    print("="*70)
    print("完整质量对比测试 - Attention Matching")
    print("="*70)

    # 加载模型（使用小模型避免 OOM）
    print("\n加载模型...")
    model_path = "mlx-community/Qwen2.5-3B-Instruct-4bit"  # 小模型
    model, tokenizer = load(model_path)
    print(f"✓ 模型加载完成: {model_path}")

    # 测试场景
    scenarios = [
        {
            "name": "技术解释",
            "prompt": "Explain how a Transformer model works, focusing on the attention mechanism.",
            "max_tokens": 150
        },
        {
            "name": "创意写作",
            "prompt": "Write a short story about a robot learning to paint.",
            "max_tokens": 150
        },
        {
            "name": "逻辑推理",
            "prompt": "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly? Explain your reasoning.",
            "max_tokens": 100
        },
        {
            "name": "长 Context 总结",
            "prompt": """The following is a research paper abstract:

Deep learning has revolutionized artificial intelligence by enabling computers to learn from vast amounts of data. Neural networks with multiple layers can automatically discover intricate structures in high-dimensional data, making them ideal for tasks such as image recognition, speech understanding, and natural language processing. The backpropagation algorithm, combined with gradient descent optimization, allows these networks to adjust their parameters to minimize prediction errors. Recent advances include attention mechanisms, which help models focus on relevant parts of the input, and transformer architectures, which have achieved state-of-the-art results in language understanding.

Summarize the key points in 3 sentences.""",
            "max_tokens": 80
        }
    ]

    # 运行所有场景
    results = []
    for scenario in scenarios:
        result = run_test_scenario(
            model, tokenizer,
            scenario_name=scenario["name"],
            prompt=scenario["prompt"],
            compression_ratio=2.0,
            max_tokens=scenario["max_tokens"]
        )
        results.append(result)
        time.sleep(1)  # 避免过热

    # 总结
    print(f"\n\n{'='*70}")
    print("总结")
    print(f"{'='*70}")

    print(f"\n{'场景':<15} {'Token Overlap':<15} {'Exact Match':<15} {'质量'}")
    print("-" * 70)

    total_overlap = 0.0
    total_exact = 0.0

    for result in results:
        print(f"{result['scenario']:<15} {result['token_overlap']*100:>6.1f}%         {result['exact_match']*100:>6.1f}%         {result['quality']}")
        total_overlap += result['token_overlap']
        total_exact += result['exact_match']

    avg_overlap = total_overlap / len(results)
    avg_exact = total_exact / len(results)

    print("-" * 70)
    print(f"{'平均':<15} {avg_overlap*100:>6.1f}%         {avg_exact*100:>6.1f}%")

    # 最终判断
    print(f"\n{'='*70}")
    if avg_overlap >= 0.50:
        print("✅ 最终结果: PASS - 平均 Token Overlap ≥ 50%")
    elif avg_overlap >= 0.40:
        print("⚠️  最终结果: MARGINAL - 平均 Token Overlap 40-50%")
    else:
        print("❌ 最终结果: FAIL - 平均 Token Overlap < 40%")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
