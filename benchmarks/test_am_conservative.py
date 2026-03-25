#!/usr/bin/env python3
"""
AM 压缩保守配置测试

测试更保守的配置：
1. compression_ratio=2.0（而不是 5.0）
2. 更多 queries (512 而不是 256)
3. 验证 Qwen3-8B（纯 Transformer）上的表现
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "mlx-lm-source"))

from mlx_lm import load
from mlx_lm.generate_with_compaction import generate_with_compaction


def create_test_prompt():
    """创建测试 prompt"""
    return """
Machine learning is a branch of artificial intelligence that focuses on building systems that can learn from and make decisions based on data. Unlike traditional programming where rules are explicitly coded, machine learning algorithms identify patterns in data and use those patterns to make predictions or decisions.

There are several types of machine learning:

1. Supervised Learning: The algorithm learns from labeled training data. Examples include classification (identifying spam emails) and regression (predicting house prices).

2. Unsupervised Learning: The algorithm finds patterns in unlabeled data. Common applications include clustering (grouping similar customers) and dimensionality reduction.

3. Reinforcement Learning: The algorithm learns through trial and error, receiving rewards or penalties for its actions. This is used in game playing and robotics.

Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers. These networks can automatically learn hierarchical representations of data, making them particularly effective for tasks like image recognition, natural language processing, and speech recognition.

Question: What is the primary purpose of backpropagation in neural networks?
"""


def main():
    print("="*70)
    print("AM 压缩保守配置测试")
    print("="*70)

    model_path = "/Volumes/toshiba/models/qwen3-8b-mlx"
    print(f"\n加载模型: {model_path}")
    print(f"架构: Qwen3-8B（纯 Transformer，36 层 Attention）")
    model, tokenizer = load(model_path)

    prompt = create_test_prompt()
    prompt_tokens = len(tokenizer.encode(prompt))

    print(f"\nPrompt length: {prompt_tokens} tokens")

    # ========================================
    # Baseline: 无压缩
    # ========================================
    print(f"\n{'='*70}")
    print("Baseline: 无压缩")
    print("="*70)

    baseline = generate_with_compaction(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=100,
        use_compaction=False,
        verbose=False,
    )

    print(f"\nBaseline output:")
    print(f"  {baseline}")

    # ========================================
    # 测试 1: ratio=2.0（保守）
    # ========================================
    print(f"\n{'='*70}")
    print("测试 1: ratio=2.0（保守配置）")
    print("="*70)

    print(f"\n配置:")
    print(f"  max_size: 200")
    print(f"  compression_ratio: 2.0（保守，而不是 5.0）")
    print(f"  num_queries: 512（更多）")
    print(f"  预期: 压缩后质量接近 baseline")

    with_am_conservative = generate_with_compaction(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=100,
        use_compaction=True,
        compaction_config={
            "max_size": 200,
            "compression_ratio": 2.0,  # 保守
            "num_queries": 512,  # 更多 queries
            "check_interval": 50,
            "use_quality_path": True,
        },
        verbose=True,
    )

    print(f"\nAM (ratio=2.0) output:")
    print(f"  {with_am_conservative}")

    # ========================================
    # 测试 2: ratio=1.5（极保守）
    # ========================================
    print(f"\n{'='*70}")
    print("测试 2: ratio=1.5（极保守配置）")
    print("="*70)

    print(f"\n配置:")
    print(f"  max_size: 200")
    print(f"  compression_ratio: 1.5（极保守）")
    print(f"  num_queries: 512")
    print(f"  预期: 质量几乎等价于 baseline")

    with_am_ultra_conservative = generate_with_compaction(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=100,
        use_compaction=True,
        compaction_config={
            "max_size": 200,
            "compression_ratio": 1.5,  # 极保守
            "num_queries": 512,
            "check_interval": 50,
            "use_quality_path": True,
        },
        verbose=True,
    )

    print(f"\nAM (ratio=1.5) output:")
    print(f"  {with_am_ultra_conservative}")

    # ========================================
    # 质量对比
    # ========================================
    print(f"\n{'='*70}")
    print("质量对比")
    print("="*70)

    def calc_similarity(text1, text2):
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        if len(tokens1) == 0:
            return 0.0
        return len(tokens1 & tokens2) / len(tokens1)

    sim_2_0 = calc_similarity(baseline, with_am_conservative)
    sim_1_5 = calc_similarity(baseline, with_am_ultra_conservative)

    print(f"\n配置                         词汇相似度")
    print("-" * 70)
    print(f"Baseline (无压缩)            100.0%")
    print(f"AM ratio=2.0                 {sim_2_0*100:.1f}%")
    print(f"AM ratio=1.5                 {sim_1_5*100:.1f}%")

    print(f"\n质量判断:")
    if sim_1_5 >= 0.9:
        print(f"✅ ratio=1.5 质量优秀（≥90%）")
    elif sim_1_5 >= 0.8:
        print(f"⚠️  ratio=1.5 质量良好（80-90%）")
    else:
        print(f"❌ ratio=1.5 质量不佳（<80%）")

    if sim_2_0 >= 0.8:
        print(f"✅ ratio=2.0 质量良好（≥80%）")
    elif sim_2_0 >= 0.6:
        print(f"⚠️  ratio=2.0 质量可接受（60-80%）")
    else:
        print(f"❌ ratio=2.0 质量不佳（<60%）")

    print(f"\n{'='*70}")
    print("结论")
    print("="*70)

    if sim_1_5 >= 0.9 and sim_2_0 >= 0.8:
        print(f"✅ AM 压缩在纯 Transformer 上可用！")
        print(f"   推荐配置: ratio=1.5~2.0, queries=512")
    elif sim_2_0 >= 0.6:
        print(f"⚠️  AM 压缩质量可接受但不理想")
        print(f"   需要进一步优化配置")
    else:
        print(f"❌ AM 压缩质量不佳")
        print(f"   可能的原因:")
        print(f"   1. Qref 采样策略不适合")
        print(f"   2. 实现有问题")
        print(f"   3. Qwen3-8B 的特殊性")


if __name__ == "__main__":
    main()
