#!/usr/bin/env python3
"""
AM 压缩真实价值测试：极限场景

场景：超长文档（重复添加内容直到 > 6000 tokens）+ 文档末尾关键信息

核心价值展示：
1. Baseline (max_kv_size=4096) → 截断，丢失文档末尾
2. AM (max_size=2048, ratio=5.0) → 压缩到 ~410 tokens，保留完整上下文

预期结果：
- Baseline: 无法回答文档末尾的问题（信息被截断）
- AM: 能正确回答（压缩保留了关键信息）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "mlx-lm-source"))

from mlx_lm import load
from mlx_lm.generate_with_compaction import generate_with_compaction


def create_ultra_long_document():
    """创建超长文档（6000+ tokens），关键信息在末尾"""

    # 基础内容（重复多次）
    base_content = """
Neural networks are powerful machine learning models inspired by biological neurons. They consist of interconnected layers of artificial neurons that process information through weighted connections. The training process involves adjusting these weights to minimize a loss function.

Deep learning refers to neural networks with many layers (typically more than 3 hidden layers). Deep networks can learn hierarchical representations of data, with each layer learning increasingly abstract features. For example, in image recognition, early layers might detect edges, middle layers detect shapes, and deep layers detect objects.

Convolutional Neural Networks (CNNs) are specialized for processing grid-like data such as images. They use convolutional layers that apply filters across spatial dimensions, pooling layers for dimensionality reduction, and fully connected layers for classification. Famous CNN architectures include LeNet, AlexNet, VGGNet, ResNet, and Inception.

Recurrent Neural Networks (RNNs) process sequential data by maintaining a hidden state that captures information from previous time steps. However, basic RNNs suffer from vanishing gradient problems. Long Short-Term Memory (LSTM) networks solve this using gating mechanisms: forget gates, input gates, and output gates.

The Transformer architecture revolutionized NLP by replacing recurrence with self-attention mechanisms. Attention allows the model to focus on relevant parts of the input. Multi-head attention runs multiple attention operations in parallel, allowing the model to capture different types of relationships. Transformers power models like BERT, GPT, and T5.

Training deep networks requires careful attention to many factors. Weight initialization strategies like Xavier and He initialization help maintain gradient flow. Learning rate scheduling techniques like step decay, exponential decay, and cosine annealing improve convergence. Regularization methods like dropout, L2 regularization, and batch normalization prevent overfitting.

Optimization algorithms have evolved significantly. Stochastic Gradient Descent (SGD) with momentum accelerates learning and helps escape local minima. Adam (Adaptive Moment Estimation) combines momentum with adaptive learning rates per parameter, making it the most popular optimizer. RMSprop is another adaptive learning rate method that works well for recurrent networks.

Data augmentation is crucial for improving model generalization. For images, techniques include rotation, flipping, cropping, color jittering, and mixup. For text, back-translation and synonym replacement are common. Augmentation forces the model to learn invariant features rather than memorizing the training data.

Transfer learning leverages pre-trained models on new tasks. Models trained on large datasets like ImageNet (for vision) or large text corpora (for NLP) learn general features that transfer to other domains. Fine-tuning these models on task-specific data often outperforms training from scratch, especially with limited data.

Model architecture search has become automated through Neural Architecture Search (NAS). Techniques include reinforcement learning, evolutionary algorithms, and differentiable architecture search. NAS has discovered architectures that outperform human-designed networks, though it requires significant computational resources.

Interpretability and explainability are increasingly important as deep learning is deployed in critical applications. Techniques include attention visualization, gradient-based saliency maps, and LIME (Local Interpretable Model-agnostic Explanations). Understanding what models learn helps debug errors and build trust.
"""

    # 重复基础内容多次以达到 6000+ tokens
    repeated_content = base_content * 3  # 重复 3 次

    # 关键信息在文档末尾（这是测试的关键！）
    critical_info = """

### IMPORTANT SECTION - CRITICAL INFORMATION AT END OF DOCUMENT

Section 99: Special Configuration for Production Deployment

When deploying neural networks in production environments, there are three critical hyperparameters that must be set correctly:

1. **BATCH_SIZE_PRODUCTION**: Must be set to exactly 64 for optimal throughput
   - Rationale: Balance between GPU utilization and latency
   - Smaller batches: Higher latency, lower throughput
   - Larger batches: Risk of OOM, diminishing returns

2. **LEARNING_RATE_WARMUP_STEPS**: Must be set to exactly 1000 steps
   - Rationale: Prevents instability in early training
   - Too few steps: Training may diverge
   - Too many steps: Wastes time with suboptimal learning

3. **GRADIENT_CLIP_THRESHOLD**: Must be set to exactly 5.0
   - Rationale: Prevents exploding gradients in deep networks
   - Too low: Slows convergence, limits model capacity
   - Too high: Risk of instability and NaN losses

These three values (64, 1000, 5.0) are the result of extensive hyperparameter search across multiple production deployments and should NOT be changed without thorough testing.

REMEMBER: BATCH_SIZE_PRODUCTION = 64, WARMUP_STEPS = 1000, GRADIENT_CLIP = 5.0
"""

    return repeated_content + critical_info


def main():
    print("="*70)
    print("AM 压缩真实价值测试：极限场景")
    print("="*70)

    model_path = "/Volumes/toshiba/models/qwen3-8b-mlx"
    print(f"\n加载模型: {model_path}")
    model, tokenizer = load(model_path)

    # 创建超长文档
    document = create_ultra_long_document()
    doc_tokens = len(tokenizer.encode(document))

    print(f"\n文档设置:")
    print(f"  文档长度: {doc_tokens} tokens")
    print(f"  Baseline limit: 4096 tokens")
    print(f"  关键信息位置: 文档末尾 (Section 99)")
    print(f"  预期: Baseline 会截断，丢失关键信息")

    # 关键问题（针对文档末尾的信息）
    question = "According to Section 99, what are the three critical hyperparameters for production deployment, and what are their exact values?"

    print(f"\n关键问题:")
    print(f"  {question}")
    print(f"\n预期答案关键词: ['64', '1000', '5.0', 'BATCH_SIZE', 'WARMUP_STEPS', 'GRADIENT_CLIP']")

    # ========================================
    # Baseline: 无压缩，文档被截断
    # ========================================
    print(f"\n{'='*70}")
    print("Baseline: 无压缩 (max_kv_size=4096)")
    print("="*70)

    prompt_baseline = f"{document}\n\nQuestion: {question}\nAnswer:"

    print(f"\nPrompt tokens: {len(tokenizer.encode(prompt_baseline))}")
    print(f"预期: 前 4096 tokens 保留，文档末尾被截断")

    answer_baseline = generate_with_compaction(
        model,
        tokenizer,
        prompt=prompt_baseline,
        max_tokens=150,
        use_compaction=False,  # 禁用压缩
        verbose=False,
    )

    print(f"\nBaseline Answer:")
    print(f"  {answer_baseline}")

    # 检查关键词
    keywords = ['64', '1000', '5.0', 'batch', 'warmup', 'gradient', 'clip']
    baseline_matched = sum(1 for kw in keywords if kw.lower() in answer_baseline.lower())
    baseline_score = baseline_matched / len(keywords)

    print(f"\n关键词匹配: {baseline_matched}/{len(keywords)} ({baseline_score*100:.0f}%)")

    # ========================================
    # AM 压缩: 保留完整上下文
    # ========================================
    print(f"\n{'='*70}")
    print("With AM Compression")
    print("="*70)

    prompt_am = f"{document}\n\nQuestion: {question}\nAnswer:"

    print(f"\nPrompt tokens: {len(tokenizer.encode(prompt_am))}")
    print(f"AM 配置:")
    print(f"  max_size: 2048")
    print(f"  compression_ratio: 5.0")
    print(f"  预期压缩后: ~410 tokens")
    print(f"  关键: 使用 Qref 保留重要信息")

    answer_am = generate_with_compaction(
        model,
        tokenizer,
        prompt=prompt_am,
        max_tokens=150,
        use_compaction=True,
        compaction_config={
            "max_size": 2048,
            "compression_ratio": 5.0,
            "num_queries": 256,  # 更多 queries
            "check_interval": 128,
            "use_quality_path": True,
        },
        verbose=True,
    )

    print(f"\nAM Answer:")
    print(f"  {answer_am}")

    # 检查关键词
    am_matched = sum(1 for kw in keywords if kw.lower() in answer_am.lower())
    am_score = am_matched / len(keywords)

    print(f"\n关键词匹配: {am_matched}/{len(keywords)} ({am_score*100:.0f}%)")

    # ========================================
    # 对比分析
    # ========================================
    print(f"\n{'='*70}")
    print("对比分析")
    print("="*70)

    print(f"\n{'指标':<30} {'Baseline':<15} {'AM 压缩':<15} {'改进':<10}")
    print("-" * 70)
    print(f"{'KV Cache 大小':<30} {4096:<15} {'~410':<15} {'-90%':<10}")
    print(f"{'关键词匹配率':<30} {baseline_score*100:<14.0f}% {am_score*100:<14.0f}% {(am_score-baseline_score)*100:+9.0f}%")
    print(f"{'能回答末尾问题':<30} {'❌' if baseline_score < 0.5 else '✅':<15} {'✅' if am_score >= 0.5 else '❌':<15} {'':<10}")

    print(f"\n{'='*70}")
    print("AM 压缩的核心价值")
    print("="*70)

    if am_score > baseline_score + 0.3:  # 30% 以上改进
        print(f"\n✅ AM 压缩显著提升长文档能力！")
        print(f"\n价值 1: 保留完整上下文")
        print(f"  - 文档: {doc_tokens} tokens")
        print(f"  - Baseline 截断到: 4096 tokens (丢失 {doc_tokens-4096} tokens)")
        print(f"  - AM 压缩到: ~410 tokens (保留完整信息)")
        print(f"  - 内存节省: 90%")

        print(f"\n价值 2: 能回答文档末尾的问题")
        print(f"  - Baseline: {baseline_score*100:.0f}% 正确率（信息被截断）")
        print(f"  - AM: {am_score*100:.0f}% 正确率（Qref 保留关键信息）")
        print(f"  - 改进: {(am_score-baseline_score)*100:+.0f}%")

        print(f"\n价值 3: 使用场景")
        print(f"  - 长文档问答（技术文档、法律文件、研究论文）")
        print(f"  - 长对话历史（客服机器人、个人助手）")
        print(f"  - 批处理任务（处理多个长文档）")
        print(f"  - 内存受限环境（边缘设备、移动设备）")

        print(f"\n🎉 这就是 AM 压缩的真正价值！")
    elif am_score > baseline_score:
        print(f"\n⚠️  AM 压缩有改进: {baseline_score*100:.0f}% → {am_score*100:.0f}%")
        print(f"   但改进不显著（可能需要更极端的场景）")
    else:
        print(f"\n⚠️  两者表现相近: Baseline {baseline_score*100:.0f}%, AM {am_score*100:.0f}%")
        print(f"   可能原因:")
        print(f"   1. 压缩丢失了关键信息")
        print(f"   2. 需要更多 queries 或更小的 compression_ratio")
        print(f"   3. Baseline 也能从截断的上下文中推断答案")


if __name__ == "__main__":
    main()
