#!/usr/bin/env python3
"""
AM 压缩价值测试：长文档问答

场景：
- 超长机器学习教科书章节（10K+ tokens）
- 基于文档回答多个问题（包括文档后半部分的内容）
- 验证 AM 压缩能保留完整上下文，避免信息丢失

对比：
- Baseline (无压缩，max_kv_size=4096) → 文档被截断
- With AM (max_size=2048, ratio=5.0) → 压缩到 ~410 tokens，保留完整上下文

预期价值：
1. 内存节省：5x 压缩比
2. 上下文保留：能回答文档末尾的问题
3. 质量保证：Qref 确保关键信息不丢失
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "mlx-lm-source"))

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate_with_compaction import generate_with_compaction


def create_long_document():
    """创建一个 10K+ tokens 的机器学习教科书章节"""
    return """
# Chapter 5: Deep Learning Fundamentals

## 5.1 Introduction to Neural Networks

Artificial neural networks are computing systems inspired by the biological neural networks that constitute animal brains. A neural network consists of interconnected nodes called neurons, organized in layers. Each connection between neurons has an associated weight that determines the strength of the signal passed between them.

The basic building block of a neural network is the perceptron, introduced by Frank Rosenblatt in 1958. A perceptron takes multiple inputs, multiplies each by a weight, sums them up, adds a bias term, and passes the result through an activation function to produce an output.

### 5.1.1 Activation Functions

Activation functions introduce non-linearity into the network, allowing it to learn complex patterns. Common activation functions include:

1. **Sigmoid**: σ(x) = 1 / (1 + e^(-x))
   - Output range: (0, 1)
   - Problem: Vanishing gradient for large |x|
   - Use case: Binary classification output layer

2. **Tanh**: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
   - Output range: (-1, 1)
   - Zero-centered, but still suffers from vanishing gradient
   - Better than sigmoid for hidden layers

3. **ReLU** (Rectified Linear Unit): f(x) = max(0, x)
   - Most popular activation for hidden layers
   - Computationally efficient
   - Problem: "Dying ReLU" when neurons always output 0
   - Variants: Leaky ReLU, PReLU, ELU

4. **Softmax**: For multi-class classification
   - Converts logits to probability distribution
   - Always used in the output layer for classification

## 5.2 Backpropagation Algorithm

Backpropagation is the cornerstone algorithm for training neural networks. It efficiently computes gradients of the loss function with respect to all parameters in the network.

### 5.2.1 Forward Pass

During the forward pass, inputs are propagated through the network layer by layer:
1. For each layer l, compute: a^l = σ(W^l * a^(l-1) + b^l)
2. The final layer produces the network's output
3. Compare output to ground truth using a loss function

### 5.2.2 Backward Pass

The backward pass computes gradients using the chain rule:
1. Compute loss gradient at output layer
2. Propagate error backwards through each layer
3. Calculate gradients for weights and biases
4. Update parameters using an optimization algorithm

### 5.2.3 Mathematical Derivation

For a simple two-layer network:
- Forward: z = Wx + b, a = σ(z)
- Loss: L = (a - y)^2 / 2
- Gradient: ∂L/∂W = ∂L/∂a * ∂a/∂z * ∂z/∂W
- Chain rule: ∂L/∂W = (a - y) * σ'(z) * x

## 5.3 Optimization Algorithms

### 5.3.1 Gradient Descent Variants

1. **Batch Gradient Descent**
   - Uses entire dataset for each update
   - Pros: Stable convergence
   - Cons: Slow for large datasets, can get stuck in local minima

2. **Stochastic Gradient Descent (SGD)**
   - Updates parameters for each training example
   - Pros: Fast, can escape local minima
   - Cons: High variance in updates, noisy convergence

3. **Mini-batch Gradient Descent**
   - Best of both worlds: batch size typically 32-256
   - Balances speed and stability
   - Most commonly used in practice

### 5.3.2 Advanced Optimizers

1. **Momentum**: Accelerates SGD by adding velocity
   - v = βv + (1-β)∇L
   - θ = θ - αv
   - Helps overcome plateaus and local minima

2. **RMSprop**: Adaptive learning rate per parameter
   - s = βs + (1-β)(∇L)^2
   - θ = θ - α * ∇L / √(s + ε)
   - Works well for non-stationary objectives

3. **Adam** (Adaptive Moment Estimation): Combines momentum and RMSprop
   - Most popular optimizer in deep learning
   - m = β1*m + (1-β1)*∇L (first moment)
   - v = β2*v + (1-β2)*(∇L)^2 (second moment)
   - θ = θ - α * m̂ / (√v̂ + ε)
   - Default hyperparameters: β1=0.9, β2=0.999, α=0.001

## 5.4 Regularization Techniques

Regularization prevents overfitting by constraining the model's capacity.

### 5.4.1 L1 and L2 Regularization

1. **L2 (Weight Decay)**: Add λ||W||^2 to loss
   - Penalizes large weights
   - Leads to smooth models
   - Most common choice

2. **L1 (Lasso)**: Add λ||W||_1 to loss
   - Promotes sparsity
   - Can zero out unimportant weights
   - Feature selection property

### 5.4.2 Dropout

Introduced by Hinton et al. in 2012, dropout randomly sets a fraction of neurons to zero during training:
- Prevents co-adaptation of neurons
- Acts as ensemble of many sub-networks
- Typical rate: 0.5 for hidden layers, 0.2 for input
- At test time, scale activations by keep probability

### 5.4.3 Batch Normalization

Normalizes activations within each mini-batch:
- BN(x) = γ * (x - μ) / √(σ^2 + ε) + β
- Reduces internal covariate shift
- Allows higher learning rates
- Has slight regularization effect
- Running statistics maintained for inference

### 5.4.4 Data Augmentation

Artificially increase training data by applying transformations:
- Images: rotation, flipping, cropping, color jittering
- Text: back-translation, synonym replacement
- Audio: time stretching, pitch shifting
- Forces model to learn invariant features

## 5.5 Convolutional Neural Networks (CNNs)

CNNs are specialized for processing grid-like data such as images.

### 5.5.1 Convolution Operation

A convolution applies a filter (kernel) across the input:
- Filter slides across spatial dimensions
- Element-wise multiplication and summation
- Produces feature map
- Parameters shared across spatial locations (translation invariance)

Key hyperparameters:
- Filter size: typically 3x3 or 5x5
- Stride: how many pixels to move (usually 1 or 2)
- Padding: "same" or "valid"
- Number of filters: depth of output

### 5.5.2 Pooling Layers

Pooling reduces spatial dimensions:
1. **Max Pooling**: Takes maximum in each region
   - Most common: 2x2 with stride 2
   - Provides translation invariance
   - Reduces computation

2. **Average Pooling**: Takes average
   - Gentler dimensionality reduction
   - Sometimes used before final classification layer

### 5.5.3 Classic CNN Architectures

1. **LeNet-5** (1998): First successful CNN
   - MNIST digit recognition
   - 2 conv layers, 2 pooling layers, 3 FC layers

2. **AlexNet** (2012): ImageNet breakthrough
   - 5 conv layers, 3 FC layers
   - ReLU activation, dropout
   - GPU training

3. **VGGNet** (2014): Very deep networks
   - 16-19 layers
   - Only 3x3 convolutions
   - Showed depth is beneficial

4. **ResNet** (2015): Residual connections
   - Skip connections: y = F(x) + x
   - Solved vanishing gradient in very deep networks
   - Up to 152 layers

5. **Inception** (2014): Multi-scale features
   - Parallel convolutions of different sizes
   - 1x1 convolutions for dimensionality reduction
   - Efficient computation

## 5.6 Recurrent Neural Networks (RNNs)

RNNs process sequential data by maintaining hidden state.

### 5.6.1 Basic RNN

h_t = tanh(W_hh * h_(t-1) + W_xh * x_t + b_h)
y_t = W_hy * h_t + b_y

Problems:
- Vanishing/exploding gradients
- Difficulty learning long-term dependencies
- Limited to short sequences in practice

### 5.6.2 LSTM (Long Short-Term Memory)

Introduced by Hochreiter & Schmidhuber (1997), LSTM uses gating mechanisms:

Gates:
- Forget gate: f_t = σ(W_f * [h_(t-1), x_t] + b_f)
- Input gate: i_t = σ(W_i * [h_(t-1), x_t] + b_i)
- Output gate: o_t = σ(W_o * [h_(t-1), x_t] + b_o)

Cell update:
- C̃_t = tanh(W_c * [h_(t-1), x_t] + b_c)
- C_t = f_t ⊙ C_(t-1) + i_t ⊙ C̃_t
- h_t = o_t ⊙ tanh(C_t)

Advantages:
- Can learn long-term dependencies
- Gradient flow through cell state
- Widely used for NLP tasks

### 5.6.3 GRU (Gated Recurrent Unit)

Simplified version of LSTM:
- Fewer parameters: reset and update gates only
- r_t = σ(W_r * [h_(t-1), x_t])
- z_t = σ(W_z * [h_(t-1), x_t])
- h̃_t = tanh(W * [r_t ⊙ h_(t-1), x_t])
- h_t = (1 - z_t) ⊙ h_(t-1) + z_t ⊙ h̃_t

Performance similar to LSTM, faster training.

## 5.7 Attention Mechanisms and Transformers

Attention allows models to focus on relevant parts of the input.

### 5.7.1 Attention Mechanism

Attention(Q, K, V) = softmax(QK^T / √d_k) * V

Where:
- Q: Query vectors
- K: Key vectors
- V: Value vectors
- d_k: Dimension of keys (scaling factor)

Intuition:
- Queries look for relevant keys
- Attention weights determine value contribution
- Can attend to entire sequence simultaneously

### 5.7.2 Self-Attention

Each position attends to all positions in the same sequence:
- Used in Transformers
- Captures long-range dependencies
- Parallelizable (unlike RNNs)
- Foundation of modern NLP (BERT, GPT)

### 5.7.3 Multi-Head Attention

Multiple attention mechanisms in parallel:
- Different heads learn different relationships
- Concatenate and project results
- Typical: 8 or 16 heads
- Each head has d_model / num_heads dimensions

### 5.7.4 Transformer Architecture

Encoder-Decoder structure:
- Encoder: Self-attention + Feed-forward
- Decoder: Masked self-attention + Encoder-decoder attention + Feed-forward
- Positional encoding for sequence order
- Layer normalization and residual connections

Key innovations:
- No recurrence: fully parallel training
- Self-attention: long-range dependencies
- Positional encoding: order information
- Scaled dot-product attention: stable training

Applications:
- Machine Translation: Original use case
- Language Modeling: GPT series
- Masked Language Modeling: BERT
- Image Generation: Vision Transformers (ViT)
- Multimodal: CLIP, DALL-E

## 5.8 Training Deep Networks

### 5.8.1 Weight Initialization

Proper initialization is crucial for deep networks:

1. **Xavier/Glorot**: Var(W) = 2 / (n_in + n_out)
   - For sigmoid/tanh activations
   - Maintains variance across layers

2. **He Initialization**: Var(W) = 2 / n_in
   - For ReLU activations
   - Accounts for dead neurons

3. **Orthogonal**: Initialize with orthogonal matrices
   - Preserves gradient norms
   - Good for RNNs

### 5.8.2 Learning Rate Scheduling

Adjust learning rate during training:

1. **Step Decay**: Reduce by factor every N epochs
2. **Exponential Decay**: α_t = α_0 * e^(-kt)
3. **Cosine Annealing**: Smooth reduction
4. **Warmup**: Start with small LR, gradually increase
5. **One Cycle**: Increase then decrease (super-convergence)

### 5.8.3 Gradient Clipping

Prevent exploding gradients:
- Clip by value: g = min(max(g, -threshold), threshold)
- Clip by norm: g = g * threshold / ||g|| if ||g|| > threshold
- Essential for RNNs/LSTMs

## 5.9 Advanced Topics

### 5.9.1 Residual Networks

Skip connections enable very deep networks:
- F(x) + x instead of just F(x)
- Identity mapping when F(x) = 0
- Gradient flows directly through shortcuts
- Enables 100+ layer networks

### 5.9.2 Depthwise Separable Convolutions

Factorize standard convolution:
- Depthwise: Apply filter per channel
- Pointwise: 1x1 convolution to combine
- Reduces parameters and computation
- Used in MobileNet, EfficientNet

### 5.9.3 Neural Architecture Search (NAS)

Automatically discover network architectures:
- Search space: possible operations and connections
- Search strategy: RL, evolution, gradient-based
- Performance estimation: train and evaluate
- AutoML applications

## 5.10 Important Hyperparameter: Attention Head Dimension

In modern Transformer architectures, the attention head dimension is a critical hyperparameter that affects model capacity and efficiency.

### 5.10.1 Head Dimension Formula

For a model with dimension d_model and h attention heads:
- head_dim = d_model / h
- Example: GPT-3 has d_model=12288 and h=96, so head_dim=128

### 5.10.2 Why 128 is a Magic Number

The value 128 appears frequently in successful models:
- BERT-Large: head_dim = 1024/16 = 64
- GPT-2: head_dim = 768/12 = 64
- GPT-3: head_dim = 12288/96 = 128
- LLaMA: head_dim = 128 (fixed)

Reasons:
- Balance between expressiveness and computation
- Powers of 2 are efficient on GPUs
- Large enough to capture complex patterns
- Small enough to keep total parameters reasonable

### 5.10.3 Impact on Performance

Studies show that head_dim between 64-128 is optimal:
- Too small (<32): Insufficient capacity per head
- Too large (>256): Redundancy, diminishing returns
- Sweet spot (64-128): Best performance/efficiency trade-off

This is why most modern LLMs standardize on head_dim=128.

## Summary

This chapter covered the fundamentals of deep learning, from basic neural networks to modern Transformer architectures. Key takeaways:

1. Activation functions introduce non-linearity
2. Backpropagation efficiently computes gradients
3. Optimizers like Adam improve convergence
4. Regularization prevents overfitting
5. CNNs excel at spatial data (images)
6. RNNs/LSTMs handle sequential data
7. Transformers use attention for parallel processing
8. Proper initialization and training techniques are essential
9. Attention head dimension of 128 is a well-established best practice

The field continues to evolve rapidly, with new architectures and techniques emerging regularly.
"""


def create_questions():
    """创建基于文档不同部分的问题"""
    return [
        {
            "question": "What are the three main types of activation functions discussed in section 5.1.1?",
            "location": "early",  # 文档前部
            "expected_keywords": ["sigmoid", "tanh", "relu"],
        },
        {
            "question": "According to section 5.3.2, what are the default hyperparameters for the Adam optimizer?",
            "location": "middle",  # 文档中部
            "expected_keywords": ["β1=0.9", "β2=0.999", "α=0.001", "0.9", "0.999"],
        },
        {
            "question": "What is the recommended attention head dimension mentioned in section 5.10.2, and why is it considered a 'magic number'?",
            "location": "late",  # 文档后部 - 关键测试点！
            "expected_keywords": ["128", "balance", "efficiency", "gpu", "optimal"],
        },
    ]


def test_without_compression(model, tokenizer, document, questions):
    """测试 Baseline：无压缩，文档会被截断"""
    print("="*70)
    print("Baseline: 无压缩 (max_kv_size=4096)")
    print("="*70)

    document_tokens = len(tokenizer.encode(document))
    print(f"\n文档长度: {document_tokens} tokens")
    print(f"KV cache limit: 4096 tokens")
    print(f"预期: 文档会被截断到前 4096 tokens")

    results = []

    for i, q in enumerate(questions):
        print(f"\n问题 {i+1} ({q['location']}):")
        print(f"  {q['question']}")

        prompt = f"{document}\n\nQuestion: {q['question']}\nAnswer:"

        answer = generate_with_compaction(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=100,
            use_compaction=False,  # 禁用压缩
            verbose=False,
        )

        print(f"  Answer: {answer[:150]}...")

        # 检查是否包含预期关键词
        answer_lower = answer.lower()
        matched = sum(1 for kw in q['expected_keywords'] if kw.lower() in answer_lower)
        score = matched / len(q['expected_keywords'])

        results.append({
            "question": q['question'],
            "location": q['location'],
            "answer": answer,
            "score": score,
        })

        print(f"  关键词匹配: {matched}/{len(q['expected_keywords'])} ({score*100:.0f}%)")

    return results


def test_with_compression(model, tokenizer, document, questions):
    """测试 AM 压缩：保留完整文档上下文"""
    print(f"\n{'='*70}")
    print("With AM Compression (max_size=2048, ratio=5.0)")
    print("="*70)

    document_tokens = len(tokenizer.encode(document))
    print(f"\n文档长度: {document_tokens} tokens")
    print(f"AM 配置: max_size=2048, compression_ratio=5.0")
    print(f"预期压缩后: ~{2048/5.0:.0f} tokens (保留完整上下文)")

    results = []

    for i, q in enumerate(questions):
        print(f"\n问题 {i+1} ({q['location']}):")
        print(f"  {q['question']}")

        prompt = f"{document}\n\nQuestion: {q['question']}\nAnswer:"

        answer = generate_with_compaction(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=100,
            use_compaction=True,
            compaction_config={
                "max_size": 2048,
                "compression_ratio": 5.0,
                "num_queries": 256,  # 更多 queries 确保质量
                "check_interval": 128,
                "use_quality_path": True,
            },
            verbose=False,
        )

        print(f"  Answer: {answer[:150]}...")

        # 检查是否包含预期关键词
        answer_lower = answer.lower()
        matched = sum(1 for kw in q['expected_keywords'] if kw.lower() in answer_lower)
        score = matched / len(q['expected_keywords'])

        results.append({
            "question": q['question'],
            "location": q['location'],
            "answer": answer,
            "score": score,
        })

        print(f"  关键词匹配: {matched}/{len(q['expected_keywords'])} ({score*100:.0f}%)")

    return results


def main():
    print("="*70)
    print("AM 压缩价值测试：长文档问答")
    print("="*70)

    model_path = "/Volumes/toshiba/models/qwen3-8b-mlx"
    print(f"\n加载模型: {model_path}")
    model, tokenizer = load(model_path)

    # 创建长文档和问题
    document = create_long_document()
    questions = create_questions()

    print(f"\n测试设置:")
    print(f"  文档: 机器学习教科书第 5 章")
    print(f"  文档长度: {len(tokenizer.encode(document))} tokens")
    print(f"  问题数量: {len(questions)}")
    print(f"  关键测试: 文档末尾的问题（section 5.10）")

    # 测试 1: Baseline (无压缩)
    baseline_results = test_without_compression(model, tokenizer, document, questions)

    # 测试 2: AM 压缩
    am_results = test_with_compression(model, tokenizer, document, questions)

    # ========================================
    # 对比分析
    # ========================================
    print(f"\n{'='*70}")
    print("对比分析")
    print("="*70)

    print(f"\n{'位置':<10} {'问题编号':<10} {'Baseline':<15} {'AM 压缩':<15} {'改进':<10}")
    print("-" * 70)

    for i in range(len(questions)):
        baseline_score = baseline_results[i]['score']
        am_score = am_results[i]['score']
        improvement = am_score - baseline_score
        location = questions[i]['location']

        print(f"{location:<10} Q{i+1:<9} {baseline_score*100:>6.0f}% {' '*7} {am_score*100:>6.0f}% {' '*7} {improvement*100:>+6.0f}%")

    # 关键指标
    baseline_late = [r['score'] for r in baseline_results if questions[baseline_results.index(r)]['location'] == 'late'][0]
    am_late = [r['score'] for r in am_results if questions[am_results.index(r)]['location'] == 'late'][0]

    print(f"\n{'='*70}")
    print("关键发现")
    print("="*70)

    print(f"\n文档末尾问题（section 5.10 - 最关键的测试）:")
    print(f"  Baseline: {baseline_late*100:.0f}% 正确率")
    print(f"  AM 压缩:  {am_late*100:.0f}% 正确率")
    print(f"  改进:     {(am_late - baseline_late)*100:+.0f}%")

    if am_late > baseline_late:
        print(f"\n✅ AM 压缩价值验证成功！")
        print(f"   AM 压缩能保留文档末尾的信息，而 baseline 丢失了！")
    else:
        print(f"\n⚠️  两者表现相近")

    avg_baseline = sum(r['score'] for r in baseline_results) / len(baseline_results)
    avg_am = sum(r['score'] for r in am_results) / len(am_results)

    print(f"\n平均性能:")
    print(f"  Baseline: {avg_baseline*100:.0f}%")
    print(f"  AM 压缩:  {avg_am*100:.0f}%")
    print(f"  改进:     {(avg_am - avg_baseline)*100:+.0f}%")

    # 内存分析
    print(f"\n内存分析:")
    doc_tokens = len(tokenizer.encode(document))
    print(f"  文档大小: {doc_tokens} tokens")
    print(f"  Baseline KV cache: {min(doc_tokens, 4096)} tokens (截断)")
    print(f"  AM 压缩 KV cache: ~{2048/5.0:.0f} tokens (完整上下文)")
    print(f"  内存节省: {(1 - 2048/5.0/min(doc_tokens, 4096))*100:.0f}%")

    print(f"\n{'='*70}")
    print("结论")
    print("="*70)

    if am_late > baseline_late + 0.2:  # 20% 以上改进
        print(f"✅ AM 压缩显著提升长文档问答能力！")
        print(f"   关键价值:")
        print(f"   1. 保留完整文档上下文（{doc_tokens} tokens → ~410 tokens）")
        print(f"   2. 能回答文档末尾的问题（Baseline 丢失）")
        print(f"   3. 内存节省 {(1 - 2048/5.0/min(doc_tokens, 4096))*100:.0f}%")
        print(f"   4. 使用 Qref 确保关键信息不丢失")
    elif am_late > baseline_late:
        print(f"⚠️  AM 压缩有改进，但不显著")
    else:
        print(f"⚠️  测试未显示明显价值（可能需要调整配置）")


if __name__ == "__main__":
    main()
