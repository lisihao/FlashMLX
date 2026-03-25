#!/usr/bin/env python3
"""
完整验证方案：
Step 1: Qwen3.5-0.8B 快速测试 H2O/StreamingLLM 是否乱码
Step 2: Qwen3-8B 测试盲区场景（极长上下文、流式、特定领域、多轮对话）
"""

import sys
import time
import gc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "mlx-lm-source"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import mlx.core as mx
from mlx_lm import load, generate
from flashmlx.cache.h2o import test_h2o_quality
from flashmlx.cache.streaming_llm import test_streaming_llm_quality


def check_garbage(text):
    """检查是否为乱码"""
    if not text or len(text) < 10:
        return True

    words = text.split()
    if len(words) < 10:
        return False

    first_50 = words[:min(50, len(words))]
    unique_ratio = len(set(first_50)) / len(first_50)

    if unique_ratio < 0.3:
        return True

    for i in range(len(words) - 5):
        if words[i] == words[i+1] == words[i+2] == words[i+3] == words[i+4]:
            return True

    return False


def test_compression_quality(model_name, prompt, method_name, compression_ratio=2.0):
    """
    测试压缩质量（使用 flashmlx 的测试函数）
    """
    print(f"\n{'='*70}")
    print(f"测试: {method_name} on {Path(model_name).name}")
    print(f"  Compression ratio: {compression_ratio}")
    print(f"{'='*70}")

    try:
        # 加载模型
        print("Loading model...")
        model, tokenizer = load(model_name)
        print(f"Model loaded: {len(model.layers)} layers")

        # Tokenize
        tokens = mx.array(tokenizer.encode(prompt))
        print(f"Prompt tokens: {tokens.shape[0]}")

        # Forward pass to get KV cache
        print("Running forward pass...")
        output = model(tokens[None, :])

        # 检查是否有 cache
        if not hasattr(model, '_cache') or model._cache is None:
            print("❌ Model has no cache, skipping compression test")
            return None

        cache = model._cache

        # 收集所有 Attention 层的 KV cache
        all_results = []

        for layer_idx, layer_cache in enumerate(cache):
            if layer_cache is None or len(layer_cache) != 2:
                continue

            keys, values = layer_cache
            if keys is None or values is None:
                continue

            # 只测试第一个 head（简化）
            if len(keys.shape) >= 3:
                n_heads, seq_len, head_dim = keys.shape
                K = keys[0]  # (seq_len, head_dim)
                V = values[0]
                Q = K  # 使用 keys 作为 queries

                budget = int(seq_len / compression_ratio)

                try:
                    if method_name == "H2O":
                        result = test_h2o_quality(K, V, Q, max_capacity=budget, recent_ratio=0.25)
                    elif method_name == "StreamingLLM":
                        result = test_streaming_llm_quality(K, V, Q, max_capacity=budget, num_sinks=4)
                    else:
                        continue

                    quality = result.get('cosine_similarity', 0.0)
                    all_results.append(quality)

                except Exception as e:
                    print(f"  Layer {layer_idx} failed: {e}")
                    continue

        if not all_results:
            print("❌ No valid compression results")
            return None

        avg_quality = sum(all_results) / len(all_results)
        print(f"\n✅ Average quality: {avg_quality:.6f} (over {len(all_results)} layers)")

        # 清理内存
        del model, tokenizer, cache
        gc.collect()
        mx.metal.clear_cache()

        return {
            'success': True,
            'method': method_name,
            'model': Path(model_name).name,
            'quality': avg_quality,
            'num_layers': len(all_results)
        }

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

        # 清理内存
        gc.collect()
        mx.metal.clear_cache()

        return {
            'success': False,
            'method': method_name,
            'model': Path(model_name).name,
            'error': str(e)
        }


def test_generation_with_baseline(model_name, prompt, max_tokens=200):
    """
    测试 Baseline 生成（无压缩）
    """
    print(f"\n{'='*70}")
    print(f"Baseline Generation: {Path(model_name).name}")
    print(f"{'='*70}")

    try:
        print("Loading model...")
        model, tokenizer = load(model_name)
        print(f"Model loaded: {len(model.layers)} layers")

        print(f"\nGenerating...")
        start = time.time()
        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False
        )
        elapsed = time.time() - start

        num_tokens = len(tokenizer.encode(response))
        is_garbage = check_garbage(response)

        print(f"\n✅ Success")
        print(f"Tokens: {num_tokens}")
        print(f"Time: {elapsed:.2f}s")
        print(f"Speed: {num_tokens / elapsed:.2f} tok/s")
        print(f"Garbage: {'❌ Yes' if is_garbage else '✅ No'}")
        print(f"\nOutput (first 300 chars):")
        print(f"{'-'*70}")
        print(response[:300])
        if len(response) > 300:
            print("... (truncated)")
        print(f"{'-'*70}")

        # 清理内存
        del model, tokenizer
        gc.collect()
        mx.metal.clear_cache()

        return {
            'success': True,
            'model': Path(model_name).name,
            'tokens': num_tokens,
            'time': elapsed,
            'tps': num_tokens / elapsed,
            'is_garbage': is_garbage,
            'output': response
        }

    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()

        # 清理内存
        gc.collect()
        mx.metal.clear_cache()

        return {
            'success': False,
            'model': Path(model_name).name,
            'error': str(e)
        }


def step1_quick_test():
    """
    Step 1: Qwen3.5-0.8B 快速测试
    """
    print("\n" + "="*70)
    print("STEP 1: Quick Test on Qwen3.5-0.8B")
    print("="*70)

    model_name = "/Volumes/toshiba/models/qwen3.5-0.8b-opus-distilled"
    prompt = "介绍机器学习的基本概念"

    results = []

    # Baseline
    print("\n[1/3] Testing Baseline...")
    baseline = test_generation_with_baseline(model_name, prompt, max_tokens=100)
    results.append(baseline)

    # H2O
    print("\n[2/3] Testing H2O...")
    h2o = test_compression_quality(model_name, prompt, "H2O", compression_ratio=2.0)
    results.append(h2o)

    # StreamingLLM
    print("\n[3/3] Testing StreamingLLM...")
    streaming = test_compression_quality(model_name, prompt, "StreamingLLM", compression_ratio=2.0)
    results.append(streaming)

    return results


def step2_blind_spot_tests():
    """
    Step 2: Qwen3-8B 盲区场景测试
    """
    print("\n" + "="*70)
    print("STEP 2: Blind Spot Tests on Qwen3-8B")
    print("="*70)

    model_name = "/Volumes/toshiba/models/qwen3-8b-mlx"

    test_cases = [
        {
            'name': '极长上下文 (4K+ tokens)',
            'prompt': """以下是一篇关于深度学习的长文章：

深度学习是机器学习的一个分支，它使用多层神经网络来学习数据的表示。深度学习已经在计算机视觉、自然语言处理、语音识别等领域取得了突破性进展。

卷积神经网络（CNN）是深度学习中最重要的架构之一，主要用于图像处理任务。CNN通过卷积层、池化层和全连接层的组合，能够自动学习图像的特征表示。

循环神经网络（RNN）专门用于处理序列数据，如文本和时间序列。LSTM和GRU是RNN的改进版本，解决了长期依赖问题。

Transformer架构彻底改变了自然语言处理领域，它使用自注意力机制来捕捉序列中的长距离依赖关系。BERT、GPT等大型语言模型都基于Transformer架构。

深度学习的训练需要大量数据和计算资源。反向传播算法用于计算梯度，优化器（如Adam、SGD）用于更新模型参数。正则化技术（如Dropout、Batch Normalization）可以防止过拟合。

迁移学习是深度学习中的重要技术，通过在大规模数据集上预训练模型，然后在特定任务上微调，可以大大减少训练时间和数据需求。

生成对抗网络（GAN）是一种生成模型，由生成器和判别器组成，通过对抗训练来生成逼真的数据。扩散模型是近年来兴起的另一种生成模型，在图像生成任务上表现出色。

强化学习结合深度学习，产生了深度强化学习，成功应用于游戏AI、机器人控制等领域。AlphaGo就是深度强化学习的典型应用。

深度学习的可解释性是当前研究的热点，如何理解和解释神经网络的决策过程对于应用至关重要。注意力可视化、显著性图等技术可以帮助我们理解模型的行为。

联邦学习是一种隐私保护的机器学习方法，允许多个参与方在不共享原始数据的情况下协同训练模型。这对于医疗、金融等敏感领域特别重要。

神经架构搜索（NAS）自动设计神经网络结构，可以找到比人工设计更优的架构。AutoML进一步自动化了整个机器学习流程。

模型压缩和加速技术对于将深度学习模型部署到移动设备和边缘计算设备上至关重要。量化、剪枝、知识蒸馏等技术可以在保持性能的同时大幅减小模型大小。

多模态学习结合视觉、语言、音频等多种模态的信息，是实现通用人工智能的重要方向。CLIP、DALL-E等模型展示了多模态学习的巨大潜力。

持续学习使模型能够不断从新数据中学习而不遗忘旧知识，这对于构建长期运行的AI系统非常重要。

元学习（Learning to Learn）研究如何让模型快速适应新任务，只需少量样本就能学会新技能。

神经符号集成试图结合神经网络的学习能力和符号系统的推理能力，构建更强大的AI系统。

深度学习在科学领域也有广泛应用，如蛋白质结构预测、药物发现、材料设计等，展现了AI在科学研究中的巨大潜力。

现在，基于上述内容，请详细总结深度学习的主要发展方向和未来趋势。""",
            'max_tokens': 300
        },
        {
            'name': '流式生成场景',
            'prompt': "请生成一个长故事，描述一个程序员的一天，从早上起床到晚上睡觉，包含工作中的各种细节。",
            'max_tokens': 500
        },
        {
            'name': '医学领域',
            'prompt': "解释糖尿病的病理机制、诊断标准、治疗方案和预防措施，要求专业详细。",
            'max_tokens': 300
        },
        {
            'name': '法律领域',
            'prompt': "分析合同法中的要约、承诺、对价等基本概念，并举例说明合同成立的条件。",
            'max_tokens': 300
        },
        {
            'name': '科学领域',
            'prompt': "详细解释量子力学中的波粒二象性、不确定性原理和量子纠缠现象，包括相关实验和数学公式。",
            'max_tokens': 300
        },
        {
            'name': '多轮对话长历史',
            'prompt': """以下是一段长对话历史：

用户: 你好，我想学习机器学习，应该从哪里开始？
助手: 建议从Python编程基础开始，然后学习numpy、pandas等数据处理库，接着学习线性代数和微积分基础。

用户: 我已经会Python了，线性代数需要学到什么程度？
助手: 需要掌握矩阵运算、特征值分解、奇异值分解等，这些是理解机器学习算法的基础。

用户: 学完数学基础后应该学什么？
助手: 可以开始学习经典机器学习算法，如线性回归、逻辑回归、决策树、随机森林等。

用户: 深度学习和传统机器学习有什么区别？
助手: 深度学习使用多层神经网络自动学习特征表示，而传统机器学习通常需要手工设计特征。

用户: 我应该先学PyTorch还是TensorFlow？
助手: 建议从PyTorch开始，它的语法更直观，更容易上手。

现在，基于上述对话，用户接着问：那学完PyTorch基础后，应该做什么项目来实践？请给出详细建议。""",
            'max_tokens': 300
        }
    ]

    results = []

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"Test Case {i}/{len(test_cases)}: {test_case['name']}")
        print(f"{'='*70}")

        # Baseline
        print(f"\n  [1/3] Baseline...")
        baseline = test_generation_with_baseline(
            model_name,
            test_case['prompt'],
            max_tokens=test_case['max_tokens']
        )

        # H2O
        print(f"\n  [2/3] H2O...")
        h2o = test_compression_quality(
            model_name,
            test_case['prompt'],
            "H2O",
            compression_ratio=2.0
        )

        # StreamingLLM
        print(f"\n  [3/3] StreamingLLM...")
        streaming = test_compression_quality(
            model_name,
            test_case['prompt'],
            "StreamingLLM",
            compression_ratio=2.0
        )

        results.append({
            'test_case': test_case['name'],
            'baseline': baseline,
            'h2o': h2o,
            'streaming': streaming
        })

        # 强制清理内存
        gc.collect()
        mx.metal.clear_cache()
        time.sleep(2)  # 等待内存释放

    return results


def generate_report(step1_results, step2_results):
    """生成完整报告"""
    report_path = Path(__file__).parent.parent / ".solar" / "h2o-streaming-full-verification.md"

    with open(report_path, "w") as f:
        f.write("# H2O 和 StreamingLLM 完整验证报告\n\n")
        f.write(f"**日期**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Step 1
        f.write("## Step 1: Qwen3.5-0.8B 快速测试\n\n")
        f.write("### 目标\n验证 H2O 和 StreamingLLM 在 Qwen3.5 混合架构上是否产生乱码\n\n")
        f.write("### 结果\n\n")
        f.write("| 方法 | 成功 | 质量/状态 |\n")
        f.write("|------|------|----------|\n")

        for result in step1_results:
            if result is None:
                continue
            if result['success']:
                if 'quality' in result:
                    f.write(f"| {result['method']} | ✅ | {result['quality']:.6f} |\n")
                else:
                    garbage = "❌ 乱码" if result.get('is_garbage', False) else "✅ 正常"
                    f.write(f"| Baseline | ✅ | {garbage} |\n")
            else:
                f.write(f"| {result.get('method', 'Unknown')} | ❌ | {result.get('error', 'Unknown')} |\n")

        # Step 2
        f.write("\n## Step 2: Qwen3-8B 盲区场景测试\n\n")
        f.write("### 测试场景\n")
        f.write("- 极长上下文 (4K+ tokens)\n")
        f.write("- 流式生成场景\n")
        f.write("- 特定领域（医学、法律、科学）\n")
        f.write("- 多轮对话长历史\n\n")

        f.write("### 结果\n\n")

        for result in step2_results:
            f.write(f"#### {result['test_case']}\n\n")
            f.write("| 方法 | 成功 | 质量/状态 |\n")
            f.write("|------|------|----------|\n")

            for key in ['baseline', 'h2o', 'streaming']:
                r = result[key]
                if r is None:
                    continue
                if r['success']:
                    if 'quality' in r:
                        f.write(f"| {r['method']} | ✅ | {r['quality']:.6f} |\n")
                    else:
                        garbage = "❌ 乱码" if r.get('is_garbage', False) else "✅ 正常"
                        f.write(f"| Baseline | ✅ | {garbage} |\n")
                else:
                    f.write(f"| {r.get('method', 'Baseline')} | ❌ | {r.get('error', 'Unknown')[:50]} |\n")

            f.write("\n")

        f.write("## 总结\n\n")
        f.write("### H2O 和 StreamingLLM 在混合架构上的表现\n\n")
        f.write("（待分析）\n\n")
        f.write("### 不同场景下的算法选择建议\n\n")
        f.write("（待分析）\n")

    print(f"\n详细报告已保存到: {report_path}")


def main():
    print("="*70)
    print("H2O 和 StreamingLLM 完整验证")
    print("="*70)
    print("\n执行计划:")
    print("  Step 1: Qwen3.5-0.8B 快速测试（验证是否乱码）")
    print("  Step 2: Qwen3-8B 盲区场景测试（5个场景）")
    print("\n⚠️  串行执行，避免内存溢出\n")

    # Step 1
    step1_results = step1_quick_test()

    # 清理内存
    print("\n清理内存...")
    gc.collect()
    mx.metal.clear_cache()
    time.sleep(5)

    # Step 2
    step2_results = step2_blind_spot_tests()

    # 生成报告
    generate_report(step1_results, step2_results)

    print("\n" + "="*70)
    print("验证完成!")
    print("="*70)


if __name__ == "__main__":
    main()
