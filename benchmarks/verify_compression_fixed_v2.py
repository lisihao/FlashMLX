#!/usr/bin/env python3
"""
任务 1 & 2: 修复版本 v2
使用正确的压缩实现
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
from mlx_lm.compaction.quality import compact_single_head_quality


def test_am_quality(K, V, Q, budget):
    """测试 AM 质量（使用 quality.py）"""
    try:
        # 使用 quality path 压缩
        C1, beta, C2 = compact_single_head_quality(
            Q, K, V, budget,
            scale=None,
            fit_beta=True,
            fit_c2=True,
            nnls_method="clamped",
            lsq_method="lstsq"
        )

        # 计算质量
        head_dim = K.shape[1]
        scale = 1.0 / mx.sqrt(mx.array(head_dim, dtype=K.dtype))

        # 原始输出
        attn_orig = mx.softmax(Q @ K.T * scale, axis=-1)
        out_orig = attn_orig @ V

        # 压缩输出
        attn_comp = mx.softmax(Q @ C1.T * scale + beta, axis=-1)
        out_comp = attn_comp @ C2

        # Cosine similarity
        out_orig_flat = mx.reshape(out_orig, (-1,))
        out_comp_flat = mx.reshape(out_comp, (-1,))
        cos_sim = float(
            mx.sum(out_orig_flat * out_comp_flat) /
            (mx.linalg.norm(out_orig_flat) * mx.linalg.norm(out_comp_flat))
        )

        return {'cosine_similarity': cos_sim}

    except Exception as e:
        print(f"    AM failed: {e}")
        return None


def test_compression_on_random_kv(model_name, seq_len=512, compression_ratio=2.0):
    """
    在随机 KV 上测试压缩质量
    """
    print(f"\n{'='*70}")
    print(f"压缩质量测试: {Path(model_name).name}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Compression ratio: {compression_ratio}")
    print(f"{'='*70}")

    try:
        # 加载模型（只用于获取架构信息）
        print("Loading model...")
        model, tokenizer = load(model_name)
        print(f"Model loaded: {len(model.model.layers)} layers")

        # 识别 Attention 层
        attention_layers = []
        for i, layer in enumerate(model.model.layers):
            if hasattr(layer, 'self_attn'):
                attention_layers.append(i)

        print(f"Attention layers: {len(attention_layers)} total")

        if not attention_layers:
            print("❌ No attention layers found")
            del model, tokenizer
            return None

        # 获取 head_dim
        first_attn_layer = model.model.layers[attention_layers[0]]
        head_dim = first_attn_layer.self_attn.head_dim

        print(f"Head dimension: {head_dim}")

        # 清理模型（不再需要）
        del model, tokenizer
        gc.collect()
        mx.clear_cache()

        # 创建随机 KV cache
        print(f"\nCreating random KV cache...")
        K = mx.random.normal((seq_len, head_dim))
        V = mx.random.normal((seq_len, head_dim))
        Q = K  # 使用 K 作为 Q

        budget = max(int(seq_len / compression_ratio), 10)
        print(f"Compression budget: {seq_len} → {budget}")

        # 测试三种方法
        results = {}

        # Test AM
        print(f"\n  [1/3] Testing AM...")
        am_result = test_am_quality(K, V, Q, budget)
        if am_result:
            results['AM'] = am_result['cosine_similarity']
            print(f"    Quality: {am_result['cosine_similarity']:.6f}")
        else:
            results['AM'] = 0.0

        # Test H2O
        print(f"\n  [2/3] Testing H2O...")
        try:
            h2o_result = test_h2o_quality(K, V, Q, max_capacity=budget, recent_ratio=0.25)
            if h2o_result:
                results['H2O'] = h2o_result['cosine_similarity']
                print(f"    Quality: {h2o_result['cosine_similarity']:.6f}")
            else:
                results['H2O'] = 0.0
        except Exception as e:
            print(f"    H2O failed: {e}")
            results['H2O'] = 0.0

        # Test StreamingLLM
        print(f"\n  [3/3] Testing StreamingLLM...")
        try:
            stream_result = test_streaming_llm_quality(K, V, Q, max_capacity=budget, num_sinks=4)
            if stream_result:
                results['StreamingLLM'] = stream_result['cosine_similarity']
                print(f"    Quality: {stream_result['cosine_similarity']:.6f}")
            else:
                results['StreamingLLM'] = 0.0
        except Exception as e:
            print(f"    StreamingLLM failed: {e}")
            results['StreamingLLM'] = 0.0

        print(f"\n✅ Test completed:")
        for method, quality in results.items():
            print(f"  {method}: {quality:.6f}")

        # 清理
        del K, V, Q
        gc.collect()
        mx.clear_cache()

        return {
            'success': True,
            'model': Path(model_name).name,
            'seq_len': seq_len,
            'budget': budget,
            'results': results
        }

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

        gc.collect()
        mx.clear_cache()

        return {
            'success': False,
            'model': Path(model_name).name,
            'error': str(e)
        }


def analyze_multi_turn_failure(model_name):
    """
    任务 2: 分析多轮对话场景失败的根本原因
    """
    print(f"\n{'='*70}")
    print("任务 2: 多轮对话场景失败分析")
    print(f"{'='*70}")

    # 测试用例
    test_cases = [
        {
            'name': "原始 prompt (长对话历史)",
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
            'max_tokens': 200
        },
        {
            'name': "简化 prompt (压缩历史)",
            'prompt': """对话历史摘要：用户学习路径是 Python → 数学 → 机器学习 → PyTorch。

问题：学完PyTorch基础后，应该做什么项目来实践？请给出详细建议。""",
            'max_tokens': 200
        },
        {
            'name': "纯问题 (无历史)",
            'prompt': "学完PyTorch基础后，应该做什么项目来实践？请给出详细建议。",
            'max_tokens': 200
        }
    ]

    results = []

    try:
        print("Loading model...")
        model, tokenizer = load(model_name)

        for test_case in test_cases:
            print(f"\n{'='*70}")
            print(f"测试: {test_case['name']}")
            print(f"{'='*70}")

            tokens = tokenizer.encode(test_case['prompt'])
            print(f"Prompt tokens: {len(tokens)}")

            print("Generating...")
            start = time.time()

            try:
                response = generate(
                    model,
                    tokenizer,
                    prompt=test_case['prompt'],
                    max_tokens=test_case['max_tokens'],
                    verbose=False
                )

                elapsed = time.time() - start
                num_tokens = len(tokenizer.encode(response))

                # 检查乱码
                is_garbage = check_garbage(response)

                print(f"\n✅ Success")
                print(f"Tokens: {num_tokens}")
                print(f"Time: {elapsed:.2f}s")
                print(f"Garbage: {'❌ Yes' if is_garbage else '✅ No'}")
                print(f"\nFirst 200 chars:")
                print(f"{'-'*70}")
                print(response[:200])
                if len(response) > 200:
                    print("...")
                print(f"{'-'*70}")

                results.append({
                    'name': test_case['name'],
                    'success': True,
                    'prompt_tokens': len(tokens),
                    'output_tokens': num_tokens,
                    'is_garbage': is_garbage,
                    'output_sample': response[:500]
                })

            except Exception as e:
                print(f"❌ Failed: {e}")
                results.append({
                    'name': test_case['name'],
                    'success': False,
                    'prompt_tokens': len(tokens),
                    'error': str(e)
                })

            # 清理
            gc.collect()
            mx.clear_cache()
            time.sleep(1)

        del model, tokenizer
        gc.collect()
        mx.clear_cache()

        return results

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

        gc.collect()
        mx.clear_cache()

        return None


def check_garbage(text):
    """检查是否为乱码"""
    if not text or len(text) < 10:
        return True

    words = text.split()
    if len(words) < 10:
        return False

    # 检查重复率
    first_50 = words[:min(50, len(words))]
    unique_ratio = len(set(first_50)) / len(first_50)

    if unique_ratio < 0.3:
        return True

    # 检查连续重复
    for i in range(len(words) - 5):
        if words[i] == words[i+1] == words[i+2] == words[i+3] == words[i+4]:
            return True

    return False


def main():
    print("="*70)
    print("任务 1 & 2: 压缩方法验证 + 失败分析")
    print("="*70)

    all_results = {}

    # 任务 1: 测试压缩方法
    print("\n" + "="*70)
    print("任务 1: 压缩方法质量测试")
    print("="*70)

    models = [
        ("/Volumes/toshiba/models/qwen3.5-0.8b-opus-distilled", "Qwen3.5-0.8B (混合架构)"),
        ("/Volumes/toshiba/models/qwen3-8b-mlx", "Qwen3-8B (纯 Transformer)")
    ]

    for model_path, model_desc in models:
        print(f"\n{'='*70}")
        print(f"测试: {model_desc}")
        print(f"{'='*70}")

        result = test_compression_on_random_kv(
            model_path,
            seq_len=512,
            compression_ratio=2.0
        )

        all_results[model_desc] = result

        # 清理
        gc.collect()
        mx.clear_cache()
        time.sleep(3)

    # 任务 2: 多轮对话失败分析
    print("\n" + "="*70)
    print("任务 2: 多轮对话场景失败分析")
    print("="*70)

    failure_analysis = analyze_multi_turn_failure(
        "/Volumes/toshiba/models/qwen3-8b-mlx"
    )

    all_results['failure_analysis'] = failure_analysis

    # 生成报告
    report_path = Path(__file__).parent.parent / ".solar" / "compression-final-report.md"

    with open(report_path, "w") as f:
        f.write("# 压缩方法验证与失败分析 - 最终报告\n\n")
        f.write(f"**日期**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # 任务 1
        f.write("## 任务 1: 压缩方法质量对比\n\n")
        f.write("### 测试方法\n")
        f.write("- 使用随机生成的 KV cache (512 tokens, head_dim=模型配置)\n")
        f.write("- Compression ratio: 2.0 (512 → 256)\n")
        f.write("- 质量指标: Cosine Similarity\n\n")

        for model_desc in ["Qwen3.5-0.8B (混合架构)", "Qwen3-8B (纯 Transformer)"]:
            if model_desc in all_results:
                result = all_results[model_desc]

                f.write(f"### {model_desc}\n\n")

                if result and result.get('success'):
                    f.write("| 方法 | 质量 |\n")
                    f.write("|------|------|\n")
                    for method, quality in result['results'].items():
                        f.write(f"| {method} | {quality:.6f} |\n")
                    f.write("\n")
                else:
                    f.write(f"测试失败: {result.get('error', 'Unknown') if result else 'No result'}\n\n")

        # 任务 2
        f.write("## 任务 2: 多轮对话场景失败分析\n\n")

        if failure_analysis:
            f.write("| Prompt 类型 | Tokens | 乱码 | 状态 |\n")
            f.write("|------------|--------|------|------|\n")

            for result in failure_analysis:
                if result.get('success'):
                    garbage = "❌ 是" if result.get('is_garbage') else "✅ 否"
                    f.write(f"| {result['name']} | {result['prompt_tokens']} | {garbage} | ✅ |\n")
                else:
                    f.write(f"| {result['name']} | {result.get('prompt_tokens', '-')} | - | ❌ |\n")

            f.write("\n### 分析结论\n\n")

            original = next((r for r in failure_analysis if '原始' in r['name']), None)
            simplified = next((r for r in failure_analysis if '简化' in r['name']), None)
            pure = next((r for r in failure_analysis if '纯问题' in r['name']), None)

            if original and original.get('success'):
                if original.get('is_garbage'):
                    f.write("**根因分析：原始 prompt 产生乱码**\n\n")

                    if simplified and simplified.get('success'):
                        if simplified.get('is_garbage'):
                            f.write("- 简化后仍乱码 → 可能是对话格式或 prompt 长度问题\n")
                        else:
                            f.write("- 简化后正常 → 问题在于对话历史的复杂度或格式\n")

                    if pure and pure.get('success'):
                        if pure.get('is_garbage'):
                            f.write("- 纯问题也乱码 → 可能是模型本身或问题内容问题\n")
                        else:
                            f.write("- 纯问题正常 → **确认问题在于对话历史处理**\n")
                            f.write("\n**建议**: 在多轮对话场景下，应该使用更简洁的历史摘要格式。\n")
                else:
                    f.write("原始 prompt 正常生成。\n")

    print(f"\n详细报告已保存到: {report_path}")

    print("\n" + "="*70)
    print("任务完成!")
    print("="*70)


if __name__ == "__main__":
    main()
