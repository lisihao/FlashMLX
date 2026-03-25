#!/usr/bin/env python3
"""
端到端测试：AM 压缩在 Qwen3.5 系列上是否真的能用于推理
（不只是测试压缩质量，而是真正用压缩后的 cache 生成文本）
"""

import sys
import time
import gc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "mlx-lm-source"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import mlx.core as mx
from mlx_lm import load, generate


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

    # 检查句子级别的重复
    sentences = text.split('。')
    if len(sentences) >= 3:
        # 如果前三句有两句一样，认为是乱码
        if sentences[0] == sentences[1] or sentences[1] == sentences[2]:
            return True

    return False


def test_am_on_model_e2e(model_path, test_prompt, max_tokens=200, compression_ratio=2.0):
    """
    端到端测试 AM 压缩

    流程：
    1. 加载模型
    2. 启用 AM 压缩（如果可能）
    3. 生成文本
    4. 检查是否乱码
    """
    print(f"\n{'='*70}")
    print(f"端到端测试: {Path(model_path).name}")
    print(f"  Prompt: {test_prompt[:50]}...")
    print(f"  Max tokens: {max_tokens}")
    print(f"  Compression ratio: {compression_ratio}")
    print(f"{'='*70}")

    try:
        # 加载模型
        print("\nLoading model...")
        model, tokenizer = load(model_path)

        # 检查模型类型
        model_type = type(model).__name__
        print(f"Model type: {model_type}")

        # 检查是否是混合架构
        if hasattr(model, 'layers'):
            layer_types = set()
            for layer in model.layers:
                if 'self_attn' in layer.state:
                    layer_types.add('Attention')
                elif 'linear_attn' in layer.state:
                    layer_types.add('SSM')

            is_hybrid = len(layer_types) > 1
            print(f"Architecture: {'Hybrid (SSM + Attention)' if is_hybrid else 'Pure Transformer'}")
            print(f"Layer types: {layer_types}")

        # Tokenize
        tokens = tokenizer.encode(test_prompt)
        print(f"Prompt tokens: {len(tokens)}")

        # Test 1: Baseline (无压缩)
        print(f"\n{'='*70}")
        print("Test 1: Baseline (无压缩)")
        print(f"{'='*70}")

        start = time.time()
        baseline_response = generate(
            model,
            tokenizer,
            prompt=test_prompt,
            max_tokens=max_tokens,
            verbose=False
        )
        baseline_time = time.time() - start
        baseline_garbage = check_garbage(baseline_response)

        print(f"✅ Baseline generated")
        print(f"Tokens: {len(tokenizer.encode(baseline_response))}")
        print(f"Time: {baseline_time:.2f}s")
        print(f"Garbage: {'❌ Yes' if baseline_garbage else '✅ No'}")
        print(f"\nFirst 200 chars:")
        print(f"{'-'*70}")
        print(baseline_response[:200])
        if len(baseline_response) > 200:
            print("...")
        print(f"{'-'*70}")

        # 清理 cache
        gc.collect()
        mx.clear_cache()

        # Test 2: 使用 AM 压缩
        # 注意：这里我们需要修改 mlx-lm 来支持 AM 压缩
        # 目前 mlx-lm 的 generate 函数不支持传入压缩配置
        # 所以这个测试暂时无法完成

        print(f"\n{'='*70}")
        print("Test 2: 使用 AM 压缩")
        print(f"{'='*70}")
        print("⚠️  mlx-lm 的 generate() 函数目前不支持传入压缩配置")
        print("⚠️  需要修改 mlx-lm 或使用低级 API")
        print("⚠️  跳过此测试")

        # 清理
        del model, tokenizer
        gc.collect()
        mx.clear_cache()

        return {
            'success': True,
            'model': Path(model_path).name,
            'is_hybrid': is_hybrid if 'is_hybrid' in locals() else False,
            'baseline': {
                'tokens': len(tokenizer.encode(baseline_response)),
                'time': baseline_time,
                'is_garbage': baseline_garbage,
                'sample': baseline_response[:500]
            }
        }

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

        gc.collect()
        mx.clear_cache()

        return {
            'success': False,
            'model': Path(model_path).name,
            'error': str(e)
        }


def main():
    print("="*70)
    print("Qwen3.5 系列 AM 压缩端到端测试")
    print("="*70)

    # 测试 prompt
    test_prompt = """学完PyTorch基础后，应该做什么项目来实践？请给出详细建议。"""

    models = [
        "/Volumes/toshiba/models/qwen3.5-0.8b-opus-distilled",
        "/Volumes/toshiba/models/qwen3.5-2b-opus-distilled",
        "/Volumes/toshiba/models/qwen3.5-35b-mlx",
    ]

    all_results = {}

    for model_path in models:
        result = test_am_on_model_e2e(
            model_path,
            test_prompt,
            max_tokens=200,
            compression_ratio=2.0
        )

        all_results[Path(model_path).name] = result

        # 清理
        gc.collect()
        mx.clear_cache()
        time.sleep(3)

    # 生成报告
    report_path = Path(__file__).parent.parent / ".solar" / "qwen35-am-e2e-report.md"

    with open(report_path, "w") as f:
        f.write("# Qwen3.5 系列 AM 压缩端到端测试报告\n\n")
        f.write(f"**日期**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## 测试说明\n\n")
        f.write("**重要发现**：之前的压缩质量测试（cosine similarity）显示 AM 在 Qwen3.5 上质量为 1.0，\n")
        f.write("但这只是数学近似质量，不代表能用于真实推理。\n\n")
        f.write("本测试尝试在真实推理中使用 AM 压缩，检查是否产生乱码。\n\n")

        f.write("## 测试结果\n\n")

        for model_name, result in all_results.items():
            f.write(f"### {model_name}\n\n")

            if result.get('success'):
                f.write(f"- 架构: {'混合 (SSM + Attention)' if result.get('is_hybrid') else '纯 Transformer'}\n")
                f.write(f"- Baseline 生成: {'✅ 正常' if not result['baseline']['is_garbage'] else '❌ 乱码'}\n")
                f.write(f"- 生成 tokens: {result['baseline']['tokens']}\n")
                f.write(f"- 生成时间: {result['baseline']['time']:.2f}s\n\n")

                f.write("**Baseline 输出示例**:\n")
                f.write("```\n")
                f.write(result['baseline']['sample'][:300])
                if len(result['baseline']['sample']) > 300:
                    f.write("\n...")
                f.write("\n```\n\n")
            else:
                f.write(f"❌ 测试失败: {result.get('error')}\n\n")

        f.write("## 结论\n\n")
        f.write("⚠️  **当前限制**: mlx-lm 的 `generate()` 函数不支持传入压缩配置，\n")
        f.write("无法完成端到端的 AM 压缩推理测试。\n\n")
        f.write("**需要**:\n")
        f.write("1. 修改 mlx-lm 添加压缩配置参数，或\n")
        f.write("2. 使用低级 API 手动实现带 AM 压缩的推理循环\n\n")

    print(f"\n详细报告已保存到: {report_path}")

    print("\n" + "="*70)
    print("测试完成!")
    print("="*70)
    print("\n⚠️  注意：由于 mlx-lm API 限制，无法完成真正的端到端 AM 压缩测试")
    print("⚠️  之前的测试只能说明 AM 在数学上能很好地近似 attention 输出")
    print("⚠️  但不能证明在真实推理中不会产生乱码")


if __name__ == "__main__":
    main()
