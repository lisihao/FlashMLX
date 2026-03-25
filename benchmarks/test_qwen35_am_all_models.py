#!/usr/bin/env python3
"""
在所有 Qwen3.5 模型上测试 AM 压缩是否产生乱码
"""

import sys
import time
import gc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "mlx-lm-source"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import ArraysCache
from mlx_lm.models.compacted_cache import CompactedKVCache
from mlx_lm.models.cache import KVCache


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


def generate_with_cache(model, tokenizer, prompt, cache, max_tokens=100, temp=0.0):
    """
    使用指定的 cache 生成文本

    Args:
        model: 模型
        tokenizer: tokenizer
        prompt: 输入 prompt
        cache: KVCache 或 CompactedKVCache
        max_tokens: 最大生成 token 数
        temp: 温度参数

    Returns:
        str: 生成的文本
    """
    # Tokenize
    tokens = mx.array([tokenizer.encode(prompt)])

    # 生成
    generated_tokens = []

    for _ in range(max_tokens):
        # 前向传播
        logits = model(tokens, cache=cache)

        # 获取最后一个 token 的 logits
        logits = logits[:, -1, :]

        # 采样
        if temp == 0:
            # Greedy
            next_token = mx.argmax(logits, axis=-1)
        else:
            # Temperature sampling
            logits = logits / temp
            probs = mx.softmax(logits, axis=-1)
            next_token = mx.random.categorical(mx.log(probs))

        next_token = next_token.item()

        # 检查是否结束
        if next_token == tokenizer.eos_token_id:
            break

        generated_tokens.append(next_token)

        # 准备下一轮
        tokens = mx.array([[next_token]])

    # Decode
    return tokenizer.decode(generated_tokens)


def test_am_on_model(model_path, test_prompt, compression_ratio=2.0, max_tokens=100):
    """
    测试 AM 压缩在指定模型上是否产生乱码
    """
    print(f"\n{'='*70}")
    print(f"测试: {Path(model_path).name}")
    print(f"  Compression ratio: {compression_ratio}")
    print(f"{'='*70}")

    try:
        # 加载模型
        print("\nLoading model...")
        model, tokenizer = load(model_path)

        # 检查架构
        if hasattr(model, 'layers'):
            attention_layers = []
            ssm_layers = []

            for i, layer in enumerate(model.layers):
                if 'self_attn' in layer.state:
                    attention_layers.append(i)
                elif 'linear_attn' in layer.state:
                    ssm_layers.append(i)

            is_hybrid = len(attention_layers) > 0 and len(ssm_layers) > 0
            print(f"Architecture: {'Hybrid' if is_hybrid else 'Pure'}")
            print(f"  Attention layers: {len(attention_layers)}")
            print(f"  SSM layers: {len(ssm_layers)}")

        results = {}

        # Test 1: Baseline (无压缩)
        print(f"\n{'='*70}")
        print("Test 1: Baseline (无压缩)")
        print(f"{'='*70}")

        baseline_cache = ArraysCache(size=len(model.layers))
        for i in range(len(model.layers)):
            baseline_cache[i] = KVCache()

        start = time.time()
        baseline_output = generate_with_cache(
            model, tokenizer, test_prompt, baseline_cache, max_tokens=max_tokens
        )
        baseline_time = time.time() - start

        baseline_garbage = check_garbage(baseline_output)

        print(f"Generated {len(tokenizer.encode(baseline_output))} tokens in {baseline_time:.2f}s")
        print(f"Garbage: {'❌ Yes' if baseline_garbage else '✅ No'}")
        print(f"\nOutput:")
        print(f"{'-'*70}")
        print(baseline_output[:200])
        if len(baseline_output) > 200:
            print("...")
        print(f"{'-'*70}")

        results['baseline'] = {
            'output': baseline_output,
            'is_garbage': baseline_garbage,
            'time': baseline_time
        }

        # 清理
        del baseline_cache
        gc.collect()
        mx.clear_cache()

        # Test 2: AM 压缩（只在 Attention 层）
        print(f"\n{'='*70}")
        print(f"Test 2: AM 压缩 (ratio={compression_ratio})")
        print(f"{'='*70}")

        am_cache = ArraysCache(size=len(model.layers))
        for i in range(len(model.layers)):
            if i in attention_layers:
                # Attention 层使用 AM 压缩
                am_cache[i] = CompactedKVCache(
                    max_size=512,
                    compression_ratio=compression_ratio
                )
                print(f"Layer {i}: CompactedKVCache (AM compression)")
            else:
                # SSM 层使用标准 cache
                am_cache[i] = KVCache()

        start = time.time()
        am_output = generate_with_cache(
            model, tokenizer, test_prompt, am_cache, max_tokens=max_tokens
        )
        am_time = time.time() - start

        am_garbage = check_garbage(am_output)

        print(f"Generated {len(tokenizer.encode(am_output))} tokens in {am_time:.2f}s")
        print(f"Garbage: {'❌ Yes' if am_garbage else '✅ No'}")
        print(f"\nOutput:")
        print(f"{'-'*70}")
        print(am_output[:200])
        if len(am_output) > 200:
            print("...")
        print(f"{'-'*70}")

        results['am_compressed'] = {
            'output': am_output,
            'is_garbage': am_garbage,
            'time': am_time
        }

        # 清理
        del am_cache, model, tokenizer
        gc.collect()
        mx.clear_cache()

        return {
            'success': True,
            'model': Path(model_path).name,
            'is_hybrid': is_hybrid if 'is_hybrid' in locals() else False,
            'attention_layers': len(attention_layers) if 'attention_layers' in locals() else 0,
            'ssm_layers': len(ssm_layers) if 'ssm_layers' in locals() else 0,
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
            'model': Path(model_path).name,
            'error': str(e)
        }


def main():
    print("="*70)
    print("Qwen3.5 系列 AM 压缩端到端测试")
    print("="*70)

    test_prompt = "学完PyTorch基础后，应该做什么项目来实践？请给出详细建议。"

    models = [
        "/Volumes/toshiba/models/qwen3.5-0.8b-opus-distilled",
        "/Volumes/toshiba/models/qwen3.5-2b-opus-distilled",
        "/Volumes/toshiba/models/qwen3.5-35b-mlx",
    ]

    all_results = {}

    for model_path in models:
        result = test_am_on_model(
            model_path,
            test_prompt,
            compression_ratio=2.0,
            max_tokens=100
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

        f.write("## 测试目的\n\n")
        f.write("验证 AM 压缩在 Qwen3.5 混合架构上是否真的能用于推理\n\n")
        f.write("**关键区别**：\n")
        f.write("- 之前的测试只计算压缩质量（cosine similarity）\n")
        f.write("- 本测试在**真实推理**中使用 AM 压缩，检查是否产生乱码\n\n")

        f.write("## 测试方法\n\n")
        f.write("- 在 Attention 层使用 CompactedKVCache (AM 压缩)\n")
        f.write("- 在 SSM 层使用标准 KVCache (无压缩)\n")
        f.write("- Compression ratio: 2.0\n")
        f.write("- Max tokens: 100\n\n")

        f.write("## 测试结果\n\n")

        for model_name, result in all_results.items():
            f.write(f"### {model_name}\n\n")

            if result.get('success'):
                f.write(f"**架构**: {'混合 (SSM + Attention)' if result['is_hybrid'] else '纯 Transformer'}\n")
                f.write(f"- Attention 层: {result['attention_layers']}\n")
                f.write(f"- SSM 层: {result['ssm_layers']}\n\n")

                # Baseline
                baseline = result['results']['baseline']
                f.write(f"**Baseline (无压缩)**:\n")
                f.write(f"- 乱码: {'❌ 是' if baseline['is_garbage'] else '✅ 否'}\n")
                f.write(f"- 时间: {baseline['time']:.2f}s\n\n")

                f.write("输出示例:\n")
                f.write("```\n")
                f.write(baseline['output'][:300])
                if len(baseline['output']) > 300:
                    f.write("\n...")
                f.write("\n```\n\n")

                # AM compressed
                am = result['results']['am_compressed']
                f.write(f"**AM 压缩 (ratio=2.0)**:\n")
                f.write(f"- 乱码: {'❌ 是' if am['is_garbage'] else '✅ 否'}\n")
                f.write(f"- 时间: {am['time']:.2f}s\n\n")

                f.write("输出示例:\n")
                f.write("```\n")
                f.write(am['output'][:300])
                if len(am['output']) > 300:
                    f.write("\n...")
                f.write("\n```\n\n")

                # 对比
                if baseline['is_garbage'] == False and am['is_garbage'] == True:
                    f.write("🔴 **结论**: AM 压缩导致乱码！\n\n")
                elif baseline['is_garbage'] == False and am['is_garbage'] == False:
                    f.write("✅ **结论**: AM 压缩正常工作！\n\n")
                else:
                    f.write("⚠️  **结论**: Baseline 本身就有问题\n\n")

            else:
                f.write(f"❌ 测试失败: {result.get('error')}\n\n")

        f.write("## 总结\n\n")

        success_models = [name for name, r in all_results.items() if r.get('success')]
        failed_models = [name for name, r in all_results.items() if not r.get('success')]

        if success_models:
            f.write("### 成功测试的模型\n\n")
            for model in success_models:
                result = all_results[model]
                baseline_ok = not result['results']['baseline']['is_garbage']
                am_ok = not result['results']['am_compressed']['is_garbage']

                status = "✅ AM 正常" if (baseline_ok and am_ok) else "🔴 AM 产生乱码" if (baseline_ok and not am_ok) else "⚠️  Baseline 异常"
                f.write(f"- {model}: {status}\n")

            f.write("\n")

        if failed_models:
            f.write("### 测试失败的模型\n\n")
            for model in failed_models:
                f.write(f"- {model}\n")

            f.write("\n")

    print(f"\n详细报告已保存到: {report_path}")

    print("\n" + "="*70)
    print("测试完成!")
    print("="*70)


if __name__ == "__main__":
    main()
