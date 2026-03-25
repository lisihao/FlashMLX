#!/usr/bin/env python3
"""
验证 H2O 和 StreamingLLM 是否适合 Qwen3.5 混合架构
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "mlx-lm-source"))

import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.models.cache import ArraysCache
from mlx_lm.models.compacted_cache import CompactedKVCache


def classify_memory_type(layer, layer_idx):
    """分类记忆类型"""
    if hasattr(layer, 'linear_attn') or hasattr(layer, 'mamba_block'):
        return "state_memory"
    elif hasattr(layer, 'self_attn'):
        return "attention_memory"
    else:
        return "unknown"


class H2OHeterogeneousCacheManager:
    """H2O 异构缓存管理器"""

    def __init__(self, model, max_size=8192, compression_ratio=2.0):
        self.model = model
        self.max_size = max_size
        self.compression_ratio = compression_ratio
        self.caches = self._create_caches()

    def _create_caches(self):
        """创建 H2O 压缩缓存"""
        num_layers = len(self.model.layers)
        cache_container = ArraysCache(size=num_layers)

        for i, layer in enumerate(self.model.layers):
            memory_type = classify_memory_type(layer, i)

            if memory_type == "attention_memory":
                # 使用 H2O 压缩
                cache = CompactedKVCache(
                    max_size=self.max_size,
                    compression_ratio=self.compression_ratio,
                    compression_method="h2o"  # H2O 方法
                )
            else:
                cache = None

            cache_container[i] = cache

        return cache_container


class StreamingLLMHeterogeneousCacheManager:
    """StreamingLLM 异构缓存管理器"""

    def __init__(self, model, max_size=8192, compression_ratio=2.0):
        self.model = model
        self.max_size = max_size
        self.compression_ratio = compression_ratio
        self.caches = self._create_caches()

    def _create_caches(self):
        """创建 StreamingLLM 压缩缓存"""
        num_layers = len(self.model.layers)
        cache_container = ArraysCache(size=num_layers)

        for i, layer in enumerate(self.model.layers):
            memory_type = classify_memory_type(layer, i)

            if memory_type == "attention_memory":
                # 使用 StreamingLLM 压缩
                cache = CompactedKVCache(
                    max_size=self.max_size,
                    compression_ratio=self.compression_ratio,
                    compression_method="streaming_llm"  # StreamingLLM 方法
                )
            else:
                cache = None

            cache_container[i] = cache

        return cache_container


def check_garbage(text):
    """检查是否为乱码"""
    words = text.split()
    if len(words) < 10:
        return False

    first_50 = words[:50]
    unique_ratio = len(set(first_50)) / len(first_50)

    if unique_ratio < 0.3:
        return True

    for i in range(len(words) - 5):
        if words[i] == words[i+1] == words[i+2] == words[i+3] == words[i+4]:
            return True

    return False


def test_method(model, tokenizer, prompt, max_tokens, method_name, cache_manager_class):
    """测试特定压缩方法"""
    print(f"\n{'='*70}")
    print(f"测试: {method_name}")
    print(f"{'='*70}")

    start = time.time()

    try:
        if cache_manager_class is None:
            # Baseline
            response = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                verbose=False
            )
        else:
            # 压缩方法
            cache_manager = cache_manager_class(
                model,
                max_size=8192,
                compression_ratio=2.0
            )

            response = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                verbose=False,
                prompt_cache=cache_manager.caches
            )

        elapsed = time.time() - start
        num_tokens = len(tokenizer.encode(response))

        is_garbage = check_garbage(response)

        print(f"\n✅ 成功")
        print(f"Tokens: {num_tokens}")
        print(f"Time: {elapsed:.2f}s")
        print(f"Speed: {num_tokens / elapsed:.2f} tok/s")
        print(f"乱码检测: {'❌ 是' if is_garbage else '✅ 否'}")
        print(f"\n生成内容 (前 300 字符):")
        print(f"{'-'*70}")
        print(response[:300])
        if len(response) > 300:
            print(f"... (截断)")
        print(f"{'-'*70}")

        return {
            'success': True,
            'method': method_name,
            'tokens': num_tokens,
            'time': elapsed,
            'tps': num_tokens / elapsed,
            'is_garbage': is_garbage,
            'output': response
        }

    except Exception as e:
        elapsed = time.time() - start
        print(f"\n❌ 失败: {str(e)}")
        import traceback
        traceback.print_exc()

        return {
            'success': False,
            'method': method_name,
            'error': str(e),
            'time': elapsed
        }


def main():
    model_name = "/Volumes/toshiba/models/qwen3.5-35b-mlx"
    prompt = "介绍机器学习的基本概念和应用场景"
    max_tokens = 200

    print("="*70)
    print("H2O 和 StreamingLLM 在 Qwen3.5 混合架构上的验证")
    print("="*70)
    print(f"\n模型: {model_name}")
    print(f"Prompt: '{prompt}'")
    print(f"Max tokens: {max_tokens}\n")

    # 加载模型
    print("Loading model...")
    model, tokenizer = load(model_name)
    print(f"Model loaded: {len(model.layers)} layers\n")

    # 测试配置
    tests = [
        ("Baseline (无压缩)", None),
        ("H2O (compression_ratio=2.0)", H2OHeterogeneousCacheManager),
        ("StreamingLLM (compression_ratio=2.0)", StreamingLLMHeterogeneousCacheManager)
    ]

    results = []

    for method_name, cache_manager_class in tests:
        result = test_method(
            model, tokenizer, prompt, max_tokens,
            method_name, cache_manager_class
        )
        results.append(result)

    # 生成对比报告
    print("\n" + "="*70)
    print("对比报告")
    print("="*70 + "\n")

    print(f"{'方法':<40} {'成功':<6} {'乱码':<6} {'Tokens':<8} {'速度':<12}")
    print("-"*80)

    for result in results:
        if result['success']:
            garbage_mark = "❌" if result.get('is_garbage', False) else "✅"
            print(f"{result['method']:<40} ✅     {garbage_mark:<6} {result['tokens']:<8} {result['tps']:<12.2f}")
        else:
            print(f"{result['method']:<40} ❌     -      -        -")

    # 保存报告
    report_path = Path(__file__).parent.parent / ".solar" / "h2o-streaming-qwen-test.md"
    with open(report_path, "w") as f:
        f.write("# H2O 和 StreamingLLM 在 Qwen3.5 混合架构上的验证\n\n")
        f.write(f"**日期**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**模型**: {model_name}\n")
        f.write(f"**目的**: 验证 H2O 和 StreamingLLM 是否能在混合架构上工作\n\n")

        f.write("## 对比结果\n\n")
        f.write("| 方法 | 成功 | 乱码 | Tokens | 耗时 | 速度 |\n")
        f.write("|------|------|------|--------|------|------|\n")

        for result in results:
            if result['success']:
                garbage_mark = "❌ 是" if result.get('is_garbage', False) else "✅ 否"
                f.write(f"| {result['method']} | ✅ | {garbage_mark} | {result['tokens']} | {result['time']:.2f}s | {result['tps']:.2f} tok/s |\n")
            else:
                f.write(f"| {result['method']} | ❌ | - | - | - | - |\n")

        f.write("\n## 生成内容对比\n\n")

        for result in results:
            if result['success']:
                f.write(f"### {result['method']}\n\n")
                if result.get('is_garbage', False):
                    f.write("**状态**: ❌ 乱码\n\n")
                else:
                    f.write("**状态**: ✅ 正常\n\n")
                f.write("```\n")
                f.write(result['output'][:500])
                if len(result['output']) > 500:
                    f.write(f"\n... (截断, 总共 {len(result['output'])} 字符)")
                f.write("\n```\n\n")

        f.write("## 结论\n\n")

        h2o_result = next((r for r in results if 'H2O' in r['method']), None)
        streaming_result = next((r for r in results if 'StreamingLLM' in r['method']), None)

        if h2o_result and h2o_result['success'] and not h2o_result.get('is_garbage', False):
            f.write("### ✅ H2O 可以在 Qwen3.5 混合架构上工作\n\n")
            f.write("- H2O 生成质量正常\n")
            f.write("- 可以作为混合架构的备选压缩方案\n\n")
        elif h2o_result:
            f.write("### ❌ H2O 在 Qwen3.5 混合架构上失败\n\n")
            if h2o_result.get('is_garbage', False):
                f.write("- H2O 也产生乱码\n")
            else:
                f.write(f"- H2O 执行失败: {h2o_result.get('error', 'Unknown')}\n")

        if streaming_result and streaming_result['success'] and not streaming_result.get('is_garbage', False):
            f.write("### ✅ StreamingLLM 可以在 Qwen3.5 混合架构上工作\n\n")
            f.write("- StreamingLLM 生成质量正常\n")
            f.write("- 可以作为混合架构的备选压缩方案\n\n")
        elif streaming_result:
            f.write("### ❌ StreamingLLM 在 Qwen3.5 混合架构上失败\n\n")
            if streaming_result.get('is_garbage', False):
                f.write("- StreamingLLM 也产生乱码\n")
            else:
                f.write(f"- StreamingLLM 执行失败: {streaming_result.get('error', 'Unknown')}\n")

    print(f"\n详细报告已保存到: {report_path}")


if __name__ == "__main__":
    main()
