#!/usr/bin/env python3
"""
Heterogeneous Memory Compaction 概念验证

测试：将 Qwen3.5 的记忆系统拆分为两类
- Attention-Memory: 使用 AM 压缩
- State-Memory: 不压缩（待设计专用方法）
"""

import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "mlx-lm-source"))

import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.models.cache import KVCache
from mlx_lm.models.compacted_cache import CompactedKVCache


def classify_memory_type(layer, layer_idx):
    """
    分类记忆类型

    Args:
        layer: 模型层对象
        layer_idx: 层索引

    Returns:
        str: "attention_memory" | "state_memory"
    """
    # SSM/Mamba/Linear Attention → State-Memory
    if hasattr(layer, 'linear_attn') or hasattr(layer, 'mamba_block'):
        return "state_memory"

    # Standard Attention → Attention-Memory
    elif hasattr(layer, 'self_attn'):
        return "attention_memory"

    else:
        return "unknown"


class HeterogeneousCacheManager:
    """
    异构缓存管理器

    根据记忆类型选择不同的压缩策略：
    - Attention-Memory: AM 压缩（CompactedKVCache）
    - State-Memory: 标准缓存（KVCache，不压缩）
    """

    def __init__(self, model, max_size=4096, compression_ratio=5.0):
        """
        Args:
            model: 模型对象
            max_size: Attention-Memory 的压缩阈值
            compression_ratio: Attention-Memory 的压缩比例
        """
        self.model = model
        self.max_size = max_size
        self.compression_ratio = compression_ratio

        # 统计（必须在 _create_caches 之前初始化）
        self.stats = {
            'attention_memory_count': 0,
            'state_memory_count': 0,
            'unknown_count': 0
        }

        # 创建异构缓存
        self.caches = self._create_caches()

    def _create_caches(self):
        """创建异构缓存列表"""
        from mlx_lm.models.cache import ArraysCache

        num_layers = len(self.model.layers)
        cache_container = ArraysCache(size=num_layers)

        print("\n" + "="*60)
        print("Heterogeneous Cache Manager: Creating Caches")
        print("="*60)

        for i, layer in enumerate(self.model.layers):
            memory_type = classify_memory_type(layer, i)

            if memory_type == "attention_memory":
                # Attention-Memory: 使用 AM 压缩
                cache = CompactedKVCache(
                    max_size=self.max_size,
                    compression_ratio=self.compression_ratio
                )
                self.stats['attention_memory_count'] += 1
                print(f"Layer {i:2d}: Attention-Memory (AM compression)")

            elif memory_type == "state_memory":
                # State-Memory: 标准缓存（暂不压缩）
                cache = None  # 使用默认缓存
                self.stats['state_memory_count'] += 1
                print(f"Layer {i:2d}: State-Memory (no compression)")

            else:
                cache = None
                self.stats['unknown_count'] += 1
                print(f"Layer {i:2d}: Unknown (no compression)")

            cache_container[i] = cache

        print("="*60)
        print(f"Attention-Memory layers: {self.stats['attention_memory_count']}")
        print(f"State-Memory layers: {self.stats['state_memory_count']}")
        print(f"Unknown layers: {self.stats['unknown_count']}")
        print("="*60 + "\n")

        return cache_container

    def get_stats(self):
        """获取统计信息"""
        return self.stats


def test_heterogeneous_compaction(
    model_name="/Volumes/toshiba/models/qwen3.5-35b-mlx",
    prompt="介绍机器学习的基本概念和应用场景",
    max_tokens=300
):
    """
    测试异构记忆压缩

    Args:
        model_name: 模型名称
        prompt: 测试 prompt
        max_tokens: 最大生成 token 数

    Returns:
        dict: 测试结果
    """
    print("\n" + "#"*60)
    print("# Heterogeneous Memory Compaction 概念验证")
    print("#"*60)
    print(f"\n模型: {model_name}")
    print(f"Prompt: '{prompt}'")
    print(f"Max tokens: {max_tokens}\n")

    # 加载模型
    print("Loading model...")
    model, tokenizer = load(model_name)
    print(f"Model loaded: {len(model.layers)} layers\n")

    # 创建异构缓存管理器
    cache_manager = HeterogeneousCacheManager(
        model,
        max_size=4096,
        compression_ratio=5.0
    )

    # 运行生成测试
    print("Generating with Heterogeneous Cache...")
    print("="*60)

    start_time = time.time()

    try:
        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,
            prompt_cache=cache_manager.caches
        )

        elapsed = time.time() - start_time
        num_tokens = len(tokenizer.encode(response))

        print("\n" + "="*60)
        print("Generated text:")
        print("="*60)
        print(response[:500])
        if len(response) > 500:
            print(f"... (truncated, total {len(response)} chars)")
        print("="*60)

        # 结果
        result = {
            'success': True,
            'num_tokens': num_tokens,
            'time_elapsed': elapsed,
            'tokens_per_sec': num_tokens / elapsed if elapsed > 0 else 0,
            'output': response,
            'cache_stats': cache_manager.get_stats()
        }

        print(f"\n✅ Generation Success!")
        print(f"Tokens: {num_tokens}")
        print(f"Time: {elapsed:.2f}s")
        print(f"Speed: {result['tokens_per_sec']:.2f} tokens/s")
        print(f"\nCache Stats:")
        print(f"  Attention-Memory layers (compressed): {result['cache_stats']['attention_memory_count']}")
        print(f"  State-Memory layers (uncompressed): {result['cache_stats']['state_memory_count']}")

        # 计算有效压缩比
        total_layers = result['cache_stats']['attention_memory_count'] + result['cache_stats']['state_memory_count']
        compressed_layers = result['cache_stats']['attention_memory_count']
        effective_ratio = compressed_layers / total_layers if total_layers > 0 else 0

        print(f"  Effective compression coverage: {effective_ratio:.1%} ({compressed_layers}/{total_layers} layers)")

        return result

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ Generation failed: {str(e)}")

        result = {
            'success': False,
            'error': str(e),
            'time_elapsed': elapsed,
            'cache_stats': cache_manager.get_stats()
        }

        return result


def compare_baseline_vs_hetero(model_name="/Volumes/toshiba/models/qwen3.5-35b-mlx"):
    """
    对比 Baseline (无压缩) vs Heterogeneous Compaction

    Args:
        model_name: 模型名称

    Returns:
        dict: 对比结果
    """
    print("\n" + "#"*60)
    print("# Baseline vs Heterogeneous Compaction 对比")
    print("#"*60 + "\n")

    prompt = "介绍机器学习的基本概念"
    max_tokens = 200

    # 加载模型
    print("Loading model...")
    model, tokenizer = load(model_name)

    results = {}

    # Test 1: Baseline (无压缩)
    print("\n" + "="*60)
    print("Test 1: Baseline (No Compression)")
    print("="*60)

    start = time.time()
    try:
        response_baseline = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False
        )
        elapsed_baseline = time.time() - start
        tokens_baseline = len(tokenizer.encode(response_baseline))

        results['baseline'] = {
            'success': True,
            'tokens': tokens_baseline,
            'time': elapsed_baseline,
            'tps': tokens_baseline / elapsed_baseline
        }

        print(f"✅ Baseline: {tokens_baseline} tokens, {elapsed_baseline:.2f}s, {results['baseline']['tps']:.2f} tok/s")

    except Exception as e:
        results['baseline'] = {'success': False, 'error': str(e)}
        print(f"❌ Baseline failed: {e}")

    # Test 2: Heterogeneous Compaction
    print("\n" + "="*60)
    print("Test 2: Heterogeneous Compaction")
    print("="*60)

    cache_manager = HeterogeneousCacheManager(model, max_size=4096, compression_ratio=5.0)

    start = time.time()
    try:
        response_hetero = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,
            prompt_cache=cache_manager.caches
        )
        elapsed_hetero = time.time() - start
        tokens_hetero = len(tokenizer.encode(response_hetero))

        results['hetero'] = {
            'success': True,
            'tokens': tokens_hetero,
            'time': elapsed_hetero,
            'tps': tokens_hetero / elapsed_hetero,
            'cache_stats': cache_manager.get_stats()
        }

        print(f"✅ Hetero: {tokens_hetero} tokens, {elapsed_hetero:.2f}s, {results['hetero']['tps']:.2f} tok/s")

    except Exception as e:
        results['hetero'] = {'success': False, 'error': str(e)}
        print(f"❌ Hetero failed: {e}")

    # 对比
    print("\n" + "="*60)
    print("Comparison Results")
    print("="*60)

    if results['baseline']['success'] and results['hetero']['success']:
        speedup = results['baseline']['time'] / results['hetero']['time']
        token_diff = results['hetero']['tokens'] - results['baseline']['tokens']

        print(f"Baseline: {results['baseline']['time']:.2f}s, {results['baseline']['tps']:.2f} tok/s")
        print(f"Hetero:   {results['hetero']['time']:.2f}s, {results['hetero']['tps']:.2f} tok/s")
        print(f"Speedup:  {speedup:.2f}x")
        print(f"Token diff: {token_diff:+d} ({results['hetero']['tokens']} vs {results['baseline']['tokens']})")
        print(f"\nCache coverage: {results['hetero']['cache_stats']['attention_memory_count']} / {results['hetero']['cache_stats']['attention_memory_count'] + results['hetero']['cache_stats']['state_memory_count']} layers compressed")

    return results


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Heterogeneous Memory Compaction 测试")
    parser.add_argument(
        '--mode',
        choices=['test', 'compare'],
        default='test',
        help='测试模式: test (单次测试) 或 compare (对比测试)'
    )
    parser.add_argument(
        '--model',
        default='/Volumes/toshiba/models/qwen3.5-35b-mlx',
        help='模型名称'
    )

    args = parser.parse_args()

    if args.mode == 'test':
        # 单次测试
        result = test_heterogeneous_compaction(model_name=args.model)

        # 保存结果
        report_path = Path(__file__).parent.parent / ".solar" / "hetero-cache-test-report.md"
        with open(report_path, "w") as f:
            f.write("# Heterogeneous Memory Compaction 测试报告\n\n")
            f.write(f"**日期**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**模型**: {args.model}\n\n")

            f.write("## 测试结果\n\n")
            if result['success']:
                f.write(f"- ✅ 成功\n")
                f.write(f"- Tokens: {result['num_tokens']}\n")
                f.write(f"- Time: {result['time_elapsed']:.2f}s\n")
                f.write(f"- Speed: {result['tokens_per_sec']:.2f} tok/s\n\n")

                f.write("## Cache 统计\n\n")
                f.write(f"- Attention-Memory layers (compressed): {result['cache_stats']['attention_memory_count']}\n")
                f.write(f"- State-Memory layers (uncompressed): {result['cache_stats']['state_memory_count']}\n")
                f.write(f"- Unknown layers: {result['cache_stats']['unknown_count']}\n\n")

                total = result['cache_stats']['attention_memory_count'] + result['cache_stats']['state_memory_count']
                coverage = result['cache_stats']['attention_memory_count'] / total if total > 0 else 0
                f.write(f"- Compression coverage: {coverage:.1%}\n\n")

                f.write("## 生成内容\n\n")
                f.write("```\n")
                f.write(result['output'][:500])
                if len(result['output']) > 500:
                    f.write(f"\n... (截断, 总共 {len(result['output'])} 字符)\n")
                f.write("```\n")
            else:
                f.write(f"- ❌ 失败\n")
                f.write(f"- Error: {result['error']}\n")

        print(f"\n报告已保存到: {report_path}")

    elif args.mode == 'compare':
        # 对比测试
        results = compare_baseline_vs_hetero(model_name=args.model)

        # 保存对比报告
        report_path = Path(__file__).parent.parent / ".solar" / "hetero-cache-comparison-report.md"
        with open(report_path, "w") as f:
            f.write("# Baseline vs Heterogeneous Compaction 对比报告\n\n")
            f.write(f"**日期**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**模型**: {args.model}\n\n")

            f.write("## 对比结果\n\n")
            f.write("| 配置 | 成功 | Tokens | 耗时 | 速度 |\n")
            f.write("|------|------|--------|------|------|\n")

            if results['baseline']['success']:
                f.write(f"| Baseline | ✅ | {results['baseline']['tokens']} | {results['baseline']['time']:.2f}s | {results['baseline']['tps']:.2f} tok/s |\n")
            else:
                f.write(f"| Baseline | ❌ | - | - | - |\n")

            if results['hetero']['success']:
                f.write(f"| Hetero | ✅ | {results['hetero']['tokens']} | {results['hetero']['time']:.2f}s | {results['hetero']['tps']:.2f} tok/s |\n")
            else:
                f.write(f"| Hetero | ❌ | - | - | - |\n")

            if results['baseline']['success'] and results['hetero']['success']:
                speedup = results['baseline']['time'] / results['hetero']['time']
                f.write(f"\n**Speedup**: {speedup:.2f}x\n")

        print(f"\n对比报告已保存到: {report_path}")


if __name__ == "__main__":
    main()
