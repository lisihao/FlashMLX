#!/usr/bin/env python3
"""
CompactedKVCache Demo

演示如何使用 CompactedKVCache 进行长文本生成。
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'mlx-lm-source')))

import mlx.core as mx
from mlx_lm.models.compacted_cache import CompactedKVCache


def demo_basic_usage():
    """演示基本使用"""
    print("=" * 60)
    print("Demo 1: 基本使用")
    print("=" * 60)

    # 创建压缩缓存
    cache = CompactedKVCache(
        max_size=1000,         # 最大 1000 tokens
        compression_ratio=5.0, # 压缩到 1/5
        recent_ratio=0.5,      # 50% recent + 50% random
    )

    print(f"\n配置:")
    print(f"  max_size: {cache.max_size}")
    print(f"  compression_ratio: {cache.compression_ratio}x")
    print(f"  recent_ratio: {cache.recent_ratio}")

    # 模拟生成过程
    B, n_heads, num_steps, head_dim = 1, 32, 100, 128

    print(f"\n模拟生成 1500 tokens (15 steps × 100 tokens/step)...")
    for i in range(15):
        keys = mx.random.normal((B, n_heads, num_steps, head_dim))
        values = mx.random.normal((B, n_heads, num_steps, head_dim))

        cached_keys, cached_values = cache.update_and_fetch(keys, values)

        current_size = cached_keys.shape[2]
        print(f"  Step {i+1:2d}: Added 100 tokens → Cache size: {current_size}")

    # 输出统计
    stats = cache.get_stats()
    print(f"\n最终统计:")
    print(f"  压缩次数: {stats['num_compressions']}")
    print(f"  当前大小: {stats['current_size']} tokens")
    print(f"  平均压缩比: {stats['avg_compression_ratio']:.2f}x")
    print(f"  内存节省: {(1 - 1/stats['avg_compression_ratio']) * 100:.1f}%")


def demo_compression_comparison():
    """演示不同压缩比的效果"""
    print("\n" + "=" * 60)
    print("Demo 2: 不同压缩比对比")
    print("=" * 60)

    ratios = [3.0, 5.0, 10.0]
    B, n_heads, num_steps, head_dim = 1, 32, 100, 128

    print(f"\n添加 1500 tokens，观察不同压缩比的效果:\n")

    for ratio in ratios:
        cache = CompactedKVCache(max_size=1000, compression_ratio=ratio)

        for i in range(15):
            keys = mx.random.normal((B, n_heads, num_steps, head_dim))
            values = mx.random.normal((B, n_heads, num_steps, head_dim))
            cache.update_and_fetch(keys, values)

        stats = cache.get_stats()
        memory_saved = (1 - stats['current_size'] / 1500) * 100

        print(f"  {ratio}x 压缩:")
        print(f"    最终大小: {stats['current_size']} tokens")
        print(f"    内存节省: {memory_saved:.1f}%")
        print(f"    压缩次数: {stats['num_compressions']}")
        print()


def demo_disable_compression():
    """演示禁用压缩"""
    print("=" * 60)
    print("Demo 3: 禁用 vs 启用压缩")
    print("=" * 60)

    B, n_heads, num_steps, head_dim = 1, 32, 100, 128

    # 禁用压缩
    cache_no_compress = CompactedKVCache(
        max_size=1000,
        enable_compression=False
    )

    # 启用压缩
    cache_compress = CompactedKVCache(
        max_size=1000,
        compression_ratio=5.0,
        enable_compression=True
    )

    print(f"\n添加 1500 tokens...\n")

    for i in range(15):
        keys = mx.random.normal((B, n_heads, num_steps, head_dim))
        values = mx.random.normal((B, n_heads, num_steps, head_dim))

        cache_no_compress.update_and_fetch(keys, values)
        cache_compress.update_and_fetch(keys, values)

    stats_no = cache_no_compress.get_stats()
    stats_yes = cache_compress.get_stats()

    print(f"禁用压缩:")
    print(f"  最终大小: {stats_no['current_size']} tokens")
    print(f"  压缩次数: {stats_no['num_compressions']}")

    print(f"\n启用压缩:")
    print(f"  最终大小: {stats_yes['current_size']} tokens")
    print(f"  压缩次数: {stats_yes['num_compressions']}")

    memory_saved = (1 - stats_yes['current_size'] / stats_no['current_size']) * 100
    print(f"\n内存节省: {memory_saved:.1f}%")


def main():
    """运行所有 demo"""
    print("\n" + "=" * 60)
    print(" CompactedKVCache Demo")
    print("=" * 60)

    demo_basic_usage()
    demo_compression_comparison()
    demo_disable_compression()

    print("\n" + "=" * 60)
    print("Demo 完成！")
    print("=" * 60)
    print("\n使用建议:")
    print("  - 中等对话 (2K-8K): compression_ratio=5.0, recent_ratio=0.5")
    print("  - 长对话 (8K-32K): compression_ratio=10.0, recent_ratio=0.6")
    print("  - 超长文档 (> 32K): compression_ratio=15.0, recent_ratio=0.7")
    print("\n详细文档: docs/COMPACTED_CACHE_USAGE.md")
    print()


if __name__ == "__main__":
    main()
