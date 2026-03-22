#!/usr/bin/env python3
"""
测试简化 SSM 缓存

⚠️  DEPRECATED: 2026-03-22 ⚠️
This test is DISABLED. SSM cache has been deprecated.
See: SSM_CACHE_DEPRECATION.md

验证：
1. 开销降低（16x → 11x）
2. GPU bug 修复
3. 跨请求复用功能
"""

import sys
print("⚠️  This test is deprecated and disabled.")
print("SSM cache has been sealed. See SSM_CACHE_DEPRECATION.md")
sys.exit(0)

# Original test code preserved below (not executed)
# ============================================================================

import time
import gc
import mlx.core as mx
import numpy as np
from mlx_lm import load, generate
from flashmlx.cache import (
    SimplifiedSSMCacheManager,
    PerLayerSSMCache,
    create_layer_types_from_model,
    LayerType
)


def test_overhead():
    """测试缓存开销。"""
    print("="*70)
    print("测试 1: 缓存开销")
    print("="*70)

    # 创建管理器
    manager = SimplifiedSSMCacheManager()

    # 创建测试数据
    test_state = mx.array(np.random.randn(256).astype(np.float32))

    # 测试
    iterations = 10000
    print(f"\n运行 {iterations} 次 store/retrieve...")

    start = time.time()
    for i in range(iterations):
        manager.store(i % 30, test_state)
        _ = manager.retrieve(i % 30)
    end = time.time()

    elapsed_us = (end - start) * 1000000
    latency_per_op = elapsed_us / (iterations * 2)

    print(f"\n结果:")
    print(f"  总时间:     {elapsed_us:.1f} μs")
    print(f"  平均延迟:   {latency_per_op:.3f} μs/op")

    # 对比
    print(f"\n对比:")
    print(f"  Simplified (NEW): {latency_per_op:.3f} μs/op (11x)")
    print(f"  Hot/Warm/Cold:    0.177 μs/op (16x)")
    print(f"  Direct dict:      0.011 μs/op (1x)")

    assert latency_per_op < 0.15, f"开销过高: {latency_per_op:.3f} μs/op"
    print(f"\n✅ 开销降低成功！")


def test_cross_request_reuse():
    """测试跨请求复用。"""
    print(f"\n{'='*70}")
    print("测试 2: 跨请求复用")
    print("="*70)

    # 创建管理器
    manager = SimplifiedSSMCacheManager(max_size_bytes=10 * 1024 * 1024)  # 10MB

    # 模拟 30 层 SSM
    num_layers = 30

    # 创建缓存
    caches = []
    for layer_idx in range(num_layers):
        cache = PerLayerSSMCache(
            manager=manager,
            layer_idx=layer_idx,
            size=2
        )
        cache.enable_managed_cache()
        caches.append(cache)

    # 模拟首次请求（存储状态）
    print(f"\n首次请求: 存储 {num_layers} 层 SSM 状态...")
    for layer_idx in range(num_layers):
        state = mx.array(np.random.randn(256).astype(np.float32))
        caches[layer_idx][0] = state

    stats = manager.get_statistics()
    print(f"  缓存条目: {stats['entry_count']}")
    print(f"  缓存大小: {stats['size_bytes'] / 1024:.1f} KB")

    # 模拟后续请求（复用状态）
    print(f"\n后续请求: 复用状态...")
    for _ in range(3):
        for layer_idx in range(num_layers):
            _ = caches[layer_idx][0]

    stats = manager.get_statistics()
    print(f"\n统计:")
    print(f"  总访问:   {stats['hits'] + stats['misses']}")
    print(f"  命中:     {stats['hits']}")
    print(f"  缺失:     {stats['misses']}")
    print(f"  命中率:   {stats['hit_rate']:.1%}")

    assert stats['hit_rate'] > 0.7, f"命中率过低: {stats['hit_rate']:.1%}"
    print(f"\n✅ 跨请求复用成功！")


def test_no_gpu_hang():
    """测试是否解决 GPU hang 问题。"""
    print(f"\n{'='*70}")
    print("测试 3: GPU Stability (简化测试)")
    print("="*70)

    print("\n注意: 完整端到端测试需要加载模型，这里只测试缓存逻辑")

    # 创建管理器
    manager = SimplifiedSSMCacheManager()

    # 创建大量缓存（模拟实际使用）
    num_layers = 30
    caches = []

    for layer_idx in range(num_layers):
        cache = PerLayerSSMCache(
            manager=manager,
            layer_idx=layer_idx,
            size=2
        )
        cache.enable_managed_cache()
        caches.append(cache)

    # 大量读写操作
    print(f"\n执行大量缓存操作（{num_layers} 层 × 1000 次）...")
    for iteration in range(1000):
        for layer_idx in range(num_layers):
            # 写入
            state = mx.array(np.random.randn(256).astype(np.float32))
            caches[layer_idx][0] = state

            # 读取
            _ = caches[layer_idx][0]

        if (iteration + 1) % 200 == 0:
            print(f"  进度: {iteration + 1}/1000")

    print(f"\n✅ 无 GPU hang，稳定性良好！")


def main():
    print("="*70)
    print("简化 SSM 缓存测试套件")
    print("="*70)

    # 测试 1: 开销
    test_overhead()

    # 测试 2: 跨请求复用
    test_cross_request_reuse()

    # 测试 3: GPU 稳定性
    test_no_gpu_hang()

    # 总结
    print(f"\n{'='*70}")
    print("测试总结")
    print(f"{'='*70}")

    print(f"\n✅ 所有测试通过！")
    print(f"\n改进:")
    print(f"  1. 开销降低: 16x → 11x (改进 1.5倍)")
    print(f"  2. 去除 Hot/Warm/Cold 三层架构")
    print(f"  3. 内存优先，不外溢到磁盘")
    print(f"  4. GPU 稳定性提升（无 hang/page fault）")
    print(f"  5. 支持跨请求复用（命中率 > 70%）")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
