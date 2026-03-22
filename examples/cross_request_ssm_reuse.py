#!/usr/bin/env python3
"""
跨请求 SSM 状态复用示例

⚠️  DEPRECATED: 2026-03-22 ⚠️
This example is DISABLED. SSM cache has been deprecated.
See: SSM_CACHE_DEPRECATION.md

展示如何在多轮对话中复用 system prompt 的 SSM 状态，避免重复计算。

性能收益：
- 首次计算：system prompt forward pass (125 ms @ 100 tokens)
- 后续复用：缓存读取 (0.011 μs × 15 layers = 0.0002 ms)
- 净收益：~125 ms / request

替代方案：
- Use ThunderLLAMA prefix caching for cross-request reuse
- See: ThunderLLAMA/thunderllama.conf
"""

import sys
print("⚠️  This example is deprecated and disabled.")
print("Use ThunderLLAMA prefix caching instead for the same functionality.")
sys.exit(0)

# Original example code preserved below (not executed)
# ============================================================================

import time
import mlx.core as mx
from mlx_lm import load
from flashmlx.cache import (
    SimplifiedSSMCacheManager,
    PerLayerSSMCache,
    create_layer_types_from_model,
    LayerType
)


def create_ssm_caches_with_simplified_manager(
    model,
    max_size_bytes: int = 100 * 1024 * 1024  # 100MB
):
    """
    为模型创建使用简化管理器的 SSM 缓存。

    Args:
        model: MLX-LM 模型
        max_size_bytes: 最大缓存内存（字节）

    Returns:
        (cache_list, simplified_manager)
    """
    # 自动检测层类型
    layer_types = create_layer_types_from_model(model)

    # 创建简化的 SSM 缓存管理器
    simplified_manager = SimplifiedSSMCacheManager(max_size_bytes=max_size_bytes)

    # 为每层创建缓存
    cache_list = []
    for layer_idx, layer_type in layer_types.items():
        if layer_type == LayerType.SSM:
            cache = PerLayerSSMCache(
                manager=simplified_manager,
                layer_idx=layer_idx,
                size=2
            )
            # 启用管理缓存（用于跨请求复用）
            cache.enable_managed_cache()
            cache_list.append(cache)
        else:
            # Attention 层使用默认缓存（或 PerLayerAttentionCache）
            cache_list.append(None)

    return cache_list, simplified_manager


def simulate_multi_turn_conversation():
    """模拟多轮对话，展示 SSM 状态复用。"""
    print("="*70)
    print("跨请求 SSM 状态复用示例")
    print("="*70)

    # 加载模型
    print("\n加载模型...")
    model_path = "/Volumes/toshiba/models/qwen3.5-35b-mlx/"
    model, tokenizer = load(model_path)

    # 创建 SSM 缓存
    print("创建简化的 SSM 缓存管理器...")
    cache_list, ssm_manager = create_ssm_caches_with_simplified_manager(model)

    # System prompt (固定，可复用)
    system_prompt = """You are a helpful AI assistant specializing in machine learning.
You provide clear, accurate, and concise answers."""

    # 模拟 3 轮对话
    conversations = [
        ("What is deep learning?", "Request 1 (首次计算 system prompt)"),
        ("Explain transformers.", "Request 2 (复用 system prompt)"),
        ("What are attention mechanisms?", "Request 3 (复用 system prompt)")
    ]

    results = []

    for i, (user_question, description) in enumerate(conversations, 1):
        print(f"\n{'='*70}")
        print(f"{description}")
        print(f"{'='*70}")

        # 构建完整 prompt
        full_prompt = f"{system_prompt}\n\nUser: {user_question}\nAssistant:"

        # 测量时间
        start_time = time.time()

        # 这里应该调用模型的 forward pass
        # 为了演示，我们直接测量缓存操作的时间
        # 实际使用时，MLX-LM 会自动使用我们注入的缓存

        # 模拟 SSM 状态访问
        for cache in cache_list:
            if isinstance(cache, PerLayerSSMCache):
                # 读取缓存（首次会 miss，后续会 hit）
                _ = cache[0]  # Convolution state
                _ = cache[1]  # SSM state

        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000

        # 获取缓存统计
        stats = ssm_manager.get_statistics()

        print(f"\n用户问题: {user_question}")
        print(f"缓存访问时间: {elapsed_ms:.6f} ms")
        print(f"\n缓存统计:")
        print(f"  条目数:     {stats['entry_count']}")
        print(f"  缓存大小:   {stats['size_bytes'] / 1024:.1f} KB")
        print(f"  命中次数:   {stats['hits']}")
        print(f"  缺失次数:   {stats['misses']}")
        print(f"  命中率:     {stats['hit_rate']:.1%}")

        results.append({
            'request': i,
            'description': description,
            'cache_time_ms': elapsed_ms,
            'hit_rate': stats['hit_rate']
        })

    # 对比分析
    print(f"\n{'='*70}")
    print("性能对比")
    print(f"{'='*70}")

    print(f"\n{'Request':<10} {'Description':<35} {'Hit Rate':<12} {'Cache Time':<15}")
    print("-"*70)

    for result in results:
        print(f"{result['request']:<10} "
              f"{result['description']:<35} "
              f"{result['hit_rate']:>10.1%}  "
              f"{result['cache_time_ms']:>12.6f} ms")

    # 理论收益分析
    print(f"\n{'='*70}")
    print("理论收益分析")
    print(f"{'='*70}")

    system_tokens = len(tokenizer.encode(system_prompt))
    forward_time_per_token = 1.25  # ms (from benchmark)
    cache_overhead_per_layer = 0.011 / 1000  # ms (simplified cache)
    num_ssm_layers = sum(1 for c in cache_list if isinstance(c, PerLayerSSMCache))

    print(f"\nSystem prompt:")
    print(f"  Tokens: {system_tokens}")
    print(f"  Forward pass (首次): {system_tokens * forward_time_per_token:.1f} ms")
    print(f"  Cache read (复用):   {num_ssm_layers * cache_overhead_per_layer:.6f} ms")
    print(f"  净收益 (每次):       {system_tokens * forward_time_per_token:.1f} ms")

    print(f"\n总收益 (3 轮对话):")
    print(f"  无缓存:   {system_tokens * forward_time_per_token * 3:.1f} ms")
    print(f"  有缓存:   {system_tokens * forward_time_per_token + num_ssm_layers * cache_overhead_per_layer * 2:.1f} ms")
    print(f"  节省:     {system_tokens * forward_time_per_token * 2:.1f} ms ({system_tokens * forward_time_per_token * 2 / (system_tokens * forward_time_per_token * 3) * 100:.1f}%)")

    print(f"\n{'='*70}")
    print("✓ 示例完成！")
    print(f"{'='*70}")


def demonstrate_simplified_cache_overhead():
    """演示简化缓存的低开销。"""
    print("\n" + "="*70)
    print("简化 SSM 缓存 vs 三层缓存开销对比")
    print("="*70)

    # 创建测试数据
    import numpy as np
    test_state = mx.array(np.random.randn(256).astype(np.float32))  # 1KB 状态

    # 简化缓存
    simplified_manager = SimplifiedSSMCacheManager()

    # 测试简化缓存
    print("\n测试简化缓存（单层 dict）...")
    iterations = 10000

    start = time.time()
    for i in range(iterations):
        simplified_manager.store(i % 30, test_state)
        _ = simplified_manager.retrieve(i % 30)
    end = time.time()

    elapsed_us = (end - start) * 1000000
    ops_per_second = (iterations * 2) / (end - start)
    latency_per_op = elapsed_us / (iterations * 2)

    print(f"  迭代次数:   {iterations}")
    print(f"  总时间:     {elapsed_us:.1f} μs")
    print(f"  吞吐量:     {ops_per_second:,.0f} ops/s")
    print(f"  平均延迟:   {latency_per_op:.3f} μs/op")

    # 对比之前的微基准测试结果
    print(f"\n{'='*70}")
    print("对比分析")
    print(f"{'='*70}")

    print(f"\n{'方案':<30} {'延迟 (μs/op)':<15} {'相对开销':<15}")
    print("-"*70)
    print(f"{'Direct dict access':<30} {0.011:<15.3f} {'基准 (1x)':<15}")
    print(f"{'Simplified cache (NEW)':<30} {latency_per_op:<15.3f} {f'{latency_per_op / 0.011:.1f}x':<15}")
    print(f"{'Hot/Warm/Cold (OLD)':<30} {0.177:<15.3f} {'16.1x':<15}")

    print(f"\n{'='*70}")
    print(f"✓ 简化缓存开销降低 {0.177 / latency_per_op:.1f}x!")
    print(f"{'='*70}")


if __name__ == "__main__":
    # 演示简化缓存的低开销
    demonstrate_simplified_cache_overhead()

    # 演示跨请求复用（注意：需要实际模型，这里只是框架）
    # simulate_multi_turn_conversation()
