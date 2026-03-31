#!/usr/bin/env python3
"""
简单验证 attention patcher 的功能
不需要加载完整模型
"""
import mlx.core as mx
from flashmlx.cache.attention_patcher import repeat_kv, patch_attention_for_compacted_cache
from flashmlx.cache.compacted_kv_cache import CompactedKVCache


def test_repeat_kv():
    """测试 GQA head repetition"""
    print("=" * 80)
    print("测试 repeat_kv 函数")
    print("=" * 80)

    # Test case 1: Beta (2D)
    B, n_kv_heads, t = 2, 4, 16
    n_rep = 2
    beta = mx.random.uniform(shape=(B, n_kv_heads, t))
    beta_repeated = repeat_kv(beta, n_rep)

    print(f"\n✓ Beta repetition:")
    print(f"  Input:  {beta.shape} (B, n_kv_heads, t)")
    print(f"  Output: {beta_repeated.shape} (B, n_heads, t)")
    print(f"  n_rep: {n_rep}")
    assert beta_repeated.shape == (B, n_kv_heads * n_rep, t)

    # Test case 2: KV cache (3D)
    head_dim = 128
    kv = mx.random.uniform(shape=(B, n_kv_heads, t, head_dim))
    kv_repeated = repeat_kv(kv, n_rep)

    print(f"\n✓ KV cache repetition:")
    print(f"  Input:  {kv.shape} (B, n_kv_heads, t, head_dim)")
    print(f"  Output: {kv_repeated.shape} (B, n_heads, t, head_dim)")
    assert kv_repeated.shape == (B, n_kv_heads * n_rep, t, head_dim)

    # Verify values are repeated correctly
    for b in range(B):
        for kv_head in range(n_kv_heads):
            for rep in range(n_rep):
                head_idx = kv_head * n_rep + rep
                assert mx.allclose(beta_repeated[b, head_idx, :], beta[b, kv_head, :])

    print("\n✓ All repeat_kv tests passed!")


def test_compacted_kv_cache():
    """测试 CompactedKVCache 创建"""
    print("\n" + "=" * 80)
    print("测试 CompactedKVCache")
    print("=" * 80)

    B, n_kv_heads, t, head_dim = 1, 4, 8, 128
    num_layers = 3

    compacted_cache = []
    for i in range(num_layers):
        c1 = mx.random.uniform(shape=(B, n_kv_heads, t, head_dim))
        beta = mx.random.uniform(shape=(B, n_kv_heads, t), low=-1.0, high=1.0)
        c2 = mx.random.uniform(shape=(B, n_kv_heads, t, head_dim))
        compacted_cache.append((c1, beta, c2))

    cache = CompactedKVCache(compacted_cache)

    print(f"\n✓ Cache 创建成功:")
    print(f"  Layers: {num_layers}")
    print(f"  Keys shape: {cache.keys.shape}")
    print(f"  Values shape: {cache.values.shape}")
    print(f"  Offset: {cache.offset}")

    # Test beta_for_layer
    for layer_idx in range(num_layers):
        beta = cache.beta_for_layer(layer_idx)
        assert beta is not None
        assert beta.shape == (B, n_kv_heads, t)
        print(f"  Layer {layer_idx} beta: {beta.shape}")

    print("\n✓ CompactedKVCache tests passed!")


def main():
    print("=" * 80)
    print("Attention Patcher 简单验证")
    print("=" * 80)

    test_repeat_kv()
    test_compacted_kv_cache()

    print("\n" + "=" * 80)
    print("✅ 所有测试通过！")
    print("=" * 80)


if __name__ == "__main__":
    main()
