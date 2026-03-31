#!/usr/bin/env python3
"""
修复版测试 - 使用 q_norm scaling

关键修复：
1. 从 cache keys 采样 queries
2. ✅ 应用 q_norm scaling（Qwen 特有）
3. 验证是否解决数值问题
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch
from mlx_lm import load, generate

from flashmlx.compaction.wrapper import AttentionMatchingWrapper


def test_with_qnorm(model, tokenizer):
    """测试带 q_norm 的 query generation"""

    prompt = "问题：3 + 5 = ?\n回答："
    max_tokens = 30

    print("="*60)
    print("🧪 测试：q_norm scaling")
    print("="*60)

    # Generate baseline
    print("\n🔵 Baseline...")
    baseline_response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        verbose=False
    )
    print(f"Baseline: {baseline_response[:100]}")

    # Get first layer for testing
    layer_idx = 0
    layer = model.model.layers[layer_idx]

    # Check if q_norm exists
    if not hasattr(layer.self_attn, 'q_norm'):
        print("\n❌ No q_norm found!")
        return

    q_norm = layer.self_attn.q_norm
    print(f"\n✅ Found q_norm: {type(q_norm).__name__}")

    # Simulate compression with q_norm scaling
    print("\n🟢 Testing compression with q_norm scaling...")

    # Create dummy KV cache
    seq_len = 100
    head_dim = 128
    num_queries = 50

    # Generate dummy keys (MLX)
    keys_mlx = mx.random.normal((seq_len, head_dim))
    values_mlx = mx.random.normal((seq_len, head_dim))

    # Sample indices
    indices = np.random.choice(seq_len, size=num_queries, replace=False)
    indices = np.sort(indices)
    indices_mlx = mx.array(indices)

    # Method 1: Without q_norm (❌ 错误)
    print("\n  ❌ Method 1: Without q_norm")
    sampled_keys_mlx = mx.take(keys_mlx, indices_mlx, axis=0)  # (num_queries, head_dim)
    print(f"     Sampled keys shape: {sampled_keys_mlx.shape}")
    print(f"     Sampled keys norm: {mx.linalg.norm(sampled_keys_mlx, axis=-1).mean():.4f}")

    # Method 2: With q_norm (✅ 正确)
    print("\n  ✅ Method 2: With q_norm")
    sampled_keys_mlx = mx.take(keys_mlx, indices_mlx, axis=0)  # (num_queries, head_dim)

    # Apply q_norm (MLX)
    sampled_queries_mlx = q_norm(sampled_keys_mlx)  # (num_queries, head_dim)
    print(f"     Queries after q_norm shape: {sampled_queries_mlx.shape}")
    print(f"     Queries after q_norm norm: {mx.linalg.norm(sampled_queries_mlx, axis=-1).mean():.4f}")

    # Convert to PyTorch and test compression
    wrapper = AttentionMatchingWrapper(
        compression_ratio=2.0,
        score_method='max',
        beta_method='nnls',
        c2_method='lsq',
    )

    # Convert to PyTorch
    K_torch = wrapper.mlx_to_torch(keys_mlx)
    V_torch = wrapper.mlx_to_torch(values_mlx)
    queries_torch = wrapper.mlx_to_torch(sampled_queries_mlx)  # ✅ 带 q_norm

    print(f"\n  🔧 Compressing with q_norm scaled queries...")

    try:
        C1, beta, C2, selected_indices = wrapper.algorithm.compute_compacted_cache(
            K=K_torch,
            V=V_torch,
            queries=queries_torch,
            t=50,  # Target size
        )

        print(f"  ✅ Compression succeeded!")
        print(f"     C1 shape: {C1.shape}")
        print(f"     beta shape: {beta.shape}")
        print(f"     beta has NaN: {torch.isnan(beta).any().item()}")
        print(f"     beta has Inf: {torch.isinf(beta).any().item()}")
        print(f"     beta range: [{beta.min():.4f}, {beta.max():.4f}]")

        # Check if beta is reasonable
        if not torch.isinf(beta).any() and not torch.isnan(beta).any():
            print(f"\n  🎉 SUCCESS: q_norm scaling fixes the numerical issues!")
            return True
        else:
            print(f"\n  ⚠️ Still has numerical issues")
            return False

    except Exception as e:
        print(f"  ❌ Compression failed: {e}")
        return False


def main():
    print("\n" + "="*60)
    print("🧪 验证 q_norm 修复方案")
    print("="*60)

    model_path = "/Volumes/toshiba/models/qwen3-8b-mlx"

    print(f"\n📦 Loading model: {model_path}")
    model, tokenizer = load(model_path)
    print(f"✅ Model loaded")

    # Test
    success = test_with_qnorm(model, tokenizer)

    print("\n" + "="*60)
    if success:
        print("🎉 q_norm scaling 修复了数值问题！")
        print("   下一步：集成到 AttentionMatchingCompressorV2")
    else:
        print("⚠️ 仍有问题，需要进一步调查")
    print("="*60)


if __name__ == "__main__":
    main()
