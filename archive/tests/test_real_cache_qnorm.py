#!/usr/bin/env python3
"""
测试使用真实 KV cache 数据的 q_norm 修复
"""

import mlx.core as mx
from mlx_lm import load, generate

print("=" * 60)
print("🧪 测试真实 KV cache + q_norm")
print("=" * 60)

# 加载模型
model_path = "/Volumes/toshiba/models/qwen3-8b-mlx"
print(f"\n📦 Loading model: {model_path}")
model, tokenizer = load(model_path)
print(f"✅ Model loaded")

# 生成一些文本以填充 KV cache
prompt = "问题：3 + 5 = ?\n回答："
print(f"\n🔵 Generating text to populate KV cache...")
print(f"Prompt: {prompt}")

# Tokenize
input_ids = mx.array(tokenizer.encode(prompt))
if input_ids.ndim == 1:
    input_ids = mx.expand_dims(input_ids, axis=0)  # Add batch dimension
print(f"Input IDs shape: {input_ids.shape}")

# Initialize cache
print(f"Initializing cache...")
if hasattr(model, 'make_cache'):
    model.cache = model.make_cache()
    print(f"✅ Cache initialized: {len(model.cache)} layers")
else:
    print(f"⚠️ Model has no make_cache method")
    # Try to create a simple cache
    from mlx_lm.models.cache import make_prompt_cache
    model.cache = make_prompt_cache(model)
    print(f"✅ Cache created: {len(model.cache)} layers")

# Forward pass to populate cache
print(f"Running forward pass...")
logits = model(input_ids)
print(f"Logits shape: {logits.shape}")

# 检查 cache 是否被填充
if not hasattr(model, 'cache') or model.cache is None or len(model.cache) == 0:
    print("\n❌ No cache populated!")
    exit(1)

print(f"\n✅ Cache populated: {len(model.cache)} layers")

# 获取第一层的 cache
layer_idx = 0
layer_cache = model.cache[layer_idx]

# Handle KVCache object
if hasattr(layer_cache, 'keys') and hasattr(layer_cache, 'values'):
    # KVCache object: .keys and .values properties
    keys = layer_cache.keys
    values = layer_cache.values
elif hasattr(layer_cache, 'state') and len(layer_cache.state) >= 2:
    # KVCache format: .state = (keys, values)
    keys, values = layer_cache.state[0], layer_cache.state[1]
elif isinstance(layer_cache, (list, tuple)) and len(layer_cache) >= 2:
    keys, values = layer_cache[0], layer_cache[1]
else:
    print(f"❌ Unexpected cache structure: {type(layer_cache)}")
    exit(1)

print(f"\n📊 Cache shape:")
print(f"  Keys: {keys.shape}")
print(f"  Values: {values.shape}")

# 获取第一个 head 的 keys/values
batch_size, num_heads, seq_len, head_dim = keys.shape
print(f"  batch_size={batch_size}, num_heads={num_heads}, seq_len={seq_len}, head_dim={head_dim}")

# 提取第一个 head 的数据
head_idx = 0
head_keys = keys[0, head_idx, :, :]  # (seq_len, head_dim)
head_values = values[0, head_idx, :, :]  # (seq_len, head_dim)

print(f"\n📊 Head {head_idx} keys:")
print(f"  Shape: {head_keys.shape}")
print(f"  Mean norm: {mx.linalg.norm(head_keys, axis=-1).mean():.4f}")
print(f"  Has NaN: {mx.isnan(head_keys).any().item()}")
print(f"  Has Inf: {mx.isinf(head_keys).any().item()}")

# 从 keys 中采样 queries
num_queries = min(50, seq_len)
import numpy as np
indices = np.random.choice(seq_len, size=num_queries, replace=False)
indices = np.sort(indices)
indices_mlx = mx.array(indices)

# Method 1: Without q_norm
print(f"\n❌ Method 1: Without q_norm")
sampled_keys = mx.take(head_keys, indices_mlx, axis=0)
print(f"  Sampled keys shape: {sampled_keys.shape}")
print(f"  Sampled keys norm: {mx.linalg.norm(sampled_keys, axis=-1).mean():.4f}")

# Method 2: With q_norm
print(f"\n✅ Method 2: With q_norm")
layer = model.layers[layer_idx]
if hasattr(layer.self_attn, 'q_norm'):
    q_norm = layer.self_attn.q_norm
    sampled_queries = q_norm(sampled_keys)
    print(f"  Queries after q_norm shape: {sampled_queries.shape}")
    print(f"  Queries after q_norm norm: {mx.linalg.norm(sampled_queries, axis=-1).mean():.4f}")
    print(f"  Queries has NaN: {mx.isnan(sampled_queries).any().item()}")
    print(f"  Queries has Inf: {mx.isinf(sampled_queries).any().item()}")

    # Test compression with real cache data
    print(f"\n🔧 Testing compression with real cache data...")

    from flashmlx.compaction.wrapper import AttentionMatchingWrapper

    wrapper = AttentionMatchingWrapper(
        compression_ratio=2.0,
        score_method='max',
        beta_method='nnls',
        c2_method='lsq',
    )

    # Convert to PyTorch
    K_torch = wrapper.mlx_to_torch(head_keys)
    V_torch = wrapper.mlx_to_torch(head_values)
    queries_torch = wrapper.mlx_to_torch(sampled_queries)

    print(f"  K shape: {K_torch.shape}")
    print(f"  V shape: {V_torch.shape}")
    print(f"  Queries shape: {queries_torch.shape}")

    try:
        target_size = max(1, int(seq_len / 2.0))
        print(f"  Target size: {target_size}")

        C1, beta, C2, selected_indices = wrapper.algorithm.compute_compacted_cache(
            K=K_torch,
            V=V_torch,
            queries=queries_torch,
            t=target_size,
        )

        print(f"\n  ✅ Compression succeeded!")
        print(f"     C1 shape: {C1.shape}")
        print(f"     beta shape: {beta.shape}")
        print(f"     beta has NaN: {mx.isnan(wrapper.torch_to_mlx(beta)).any().item()}")
        print(f"     beta has Inf: {mx.isinf(wrapper.torch_to_mlx(beta)).any().item()}")
        print(f"     beta range: [{beta.min().item():.4f}, {beta.max().item():.4f}]")

        print(f"\n🎉 SUCCESS: Real cache + q_norm works!")

    except Exception as e:
        print(f"\n  ❌ Compression failed: {e}")
        import traceback
        traceback.print_exc()

else:
    print(f"\n❌ No q_norm found in layer {layer_idx}")

print("\n" + "=" * 60)
