"""
Debug script to understand why compression quality is poor

Tests different query sampling strategies
"""
import mlx.core as mx
from mlx_lm import load
from mlx_lm.sample_utils import make_sampler
import sys
sys.path.insert(0, '/Users/lisihao/FlashMLX/src')

from flashmlx.cache import (
    create_compaction_algorithm,
    create_compacted_cache_list,
    patch_attention_for_compacted_cache,
)
from mlx_lm.models.cache import make_prompt_cache

# Load model
print("Loading model...")
model, tokenizer = load("/Users/lisihao/.omlx/models/Qwen3-1.7B-MLX-4bit")
patch_attention_for_compacted_cache(model, verbose=False)
print("✓ Model loaded\n")

# Prefix with question
prefix = """Quantum computing is a revolutionary approach to computation that \
leverages the principles of quantum mechanics. Unlike classical computers that use \
bits (0 or 1), quantum computers use quantum bits or qubits, which can exist in \
superposition states."""

question = "\n\nQuestion: Explain quantum superposition.\nAnswer:"

full_prompt = prefix + question

# Test 1: Baseline
print("=" * 70)
print("Test 1: Baseline (no compression)")
print("=" * 70)
sampler = make_sampler(temp=0.0)

from mlx_lm import generate
baseline_output = generate(
    model, tokenizer, prompt=full_prompt,
    max_tokens=50, sampler=sampler, verbose=False
)
print(f"Output: {baseline_output}\n")

# Test 2: Analyze cache before compression
print("=" * 70)
print("Test 2: Analyzing KV cache structure")
print("=" * 70)

cache = make_prompt_cache(model)
prefix_tokens = tokenizer.encode(prefix)
y = mx.array([prefix_tokens])

# Fill cache
_ = model(y, cache=cache)

T = cache[0].keys.shape[-2]
print(f"Total prefix tokens: {T}")
print(f"Prefix text preview: {prefix[:100]}...\n")

# Test 3: Different query sampling strategies
print("=" * 70)
print("Test 3: Query sampling strategies")
print("=" * 70)

strategies = [
    ("Last 50 tokens", lambda K, n: K[-n:, :]),  # Current strategy
    ("First 50 tokens", lambda K, n: K[:n, :]),  # Beginning
    ("Evenly distributed", lambda K, n: K[::max(K.shape[0]//n, 1), :][:n, :]),  # Distributed
]

for strategy_name, strategy_fn in strategies:
    print(f"\n{strategy_name}:")

    # Re-create cache
    cache = make_prompt_cache(model)
    _ = model(y, cache=cache)

    # Compress using this strategy
    algo = create_compaction_algorithm(
        score_method='mean',
        beta_method='nnls',
        c2_method='lsq',
        c2_ridge_lambda=0.01
    )

    compression_ratio = 4
    t = T // compression_ratio
    compacted_data = []

    for layer_cache in cache:
        K = layer_cache.keys
        V = layer_cache.values
        B, n_kv_heads, T_layer, head_dim = K.shape

        # Use strategy to select queries
        n_queries = min(50, T_layer)
        K_head = K[0, 0, :, :]  # (T, head_dim)
        queries = strategy_fn(K_head, n_queries)

        # Compress each head independently
        C1_layer = mx.zeros((B, n_kv_heads, t, head_dim), dtype=K.dtype)
        beta_layer = mx.zeros((B, n_kv_heads, t), dtype=K.dtype)
        C2_layer = mx.zeros((B, n_kv_heads, t, head_dim), dtype=K.dtype)

        for head_idx in range(n_kv_heads):
            K_head = K[0, head_idx, :, :]  # (T, head_dim)
            V_head = V[0, head_idx, :, :]  # (T, head_dim)
            queries_head = strategy_fn(K_head, n_queries)

            C1, beta, C2, _ = algo.compute_compacted_cache(
                K_head, V_head, queries_head, t
            )

            C1_layer[0, head_idx, :, :] = C1
            beta_layer[0, head_idx, :] = beta
            C2_layer[0, head_idx, :, :] = C2

        compacted_data.append((C1_layer, beta_layer, C2_layer))

    compressed_cache = create_compacted_cache_list(compacted_data, T)

    # Generate with compressed cache
    question_tokens = tokenizer.encode(question)
    try:
        compressed_output = generate(
            model, tokenizer, prompt=question_tokens,
            prompt_cache=compressed_cache, max_tokens=50,
            sampler=sampler, verbose=False
        )
        print(f"  Output: {compressed_output[:100]}...")

        # Calculate overlap
        baseline_tokens = tokenizer.encode(baseline_output)
        compressed_tokens = tokenizer.encode(compressed_output)
        min_len = min(len(baseline_tokens), len(compressed_tokens))
        if min_len > 0:
            matches = sum(1 for i in range(min_len) if baseline_tokens[i] == compressed_tokens[i])
            overlap = (matches / min_len) * 100
            print(f"  Token overlap: {overlap:.1f}%")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

print("\n" + "=" * 70)
print("Analysis Complete")
print("=" * 70)
