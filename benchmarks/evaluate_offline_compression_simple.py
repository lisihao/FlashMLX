"""
Simplified Offline KV Cache Compression Evaluation

直接对比原始生成 vs 压缩后生成的质量

测试方法：
1. 使用长 prompt（模拟已有 context）
2. 生成 continuation（原始）
3. 压缩长 prompt 的 KV cache
4. 用压缩 cache 重新生成
5. 对比生成质量
"""
import mlx.core as mx
import numpy as np
from pathlib import Path
import time

# MLX-LM
from mlx_lm import load, generate

# FlashMLX
from flashmlx.compaction.offline_compressor import offline_compress_kv_cache
from flashmlx.cache import create_compacted_cache_list


def load_model(model_path: str):
    """Load MLX model"""
    print(f"Loading: {model_path}")
    model, tokenizer = load(model_path)
    print(f"  ✓ Loaded")
    return model, tokenizer


def extract_kv_cache_from_model(model):
    """
    Extract KV cache from model after forward pass

    Returns:
        List of cache layers
    """
    cache = []

    # Try to access cache from model layers
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        for layer in model.model.layers:
            # Check different possible attribute names
            if hasattr(layer, 'self_attn'):
                attn = layer.self_attn
                if hasattr(attn, 'cache') and attn.cache is not None:
                    cache.append(attn.cache)

    return cache if len(cache) > 0 else None


def manual_generate(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 50,
    temp: float = 0.0,
    cache=None
):
    """
    Manual generation with cache extraction

    Args:
        model: MLX model
        tokenizer: Tokenizer
        prompt: Input prompt
        max_tokens: Max tokens to generate
        temp: Temperature
        cache: Optional cache

    Returns:
        (generated_text, final_cache)
    """
    # Tokenize
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    # Generate token by token
    generated = []

    for _ in range(max_tokens):
        # Forward pass
        if cache is None:
            # First pass: populate cache
            logits = model(input_ids)
            cache = extract_kv_cache_from_model(model)
        else:
            # Subsequent passes: use existing cache
            # Only pass the last token
            last_token = mx.array([[tokens[-1]]])
            logits = model(last_token, cache=cache)

        # Get next token (greedy if temp=0)
        if temp == 0.0:
            next_token = int(mx.argmax(logits[0, -1, :]))
        else:
            probs = mx.softmax(logits[0, -1, :] / temp, axis=-1)
            next_token = int(mx.random.categorical(mx.log(probs)))

        # Append
        generated.append(next_token)
        tokens.append(next_token)

        # Stop at EOS
        if next_token == tokenizer.eos_token_id:
            break

    # Decode
    text = tokenizer.decode(generated)

    return text, cache


def test_compression_quality():
    """
    Test compression quality with real model
    """
    print("=" * 70)
    print("Offline Compression Quality Test (Simplified)")
    print("=" * 70)
    print()

    # Configuration
    MODEL_PATH = "/Users/lisihao/.omlx/models/Qwen3-1.7B-MLX-4bit"
    COMPRESSION_RATIO = 4
    NUM_QUERIES = 100

    # Long prompt to create substantial cache
    LONG_PROMPT = """The history of artificial intelligence begins in antiquity, with myths, stories and rumors of artificial beings endowed with intelligence or consciousness by master craftsmen. The seeds of modern AI were planted by classical philosophers who attempted to describe human thinking as a symbolic system. But the field of AI wasn't formally founded until 1956, at a conference at Dartmouth College, in Hanover, New Hampshire, where the term "artificial intelligence" was coined.

In the following decades, AI research has gone through several waves of optimism and disappointment. The 1960s and 1970s saw significant progress in machine learning and neural networks. However, the"""

    CONTINUATION_PROMPT = "Based on the above history, the next major breakthrough was"

    # Load model
    model, tokenizer = load_model(MODEL_PATH)
    print()

    # === Test 1: Original Generation ===
    print("[Test 1] Original generation (no compression)")
    print("-" * 70)

    t0 = time.time()

    # Generate with full prompt
    from mlx_lm.sample_utils import make_sampler

    original_output = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=LONG_PROMPT + " " + CONTINUATION_PROMPT,
        max_tokens=50,
        sampler=make_sampler(temp=0.0)
    )

    t_original = time.time() - t0

    print(f"Time: {t_original:.2f}s")
    print(f"Output: {original_output[:200]}...")
    print()

    original_tokens = tokenizer.encode(original_output)

    # === Test 2: With Offline Compression ===
    print("[Test 2] With offline KV cache compression")
    print("-" * 70)

    # Step 1: Build cache from long prompt
    print("  Step 1: Building cache from long prompt...")
    prompt_tokens = tokenizer.encode(LONG_PROMPT)
    print(f"    Prompt length: {len(prompt_tokens)} tokens")

    # Forward pass to populate cache
    input_ids = mx.array([prompt_tokens])
    _ = model(input_ids)

    # Extract cache
    original_cache = extract_kv_cache_from_model(model)

    if original_cache is None or len(original_cache) == 0:
        print("    ❌ Failed to extract cache from model")
        print("    Model may not support cache extraction")
        return

    print(f"    ✓ Extracted cache: {len(original_cache)} layers")

    # Step 2: Compress cache
    print("  Step 2: Compressing cache...")
    print(f"    Compression ratio: {COMPRESSION_RATIO}x")
    print(f"    Num queries: {NUM_QUERIES}")

    t0_comp = time.time()

    # Stack cache for compression
    # (Assuming cache layers have .keys and .values attributes)
    compressed_layers = []

    for layer_idx, cache_layer in enumerate(original_cache):
        # Extract K, V
        if hasattr(cache_layer, 'keys') and hasattr(cache_layer, 'values'):
            K = cache_layer.keys  # (B, n_heads, T, head_dim)
            V = cache_layer.values

            # Compress
            C1, beta, C2 = offline_compress_kv_cache(
                K, V,
                compression_ratio=COMPRESSION_RATIO,
                num_queries=NUM_QUERIES,
                use_omp=False,
                verbose=False
            )

            compressed_layers.append((C1, beta, C2))
        else:
            print(f"    ⚠ Layer {layer_idx} does not have keys/values")

    if len(compressed_layers) == 0:
        print("    ❌ No layers compressed")
        return

    # Create compacted cache
    original_seq_len = original_cache[0].keys.shape[2]
    compressed_cache = create_compacted_cache_list(
        compressed_layers,
        original_seq_len=original_seq_len
    )

    t_comp = time.time() - t0_comp
    print(f"    ✓ Compressed {len(compressed_layers)} layers in {t_comp:.2f}s")

    # Step 3: Generate with compressed cache
    print("  Step 3: Generating with compressed cache...")

    t0_gen = time.time()

    # Use compressed cache for generation
    compressed_output = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=CONTINUATION_PROMPT,  # Only continuation, cache has context
        max_tokens=50,
        sampler=make_sampler(temp=0.0),
        prompt_cache=compressed_cache
    )

    t_compressed = time.time() - t0_gen

    print(f"    ✓ Generated in {t_compressed:.2f}s")
    print()

    compressed_tokens = tokenizer.encode(compressed_output)

    # === Comparison ===
    print("=" * 70)
    print("Results")
    print("=" * 70)

    print(f"\nOriginal output:")
    print(f"  {original_output[:300]}...")

    print(f"\nCompressed cache output:")
    print(f"  {compressed_output[:300]}...")

    # Token overlap
    min_len = min(len(original_tokens), len(compressed_tokens))
    if min_len > 0:
        matches = sum(1 for i in range(min_len) if original_tokens[i] == compressed_tokens[i])
        overlap_pct = (matches / min_len) * 100
    else:
        overlap_pct = 0.0

    print(f"\nMetrics:")
    print(f"  Token overlap: {overlap_pct:.1f}%")
    print(f"  Original tokens: {len(original_tokens)}")
    print(f"  Compressed tokens: {len(compressed_tokens)}")

    # Memory savings
    orig_size = sum(layer.keys.size + layer.values.size for layer in original_cache)
    comp_size = sum(layer.keys.size + layer.values.size + layer.beta.size for layer in compressed_cache)

    memory_saved_pct = ((orig_size - comp_size) / orig_size * 100) if orig_size > 0 else 0

    print(f"\nMemory:")
    print(f"  Original cache: {orig_size:,} elements")
    print(f"  Compressed cache: {comp_size:,} elements")
    print(f"  Compression ratio: {orig_size / comp_size:.2f}x")
    print(f"  Memory saved: {memory_saved_pct:.1f}%")

    # Time
    print(f"\nTime:")
    print(f"  Original generation: {t_original:.2f}s")
    print(f"  Compression: {t_comp:.2f}s")
    print(f"  Compressed generation: {t_compressed:.2f}s")

    print()
    print("Pass Criteria:")
    print(f"  Token Overlap >= 80%: {'✅ PASS' if overlap_pct >= 80 else '❌ FAIL'} ({overlap_pct:.1f}%)")
    print(f"  Compression Ratio >= {COMPRESSION_RATIO}x: {'✅ PASS' if (orig_size/comp_size) >= COMPRESSION_RATIO else '❌ FAIL'} ({orig_size/comp_size:.2f}x)")
    print(f"  Memory Saved >= 75%: {'✅ PASS' if memory_saved_pct >= 75 else '❌ FAIL'} ({memory_saved_pct:.1f}%)")

    print("=" * 70)


if __name__ == "__main__":
    test_compression_quality()
