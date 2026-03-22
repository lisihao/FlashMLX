"""
Offline KV Cache Compression Quality Evaluation

使用真实模型评估压缩质量（论文标准）

Metrics:
1. Perplexity (主要指标)
2. Token Generation Quality (token overlap)
3. Reconstruction Error (L2 distance)
4. Memory Saving
"""
import mlx.core as mx
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import time

# MLX-LM imports
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

# FlashMLX imports
from flashmlx.compaction.offline_compressor import offline_compress_kv_cache
from flashmlx.cache import create_compacted_cache_list


def load_local_model(model_path: str):
    """Load local MLX model"""
    print(f"Loading model from: {model_path}")
    model, tokenizer = load(model_path)
    print(f"  ✓ Model loaded: {model_path}")
    return model, tokenizer


def compute_perplexity(
    model,
    tokenizer,
    prompt: str,
    continuation: str,
    cache=None,
    max_tokens: int = 100
) -> float:
    """
    计算 perplexity

    Perplexity = exp(-1/N * sum(log P(token_i | context)))

    Args:
        model: MLX-LM model
        tokenizer: Tokenizer
        prompt: Initial prompt
        continuation: Text to compute perplexity on
        cache: Optional KV cache
        max_tokens: Max tokens to evaluate

    Returns:
        Perplexity score
    """
    # Tokenize
    full_text = prompt + continuation
    tokens = tokenizer.encode(full_text)

    # Split into prompt and continuation
    prompt_tokens = tokenizer.encode(prompt)
    prompt_len = len(prompt_tokens)

    # Limit continuation length
    eval_tokens = tokens[prompt_len:prompt_len + max_tokens]

    # Compute log probabilities
    log_probs = []

    for i, token in enumerate(eval_tokens):
        # Context = prompt + continuation so far
        context = tokens[:prompt_len + i]

        # Forward pass
        logits = model(mx.array([context]), cache=cache)

        # Get probability for next token
        probs = mx.softmax(logits[0, -1, :], axis=-1)
        token_prob = float(probs[token])

        # Log probability
        log_prob = np.log(token_prob + 1e-10)
        log_probs.append(log_prob)

    # Perplexity = exp(-average log prob)
    avg_log_prob = np.mean(log_probs)
    perplexity = np.exp(-avg_log_prob)

    return perplexity


def build_cache_from_prompt(
    model,
    tokenizer,
    prompt: str
) -> List:
    """
    Build KV cache by running forward pass on prompt

    Args:
        model: MLX-LM model
        tokenizer: Tokenizer
        prompt: Input prompt

    Returns:
        Populated KV cache
    """
    # Tokenize
    prompt_tokens = tokenizer.encode(prompt)

    # Forward pass to populate cache
    # Model's cache will be populated during forward pass
    model(mx.array([prompt_tokens]))

    # Extract cache from model
    cache = []
    for layer in model.model.layers:
        if hasattr(layer, 'self_attn'):
            # Extract KV cache from attention layer
            cache_layer = layer.self_attn.cache
            if cache_layer is not None:
                cache.append(cache_layer)

    return cache


def generate_with_cache(
    model,
    tokenizer,
    prompt: str,
    cache=None,
    max_tokens: int = 100,
    temp: float = 0.0
) -> Tuple[str, List]:
    """
    Generate text with given cache

    Args:
        model: MLX-LM model
        tokenizer: Tokenizer
        prompt: Input prompt
        cache: KV cache (original or compressed)
        max_tokens: Max tokens to generate
        temp: Temperature (0 = greedy)

    Returns:
        (generated_text, cache_after_generation)
    """
    # Use MLX-LM's generate function with cache
    from mlx_lm import generate as mlx_generate

    # Generate (this will use the cache if provided)
    generated_text = mlx_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        temp=temp,
        prompt_cache=cache  # Use prompt_cache parameter
    )

    # Extract cache after generation
    cache_after = []
    for layer in model.model.layers:
        if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'cache'):
            cache_layer = layer.self_attn.cache
            if cache_layer is not None:
                cache_after.append(cache_layer)

    return generated_text, cache_after


def compute_token_overlap(tokens1: List[int], tokens2: List[int]) -> float:
    """
    计算 token overlap 百分比

    Args:
        tokens1: First token sequence
        tokens2: Second token sequence

    Returns:
        Overlap percentage (0-100)
    """
    min_len = min(len(tokens1), len(tokens2))

    if min_len == 0:
        return 0.0

    matches = sum(1 for i in range(min_len) if tokens1[i] == tokens2[i])
    overlap = (matches / min_len) * 100

    return overlap


def compute_cache_reconstruction_error(
    original_cache,
    compressed_cache
) -> Dict[str, float]:
    """
    计算 KV cache 重建误差

    Args:
        original_cache: Original cache list
        compressed_cache: Compressed cache list

    Returns:
        Dict with error metrics
    """
    errors = {
        'key_l2': [],
        'value_l2': [],
        'mean_key_l2': 0.0,
        'mean_value_l2': 0.0
    }

    # Compare each layer
    for orig_layer, comp_layer in zip(original_cache, compressed_cache):
        # Original keys/values
        K_orig = orig_layer.keys  # (B, n_heads, T_orig, head_dim)
        V_orig = orig_layer.values

        # Compressed keys/values
        C1 = comp_layer.keys  # (B, n_heads, T_comp, head_dim)
        C2 = comp_layer.values

        # Reconstruct approximate original
        # For now, just compare compressed representation
        # (Full reconstruction would need query set)

        # Compute L2 norm difference (average over compressed tokens)
        key_error = float(mx.mean(mx.sum((C1 - K_orig[:, :, :C1.shape[2], :]) ** 2, axis=-1)))
        value_error = float(mx.mean(mx.sum((C2 - V_orig[:, :, :C2.shape[2], :]) ** 2, axis=-1)))

        errors['key_l2'].append(key_error)
        errors['value_l2'].append(value_error)

    # Average across layers
    errors['mean_key_l2'] = np.mean(errors['key_l2'])
    errors['mean_value_l2'] = np.mean(errors['value_l2'])

    return errors


def compute_memory_savings(
    original_cache,
    compressed_cache
) -> Dict[str, float]:
    """
    计算内存节省

    Args:
        original_cache: Original cache
        compressed_cache: Compressed cache

    Returns:
        Memory statistics
    """
    # Original size
    orig_size = 0
    for layer in original_cache:
        orig_size += layer.keys.size + layer.values.size

    # Compressed size
    comp_size = 0
    for layer in compressed_cache:
        comp_size += layer.keys.size + layer.values.size
        if hasattr(layer, 'beta'):
            comp_size += layer.beta.size

    # Calculate savings
    ratio = orig_size / comp_size if comp_size > 0 else 0
    saved_pct = ((orig_size - comp_size) / orig_size * 100) if orig_size > 0 else 0

    return {
        'original_size': orig_size,
        'compressed_size': comp_size,
        'compression_ratio': ratio,
        'memory_saved_pct': saved_pct
    }


def evaluate_compression_quality(
    model_path: str,
    test_prompts: List[str],
    compression_ratio: int = 4,
    num_queries: int = 100,
    use_omp: bool = False,
    max_gen_tokens: int = 50
) -> Dict:
    """
    完整的压缩质量评估

    Args:
        model_path: Path to local MLX model
        test_prompts: List of test prompts
        compression_ratio: Target compression ratio
        num_queries: Number of representative queries
        use_omp: Use OMP refinement
        max_gen_tokens: Max tokens to generate for each prompt

    Returns:
        Evaluation results dict
    """
    print("=" * 70)
    print("Offline KV Cache Compression - Quality Evaluation")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Compression ratio: {compression_ratio}x")
    print(f"Num queries: {num_queries}")
    print(f"Use OMP: {use_omp}")
    print(f"Test prompts: {len(test_prompts)}")
    print()

    # Load model
    model, tokenizer = load_local_model(model_path)

    results = {
        'perplexity': {'original': [], 'compressed': [], 'increase_pct': []},
        'token_overlap': [],
        'reconstruction_error': None,
        'memory': None,
        'generation_time': {'original': [], 'compressed': []}
    }

    # Test each prompt
    for i, prompt in enumerate(test_prompts):
        print(f"[Test {i+1}/{len(test_prompts)}] Prompt: {prompt[:50]}...")
        print()

        # === Original Generation ===
        print("  1. Original cache generation...")
        t0 = time.time()

        # Generate with original cache
        original_text, original_cache = generate_with_cache(
            model, tokenizer, prompt,
            cache=None,
            max_tokens=max_gen_tokens,
            temp=0.0
        )

        t_orig = time.time() - t0
        results['generation_time']['original'].append(t_orig)

        print(f"     ✓ Generated {len(tokenizer.encode(original_text))} tokens in {t_orig:.2f}s")
        print(f"     Output: {original_text[:80]}...")
        print()

        # === Offline Compression ===
        print("  2. Offline compression...")
        t0 = time.time()

        # Extract KV cache for compression
        # (Assume cache is populated after generation)
        if original_cache is None or len(original_cache) == 0:
            print("     ⚠ No cache to compress, skipping this prompt")
            continue

        # Stack cache into (B, n_heads, T, head_dim) format
        K_list = []
        V_list = []
        for layer in original_cache:
            K_list.append(layer.keys)
            V_list.append(layer.values)

        # Compress each layer
        compressed_data = []
        for layer_idx, (K, V) in enumerate(zip(K_list, V_list)):
            C1, beta, C2 = offline_compress_kv_cache(
                K, V,
                compression_ratio=compression_ratio,
                num_queries=num_queries,
                use_omp=use_omp,
                verbose=False  # Quiet for benchmarking
            )
            compressed_data.append((C1, beta, C2))

        # Create compressed cache
        original_seq_len = K_list[0].shape[2]
        compressed_cache = create_compacted_cache_list(
            compressed_data,
            original_seq_len=original_seq_len
        )

        t_comp = time.time() - t0
        print(f"     ✓ Compressed in {t_comp:.2f}s")
        print()

        # === Compressed Generation ===
        print("  3. Compressed cache generation...")
        t0 = time.time()

        compressed_text, _ = generate_with_cache(
            model, tokenizer, prompt,
            cache=compressed_cache,
            max_tokens=max_gen_tokens,
            temp=0.0
        )

        t_comp_gen = time.time() - t0
        results['generation_time']['compressed'].append(t_comp_gen)

        print(f"     ✓ Generated {len(tokenizer.encode(compressed_text))} tokens in {t_comp_gen:.2f}s")
        print(f"     Output: {compressed_text[:80]}...")
        print()

        # === Quality Metrics ===
        print("  4. Computing quality metrics...")

        # Token overlap
        orig_tokens = tokenizer.encode(original_text)
        comp_tokens = tokenizer.encode(compressed_text)
        overlap = compute_token_overlap(orig_tokens, comp_tokens)
        results['token_overlap'].append(overlap)

        print(f"     Token overlap: {overlap:.1f}%")

        # Perplexity (if we have a continuation)
        # For now, skip perplexity computation (requires separate continuation text)

        print()

    # === Reconstruction Error ===
    print("5. Computing reconstruction error...")
    reconstruction_error = compute_cache_reconstruction_error(
        original_cache, compressed_cache
    )
    results['reconstruction_error'] = reconstruction_error

    print(f"   Mean Key L2: {reconstruction_error['mean_key_l2']:.4f}")
    print(f"   Mean Value L2: {reconstruction_error['mean_value_l2']:.4f}")
    print()

    # === Memory Savings ===
    print("6. Computing memory savings...")
    memory_stats = compute_memory_savings(original_cache, compressed_cache)
    results['memory'] = memory_stats

    print(f"   Original size: {memory_stats['original_size']:,} elements")
    print(f"   Compressed size: {memory_stats['compressed_size']:,} elements")
    print(f"   Compression ratio: {memory_stats['compression_ratio']:.2f}x")
    print(f"   Memory saved: {memory_stats['memory_saved_pct']:.1f}%")
    print()

    # === Summary ===
    print("=" * 70)
    print("Evaluation Summary")
    print("=" * 70)

    avg_overlap = np.mean(results['token_overlap'])
    print(f"Average Token Overlap: {avg_overlap:.1f}%")

    avg_orig_time = np.mean(results['generation_time']['original'])
    avg_comp_time = np.mean(results['generation_time']['compressed'])
    print(f"Average Generation Time:")
    print(f"  Original: {avg_orig_time:.2f}s")
    print(f"  Compressed: {avg_comp_time:.2f}s")

    print(f"\nCompression Ratio: {memory_stats['compression_ratio']:.2f}x")
    print(f"Memory Saved: {memory_stats['memory_saved_pct']:.1f}%")

    print()
    print("Pass Criteria:")
    print(f"  ✓ Token Overlap > 80%: {'✅ PASS' if avg_overlap > 80 else '❌ FAIL'}")
    print(f"  ✓ Compression Ratio >= {compression_ratio}x: {'✅ PASS' if memory_stats['compression_ratio'] >= compression_ratio else '❌ FAIL'}")

    print("=" * 70)

    return results


def main():
    """Run evaluation"""
    # Configuration
    MODEL_PATH = "/Users/lisihao/.omlx/models/Qwen3-1.7B-MLX-4bit"

    TEST_PROMPTS = [
        "The quick brown fox jumps over the lazy dog. This sentence contains",
        "In machine learning, attention mechanisms are used to",
        "Python is a popular programming language because",
        "The meaning of life is"
    ]

    # Run evaluation
    results = evaluate_compression_quality(
        model_path=MODEL_PATH,
        test_prompts=TEST_PROMPTS,
        compression_ratio=4,
        num_queries=100,
        use_omp=False,  # Start without OMP for speed
        max_gen_tokens=50
    )

    return results


if __name__ == "__main__":
    main()
