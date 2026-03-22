"""
Real model generation test with KV cache compression.

This script tests the actual generation quality of a real model
(Qwen3-8B-Instruct) using compressed KV cache.

Usage:
    python examples/test_real_model.py
"""
import mlx.core as mx

print("Loading dependencies...")
try:
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    MLX_LM_AVAILABLE = True
except ImportError:
    MLX_LM_AVAILABLE = False
    print("⚠️  mlx-lm not found. Install with: pip install mlx-lm")

from flashmlx.cache import (
    create_compaction_algorithm,
    create_compacted_cache_list,
    patch_attention_for_compacted_cache,
)


def test_compression_generation():
    """Test real model generation with compression"""

    if not MLX_LM_AVAILABLE:
        print("\n" + "=" * 70)
        print("❌ Cannot run test without mlx-lm")
        print("=" * 70)
        print("\nTo install:")
        print("  pip install mlx-lm")
        print("\nThen run:")
        print("  python examples/test_real_model.py")
        return

    print("\n" + "=" * 70)
    print("Real Model Generation Quality Test")
    print("=" * 70)

    # Configuration
    MODEL_NAME = "/Users/lisihao/.omlx/models/Qwen3-1.7B-MLX-4bit"  # Use local model
    COMPRESSION_RATIO = 4
    MAX_TOKENS = 50
    TEMPERATURE = 0.0  # Deterministic for comparison

    # 1. Load model
    print(f"\n[1/6] Loading model: {MODEL_NAME}")
    print("  (This may take a while on first run...)")

    try:
        model, tokenizer = load(MODEL_NAME)
        print("  ✓ Model loaded successfully")
    except Exception as e:
        print(f"  ✗ Failed to load model: {e}")
        print("\n  Note: Model must be downloaded first.")
        print("  You can use: mlx_lm.convert --hf-path Qwen/Qwen3-8B-Instruct")
        return

    # 2. Patch attention
    print("\n[2/6] Patching attention for beta support...")
    patch_attention_for_compacted_cache(model, verbose=False)
    print("  ✓ Attention patched")

    # 3. Prepare long prefix
    print("\n[3/6] Preparing long context prefix...")
    prefix = """Quantum computing is a revolutionary approach to computation that \
leverages the principles of quantum mechanics. Unlike classical computers that use \
bits (0 or 1), quantum computers use quantum bits or qubits, which can exist in \
superposition states. This allows quantum computers to process certain types of \
calculations exponentially faster than classical computers. Key quantum phenomena \
include superposition (existing in multiple states simultaneously) and entanglement \
(correlations between qubits that persist across distances)."""

    # Repeat to create longer context
    prefix = (prefix + "\n\n") * 5  # ~500 tokens
    print(f"  ✓ Prefix length: ~{len(prefix.split())} words")

    # 4. Generate baseline (no compression)
    print("\n[4/6] Generating baseline (no compression)...")
    baseline_prompt = prefix + "\n\nQuestion: Explain quantum superposition.\nAnswer:"

    # Create deterministic sampler
    sampler = make_sampler(temp=TEMPERATURE)

    try:
        baseline_output = generate(
            model,
            tokenizer,
            prompt=baseline_prompt,
            max_tokens=MAX_TOKENS,
            sampler=sampler,
            verbose=False
        )
        print(f"  ✓ Baseline generated ({len(baseline_output.split())} words)")
        print(f"  Preview: {baseline_output[:80]}...")
    except Exception as e:
        print(f"  ✗ Baseline generation failed: {e}")
        return

    # 5. Compress prefix cache
    print(f"\n[5/6] Compressing prefix cache ({COMPRESSION_RATIO}x)...")

    try:
        # Get prefix cache
        from mlx_lm.models.cache import make_prompt_cache
        cache = make_prompt_cache(model)
        prefix_tokens = tokenizer.encode(prefix)
        y = mx.array([prefix_tokens])

        # Fill cache with prefix
        _ = model(y, cache=cache)
        T = cache[0].keys.shape[-2]
        print(f"  Original cache: {T} tokens")

        # Compress each layer
        algo = create_compaction_algorithm(
            score_method='mean',
            beta_method='nnls',
            c2_method='lsq',
            c2_ridge_lambda=0.01
        )

        t = T // COMPRESSION_RATIO
        compacted_data = []

        for layer_idx, layer_cache in enumerate(cache):
            K = layer_cache.keys
            V = layer_cache.values
            B, n_kv_heads, _, head_dim = K.shape

            # Query samples (use recent tokens)
            n_queries = min(50, T)
            queries = K[0, 0, -n_queries:, :]

            # Compress each head
            C1_layer = mx.zeros((B, n_kv_heads, t, head_dim), dtype=K.dtype)
            beta_layer = mx.zeros((B, n_kv_heads, t), dtype=K.dtype)
            C2_layer = mx.zeros((B, n_kv_heads, t, head_dim), dtype=K.dtype)

            for head_idx in range(n_kv_heads):
                K_head = K[0, head_idx, :, :]
                V_head = V[0, head_idx, :, :]

                C1, beta, C2, _ = algo.compute_compacted_cache(
                    K_head, V_head, queries, t
                )

                C1_layer[0, head_idx, :, :] = C1
                beta_layer[0, head_idx, :] = beta
                C2_layer[0, head_idx, :, :] = C2

            compacted_data.append((C1_layer, beta_layer, C2_layer))

        compressed_cache = create_compacted_cache_list(
            compacted_cache=compacted_data,
            original_seq_len=T
        )

        print(f"  Compressed cache: {t} tokens")
        print(f"  Memory saved: {(1 - t/T) * 100:.1f}%")

    except Exception as e:
        print(f"  ✗ Compression failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 6. Generate with compressed cache
    print("\n[6/6] Generating with compressed cache...")
    question = "\n\nQuestion: Explain quantum superposition.\nAnswer:"
    question_tokens = tokenizer.encode(question)

    try:
        compressed_output = generate(
            model,
            tokenizer,
            prompt=question_tokens,
            prompt_cache=compressed_cache,  # Use compressed cache!
            max_tokens=MAX_TOKENS,
            sampler=sampler,
            verbose=False
        )
        print(f"  ✓ Compressed generated ({len(compressed_output.split())} words)")
        print(f"  Preview: {compressed_output[:80]}...")
    except Exception as e:
        print(f"  ✗ Compressed generation failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 7. Compare results
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)

    print(f"\nBaseline Output:")
    print("-" * 70)
    print(baseline_output)

    print(f"\nCompressed Output ({COMPRESSION_RATIO}x):")
    print("-" * 70)
    print(compressed_output)

    # Calculate token overlap
    baseline_tokens = tokenizer.encode(baseline_output)
    compressed_tokens = tokenizer.encode(compressed_output)

    min_len = min(len(baseline_tokens), len(compressed_tokens))
    if min_len > 0:
        matches = sum(
            1 for i in range(min_len)
            if baseline_tokens[i] == compressed_tokens[i]
        )
        overlap = (matches / min_len) * 100
    else:
        overlap = 0

    print("\n" + "=" * 70)
    print("Metrics")
    print("=" * 70)
    print(f"\nToken Overlap:      {overlap:.1f}%")
    print(f"Baseline Length:    {len(baseline_tokens)} tokens")
    print(f"Compressed Length:  {len(compressed_tokens)} tokens")
    print(f"Compression Ratio:  {COMPRESSION_RATIO}x")
    print(f"Memory Saved:       {(1 - t/T) * 100:.1f}%")

    # Quality grade
    if overlap >= 80:
        grade = "🟢 Excellent"
        desc = "Minimal difference"
    elif overlap >= 70:
        grade = "🟡 Good"
        desc = "Minor differences, acceptable"
    elif overlap >= 60:
        grade = "🟠 Acceptable"
        desc = "Noticeable differences"
    else:
        grade = "🔴 Poor"
        desc = "Significant differences"

    print(f"\nQuality Grade:      {grade}")
    print(f"Description:        {desc}")

    # Recommendations
    print("\n" + "=" * 70)
    print("Recommendations")
    print("=" * 70)

    if overlap >= 70:
        print(f"\n✅ {COMPRESSION_RATIO}x compression is suitable for production use.")
        print(f"   - Quality is maintained (≥70% overlap)")
        print(f"   - Memory savings: {(1 - t/T) * 100:.1f}%")
        print(f"   - Suitable for: Long context, multi-user scenarios")
    elif overlap >= 60:
        print(f"\n⚠️  {COMPRESSION_RATIO}x compression is usable but with caution.")
        print(f"   - Quality degradation noticeable")
        print(f"   - Consider reducing to 2x compression")
        print(f"   - Or use only for memory-constrained scenarios")
    else:
        print(f"\n❌ {COMPRESSION_RATIO}x compression not recommended.")
        print(f"   - Quality degradation too severe")
        print(f"   - Reduce compression ratio to 2x")
        print(f"   - Or disable compression for this use case")

    print("\n" + "=" * 70)
    print("✅ Test Complete")
    print("=" * 70)


if __name__ == "__main__":
    test_compression_generation()
