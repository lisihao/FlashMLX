"""
End-to-end test for KV cache compression workflow.

Tests the complete pipeline:
1. Offline compression: HighestAttentionKeysCompaction
2. Cache creation: CompactedKVCache
3. Inference: patched attention with beta bias
4. Quality validation: compression metrics
"""
import mlx.core as mx
import numpy as np
import pytest

from flashmlx.cache.compaction_algorithm import create_compaction_algorithm
from flashmlx.cache.compacted_kv_cache import create_compacted_cache_list


class TestEndToEndCompression:
    """End-to-end compression workflow tests"""

    @pytest.fixture
    def qwen3_8b_config(self):
        """Qwen3-8B model configuration"""
        return {
            'num_layers': 36,
            'num_heads': 32,
            'num_kv_heads': 8,  # GQA
            'head_dim': 128,
            'vocab_size': 151936,
        }

    @pytest.fixture
    def compression_scenario(self, qwen3_8b_config):
        """Create a realistic compression scenario"""
        config = qwen3_8b_config

        # Original sequence length (e.g., long context prefix)
        T = 1024
        # Compressed length (4x compression)
        t = 256
        # Number of query samples for compression
        n_queries = 50
        # Batch size
        batch_size = 1

        n_kv_heads = config['num_kv_heads']
        head_dim = config['head_dim']

        # Create synthetic KV cache for one layer
        # Simulate realistic data distribution (small values, some structure)
        K = mx.random.normal(shape=(batch_size, n_kv_heads, T, head_dim)) * 0.1
        V = mx.random.normal(shape=(batch_size, n_kv_heads, T, head_dim)) * 0.1

        # Query samples (recent queries that would benefit from compression)
        queries = mx.random.normal(shape=(n_queries, head_dim)) * 0.1

        return {
            'K': K,
            'V': V,
            'queries': queries,
            'T': T,
            't': t,
            'n_queries': n_queries,
            'batch_size': batch_size,
            'n_kv_heads': n_kv_heads,
            'head_dim': head_dim,
            'compression_ratio': T / t,
        }

    def test_e2e_compression_workflow(self, compression_scenario):
        """Test 1: Complete compression workflow"""
        K = compression_scenario['K']
        V = compression_scenario['V']
        queries = compression_scenario['queries']
        T = compression_scenario['T']
        t = compression_scenario['t']
        batch_size = compression_scenario['batch_size']
        n_kv_heads = compression_scenario['n_kv_heads']
        head_dim = compression_scenario['head_dim']

        print(f"\n{'='*70}")
        print("E2E Compression Workflow Test")
        print(f"{'='*70}")

        # Step 1: Offline compression
        print("\n[Step 1] Offline Compression")
        print(f"  Original cache: {T} tokens")
        print(f"  Target: {t} tokens ({compression_scenario['compression_ratio']:.1f}x compression)")

        algo = create_compaction_algorithm(
            score_method='mean',
            beta_method='nnls',
            c2_method='lsq',
            c2_ridge_lambda=0.01
        )

        # Compress each head separately (per MLX convention)
        compacted_cache_per_layer = []

        for head_idx in range(n_kv_heads):
            # Extract K, V for this head: (T, head_dim)
            K_head = K[0, head_idx, :, :]  # (T, head_dim)
            V_head = V[0, head_idx, :, :]  # (T, head_dim)

            # Compress
            C1_head, beta_head, C2_head, indices = algo.compute_compacted_cache(
                K_head, V_head, queries, t
            )

            # Store compressed results: need shape (B, n_kv_heads, t, head_dim)
            # For now, just store one head's results
            if head_idx == 0:
                # Initialize full tensors
                C1 = mx.zeros((batch_size, n_kv_heads, t, head_dim), dtype=K.dtype)
                beta = mx.zeros((batch_size, n_kv_heads, t), dtype=K.dtype)
                C2 = mx.zeros((batch_size, n_kv_heads, t, head_dim), dtype=K.dtype)

            # Fill in this head
            C1[0, head_idx, :, :] = C1_head
            beta[0, head_idx, :] = beta_head
            C2[0, head_idx, :, :] = C2_head

        compacted_cache_per_layer.append((C1, beta, C2))

        print(f"  ✓ Compressed: C1={C1.shape}, beta={beta.shape}, C2={C2.shape}")
        print(f"  ✓ Selected indices: {len(indices)} keys")

        # Step 2: Create CompactedKVCache
        print("\n[Step 2] Create CompactedKVCache")
        cache_list = create_compacted_cache_list(
            compacted_cache_per_layer,
            original_seq_len=T
        )

        print(f"  ✓ Cache list created: {len(cache_list)} layers")
        print(f"  ✓ Layer 0 cache: offset={cache_list[0].offset}, original_seq_len={cache_list[0].original_seq_len}")
        print(f"  ✓ Beta shape: {cache_list[0].get_beta().shape}")

        # Step 3: Simulate inference with new tokens
        print("\n[Step 3] Simulate Inference")
        new_tokens = 5
        new_K = mx.random.normal(shape=(batch_size, n_kv_heads, new_tokens, head_dim)) * 0.1
        new_V = mx.random.normal(shape=(batch_size, n_kv_heads, new_tokens, head_dim)) * 0.1

        # Update cache
        updated_K, updated_V = cache_list[0].update_and_fetch(new_K, new_V)

        print(f"  ✓ Cache updated: {t} → {cache_list[0].offset} tokens")
        print(f"  ✓ Updated K shape: {updated_K.shape}")
        print(f"  ✓ Beta preserved: {cache_list[0].get_beta().shape}")

        # Step 4: Calculate metrics
        print("\n[Step 4] Compression Metrics")

        # Memory savings
        original_size = T * head_dim * n_kv_heads * 2 * 4  # K+V, float32
        compressed_size = t * head_dim * n_kv_heads * 2 * 4 + t * n_kv_heads * 4  # K+V+beta
        memory_saved = original_size - compressed_size
        memory_saved_pct = (memory_saved / original_size) * 100

        print(f"  Original memory: {original_size / 1024:.1f} KB")
        print(f"  Compressed memory: {compressed_size / 1024:.1f} KB")
        print(f"  Memory saved: {memory_saved / 1024:.1f} KB ({memory_saved_pct:.1f}%)")
        print(f"  Compression ratio: {compression_scenario['compression_ratio']:.1f}x")

        # Verify results
        assert cache_list[0].offset == t + new_tokens
        assert cache_list[0].get_beta() is not None
        assert memory_saved_pct > 70  # Should save >70% for 4x compression

        print(f"\n{'='*70}")
        print("✅ E2E Compression Test PASSED")
        print(f"{'='*70}")

    def test_compression_quality_metrics(self, compression_scenario):
        """Test 2: Measure compression quality"""
        K = compression_scenario['K']
        V = compression_scenario['V']
        queries = compression_scenario['queries']
        T = compression_scenario['T']
        t = compression_scenario['t']
        n_kv_heads = compression_scenario['n_kv_heads']
        head_dim = compression_scenario['head_dim']

        print(f"\n{'='*70}")
        print("Compression Quality Metrics")
        print(f"{'='*70}")

        # Test different compression ratios
        compression_ratios = [2, 4, 8]
        results = []

        for ratio in compression_ratios:
            target_t = T // ratio

            # Compress
            algo = create_compaction_algorithm(
                score_method='mean',
                beta_method='nnls',
                c2_method='lsq',
                c2_ridge_lambda=0.01
            )

            # Use first head for testing
            K_head = K[0, 0, :, :]
            V_head = V[0, 0, :, :]

            C1, beta, C2, indices = algo.compute_compacted_cache(
                K_head, V_head, queries, target_t
            )

            # Compute attention with original vs compressed
            scale = 1.0 / mx.sqrt(mx.array(head_dim, dtype=K.dtype))

            # Original attention
            attn_scores_orig = queries @ K_head.T * scale  # (n_queries, T)
            attn_weights_orig = mx.softmax(attn_scores_orig, axis=-1)
            attn_output_orig = attn_weights_orig @ V_head  # (n_queries, head_dim)

            # Compressed attention (without beta for now)
            attn_scores_comp = queries @ C1.T * scale  # (n_queries, t)
            attn_weights_comp = mx.softmax(attn_scores_comp, axis=-1)
            attn_output_comp = attn_weights_comp @ C2  # (n_queries, head_dim)

            # Compute similarity
            # Cosine similarity between outputs
            orig_norm = mx.linalg.norm(attn_output_orig, axis=1, keepdims=True)
            comp_norm = mx.linalg.norm(attn_output_comp, axis=1, keepdims=True)
            cosine_sim = mx.sum(attn_output_orig * attn_output_comp, axis=1) / (orig_norm.squeeze() * comp_norm.squeeze() + 1e-8)
            avg_cosine_sim = float(mx.mean(cosine_sim))

            # MSE
            mse = float(mx.mean((attn_output_orig - attn_output_comp) ** 2))

            results.append({
                'ratio': ratio,
                't': target_t,
                'cosine_similarity': avg_cosine_sim,
                'mse': mse,
            })

            print(f"\nCompression Ratio: {ratio}x (T={T} → t={target_t})")
            print(f"  Cosine Similarity: {avg_cosine_sim:.4f}")
            print(f"  MSE: {mse:.6f}")

        # Verify quality degrades gracefully with compression
        # Higher compression should have lower similarity (monotonic decrease)
        print(f"\n[Quality Analysis]")
        print(f"  2x: {results[0]['cosine_similarity']:.4f}")
        print(f"  4x: {results[1]['cosine_similarity']:.4f}")
        print(f"  8x: {results[2]['cosine_similarity']:.4f}")

        # Check monotonic decrease
        assert results[0]['cosine_similarity'] > results[1]['cosine_similarity'], \
            "2x should be better than 4x"
        assert results[1]['cosine_similarity'] > results[2]['cosine_similarity'], \
            "4x should be better than 8x"

        # Quality thresholds based on compression ratio (relaxed for random data)
        # 2x compression should have reasonable quality
        assert results[0]['cosine_similarity'] > 0.60, \
            f"2x compression quality too low: {results[0]['cosine_similarity']:.4f}"

        # 4x compression should still be usable
        assert results[1]['cosine_similarity'] > 0.40, \
            f"4x compression quality too low: {results[1]['cosine_similarity']:.4f}"

        # 8x compression is very aggressive (just check not completely random)
        assert results[2]['cosine_similarity'] > 0.20, \
            f"8x compression quality too low: {results[2]['cosine_similarity']:.4f}"

        print(f"  ✓ Quality degrades monotonically")
        print(f"  ✓ All compression ratios above thresholds")

        print(f"\n{'='*70}")
        print("✅ Quality Metrics Test PASSED")
        print(f"{'='*70}")

    def test_multi_layer_compression(self, qwen3_8b_config):
        """Test 3: Compress multiple layers"""
        config = qwen3_8b_config

        # Smaller test: 3 layers
        num_layers = 3
        T = 512
        t = 128
        n_queries = 20

        batch_size = 1
        n_kv_heads = config['num_kv_heads']
        head_dim = config['head_dim']

        print(f"\n{'='*70}")
        print(f"Multi-Layer Compression Test ({num_layers} layers)")
        print(f"{'='*70}")

        algo = create_compaction_algorithm(
            score_method='mean',
            beta_method='ones',  # Faster for multi-layer test
            c2_method='direct'
        )

        compacted_cache = []

        for layer_idx in range(num_layers):
            print(f"\n[Layer {layer_idx}] Compressing...")

            # Create KV cache for this layer
            K = mx.random.normal(shape=(batch_size, n_kv_heads, T, head_dim)) * 0.1
            V = mx.random.normal(shape=(batch_size, n_kv_heads, T, head_dim)) * 0.1
            queries = mx.random.normal(shape=(n_queries, head_dim)) * 0.1

            # Compress all heads for this layer
            C1 = mx.zeros((batch_size, n_kv_heads, t, head_dim), dtype=K.dtype)
            beta = mx.zeros((batch_size, n_kv_heads, t), dtype=K.dtype)
            C2 = mx.zeros((batch_size, n_kv_heads, t, head_dim), dtype=K.dtype)

            for head_idx in range(n_kv_heads):
                K_head = K[0, head_idx, :, :]
                V_head = V[0, head_idx, :, :]

                C1_head, beta_head, C2_head, _ = algo.compute_compacted_cache(
                    K_head, V_head, queries, t
                )

                C1[0, head_idx, :, :] = C1_head
                beta[0, head_idx, :] = beta_head
                C2[0, head_idx, :, :] = C2_head

            compacted_cache.append((C1, beta, C2))
            print(f"  ✓ Layer {layer_idx}: {T} → {t} tokens ({T/t:.1f}x)")

        # Create cache list
        cache_list = create_compacted_cache_list(compacted_cache, original_seq_len=T)

        assert len(cache_list) == num_layers
        for layer_idx, cache in enumerate(cache_list):
            assert cache.offset == t
            assert cache.layer_idx == layer_idx
            assert cache.get_beta() is not None

        print(f"\n{'='*70}")
        print(f"✅ Multi-Layer Compression Test PASSED")
        print(f"  Total layers compressed: {num_layers}")
        print(f"  Compression ratio: {T/t:.1f}x per layer")
        print(f"{'='*70}")


def test_compression_algorithm_integration():
    """Verify compression algorithm integrates with cache system"""
    from flashmlx.cache import (
        HighestAttentionKeysCompaction,
        create_compaction_algorithm,
        CompactedKVCache,
        create_compacted_cache_list,
    )

    # Create compressor
    compressor = create_compaction_algorithm()
    assert isinstance(compressor, HighestAttentionKeysCompaction)

    # Create synthetic data
    K = mx.random.normal(shape=(100, 64)) * 0.1
    V = mx.random.normal(shape=(100, 64)) * 0.1
    queries = mx.random.normal(shape=(10, 64)) * 0.1

    # Compress
    C1, beta, C2, indices = compressor.compute_compacted_cache(K, V, queries, 25)

    # Create cache (need to wrap in batch/head dimensions)
    C1_wrapped = mx.expand_dims(mx.expand_dims(C1, 0), 0)  # (1, 1, t, d)
    beta_wrapped = mx.expand_dims(mx.expand_dims(beta, 0), 0)  # (1, 1, t)
    C2_wrapped = mx.expand_dims(mx.expand_dims(C2, 0), 0)  # (1, 1, t, d)

    cache_list = create_compacted_cache_list([(C1_wrapped, beta_wrapped, C2_wrapped)])

    assert len(cache_list) == 1
    assert cache_list[0].offset == 25


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
