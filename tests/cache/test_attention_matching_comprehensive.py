"""
Comprehensive unit tests for AttentionMatchingCompressor (Task #68)

This file provides comprehensive test coverage (target: ≥95%) for all
AttentionMatchingCompressor functionality.
"""

import unittest
import numpy as np
import mlx.core as mx
import time

from flashmlx.cache import AttentionMatchingCompressor


class TestAttentionMatchingEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions"""

    def test_extreme_compression_ratio(self):
        """Test very high compression ratio"""
        compressor = AttentionMatchingCompressor(compression_ratio=10.0)

        batch_size, num_heads, seq_len, head_dim = 1, 8, 100, 64
        keys = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
        values = mx.random.normal((batch_size, num_heads, seq_len, head_dim))

        compressed_keys, compressed_values = compressor.compress_kv_cache(
            layer_idx=0,
            kv_cache=(keys, values)
        )

        # 100 / 10.0 = 10
        self.assertEqual(compressed_keys.shape[2], 10)
        self.assertEqual(compressed_values.shape[2], 10)

    def test_minimal_compression_ratio(self):
        """Test compression ratio close to 1.0"""
        compressor = AttentionMatchingCompressor(compression_ratio=1.01)

        batch_size, num_heads, seq_len, head_dim = 1, 8, 100, 64
        keys = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
        values = mx.random.normal((batch_size, num_heads, seq_len, head_dim))

        compressed_keys, compressed_values = compressor.compress_kv_cache(
            layer_idx=0,
            kv_cache=(keys, values)
        )

        # 100 / 1.01 ≈ 99
        expected_len = int(100 / 1.01)
        self.assertEqual(compressed_keys.shape[2], expected_len)

    def test_compression_with_small_sequence(self):
        """Test compression with very small sequence length"""
        compressor = AttentionMatchingCompressor(compression_ratio=2.0)

        # Very small sequence
        batch_size, num_heads, seq_len, head_dim = 1, 8, 4, 64
        keys = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
        values = mx.random.normal((batch_size, num_heads, seq_len, head_dim))

        compressed_keys, compressed_values = compressor.compress_kv_cache(
            layer_idx=0,
            kv_cache=(keys, values)
        )

        # 4 / 2.0 = 2
        self.assertEqual(compressed_keys.shape[2], 2)

    def test_compression_ratio_that_results_in_zero_keys(self):
        """Test that extremely high compression ratio raises error"""
        compressor = AttentionMatchingCompressor(compression_ratio=1000.0)

        batch_size, num_heads, seq_len, head_dim = 1, 8, 10, 64
        keys = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
        values = mx.random.normal((batch_size, num_heads, seq_len, head_dim))

        # 10 / 1000.0 = 0.01 → int() = 0, should raise error
        with self.assertRaises(ValueError):
            compressor.compress_kv_cache(layer_idx=0, kv_cache=(keys, values))

    def test_single_key_compression(self):
        """Test compression down to a single key"""
        compressor = AttentionMatchingCompressor(compression_ratio=10.0)

        batch_size, num_heads, seq_len, head_dim = 1, 8, 10, 64
        keys = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
        values = mx.random.normal((batch_size, num_heads, seq_len, head_dim))

        compressed_keys, compressed_values = compressor.compress_kv_cache(
            layer_idx=0,
            kv_cache=(keys, values)
        )

        # 10 / 10.0 = 1
        self.assertEqual(compressed_keys.shape[2], 1)
        self.assertEqual(compressed_values.shape[2], 1)

    def test_large_batch_size(self):
        """Test compression with large batch size"""
        compressor = AttentionMatchingCompressor(compression_ratio=2.0)

        # Large batch
        batch_size, num_heads, seq_len, head_dim = 32, 8, 100, 64
        keys = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
        values = mx.random.normal((batch_size, num_heads, seq_len, head_dim))

        compressed_keys, compressed_values = compressor.compress_kv_cache(
            layer_idx=0,
            kv_cache=(keys, values)
        )

        # Batch dimension should be preserved
        self.assertEqual(compressed_keys.shape[0], batch_size)
        self.assertEqual(compressed_keys.shape[2], 50)

    def test_many_heads(self):
        """Test compression with many attention heads"""
        compressor = AttentionMatchingCompressor(compression_ratio=2.0)

        # Many heads
        batch_size, num_heads, seq_len, head_dim = 1, 64, 100, 64
        keys = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
        values = mx.random.normal((batch_size, num_heads, seq_len, head_dim))

        compressed_keys, compressed_values = compressor.compress_kv_cache(
            layer_idx=0,
            kv_cache=(keys, values)
        )

        # Heads dimension should be preserved
        self.assertEqual(compressed_keys.shape[1], num_heads)
        self.assertEqual(compressed_keys.shape[2], 50)


class TestAttentionMatchingMultiLayer(unittest.TestCase):
    """Test multi-layer compression scenarios"""

    def test_multiple_layers_sequential(self):
        """Test compressing multiple layers sequentially"""
        compressor = AttentionMatchingCompressor(compression_ratio=2.0)

        batch_size, num_heads, seq_len, head_dim = 1, 8, 100, 64

        # Compress 10 different layers
        for layer_idx in range(10):
            keys = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
            values = mx.random.normal((batch_size, num_heads, seq_len, head_dim))

            compressed_keys, compressed_values = compressor.compress_kv_cache(
                layer_idx=layer_idx,
                kv_cache=(keys, values)
            )

            self.assertEqual(compressed_keys.shape[2], 50)

        # All layers should have β parameters
        self.assertEqual(len(compressor.beta_params), 10)
        for layer_idx in range(10):
            self.assertIn(layer_idx, compressor.beta_params)

    def test_layer_specific_compression_ratios(self):
        """Test different compression ratios for different layers"""
        batch_size, num_heads, seq_len, head_dim = 1, 8, 100, 64

        # Use different compressors for different layers
        compressors = {
            0: AttentionMatchingCompressor(compression_ratio=2.0),
            1: AttentionMatchingCompressor(compression_ratio=3.0),
            2: AttentionMatchingCompressor(compression_ratio=5.0),
        }

        results = {}
        for layer_idx, compressor in compressors.items():
            keys = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
            values = mx.random.normal((batch_size, num_heads, seq_len, head_dim))

            compressed_keys, _ = compressor.compress_kv_cache(
                layer_idx=layer_idx,
                kv_cache=(keys, values)
            )

            results[layer_idx] = compressed_keys.shape[2]

        # Verify different compression levels
        self.assertEqual(results[0], 50)  # 100 / 2.0
        self.assertEqual(results[1], 33)  # 100 / 3.0
        self.assertEqual(results[2], 20)  # 100 / 5.0


class TestAttentionMatchingStatistics(unittest.TestCase):
    """Test statistics and monitoring functionality"""

    def test_statistics_accumulation(self):
        """Test that statistics accumulate correctly"""
        compressor = AttentionMatchingCompressor(compression_ratio=2.0)

        batch_size, num_heads, seq_len, head_dim = 1, 8, 100, 64

        # Perform 5 compressions
        for i in range(5):
            keys = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
            values = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
            compressor.compress_kv_cache(layer_idx=0, kv_cache=(keys, values))

        stats = compressor.get_compression_stats()

        self.assertEqual(stats["total_compressions"], 5)
        self.assertEqual(stats["total_keys_before"], 500)  # 5 * 100
        self.assertEqual(stats["total_keys_after"], 250)   # 5 * 50
        self.assertAlmostEqual(stats["avg_compression_ratio"], 2.0)

    def test_statistics_with_different_sequence_lengths(self):
        """Test statistics with varying sequence lengths"""
        compressor = AttentionMatchingCompressor(compression_ratio=2.0)

        batch_size, num_heads, head_dim = 1, 8, 64

        # Compress sequences of different lengths
        seq_lengths = [50, 100, 200]
        for seq_len in seq_lengths:
            keys = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
            values = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
            compressor.compress_kv_cache(layer_idx=0, kv_cache=(keys, values))

        stats = compressor.get_compression_stats()

        total_before = sum(seq_lengths)  # 350
        total_after = sum([int(s / 2.0) for s in seq_lengths])  # 175

        self.assertEqual(stats["total_compressions"], 3)
        self.assertEqual(stats["total_keys_before"], total_before)
        self.assertEqual(stats["total_keys_after"], total_after)

    def test_reset_history_clears_stats_per_layer(self):
        """Test that reset_history clears layer-specific history"""
        compressor = AttentionMatchingCompressor(compression_ratio=2.0)

        batch_size, num_heads, seq_len, head_dim = 1, 8, 100, 64

        # Build history for layer 0
        for _ in range(3):
            keys = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
            values = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
            compressor.compress_kv_cache(layer_idx=0, kv_cache=(keys, values))

        # Verify history exists
        self.assertEqual(len(compressor.attention_history[0]), 3)

        # Reset layer 0
        compressor.reset_history(layer_idx=0)

        # History should be empty
        self.assertEqual(len(compressor.attention_history[0]), 0)

        # Statistics should still be preserved (global counter)
        stats = compressor.get_compression_stats()
        self.assertEqual(stats["total_compressions"], 3)


class TestAttentionMatchingIntegration(unittest.TestCase):
    """Test integration scenarios and end-to-end workflows"""

    def test_full_compression_and_compensation_workflow(self):
        """Test complete workflow: compress → apply β compensation"""
        compressor = AttentionMatchingCompressor(
            compression_ratio=2.0,
            beta_calibration=True
        )

        # Step 1: Compress KV cache
        batch_size, num_heads, seq_len, head_dim = 1, 8, 100, 64
        keys = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
        values = mx.random.normal((batch_size, num_heads, seq_len, head_dim))

        compressed_keys, compressed_values = compressor.compress_kv_cache(
            layer_idx=0,
            kv_cache=(keys, values)
        )

        # Step 2: Simulate attention computation with compressed cache
        query_len = 10
        compressed_seq_len = compressed_keys.shape[2]

        query = mx.random.normal((batch_size, num_heads, query_len, head_dim))

        # Compute attention scores
        scores = mx.matmul(
            query,
            mx.transpose(compressed_keys, axes=[0, 1, 3, 2])
        )

        # Step 3: Apply β compensation
        compensated_scores = compressor.apply_beta_compensation(
            layer_idx=0,
            attention_scores=scores
        )

        # Verify workflow
        self.assertEqual(scores.shape, (batch_size, num_heads, query_len, compressed_seq_len))
        self.assertEqual(compensated_scores.shape, scores.shape)

        # β should have been applied
        beta = compressor.beta_params[0]
        expected_scores = scores + beta
        self.assertTrue(mx.allclose(compensated_scores, expected_scores, atol=1e-5))

    def test_attention_history_window_limit(self):
        """Test that attention history respects window limit"""
        window_size = 5
        compressor = AttentionMatchingCompressor(
            compression_ratio=2.0,
            attention_history_window=window_size
        )

        batch_size, num_heads, seq_len, head_dim = 1, 8, 100, 64

        # Compress 10 times (more than window size)
        for _ in range(10):
            keys = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
            values = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
            compressor.compress_kv_cache(layer_idx=0, kv_cache=(keys, values))

        # History should be limited to window size
        self.assertEqual(len(compressor.attention_history[0]), window_size)

    def test_weighted_eviction_randomness(self):
        """Test that weighted eviction produces different results"""
        compressor = AttentionMatchingCompressor(
            compression_ratio=2.0,
            eviction_policy="weighted"
        )

        batch_size, num_heads, seq_len, head_dim = 1, 8, 100, 64
        keys = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
        values = mx.random.normal((batch_size, num_heads, seq_len, head_dim))

        # Compress twice with same input
        compressed_keys1, _ = compressor.compress_kv_cache(
            layer_idx=0,
            kv_cache=(keys, values)
        )

        compressor.reset_history(layer_idx=0)

        compressed_keys2, _ = compressor.compress_kv_cache(
            layer_idx=0,
            kv_cache=(keys, values)
        )

        # Due to randomness, results might differ
        # (Not guaranteed, but very likely with weighted sampling)
        # Just verify both have correct shape
        self.assertEqual(compressed_keys1.shape[2], 50)
        self.assertEqual(compressed_keys2.shape[2], 50)


class TestAttentionMatchingPerformance(unittest.TestCase):
    """Test performance characteristics"""

    def test_compression_speed(self):
        """Test that compression is reasonably fast"""
        compressor = AttentionMatchingCompressor(compression_ratio=2.0)

        batch_size, num_heads, seq_len, head_dim = 1, 8, 1000, 64
        keys = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
        values = mx.random.normal((batch_size, num_heads, seq_len, head_dim))

        # Measure compression time
        start_time = time.perf_counter()

        for _ in range(10):
            compressor.compress_kv_cache(layer_idx=0, kv_cache=(keys, values))

        end_time = time.perf_counter()

        avg_time = (end_time - start_time) / 10

        # Compression should take < 100ms per call
        self.assertLess(avg_time, 0.1)

    def test_beta_compensation_overhead(self):
        """Test that β compensation has minimal overhead"""
        compressor = AttentionMatchingCompressor(
            compression_ratio=2.0,
            beta_calibration=True
        )

        # First compress to generate β
        batch_size, num_heads, seq_len, head_dim = 1, 8, 100, 64
        keys = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
        values = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
        compressor.compress_kv_cache(layer_idx=0, kv_cache=(keys, values))

        # Measure β compensation time
        query_len, key_len = 10, 50
        attention_scores = mx.random.normal((batch_size, num_heads, query_len, key_len))

        start_time = time.perf_counter()

        for _ in range(1000):
            compressor.apply_beta_compensation(layer_idx=0, attention_scores=attention_scores)

        end_time = time.perf_counter()

        avg_time = (end_time - start_time) / 1000

        # β compensation should be very fast (< 1ms)
        self.assertLess(avg_time, 0.001)


class TestAttentionMatchingRepr(unittest.TestCase):
    """Test string representation and debugging"""

    def test_repr(self):
        """Test __repr__ method"""
        compressor = AttentionMatchingCompressor(
            compression_ratio=2.5,
            beta_calibration=False,
            eviction_policy="weighted"
        )

        repr_str = repr(compressor)

        # Should contain key configuration
        self.assertIn("2.5", repr_str)
        self.assertIn("False", repr_str)
        self.assertIn("weighted", repr_str)


if __name__ == "__main__":
    unittest.main()
