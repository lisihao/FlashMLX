"""
Basic tests for AttentionMatchingCompressor (Task #66)

This file contains basic functionality tests to verify the core compression logic.
Full comprehensive tests will be in Task #68.
"""

import unittest
import numpy as np
import mlx.core as mx

from flashmlx.cache import AttentionMatchingCompressor


class TestAttentionMatchingBasic(unittest.TestCase):
    """Basic tests for AttentionMatchingCompressor core functionality"""

    def test_initialization(self):
        """Test compressor initialization with different configurations"""
        # Default configuration
        compressor = AttentionMatchingCompressor()
        self.assertEqual(compressor.compression_ratio, 2.0)
        self.assertTrue(compressor.beta_calibration)
        self.assertEqual(compressor.eviction_policy, "top_k")

        # Custom configuration
        compressor = AttentionMatchingCompressor(
            compression_ratio=3.0,
            beta_calibration=False,
            eviction_policy="weighted"
        )
        self.assertEqual(compressor.compression_ratio, 3.0)
        self.assertFalse(compressor.beta_calibration)
        self.assertEqual(compressor.eviction_policy, "weighted")

    def test_invalid_compression_ratio(self):
        """Test that invalid compression ratio raises error"""
        with self.assertRaises(ValueError):
            AttentionMatchingCompressor(compression_ratio=0.5)

    def test_invalid_eviction_policy(self):
        """Test that invalid eviction policy raises error"""
        with self.assertRaises(ValueError):
            AttentionMatchingCompressor(eviction_policy="invalid")

    def test_basic_compression_shape(self):
        """Test that compression produces correct output shape"""
        compressor = AttentionMatchingCompressor(compression_ratio=2.0)

        # Create mock KV cache
        batch_size, num_heads, seq_len, head_dim = 1, 8, 100, 64
        keys = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
        values = mx.random.normal((batch_size, num_heads, seq_len, head_dim))

        # Compress
        compressed_keys, compressed_values = compressor.compress_kv_cache(
            layer_idx=0,
            kv_cache=(keys, values)
        )

        # Verify shape
        expected_seq_len = int(seq_len / 2.0)
        self.assertEqual(compressed_keys.shape, (batch_size, num_heads, expected_seq_len, head_dim))
        self.assertEqual(compressed_values.shape, (batch_size, num_heads, expected_seq_len, head_dim))

    def test_compression_ratio_3x(self):
        """Test compression with 3x ratio"""
        compressor = AttentionMatchingCompressor(compression_ratio=3.0)

        batch_size, num_heads, seq_len, head_dim = 1, 8, 90, 64
        keys = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
        values = mx.random.normal((batch_size, num_heads, seq_len, head_dim))

        compressed_keys, compressed_values = compressor.compress_kv_cache(
            layer_idx=0,
            kv_cache=(keys, values)
        )

        # 90 / 3.0 = 30
        expected_seq_len = 30
        self.assertEqual(compressed_keys.shape[2], expected_seq_len)
        self.assertEqual(compressed_values.shape[2], expected_seq_len)

    def test_no_compression_when_already_small(self):
        """Test that no compression happens when seq_len already at target"""
        # Use compression_ratio=1.0 to ensure target_seq_len == seq_len
        compressor = AttentionMatchingCompressor(compression_ratio=1.0)

        # With ratio 1.0, target = seq_len, so no compression needed
        batch_size, num_heads, seq_len, head_dim = 1, 8, 10, 64
        keys = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
        values = mx.random.normal((batch_size, num_heads, seq_len, head_dim))

        compressed_keys, compressed_values = compressor.compress_kv_cache(
            layer_idx=0,
            kv_cache=(keys, values)
        )

        # Should return original arrays (no compression)
        self.assertEqual(compressed_keys.shape[2], seq_len)
        self.assertEqual(compressed_values.shape[2], seq_len)

    def test_top_k_eviction_policy(self):
        """Test top-k eviction policy"""
        compressor = AttentionMatchingCompressor(
            compression_ratio=2.0,
            eviction_policy="top_k"
        )

        batch_size, num_heads, seq_len, head_dim = 1, 8, 100, 64
        keys = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
        values = mx.random.normal((batch_size, num_heads, seq_len, head_dim))

        compressed_keys, compressed_values = compressor.compress_kv_cache(
            layer_idx=0,
            kv_cache=(keys, values)
        )

        # Verify compression happened
        self.assertEqual(compressed_keys.shape[2], 50)
        self.assertEqual(compressed_values.shape[2], 50)

    def test_weighted_eviction_policy(self):
        """Test weighted eviction policy"""
        compressor = AttentionMatchingCompressor(
            compression_ratio=2.0,
            eviction_policy="weighted"
        )

        batch_size, num_heads, seq_len, head_dim = 1, 8, 100, 64
        keys = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
        values = mx.random.normal((batch_size, num_heads, seq_len, head_dim))

        compressed_keys, compressed_values = compressor.compress_kv_cache(
            layer_idx=0,
            kv_cache=(keys, values)
        )

        # Verify compression happened
        self.assertEqual(compressed_keys.shape[2], 50)
        self.assertEqual(compressed_values.shape[2], 50)

    def test_multiple_compressions_same_layer(self):
        """Test that multiple compressions for same layer work correctly"""
        compressor = AttentionMatchingCompressor(compression_ratio=2.0)

        batch_size, num_heads, seq_len, head_dim = 1, 8, 100, 64

        # First compression
        keys1 = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
        values1 = mx.random.normal((batch_size, num_heads, seq_len, head_dim))

        compressed_keys1, compressed_values1 = compressor.compress_kv_cache(
            layer_idx=0,
            kv_cache=(keys1, values1)
        )

        # Second compression (same layer, different data)
        keys2 = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
        values2 = mx.random.normal((batch_size, num_heads, seq_len, head_dim))

        compressed_keys2, compressed_values2 = compressor.compress_kv_cache(
            layer_idx=0,
            kv_cache=(keys2, values2)
        )

        # Both should compress correctly
        self.assertEqual(compressed_keys1.shape[2], 50)
        self.assertEqual(compressed_keys2.shape[2], 50)

        # Attention history should be updated
        self.assertIn(0, compressor.attention_history)
        self.assertEqual(len(compressor.attention_history[0]), 2)

    def test_compression_stats(self):
        """Test compression statistics tracking"""
        compressor = AttentionMatchingCompressor(compression_ratio=2.0)

        batch_size, num_heads, seq_len, head_dim = 1, 8, 100, 64
        keys = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
        values = mx.random.normal((batch_size, num_heads, seq_len, head_dim))

        # First compression
        compressor.compress_kv_cache(layer_idx=0, kv_cache=(keys, values))

        stats = compressor.get_compression_stats()

        self.assertEqual(stats["total_compressions"], 1)
        self.assertEqual(stats["total_keys_before"], 100)
        self.assertEqual(stats["total_keys_after"], 50)
        self.assertAlmostEqual(stats["avg_compression_ratio"], 2.0)

    def test_reset_history(self):
        """Test resetting attention history"""
        compressor = AttentionMatchingCompressor()

        batch_size, num_heads, seq_len, head_dim = 1, 8, 100, 64
        keys = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
        values = mx.random.normal((batch_size, num_heads, seq_len, head_dim))

        # Compress to build history
        compressor.compress_kv_cache(layer_idx=0, kv_cache=(keys, values))
        compressor.compress_kv_cache(layer_idx=1, kv_cache=(keys, values))

        # Verify history exists
        self.assertIn(0, compressor.attention_history)
        self.assertIn(1, compressor.attention_history)

        # Reset specific layer
        compressor.reset_history(layer_idx=0)
        self.assertEqual(len(compressor.attention_history[0]), 0)
        self.assertGreater(len(compressor.attention_history[1]), 0)

        # Reset all
        compressor.reset_history()
        self.assertEqual(len(compressor.attention_history), 0)

    def test_mismatched_key_value_shapes(self):
        """Test that mismatched key/value shapes raise error"""
        compressor = AttentionMatchingCompressor()

        keys = mx.random.normal((1, 8, 100, 64))
        values = mx.random.normal((1, 8, 50, 64))  # Different seq_len

        with self.assertRaises(ValueError):
            compressor.compress_kv_cache(layer_idx=0, kv_cache=(keys, values))


if __name__ == "__main__":
    unittest.main()
