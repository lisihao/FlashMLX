"""
Tests for β calibration mechanism (Task #67)

Tests the beta calibration functionality in AttentionMatchingCompressor.
"""

import unittest
import numpy as np
import mlx.core as mx

from flashmlx.cache import AttentionMatchingCompressor


class TestBetaCalibration(unittest.TestCase):
    """Tests for β calibration mechanism"""

    def test_beta_calibration_enabled(self):
        """Test that β calibration is computed when enabled"""
        compressor = AttentionMatchingCompressor(
            compression_ratio=2.0,
            beta_calibration=True
        )

        batch_size, num_heads, seq_len, head_dim = 1, 8, 100, 64
        keys = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
        values = mx.random.normal((batch_size, num_heads, seq_len, head_dim))

        # Compress
        compressor.compress_kv_cache(layer_idx=0, kv_cache=(keys, values))

        # β should be computed and stored
        self.assertIn(0, compressor.beta_params)
        beta = compressor.beta_params[0]

        # β should be a finite number
        self.assertTrue(np.isfinite(beta))

    def test_beta_calibration_disabled(self):
        """Test that β calibration is skipped when disabled"""
        compressor = AttentionMatchingCompressor(
            compression_ratio=2.0,
            beta_calibration=False
        )

        batch_size, num_heads, seq_len, head_dim = 1, 8, 100, 64
        keys = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
        values = mx.random.normal((batch_size, num_heads, seq_len, head_dim))

        # Compress
        compressor.compress_kv_cache(layer_idx=0, kv_cache=(keys, values))

        # β should not be computed
        self.assertNotIn(0, compressor.beta_params)

    def test_beta_value_range(self):
        """Test that β values are in reasonable range"""
        compressor = AttentionMatchingCompressor(
            compression_ratio=2.0,
            beta_calibration=True
        )

        batch_size, num_heads, seq_len, head_dim = 1, 8, 100, 64
        keys = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
        values = mx.random.normal((batch_size, num_heads, seq_len, head_dim))

        # Compress
        compressor.compress_kv_cache(layer_idx=0, kv_cache=(keys, values))

        beta = compressor.beta_params[0]

        # For 2x compression, β should be roughly log(2) ≈ 0.693
        # Allow some variation due to distribution effects
        self.assertGreater(beta, 0.0)
        self.assertLess(beta, 2.0)

    def test_beta_increases_with_compression_ratio(self):
        """Test that β increases with higher compression ratios"""
        batch_size, num_heads, seq_len, head_dim = 1, 8, 100, 64
        keys = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
        values = mx.random.normal((batch_size, num_heads, seq_len, head_dim))

        betas = []
        for ratio in [1.5, 2.0, 3.0, 5.0]:
            compressor = AttentionMatchingCompressor(
                compression_ratio=ratio,
                beta_calibration=True
            )

            compressor.compress_kv_cache(layer_idx=0, kv_cache=(keys, values))
            betas.append(compressor.beta_params[0])

        # β should increase with compression ratio
        for i in range(len(betas) - 1):
            self.assertLess(betas[i], betas[i + 1])

    def test_apply_beta_compensation_shape(self):
        """Test that β compensation preserves shape"""
        compressor = AttentionMatchingCompressor(
            compression_ratio=2.0,
            beta_calibration=True
        )

        # First, compress to generate β
        batch_size, num_heads, seq_len, head_dim = 1, 8, 100, 64
        keys = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
        values = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
        compressor.compress_kv_cache(layer_idx=0, kv_cache=(keys, values))

        # Now test β compensation on attention scores
        query_len, key_len = 10, 50
        attention_scores = mx.random.normal((batch_size, num_heads, query_len, key_len))

        compensated_scores = compressor.apply_beta_compensation(
            layer_idx=0,
            attention_scores=attention_scores
        )

        # Shape should be preserved
        self.assertEqual(compensated_scores.shape, attention_scores.shape)

    def test_apply_beta_compensation_without_calibration(self):
        """Test that β compensation is no-op when no β available"""
        compressor = AttentionMatchingCompressor(
            compression_ratio=2.0,
            beta_calibration=True
        )

        # Apply compensation without compression (no β available)
        batch_size, num_heads, query_len, key_len = 1, 8, 10, 50
        attention_scores = mx.random.normal((batch_size, num_heads, query_len, key_len))

        compensated_scores = compressor.apply_beta_compensation(
            layer_idx=0,  # No β for layer 0 yet
            attention_scores=attention_scores
        )

        # Should return unchanged scores
        self.assertTrue(mx.allclose(compensated_scores, attention_scores))

    def test_apply_beta_compensation_shifts_distribution(self):
        """Test that β compensation actually shifts the scores"""
        compressor = AttentionMatchingCompressor(
            compression_ratio=2.0,
            beta_calibration=True
        )

        # Compress to generate β
        batch_size, num_heads, seq_len, head_dim = 1, 8, 100, 64
        keys = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
        values = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
        compressor.compress_kv_cache(layer_idx=0, kv_cache=(keys, values))

        beta = compressor.beta_params[0]

        # Apply compensation
        query_len, key_len = 10, 50
        attention_scores = mx.random.normal((batch_size, num_heads, query_len, key_len))

        compensated_scores = compressor.apply_beta_compensation(
            layer_idx=0,
            attention_scores=attention_scores
        )

        # Compensated scores should be original + β
        expected_scores = attention_scores + beta
        self.assertTrue(mx.allclose(compensated_scores, expected_scores, atol=1e-5))

    def test_beta_per_layer_independence(self):
        """Test that different layers have independent β values"""
        compressor = AttentionMatchingCompressor(
            compression_ratio=2.0,
            beta_calibration=True
        )

        batch_size, num_heads, seq_len, head_dim = 1, 8, 100, 64

        # Compress two different layers
        keys1 = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
        values1 = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
        compressor.compress_kv_cache(layer_idx=0, kv_cache=(keys1, values1))

        keys2 = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
        values2 = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
        compressor.compress_kv_cache(layer_idx=1, kv_cache=(keys2, values2))

        # Both layers should have β
        self.assertIn(0, compressor.beta_params)
        self.assertIn(1, compressor.beta_params)

        # β values might be different (due to different weight distributions)
        # Just verify they're both finite
        self.assertTrue(np.isfinite(compressor.beta_params[0]))
        self.assertTrue(np.isfinite(compressor.beta_params[1]))

    def test_beta_stability_across_compressions(self):
        """Test that β remains stable across multiple compressions"""
        compressor = AttentionMatchingCompressor(
            compression_ratio=2.0,
            beta_calibration=True
        )

        batch_size, num_heads, seq_len, head_dim = 1, 8, 100, 64

        betas = []
        for _ in range(5):
            keys = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
            values = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
            compressor.compress_kv_cache(layer_idx=0, kv_cache=(keys, values))
            betas.append(compressor.beta_params[0])

        # β should be relatively stable (within 50% of mean)
        mean_beta = np.mean(betas)
        for beta in betas:
            self.assertGreater(beta, mean_beta * 0.5)
            self.assertLess(beta, mean_beta * 1.5)


if __name__ == "__main__":
    unittest.main()
