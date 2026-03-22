"""
Test Quality Path B.1: Attention-Aware Key Selection

Verifies that select_keys_attention_aware() selects keys based on real attention weights.
"""

import unittest
import mlx.core as mx

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'mlx-lm-source')))

from mlx_lm.compaction.quality import select_keys_attention_aware
from mlx_lm.compaction.base import compute_attention_output, safe_softmax


class TestAttentionAwareSelection(unittest.TestCase):
    """Test attention-aware key selection"""

    def test_selection_basic(self):
        """Test basic attention-aware selection"""
        # Create simple data where certain keys should be selected
        seq_len = 10
        head_dim = 8
        budget = 5

        # Create keys with different patterns
        keys = mx.random.normal((seq_len, head_dim))

        # Create queries that attend to specific keys
        # Query 1 attends to key 3
        # Query 2 attends to key 7
        queries = mx.zeros((2, head_dim))
        queries[0, :] = keys[3, :]  # Identical to key 3
        queries[1, :] = keys[7, :]  # Identical to key 7

        # Select keys
        indices = select_keys_attention_aware(queries, keys, budget)

        # Verify
        self.assertEqual(indices.shape, (budget,))

        # Keys 3 and 7 should be selected (highest attention)
        indices_list = indices.tolist()
        self.assertIn(3, indices_list)
        self.assertIn(7, indices_list)

        # Indices should be sorted
        self.assertEqual(indices_list, sorted(indices_list))

    def test_selection_uniform_attention(self):
        """Test when all keys have equal attention"""
        seq_len = 8
        head_dim = 4
        budget = 4

        # Create keys
        keys = mx.random.normal((seq_len, head_dim))

        # Create random queries
        queries = mx.random.normal((5, head_dim))

        # Select keys
        indices = select_keys_attention_aware(queries, keys, budget)

        # Verify shape
        self.assertEqual(indices.shape, (budget,))

        # Verify sorted
        indices_list = indices.tolist()
        self.assertEqual(indices_list, sorted(indices_list))

        # Verify all indices in range
        for idx in indices_list:
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, seq_len)

    def test_selection_single_query(self):
        """Test with single query"""
        seq_len = 10
        head_dim = 8
        budget = 3

        keys = mx.random.normal((seq_len, head_dim))
        queries = mx.random.normal((1, head_dim))

        indices = select_keys_attention_aware(queries, keys, budget)

        self.assertEqual(indices.shape, (budget,))

    def test_selection_budget_exceeds_seq_len(self):
        """Test when budget >= seq_len (should return all indices)"""
        seq_len = 5
        head_dim = 4
        budget = 10  # Exceeds seq_len

        keys = mx.random.normal((seq_len, head_dim))
        queries = mx.random.normal((3, head_dim))

        indices = select_keys_attention_aware(queries, keys, budget)

        # Should return all indices
        self.assertEqual(indices.shape, (seq_len,))
        self.assertEqual(set(indices.tolist()), set(range(seq_len)))

    def test_attention_computation_consistency(self):
        """
        Test that selected keys actually have high attention.

        Verify that the sum of attention weights for selected keys
        is higher than random selection.
        """
        seq_len = 20
        head_dim = 16
        budget = 8
        scale = head_dim ** 0.5

        keys = mx.random.normal((seq_len, head_dim))
        queries = mx.random.normal((5, head_dim))

        # Select using attention-aware method
        indices = select_keys_attention_aware(queries, keys, budget, scale=scale)

        # Compute attention scores
        scores = (queries @ keys.T) / scale  # (5, 20)
        attn_weights = safe_softmax(scores, axis=1)  # (5, 20)

        # Sum attention for selected keys
        selected_attn = mx.sum(attn_weights[:, indices])

        # Sum attention for random selection (repeat 10 times, take min)
        random_attns = []
        for _ in range(10):
            random_indices = mx.random.randint(0, seq_len, (budget,))
            random_attn = mx.sum(attn_weights[:, random_indices])
            random_attns.append(float(random_attn))

        min_random_attn = min(random_attns)

        # Attention-aware selection should have higher attention sum
        self.assertGreater(float(selected_attn), min_random_attn)

    def test_scale_parameter(self):
        """Test custom scale parameter"""
        seq_len = 8
        head_dim = 4
        budget = 4

        keys = mx.random.normal((seq_len, head_dim))
        queries = mx.random.normal((3, head_dim))

        # Test with different scales
        for scale in [1.0, 2.0, head_dim ** 0.5]:
            indices = select_keys_attention_aware(queries, keys, budget, scale=scale)
            self.assertEqual(indices.shape, (budget,))

    def test_large_scale(self):
        """Test with larger dimensions"""
        seq_len = 100
        head_dim = 64
        budget = 20

        keys = mx.random.normal((seq_len, head_dim))
        queries = mx.random.normal((10, head_dim))

        indices = select_keys_attention_aware(queries, keys, budget)

        self.assertEqual(indices.shape, (budget,))

        # Verify sorted and unique
        indices_list = indices.tolist()
        self.assertEqual(indices_list, sorted(indices_list))
        self.assertEqual(len(indices_list), len(set(indices_list)))


if __name__ == '__main__':
    unittest.main(verbosity=2)
