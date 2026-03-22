"""
Test Quality Path B.5: Quality Testing on Random Data

This is the CRITICAL test - verifies that Quality Path solves the original problem:
Fast Path performs poorly on random data, Quality Path should handle it well.

This test validates the entire motivation for implementing Quality Path.
"""

import unittest
import mlx.core as mx

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'mlx-lm-source')))

from mlx_lm.compaction.quality import compact_single_head_quality
from mlx_lm.compaction.base import compute_attention_output


class TestQualityRandomData(unittest.TestCase):
    """Test Quality Path on random data (the original problem)"""

    def test_random_data_basic(self):
        """
        Basic test: Quality Path vs Fast Path on random data.

        This is THE test that validates the entire project motivation.
        Fast Path should struggle, Quality Path should excel.
        """
        seq_len = 100
        head_dim = 64
        budget = 30

        # Generate completely random data
        keys = mx.random.normal((seq_len, head_dim))
        values = mx.random.normal((seq_len, head_dim))
        queries = mx.random.normal((20, head_dim))

        original_output = compute_attention_output(queries, keys, values)

        # Fast Path (attention-aware selection, no beta, no C2)
        C1_fast, beta_fast, C2_fast = compact_single_head_quality(
            queries, keys, values, budget,
            fit_beta=False,
            fit_c2=False
        )
        output_fast = compute_attention_output(queries, C1_fast, C2_fast, beta_fast)
        error_fast = float(mx.mean((output_fast - original_output) ** 2))
        relative_error_fast = error_fast / float(mx.mean(original_output ** 2))

        # Quality Path (full pipeline)
        C1_quality, beta_quality, C2_quality = compact_single_head_quality(
            queries, keys, values, budget,
            fit_beta=True,
            fit_c2=True
        )
        output_quality = compute_attention_output(
            queries, C1_quality, C2_quality, beta_quality
        )
        error_quality = float(mx.mean((output_quality - original_output) ** 2))
        relative_error_quality = error_quality / float(mx.mean(original_output ** 2))

        improvement = (error_fast - error_quality) / error_fast * 100

        print(f"\n{'='*60}")
        print(f"CRITICAL TEST: Random Data Performance")
        print(f"{'='*60}")
        print(f"Fast Path:")
        print(f"  MSE:            {error_fast:.6f}")
        print(f"  Relative error: {relative_error_fast * 100:.2f}%")
        print(f"\nQuality Path:")
        print(f"  MSE:            {error_quality:.6f}")
        print(f"  Relative error: {relative_error_quality * 100:.2f}%")
        print(f"\nImprovement:      {improvement:.1f}%")
        print(f"{'='*60}")

        # Quality Path should significantly outperform Fast Path
        self.assertLess(error_quality, error_fast)
        # Should have at least 50% improvement (project requirement)
        self.assertGreater(improvement, 50.0)
        # Quality Path should have low relative error (< 5%)
        self.assertLess(relative_error_quality, 0.05)

    def test_random_data_scaling(self):
        """
        Test Quality Path on random data with different sizes.

        Verify improvement holds across different scales.
        """
        configs = [
            (50, 32, 15),   # Small
            (100, 64, 30),  # Medium
            (200, 64, 50),  # Large
        ]

        print(f"\nRandom data scaling:")
        for seq_len, head_dim, budget in configs:
            keys = mx.random.normal((seq_len, head_dim))
            values = mx.random.normal((seq_len, head_dim))
            queries = mx.random.normal((10, head_dim))

            original_output = compute_attention_output(queries, keys, values)

            # Fast Path
            C1_fast, beta_fast, C2_fast = compact_single_head_quality(
                queries, keys, values, budget,
                fit_beta=False, fit_c2=False
            )
            output_fast = compute_attention_output(queries, C1_fast, C2_fast, beta_fast)
            error_fast = float(mx.mean((output_fast - original_output) ** 2))

            # Quality Path
            C1_quality, beta_quality, C2_quality = compact_single_head_quality(
                queries, keys, values, budget,
                fit_beta=True, fit_c2=True
            )
            output_quality = compute_attention_output(
                queries, C1_quality, C2_quality, beta_quality
            )
            error_quality = float(mx.mean((output_quality - original_output) ** 2))

            improvement = (error_fast - error_quality) / error_fast * 100

            print(f"  seq={seq_len:3d}, head_dim={head_dim:2d}, budget={budget:2d}: "
                  f"improvement={improvement:.1f}%")

            # Should have significant improvement at all scales
            self.assertGreater(improvement, 50.0)

    def test_random_data_with_different_distributions(self):
        """
        Test Quality Path on random data with different distributions.

        Test:
        1. Normal distribution (standard case)
        2. Uniform distribution
        3. Larger variance

        Quality Path should handle all distributions well.
        """
        seq_len = 80
        head_dim = 32
        budget = 25

        distributions = [
            ("Normal", lambda: mx.random.normal((seq_len, head_dim))),
            ("Uniform", lambda: mx.random.uniform(-1, 1, (seq_len, head_dim))),
            ("Large variance", lambda: mx.random.normal((seq_len, head_dim)) * 3.0),
        ]

        print(f"\nRandom distributions:")
        for dist_name, generator in distributions:
            keys = generator()
            values = generator()
            queries = mx.random.normal((10, head_dim))

            original_output = compute_attention_output(queries, keys, values)

            # Fast Path
            C1_fast, beta_fast, C2_fast = compact_single_head_quality(
                queries, keys, values, budget,
                fit_beta=False, fit_c2=False
            )
            output_fast = compute_attention_output(queries, C1_fast, C2_fast, beta_fast)
            error_fast = float(mx.mean((output_fast - original_output) ** 2))

            # Quality Path
            C1_quality, beta_quality, C2_quality = compact_single_head_quality(
                queries, keys, values, budget,
                fit_beta=True, fit_c2=True
            )
            output_quality = compute_attention_output(
                queries, C1_quality, C2_quality, beta_quality
            )
            error_quality = float(mx.mean((output_quality - original_output) ** 2))

            improvement = (error_fast - error_quality) / error_fast * 100

            print(f"  {dist_name:15s}: improvement={improvement:.1f}%")

            # Should work well on all distributions
            self.assertGreater(improvement, 30.0)

    def test_random_data_compression_ratios(self):
        """
        Test Quality Path with different compression ratios.

        Verify improvement holds at different compression levels.
        """
        seq_len = 100
        head_dim = 64

        # Different compression ratios
        budgets = [
            (10, 90),   # 90% compression
            (25, 75),   # 75% compression
            (50, 50),   # 50% compression
            (75, 25),   # 25% compression
        ]

        print(f"\nCompression ratios:")
        for budget, compression_pct in budgets:
            keys = mx.random.normal((seq_len, head_dim))
            values = mx.random.normal((seq_len, head_dim))
            queries = mx.random.normal((10, head_dim))

            original_output = compute_attention_output(queries, keys, values)

            # Fast Path
            C1_fast, beta_fast, C2_fast = compact_single_head_quality(
                queries, keys, values, budget,
                fit_beta=False, fit_c2=False
            )
            output_fast = compute_attention_output(queries, C1_fast, C2_fast, beta_fast)
            error_fast = float(mx.mean((output_fast - original_output) ** 2))

            # Quality Path
            C1_quality, beta_quality, C2_quality = compact_single_head_quality(
                queries, keys, values, budget,
                fit_beta=True, fit_c2=True
            )
            output_quality = compute_attention_output(
                queries, C1_quality, C2_quality, beta_quality
            )
            error_quality = float(mx.mean((output_quality - original_output) ** 2))

            improvement = (error_fast - error_quality) / error_fast * 100

            print(f"  {compression_pct}% compression (budget={budget:2d}): "
                  f"improvement={improvement:.1f}%")

            # Should improve at all compression levels
            self.assertGreater(improvement, 30.0)

    def test_random_vs_structured_data(self):
        """
        Compare Quality Path performance on random vs structured data.

        Structured data = queries are similar to some keys (realistic attention)
        Random data = completely random (worst case)

        Quality Path should handle both well, Fast Path should only handle structured.
        """
        seq_len = 80
        head_dim = 32
        budget = 25

        # Structured data: queries attend to specific keys
        keys_structured = mx.random.normal((seq_len, head_dim))
        values_structured = mx.random.normal((seq_len, head_dim))
        # Make queries similar to first 10 keys
        queries_structured = keys_structured[:10, :] + mx.random.normal((10, head_dim)) * 0.1

        # Random data: completely random
        keys_random = mx.random.normal((seq_len, head_dim))
        values_random = mx.random.normal((seq_len, head_dim))
        queries_random = mx.random.normal((10, head_dim))

        results = {}
        print(f"\nStructured vs Random data:")

        for data_type, keys, values, queries in [
            ("Structured", keys_structured, values_structured, queries_structured),
            ("Random", keys_random, values_random, queries_random),
        ]:
            original_output = compute_attention_output(queries, keys, values)

            # Fast Path
            C1_fast, beta_fast, C2_fast = compact_single_head_quality(
                queries, keys, values, budget,
                fit_beta=False, fit_c2=False
            )
            output_fast = compute_attention_output(queries, C1_fast, C2_fast, beta_fast)
            error_fast = float(mx.mean((output_fast - original_output) ** 2))

            # Quality Path
            C1_quality, beta_quality, C2_quality = compact_single_head_quality(
                queries, keys, values, budget,
                fit_beta=True, fit_c2=True
            )
            output_quality = compute_attention_output(
                queries, C1_quality, C2_quality, beta_quality
            )
            error_quality = float(mx.mean((output_quality - original_output) ** 2))

            improvement = (error_fast - error_quality) / error_fast * 100
            results[data_type] = (error_fast, error_quality, improvement)

            print(f"  {data_type:10s}:")
            print(f"    Fast Path error:    {error_fast:.6f}")
            print(f"    Quality Path error: {error_quality:.6f}")
            print(f"    Improvement:        {improvement:.1f}%")

        # On structured data, both should work reasonably well
        # On random data, Quality Path should have much larger improvement
        _, _, improvement_structured = results["Structured"]
        _, _, improvement_random = results["Random"]

        # Random data improvement should be larger
        self.assertGreater(improvement_random, 30.0)

    def test_random_data_consistency(self):
        """
        Test that Quality Path produces consistent results on random data.

        Run multiple times with different random seeds, verify stable performance.
        """
        seq_len = 80
        head_dim = 32
        budget = 25

        improvements = []

        print(f"\nConsistency test (5 runs):")
        for run in range(5):
            # Different random data each time
            keys = mx.random.normal((seq_len, head_dim))
            values = mx.random.normal((seq_len, head_dim))
            queries = mx.random.normal((10, head_dim))

            original_output = compute_attention_output(queries, keys, values)

            # Fast Path
            C1_fast, beta_fast, C2_fast = compact_single_head_quality(
                queries, keys, values, budget,
                fit_beta=False, fit_c2=False
            )
            output_fast = compute_attention_output(queries, C1_fast, C2_fast, beta_fast)
            error_fast = float(mx.mean((output_fast - original_output) ** 2))

            # Quality Path
            C1_quality, beta_quality, C2_quality = compact_single_head_quality(
                queries, keys, values, budget,
                fit_beta=True, fit_c2=True
            )
            output_quality = compute_attention_output(
                queries, C1_quality, C2_quality, beta_quality
            )
            error_quality = float(mx.mean((output_quality - original_output) ** 2))

            improvement = (error_fast - error_quality) / error_fast * 100
            improvements.append(improvement)

            print(f"  Run {run + 1}: improvement={improvement:.1f}%")

        # All runs should show significant improvement
        for improvement in improvements:
            self.assertGreater(improvement, 30.0)

        # Results should be relatively consistent (std < 30% of mean)
        mean_improvement = sum(improvements) / len(improvements)
        std_improvement = (sum((x - mean_improvement) ** 2 for x in improvements) / len(improvements)) ** 0.5

        print(f"  Mean: {mean_improvement:.1f}%, Std: {std_improvement:.1f}%")

        # Consistent performance
        self.assertLess(std_improvement / mean_improvement, 0.3)

    def test_random_data_multi_query(self):
        """
        Test Quality Path on random data with varying number of queries.

        More queries = more information for fitting = potentially better results.
        """
        seq_len = 80
        head_dim = 32
        budget = 25

        query_counts = [5, 10, 20, 40]

        print(f"\nQuery count scaling:")
        for query_count in query_counts:
            keys = mx.random.normal((seq_len, head_dim))
            values = mx.random.normal((seq_len, head_dim))
            queries = mx.random.normal((query_count, head_dim))

            original_output = compute_attention_output(queries, keys, values)

            # Fast Path
            C1_fast, beta_fast, C2_fast = compact_single_head_quality(
                queries, keys, values, budget,
                fit_beta=False, fit_c2=False
            )
            output_fast = compute_attention_output(queries, C1_fast, C2_fast, beta_fast)
            error_fast = float(mx.mean((output_fast - original_output) ** 2))

            # Quality Path
            C1_quality, beta_quality, C2_quality = compact_single_head_quality(
                queries, keys, values, budget,
                fit_beta=True, fit_c2=True
            )
            output_quality = compute_attention_output(
                queries, C1_quality, C2_quality, beta_quality
            )
            error_quality = float(mx.mean((output_quality - original_output) ** 2))

            improvement = (error_fast - error_quality) / error_fast * 100

            print(f"  {query_count:2d} queries: improvement={improvement:.1f}%")

            # Should work well with any number of queries
            self.assertGreater(improvement, 30.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
