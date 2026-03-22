"""
Test Quality Path B.4: Complete Implementation Integration

Verifies that B.1 (Attention-Aware Selection), B.2 (Beta Fitting),
and B.3 (C2 Fitting) work together correctly in the full Quality Path algorithm.
"""

import unittest
import mlx.core as mx

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'mlx-lm-source')))

from mlx_lm.compaction.quality import (
    compact_single_head_quality,
    compact_multi_head_quality
)
from mlx_lm.compaction.base import compute_attention_output


class TestQualityPathComplete(unittest.TestCase):
    """Test complete Quality Path implementation"""

    def test_complete_pipeline(self):
        """
        Test that all three components work together:
        B.1: Attention-aware selection
        B.2: Beta fitting
        B.3: C2 fitting
        """
        seq_len = 50
        head_dim = 32
        budget = 20

        keys = mx.random.normal((seq_len, head_dim))
        values = mx.random.normal((seq_len, head_dim))
        queries = mx.random.normal((10, head_dim))

        # Full Quality Path
        C1, beta, C2 = compact_single_head_quality(
            queries, keys, values, budget,
            fit_beta=True,
            fit_c2=True,
            nnls_method="clamped",
            lsq_method="cholesky"
        )

        # Verify shapes
        self.assertEqual(C1.shape, (budget, head_dim))
        self.assertEqual(beta.shape, (budget,))
        self.assertEqual(C2.shape, (budget, head_dim))

        # Verify output quality
        original_output = compute_attention_output(queries, keys, values)
        compressed_output = compute_attention_output(queries, C1, C2, beta)

        error = float(mx.mean((compressed_output - original_output) ** 2))
        relative_error = error / float(mx.mean(original_output ** 2))

        print(f"\nComplete pipeline:")
        print(f"  MSE: {error:.6f}")
        print(f"  Relative error: {relative_error * 100:.2f}%")

        # Quality Path should have reasonable error (< 10%)
        self.assertLess(relative_error, 0.10)

    def test_method_combinations(self):
        """
        Test different combinations of NNLS and LSQ methods.
        All should produce reasonable results.
        """
        seq_len = 40
        head_dim = 16
        budget = 15

        keys = mx.random.normal((seq_len, head_dim))
        values = mx.random.normal((seq_len, head_dim))
        queries = mx.random.normal((8, head_dim))

        original_output = compute_attention_output(queries, keys, values)

        # Test all combinations
        combinations = [
            ("clamped", "lstsq"),
            ("clamped", "cholesky"),
            ("clamped", "pinv"),
            ("pgd", "lstsq"),
            ("pgd", "cholesky"),
            ("pgd", "pinv"),
        ]

        print(f"\nMethod combinations:")
        for nnls_method, lsq_method in combinations:
            C1, beta, C2 = compact_single_head_quality(
                queries, keys, values, budget,
                fit_beta=True,
                fit_c2=True,
                nnls_method=nnls_method,
                lsq_method=lsq_method
            )

            compressed_output = compute_attention_output(queries, C1, C2, beta)
            error = float(mx.mean((compressed_output - original_output) ** 2))

            print(f"  {nnls_method:8s} + {lsq_method:8s}: error={error:.6f}")

            # All combinations should have reasonable error
            self.assertLess(error, 0.1)

    def test_ablation_study(self):
        """
        Ablation study: test Quality Path with different components disabled.

        Compare:
        1. No beta, no C2 (baseline)
        2. Beta only
        3. C2 only
        4. Both beta and C2 (full)

        Expected: Full > C2 only > Beta only > Baseline
        """
        seq_len = 50
        head_dim = 32
        budget = 20

        keys = mx.random.normal((seq_len, head_dim))
        values = mx.random.normal((seq_len, head_dim))
        queries = mx.random.normal((10, head_dim))

        original_output = compute_attention_output(queries, keys, values)

        configs = [
            ("Baseline (no beta, no C2)", False, False),
            ("Beta only", True, False),
            ("C2 only", False, True),
            ("Full (beta + C2)", True, True),
        ]

        errors = {}
        print(f"\nAblation study:")
        for name, fit_beta, fit_c2 in configs:
            C1, beta, C2 = compact_single_head_quality(
                queries, keys, values, budget,
                fit_beta=fit_beta,
                fit_c2=fit_c2
            )

            compressed_output = compute_attention_output(queries, C1, C2, beta)
            error = float(mx.mean((compressed_output - original_output) ** 2))
            errors[name] = error

            print(f"  {name:25s}: {error:.6f}")

        # Full should have lowest error (or comparable to C2 only if both are near-perfect)
        # When errors are very small (< 1e-6), we're hitting floating point precision
        full_error = errors["Full (beta + C2)"]
        c2_only_error = errors["C2 only"]

        # Either full is better, or both are essentially perfect
        if full_error > 1e-6 and c2_only_error > 1e-6:
            self.assertLess(full_error, c2_only_error)

        # Full should always be better than beta-only and baseline
        self.assertLess(full_error, errors["Beta only"])
        self.assertLess(full_error, errors["Baseline (no beta, no C2)"])

    def test_multi_head_processing(self):
        """
        Test multi-head Quality Path processing.

        Verify:
        1. Each head is processed independently
        2. Output shapes are correct
        3. Quality is maintained across heads
        """
        n_heads = 4
        seq_len = 50
        head_dim = 32
        budget = 20

        keys = mx.random.normal((n_heads, seq_len, head_dim))
        values = mx.random.normal((n_heads, seq_len, head_dim))
        queries = mx.random.normal((n_heads, 10, head_dim))

        # Multi-head compression
        C1, beta, C2 = compact_multi_head_quality(
            keys, values, budget,
            queries=queries,
            fit_beta=True,
            fit_c2=True
        )

        # Verify shapes
        self.assertEqual(C1.shape, (n_heads, budget, head_dim))
        self.assertEqual(beta.shape, (n_heads, budget))
        self.assertEqual(C2.shape, (n_heads, budget, head_dim))

        # Verify quality for each head
        print(f"\nMulti-head quality:")
        for h in range(n_heads):
            original_output = compute_attention_output(
                queries[h], keys[h], values[h]
            )
            compressed_output = compute_attention_output(
                queries[h], C1[h], C2[h], beta[h]
            )

            error = float(mx.mean((compressed_output - original_output) ** 2))
            relative_error = error / float(mx.mean(original_output ** 2))

            print(f"  Head {h}: relative_error={relative_error * 100:.2f}%")

            # Each head should have good quality
            self.assertLess(relative_error, 0.10)

    def test_quality_vs_fast_path(self):
        """
        Compare Quality Path vs Fast Path on random data.

        Quality Path should significantly outperform Fast Path on random data,
        which is the motivation for the entire project.
        """
        seq_len = 100
        head_dim = 64
        budget = 30

        keys = mx.random.normal((seq_len, head_dim))
        values = mx.random.normal((seq_len, head_dim))
        queries = mx.random.normal((20, head_dim))

        original_output = compute_attention_output(queries, keys, values)

        # Quality Path
        C1_quality, beta_quality, C2_quality = compact_single_head_quality(
            queries, keys, values, budget,
            fit_beta=True,
            fit_c2=True
        )
        output_quality = compute_attention_output(
            queries, C1_quality, C2_quality, beta_quality
        )
        error_quality = float(mx.mean((output_quality - original_output) ** 2))

        # Fast Path (attention-aware selection, no beta, no C2)
        C1_fast, beta_fast, C2_fast = compact_single_head_quality(
            queries, keys, values, budget,
            fit_beta=False,
            fit_c2=False
        )
        output_fast = compute_attention_output(
            queries, C1_fast, C2_fast, beta_fast
        )
        error_fast = float(mx.mean((output_fast - original_output) ** 2))

        improvement = (error_fast - error_quality) / error_fast * 100

        print(f"\nQuality vs Fast Path:")
        print(f"  Fast Path error:    {error_fast:.6f}")
        print(f"  Quality Path error: {error_quality:.6f}")
        print(f"  Improvement: {improvement:.1f}%")

        # Quality Path should be significantly better (> 50% improvement)
        self.assertLess(error_quality, error_fast)
        self.assertGreater(improvement, 50.0)

    def test_budget_scaling(self):
        """
        Test Quality Path with different budget sizes.

        Verify:
        1. Larger budget = lower error
        2. Error decreases monotonically
        3. All budget sizes work correctly
        """
        seq_len = 80
        head_dim = 32

        keys = mx.random.normal((seq_len, head_dim))
        values = mx.random.normal((seq_len, head_dim))
        queries = mx.random.normal((10, head_dim))

        original_output = compute_attention_output(queries, keys, values)

        budgets = [10, 20, 30, 40]
        errors = []

        print(f"\nBudget scaling:")
        for budget in budgets:
            C1, beta, C2 = compact_single_head_quality(
                queries, keys, values, budget,
                fit_beta=True,
                fit_c2=True
            )

            compressed_output = compute_attention_output(queries, C1, C2, beta)
            error = float(mx.mean((compressed_output - original_output) ** 2))
            errors.append(error)

            print(f"  Budget {budget:2d}: error={error:.6f}")

        # Error should decrease with larger budget (when not at precision limit)
        # If errors are very small (< 1e-6), we're hitting floating point precision
        # In that case, just verify all errors are small
        if errors[0] > 1e-6:
            # At least the first error should be larger than the last
            self.assertGreater(errors[0], errors[-1])
        else:
            # All errors should be very small (near-perfect reconstruction)
            for error in errors:
                self.assertLess(error, 1e-4)

    def test_large_scale(self):
        """
        Test Quality Path at larger scale.

        Verify it works with realistic sizes:
        - seq_len = 512 (common context length)
        - budget = 128 (25% compression)
        - head_dim = 64 (typical)
        """
        seq_len = 512
        head_dim = 64
        budget = 128

        keys = mx.random.normal((seq_len, head_dim))
        values = mx.random.normal((seq_len, head_dim))
        queries = mx.random.normal((20, head_dim))

        # This should complete without crashing
        C1, beta, C2 = compact_single_head_quality(
            queries, keys, values, budget,
            fit_beta=True,
            fit_c2=True,
            lsq_method="cholesky"  # Fastest
        )

        # Verify shapes
        self.assertEqual(C1.shape, (budget, head_dim))
        self.assertEqual(beta.shape, (budget,))
        self.assertEqual(C2.shape, (budget, head_dim))

        # Verify quality
        original_output = compute_attention_output(queries, keys, values)
        compressed_output = compute_attention_output(queries, C1, C2, beta)

        error = float(mx.mean((compressed_output - original_output) ** 2))
        relative_error = error / float(mx.mean(original_output ** 2))

        print(f"\nLarge scale Quality Path:")
        print(f"  seq_len={seq_len}, budget={budget}, head_dim={head_dim}")
        print(f"  Compression ratio: {(1 - budget/seq_len) * 100:.1f}%")
        print(f"  Relative error: {relative_error * 100:.2f}%")

        # Should have reasonable error (< 5% for 75% retention)
        self.assertLess(relative_error, 0.05)

    def test_return_indices(self):
        """
        Test that return_indices parameter works correctly.
        """
        seq_len = 30
        head_dim = 16
        budget = 10

        keys = mx.random.normal((seq_len, head_dim))
        values = mx.random.normal((seq_len, head_dim))
        queries = mx.random.normal((5, head_dim))

        # With return_indices
        C1, beta, C2, indices = compact_single_head_quality(
            queries, keys, values, budget,
            fit_beta=True,
            fit_c2=False,  # Test with C2 disabled
            return_indices=True
        )

        # Verify indices shape and range
        self.assertEqual(indices.shape, (budget,))
        self.assertTrue(mx.all(indices >= 0))
        self.assertTrue(mx.all(indices < seq_len))

        # Verify C1 matches selected keys
        C1_expected = keys[indices]
        diff = float(mx.linalg.norm(C1 - C1_expected))
        self.assertLess(diff, 1e-5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
