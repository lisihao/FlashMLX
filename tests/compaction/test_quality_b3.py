"""
Test Quality Path B.3: LSQ C2 Fitting

Verifies that C2 fitting improves attention output matching.
"""

import unittest
import mlx.core as mx

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'mlx-lm-source')))

from mlx_lm.compaction.quality import compact_single_head_quality
from mlx_lm.compaction.base import compute_attention_output


class TestLSQC2Fitting(unittest.TestCase):
    """Test LSQ C2 fitting"""

    def test_c2_improves_output_match(self):
        """
        Test that C2 fitting improves attention output matching.

        Compare:
        - Direct C2 (no fitting): C2 = V[indices]
        - Fitted C2: C2 = LSQ solution

        Expected: fitted C2 should reduce output error.
        """
        seq_len = 50
        head_dim = 32
        budget = 20
        scale = head_dim ** 0.5

        # Create data
        keys = mx.random.normal((seq_len, head_dim))
        values = mx.random.normal((seq_len, head_dim))
        queries = mx.random.normal((10, head_dim))

        # Compress WITHOUT C2 fitting
        C1_no_fit, beta_no_fit, C2_no_fit = compact_single_head_quality(
            queries, keys, values, budget, scale,
            fit_beta=True,
            fit_c2=False  # Disabled
        )

        # Compute output error without C2 fitting
        output_no_fit = compute_attention_output(queries, C1_no_fit, C2_no_fit, beta_no_fit, scale)
        original_output = compute_attention_output(queries, keys, values, beta=None, scale=scale)
        error_no_fit = float(mx.mean((output_no_fit - original_output) ** 2))

        # Compress WITH C2 fitting
        C1_fit, beta_fit, C2_fit = compact_single_head_quality(
            queries, keys, values, budget, scale,
            fit_beta=True,
            fit_c2=True,  # Enabled
            lsq_method="lstsq"
        )

        # Compute output error with C2 fitting
        output_fit = compute_attention_output(queries, C1_fit, C2_fit, beta_fit, scale)
        error_fit = float(mx.mean((output_fit - original_output) ** 2))

        print(f"\nC2 fitting effect:")
        print(f"  Error without C2 fit: {error_no_fit:.6f}")
        print(f"  Error with C2 fit: {error_fit:.6f}")
        improvement = (error_no_fit - error_fit) / error_no_fit * 100
        print(f"  Improvement: {improvement:.1f}%")

        # C2 fitting should improve output matching
        self.assertLess(error_fit, error_no_fit)
        # Should have at least 10% improvement
        self.assertGreater(improvement, 10.0)

    def test_lsq_methods_comparison(self):
        """
        Compare three LSQ methods: lstsq, cholesky, pinv.

        All should produce similar results.
        """
        seq_len = 40
        head_dim = 16
        budget = 15

        keys = mx.random.normal((seq_len, head_dim))
        values = mx.random.normal((seq_len, head_dim))
        queries = mx.random.normal((8, head_dim))

        methods = ["lstsq", "cholesky", "pinv"]
        C2_results = {}
        errors = {}

        original_output = compute_attention_output(queries, keys, values)

        for method in methods:
            C1, beta, C2 = compact_single_head_quality(
                queries, keys, values, budget,
                fit_beta=True, fit_c2=True,
                lsq_method=method
            )
            C2_results[method] = C2

            # Compute error
            output = compute_attention_output(queries, C1, C2, beta)
            error = float(mx.mean((output - original_output) ** 2))
            errors[method] = error

        print(f"\nLSQ method comparison:")
        for method in methods:
            print(f"  {method}: error={errors[method]:.6f}")

        # All methods should produce low error
        for method in methods:
            self.assertLess(errors[method], 0.1)

        # Methods should produce similar results (within 50% relative difference)
        lstsq_norm = float(mx.linalg.norm(C2_results["lstsq"]))
        for method in ["cholesky", "pinv"]:
            diff = float(mx.linalg.norm(C2_results[method] - C2_results["lstsq"]))
            relative_diff = diff / lstsq_norm
            print(f"  {method} vs lstsq relative diff: {relative_diff * 100:.1f}%")
            self.assertLess(relative_diff, 0.5)

    def test_c2_no_fitting(self):
        """
        Test that disabling C2 fitting returns direct copy.
        """
        seq_len = 30
        head_dim = 16
        budget = 10

        keys = mx.random.normal((seq_len, head_dim))
        values = mx.random.normal((seq_len, head_dim))
        queries = mx.random.normal((5, head_dim))

        # Get indices from attention-aware selection
        from mlx_lm.compaction.quality import select_keys_attention_aware
        indices = select_keys_attention_aware(queries, keys, budget)

        # Compress without C2 fitting
        _, _, C2 = compact_single_head_quality(
            queries, keys, values, budget,
            fit_beta=False,
            fit_c2=False
        )

        # C2 should be direct copy of V[indices]
        C2_expected = values[indices]

        # Should be identical
        diff = float(mx.linalg.norm(C2 - C2_expected))
        print(f"\nC2 no fitting diff: {diff:.10f}")

        self.assertLess(diff, 1e-5)

    def test_c2_quality_with_budget_variation(self):
        """
        Test C2 fitting quality with different budget sizes.

        Larger budget should have lower error.
        """
        seq_len = 60
        head_dim = 32

        keys = mx.random.normal((seq_len, head_dim))
        values = mx.random.normal((seq_len, head_dim))
        queries = mx.random.normal((10, head_dim))

        original_output = compute_attention_output(queries, keys, values)

        budgets = [10, 20, 30]
        errors = []

        for budget in budgets:
            C1, beta, C2 = compact_single_head_quality(
                queries, keys, values, budget,
                fit_beta=True, fit_c2=True,
                lsq_method="cholesky"
            )

            output = compute_attention_output(queries, C1, C2, beta)
            error = float(mx.mean((output - original_output) ** 2))
            errors.append(error)

            print(f"\nBudget {budget}: error={error:.6f}")

        # Error should decrease with larger budget
        self.assertGreater(errors[0], errors[1])  # 10 > 20
        self.assertGreater(errors[1], errors[2])  # 20 > 30

    def test_c2_numerical_stability(self):
        """
        Test C2 fitting numerical stability.

        Even with ill-conditioned attention weights, C2 should not explode.
        """
        seq_len = 40
        head_dim = 16
        budget = 15

        keys = mx.random.normal((seq_len, head_dim))
        values = mx.random.normal((seq_len, head_dim))

        # Create queries that produce ill-conditioned attention
        # (very peaked attention on few keys)
        queries = keys[:5, :] + mx.random.normal((5, head_dim)) * 0.1

        # Use cholesky method (most stable)
        C1, beta, C2 = compact_single_head_quality(
            queries, keys, values, budget,
            fit_beta=True, fit_c2=True,
            lsq_method="cholesky"
        )

        # C2 should not have NaN or Inf
        self.assertFalse(mx.any(mx.isnan(C2)))
        self.assertFalse(mx.any(mx.isinf(C2)))

        # C2 norm should be reasonable (not exploded)
        c2_norm = float(mx.linalg.norm(C2))
        values_norm = float(mx.linalg.norm(values))

        print(f"\nNumerical stability:")
        print(f"  C2 norm: {c2_norm:.4f}")
        print(f"  Values norm: {values_norm:.4f}")
        print(f"  Ratio: {c2_norm / values_norm:.4f}")

        # C2 norm should be within 10x of original values norm
        self.assertLess(c2_norm, values_norm * 10)

    def test_c2_large_scale(self):
        """
        Test C2 fitting at larger scale.
        """
        seq_len = 200
        head_dim = 64
        budget = 50

        keys = mx.random.normal((seq_len, head_dim))
        values = mx.random.normal((seq_len, head_dim))
        queries = mx.random.normal((20, head_dim))

        # Use cholesky (fastest)
        C1, beta, C2 = compact_single_head_quality(
            queries, keys, values, budget,
            fit_beta=True, fit_c2=True,
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

        print(f"\nLarge scale C2 fitting:")
        print(f"  MSE: {error:.6f}")
        print(f"  Relative error: {relative_error * 100:.2f}%")

        # Should have reasonable error (< 10%)
        self.assertLess(relative_error, 0.10)


if __name__ == '__main__':
    unittest.main(verbosity=2)
