"""Basic tests for compaction algorithm"""
import mlx.core as mx
import pytest

from flashmlx.cache.compaction_algorithm import (
    HighestAttentionKeysCompaction,
    create_compaction_algorithm,
)


class TestHighestAttentionKeysCompaction:
    """Test suite for compression algorithm"""

    @pytest.fixture
    def sample_kv_data(self):
        """Create sample K, V, queries for testing"""
        T = 100  # Original sequence length
        t = 25   # Compressed length
        d = 64   # Head dimension
        n = 10   # Number of query samples

        K = mx.random.uniform(shape=(T, d))
        V = mx.random.uniform(shape=(T, d))
        queries = mx.random.uniform(shape=(n, d))

        return K, V, queries, t, d, n, T

    def test_construction(self):
        """Test 1: Verify correct initialization"""
        algo = HighestAttentionKeysCompaction(
            score_method='mean',
            beta_method='nnls',
            c2_method='lsq',
            c2_ridge_lambda=0.01,
            c2_solver='lstsq'
        )

        assert algo.score_method == 'mean'
        assert algo.beta_method == 'nnls'
        assert algo.c2_method == 'lsq'
        assert algo.c2_ridge_lambda == 0.01
        assert algo.c2_solver == 'lstsq'

    def test_invalid_score_method(self):
        """Test 2: Verify error on invalid score_method"""
        with pytest.raises(ValueError, match="score_method must be"):
            HighestAttentionKeysCompaction(score_method='invalid')

    def test_invalid_solver(self):
        """Test 3: Verify error on invalid solver"""
        with pytest.raises(ValueError, match="c2_solver must be"):
            HighestAttentionKeysCompaction(c2_solver='invalid')

    def test_compute_compacted_cache_basic(self, sample_kv_data):
        """Test 4: Basic compression works"""
        K, V, queries, t, d, n, T = sample_kv_data

        algo = HighestAttentionKeysCompaction(
            score_method='mean',
            beta_method='ones',  # Use simple method first
            c2_method='direct'   # Use direct method first
        )

        C1, beta, C2, indices = algo.compute_compacted_cache(K, V, queries, t)

        # Verify shapes
        assert C1.shape == (t, d)
        assert beta.shape == (t,)
        assert C2.shape == (t, d)
        assert len(indices) == t

        # Verify indices are valid
        assert all(0 <= i < T for i in indices)

        # Verify C1 contains selected keys
        for idx, i in enumerate(indices):
            assert mx.allclose(C1[idx], K[i], atol=1e-5)

    def test_compute_compacted_cache_with_nnls(self, sample_kv_data):
        """Test 5: Compression with NNLS beta solving"""
        K, V, queries, t, d, n, T = sample_kv_data

        algo = HighestAttentionKeysCompaction(
            score_method='mean',
            beta_method='nnls',  # Use NNLS
            c2_method='direct'
        )

        C1, beta, C2, indices = algo.compute_compacted_cache(K, V, queries, t)

        # Verify shapes
        assert C1.shape == (t, d)
        assert beta.shape == (t,)
        assert C2.shape == (t, d)
        assert len(indices) == t

        # Beta should not be all ones (NNLS should compute different values)
        assert not mx.allclose(beta, mx.ones_like(beta))

    def test_compute_compacted_cache_with_ridge(self, sample_kv_data):
        """Test 6: Compression with Ridge Regression for C2"""
        K, V, queries, t, d, n, T = sample_kv_data

        algo = HighestAttentionKeysCompaction(
            score_method='mean',
            beta_method='ones',
            c2_method='lsq',  # Use Ridge Regression
            c2_ridge_lambda=0.01,
            c2_solver='lstsq'
        )

        C1, beta, C2, indices = algo.compute_compacted_cache(K, V, queries, t)

        # Verify shapes
        assert C1.shape == (t, d)
        assert beta.shape == (t,)
        assert C2.shape == (t, d)

        # C2 should be different from direct selection
        # (We can't easily verify this without running direct method,
        #  but at least verify it's not all zeros or all same values)
        assert not mx.allclose(C2, mx.zeros_like(C2))
        assert not mx.allclose(C2[0], C2[1])  # Different rows

    def test_score_methods(self, sample_kv_data):
        """Test 7: Different score aggregation methods"""
        K, V, queries, t, d, n, T = sample_kv_data

        results = {}
        for method in ['mean', 'max', 'sum']:
            algo = HighestAttentionKeysCompaction(
                score_method=method,
                beta_method='ones',
                c2_method='direct'
            )
            C1, beta, C2, indices = algo.compute_compacted_cache(K, V, queries, t)
            results[method] = indices

        # Different methods should produce different selections
        # (at least some difference)
        assert results['mean'] != results['max'] or results['mean'] != results['sum']

    def test_factory_function(self):
        """Test 8: Factory function works"""
        algo = create_compaction_algorithm(
            score_method='max',
            beta_method='zeros',
            c2_method='lsq',
            c2_ridge_lambda=0.05
        )

        assert isinstance(algo, HighestAttentionKeysCompaction)
        assert algo.score_method == 'max'
        assert algo.beta_method == 'zeros'
        assert algo.c2_ridge_lambda == 0.05

    def test_edge_case_t_equals_T(self, sample_kv_data):
        """Test 9: Edge case where t = T (no compression)"""
        K, V, queries, t, d, n, T = sample_kv_data

        algo = HighestAttentionKeysCompaction()

        # Request full size (no compression)
        C1, beta, C2, indices = algo.compute_compacted_cache(K, V, queries, T)

        assert C1.shape == (T, d)
        assert len(indices) == T

    def test_error_on_t_larger_than_T(self, sample_kv_data):
        """Test 10: Error when t > T"""
        K, V, queries, t, d, n, T = sample_kv_data

        algo = HighestAttentionKeysCompaction()

        with pytest.raises(ValueError, match="Cannot compact"):
            algo.compute_compacted_cache(K, V, queries, T + 10)

    def test_error_on_mismatched_shapes(self, sample_kv_data):
        """Test 11: Error on mismatched K, V shapes"""
        K, V, queries, t, d, n, T = sample_kv_data

        algo = HighestAttentionKeysCompaction()

        # K and V different lengths
        V_wrong = mx.random.uniform(shape=(T + 10, d))
        with pytest.raises(ValueError, match="K and V must have same first dimension"):
            algo.compute_compacted_cache(K, V_wrong, queries, t)

        # Queries wrong dimension
        queries_wrong = mx.random.uniform(shape=(n, d + 10))
        with pytest.raises(ValueError, match="Query dimension.*must match key dimension"):
            algo.compute_compacted_cache(K, V, queries_wrong, t)


def test_import():
    """Verify module can be imported from package"""
    from flashmlx.cache.compaction_algorithm import (
        HighestAttentionKeysCompaction,
        create_compaction_algorithm,
    )
    assert HighestAttentionKeysCompaction is not None
    assert create_compaction_algorithm is not None


def test_package_export():
    """Verify exports are available from package __init__"""
    from flashmlx.cache import (
        HighestAttentionKeysCompaction,
        create_compaction_algorithm,
    )
    assert HighestAttentionKeysCompaction is not None
    assert create_compaction_algorithm is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
