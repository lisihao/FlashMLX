"""Unit tests for CompactedKVCache"""
import mlx.core as mx
import pytest

from flashmlx.cache.compacted_kv_cache import CompactedKVCache, CompactedKVCacheLayer, create_compacted_cache_list


class TestCompactedKVCache:
    """Test suite for CompactedKVCache class"""

    @pytest.fixture
    def sample_compacted_cache(self):
        """Create sample compacted cache for testing"""
        batch_size = 2
        n_kv_heads = 4
        compacted_len = 16
        head_dim = 64
        num_layers = 3

        compacted_cache = []
        for _ in range(num_layers):
            c1 = mx.random.uniform(shape=(batch_size, n_kv_heads, compacted_len, head_dim))
            beta = mx.random.uniform(shape=(batch_size, n_kv_heads, compacted_len))
            c2 = mx.random.uniform(shape=(batch_size, n_kv_heads, compacted_len, head_dim))
            compacted_cache.append((c1, beta, c2))

        return compacted_cache, batch_size, n_kv_heads, compacted_len, head_dim, num_layers

    def test_construction(self, sample_compacted_cache):
        """Test 1: Verify correct initialization"""
        cache_data, B, KV, t, D, num_layers = sample_compacted_cache
        original_len = 1024

        cache = CompactedKVCache(cache_data, original_seq_len=original_len)

        # Verify keys and values are set from first layer
        expected_c1, _, expected_c2 = cache_data[0]
        assert cache.keys is not None
        assert cache.values is not None
        assert cache.keys.shape == (B, KV, t, D)
        assert cache.values.shape == (B, KV, t, D)
        assert mx.allclose(cache.keys, expected_c1)
        assert mx.allclose(cache.values, expected_c2)

        # Verify offset
        assert cache.offset == t

        # Verify original_seq_len stored
        assert cache.original_seq_len == original_len

    def test_beta_for_layer(self, sample_compacted_cache):
        """Test 2: Verify beta_for_layer returns correct beta"""
        cache_data, B, KV, t, D, num_layers = sample_compacted_cache

        cache = CompactedKVCache(cache_data)

        # Verify beta for each layer
        for layer_idx in range(num_layers):
            _, expected_beta, _ = cache_data[layer_idx]
            actual_beta = cache.beta_for_layer(layer_idx)

            assert actual_beta is not None
            assert actual_beta.shape == (B, KV, t)
            assert mx.allclose(actual_beta, expected_beta)

        # Verify None for non-existent layer
        assert cache.beta_for_layer(999) is None

    def test_update_and_fetch_initial(self, sample_compacted_cache):
        """Test 3a: Verify update_and_fetch with initial None state"""
        cache_data, B, KV, t, D, num_layers = sample_compacted_cache

        cache = CompactedKVCache(cache_data)

        # New data to append
        new_len = 8
        new_keys = mx.random.uniform(shape=(B, KV, new_len, D))
        new_values = mx.random.uniform(shape=(B, KV, new_len, D))

        result_keys, result_values = cache.update_and_fetch(new_keys, new_values)

        # Verify concatenation
        expected_len = t + new_len
        assert result_keys.shape == (B, KV, expected_len, D)
        assert result_values.shape == (B, KV, expected_len, D)
        assert cache.offset == expected_len

    def test_update_and_fetch_multiple(self, sample_compacted_cache):
        """Test 3b: Verify multiple update_and_fetch calls"""
        cache_data, B, KV, t, D, num_layers = sample_compacted_cache

        cache = CompactedKVCache(cache_data)

        # First update
        new_len1 = 8
        new_keys1 = mx.random.uniform(shape=(B, KV, new_len1, D))
        new_values1 = mx.random.uniform(shape=(B, KV, new_len1, D))
        cache.update_and_fetch(new_keys1, new_values1)

        # Second update
        new_len2 = 12
        new_keys2 = mx.random.uniform(shape=(B, KV, new_len2, D))
        new_values2 = mx.random.uniform(shape=(B, KV, new_len2, D))
        result_keys, result_values = cache.update_and_fetch(new_keys2, new_values2)

        # Verify final length
        expected_len = t + new_len1 + new_len2
        assert result_keys.shape[-2] == expected_len
        assert cache.offset == expected_len

    def test_empty_compacted_cache_raises(self):
        """Test 4: Verify error on empty compacted_cache"""
        with pytest.raises(ValueError, match="cannot be empty"):
            CompactedKVCache([])

    def test_inherits_from_kv_cache(self, sample_compacted_cache):
        """Test 5: Verify inheritance from KVCache"""
        from mlx_lm.models.cache import KVCache

        cache_data = sample_compacted_cache[0]
        cache = CompactedKVCache(cache_data)

        assert isinstance(cache, KVCache)


class TestCreateCompactedCacheList:
    """Test suite for create_compacted_cache_list function"""

    @pytest.fixture
    def sample_compacted_cache(self):
        """Create sample compacted cache for testing"""
        batch_size = 2
        n_kv_heads = 4
        compacted_len = 16
        head_dim = 64
        num_layers = 3

        compacted_cache = []
        for _ in range(num_layers):
            c1 = mx.random.uniform(shape=(batch_size, n_kv_heads, compacted_len, head_dim))
            beta = mx.random.uniform(shape=(batch_size, n_kv_heads, compacted_len))
            c2 = mx.random.uniform(shape=(batch_size, n_kv_heads, compacted_len, head_dim))
            compacted_cache.append((c1, beta, c2))

        return compacted_cache, batch_size, n_kv_heads, compacted_len, head_dim, num_layers

    def test_create_cache_list(self, sample_compacted_cache):
        """Test 7: Verify create_compacted_cache_list creates per-layer caches"""
        cache_data, B, KV, t, D, num_layers = sample_compacted_cache

        cache_list = create_compacted_cache_list(cache_data, original_seq_len=1024)

        # Verify list length
        assert len(cache_list) == num_layers

        # Verify each layer
        for layer_idx, layer_cache in enumerate(cache_list):
            assert isinstance(layer_cache, CompactedKVCacheLayer)
            assert layer_cache.layer_idx == layer_idx

            # Verify data
            expected_c1, expected_beta, expected_c2 = cache_data[layer_idx]
            assert layer_cache.keys.shape == (B, KV, t, D)
            assert layer_cache.values.shape == (B, KV, t, D)
            assert layer_cache.get_beta().shape == (B, KV, t)

            assert mx.allclose(layer_cache.keys, expected_c1)
            assert mx.allclose(layer_cache.values, expected_c2)
            assert mx.allclose(layer_cache.get_beta(), expected_beta)

            # Verify offset
            assert layer_cache.offset == t

            # Verify original_seq_len
            assert layer_cache.original_seq_len == 1024


def test_import():
    """Verify module can be imported"""
    from flashmlx.cache.compacted_kv_cache import CompactedKVCache, create_compacted_cache_list
    assert CompactedKVCache is not None
    assert create_compacted_cache_list is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
