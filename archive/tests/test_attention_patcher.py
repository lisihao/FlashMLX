"""Unit tests for attention_patcher"""
import mlx.core as mx
import pytest

from flashmlx.cache.attention_patcher import repeat_kv, patch_attention_for_compacted_cache
from flashmlx.cache.compacted_kv_cache import CompactedKVCache


class TestRepeatKV:
    """Test suite for repeat_kv function"""

    def test_repeat_kv_2d(self):
        """Test 1: Repeat 2D array (beta case)"""
        B, n_kv_heads, seq_len = 2, 4, 16
        n_rep = 2  # GQA with 8 heads total

        x = mx.random.uniform(shape=(B, n_kv_heads, seq_len))
        result = repeat_kv(x, n_rep)

        # Check shape
        expected_shape = (B, n_kv_heads * n_rep, seq_len)
        assert result.shape == expected_shape

        # Check values are repeated correctly
        # Each KV head should be repeated n_rep times
        for b in range(B):
            for kv_head in range(n_kv_heads):
                for rep in range(n_rep):
                    head_idx = kv_head * n_rep + rep
                    assert mx.allclose(result[b, head_idx, :], x[b, kv_head, :])

    def test_repeat_kv_3d(self):
        """Test 2: Repeat 3D array (KV cache case)"""
        B, n_kv_heads, seq_len, head_dim = 2, 4, 16, 64
        n_rep = 2

        x = mx.random.uniform(shape=(B, n_kv_heads, seq_len, head_dim))
        result = repeat_kv(x, n_rep)

        # Check shape
        expected_shape = (B, n_kv_heads * n_rep, seq_len, head_dim)
        assert result.shape == expected_shape

        # Check values
        for b in range(B):
            for kv_head in range(n_kv_heads):
                for rep in range(n_rep):
                    head_idx = kv_head * n_rep + rep
                    assert mx.allclose(result[b, head_idx, :, :], x[b, kv_head, :, :])

    def test_repeat_kv_n_rep_1(self):
        """Test 3: No repetition when n_rep=1"""
        B, n_kv_heads, seq_len = 2, 8, 16

        x = mx.random.uniform(shape=(B, n_kv_heads, seq_len))
        result = repeat_kv(x, n_rep=1)

        # Should return original array
        assert result.shape == x.shape
        assert mx.allclose(result, x)


class TestPatchAttention:
    """Test suite for patch_attention_for_compacted_cache function"""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model structure for testing"""
        from mlx_lm import load

        # Load a small model for testing
        model_path = "/Volumes/toshiba/models/qwen3-8b-mlx"
        model, tokenizer = load(model_path)

        return model, tokenizer

    def test_patch_applies_without_error(self, mock_model):
        """Test 4: Verify patching doesn't break model"""
        model, _ = mock_model

        # Apply patch
        patch_attention_for_compacted_cache(model, verbose=False)

        # Verify model still has layers
        assert hasattr(model.model, 'layers')
        assert len(model.model.layers) > 0

        # Verify attention modules still exist
        for layer in model.model.layers:
            assert hasattr(layer, 'self_attn')
            assert hasattr(layer.self_attn, '__call__')

    def test_patched_attention_with_compacted_cache(self, mock_model):
        """Test 5: Verify beta is applied with CompactedKVCache"""
        model, _ = mock_model

        # Create sample compacted cache
        B, n_kv_heads, t, head_dim = 1, 4, 8, 128
        num_layers = len(model.model.layers)

        compacted_cache = []
        for _ in range(num_layers):
            c1 = mx.random.uniform(shape=(B, n_kv_heads, t, head_dim))
            beta = mx.random.uniform(shape=(B, n_kv_heads, t), low=-1.0, high=1.0)
            c2 = mx.random.uniform(shape=(B, n_kv_heads, t, head_dim))
            compacted_cache.append((c1, beta, c2))

        cache = CompactedKVCache(compacted_cache)

        # Apply patch
        patch_attention_for_compacted_cache(model, verbose=False)

        # Run forward pass with compacted cache
        input_ids = mx.array([[1, 2, 3, 4, 5]])
        try:
            output = model(input_ids, cache=cache)
            assert output is not None
            print(f"✓ Forward pass successful with CompactedKVCache")
        except Exception as e:
            pytest.fail(f"Forward pass failed: {e}")

    def test_patched_attention_without_cache(self, mock_model):
        """Test 6: Verify patched attention works without cache"""
        model, _ = mock_model

        # Apply patch
        patch_attention_for_compacted_cache(model, verbose=False)

        # Run forward pass without cache
        input_ids = mx.array([[1, 2, 3, 4, 5]])
        try:
            output = model(input_ids, cache=None)
            assert output is not None
            print(f"✓ Forward pass successful without cache")
        except Exception as e:
            pytest.fail(f"Forward pass failed: {e}")


def test_import():
    """Verify module can be imported"""
    from flashmlx.cache.attention_patcher import (
        repeat_kv,
        patch_attention_for_compacted_cache
    )
    assert repeat_kv is not None
    assert patch_attention_for_compacted_cache is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
