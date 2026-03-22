"""
Unit tests for injection mechanism (Task #77)

Tests hybrid cache injection into MLX-LM models.
"""

import unittest
import mlx.core as mx

from flashmlx.cache.injection import (
    HybridCacheWrapper,
    inject_hybrid_cache_manager,
    restore_original_cache,
    create_layer_types_from_model
)
from flashmlx.cache.hybrid_cache_manager import HybridCacheConfig, LayerType
from flashmlx.cache.layer_scheduler import LayerScheduler
from flashmlx.cache.managed_arrays_cache import ManagedArraysCache
from flashmlx.cache.compressed_kv_cache import CompressedKVCache


class MockLayer:
    """Mock layer for testing"""

    def __init__(self, has_attention: bool = False):
        if has_attention:
            self.self_attn = "mock_attention"


class MockModel:
    """Mock model for testing injection"""

    def __init__(self, num_layers: int = 5, layer_types: dict = None):
        """
        Initialize mock model.

        Args:
            num_layers: Number of layers
            layer_types: Dictionary mapping layer_idx → has_attention bool
        """
        self.layers = []
        for i in range(num_layers):
            has_attn = layer_types.get(i, False) if layer_types else False
            self.layers.append(MockLayer(has_attention=has_attn))

        self.cache = None  # Will be injected


class TestHybridCacheWrapper(unittest.TestCase):
    """Tests for HybridCacheWrapper"""

    def setUp(self):
        """Set up test wrapper"""
        config = HybridCacheConfig(total_budget_bytes=64 * 1024)  # 64KB

        # 5 layers: 3 SSM + 2 Attention
        layer_types = {
            0: LayerType.SSM,
            1: LayerType.ATTENTION,
            2: LayerType.SSM,
            3: LayerType.ATTENTION,
            4: LayerType.SSM
        }

        from flashmlx.cache.hybrid_cache_manager import HybridCacheManager

        manager = HybridCacheManager(config=config, layer_types=layer_types)
        scheduler = LayerScheduler(manager)
        ssm_cache = ManagedArraysCache(scheduler)
        attention_cache = CompressedKVCache(scheduler)

        self.wrapper = HybridCacheWrapper(
            scheduler=scheduler,
            ssm_cache=ssm_cache,
            attention_cache=attention_cache
        )

    def test_initialization(self):
        """Test wrapper initialization"""
        self.assertIsNotNone(self.wrapper.scheduler)
        self.assertIsNotNone(self.wrapper.ssm_cache)
        self.assertIsNotNone(self.wrapper.attention_cache)

    def test_update_and_fetch_ssm(self):
        """Test SSM update and fetch"""
        state = mx.zeros((10, 64))
        result = self.wrapper.update_and_fetch_ssm(layer_idx=0, state=state)

        self.assertTrue(mx.allclose(result, state))

    def test_update_and_fetch_attention(self):
        """Test Attention update and fetch"""
        keys = mx.zeros((1, 8, 100, 64))
        values = mx.zeros((1, 8, 100, 64))

        compressed_k, compressed_v = self.wrapper.update_and_fetch_attention(
            layer_idx=1, keys=keys, values=values
        )

        self.assertIsNotNone(compressed_k)
        self.assertIsNotNone(compressed_v)

    def test_retrieve_ssm(self):
        """Test SSM retrieval"""
        # Update first
        state = mx.zeros((10, 64))
        self.wrapper.update_and_fetch_ssm(layer_idx=0, state=state)

        # Retrieve
        retrieved = self.wrapper.retrieve_ssm(0)

        self.assertIsNotNone(retrieved)
        self.assertTrue(mx.allclose(retrieved, state))

    def test_retrieve_attention(self):
        """Test Attention retrieval"""
        # Update first
        keys = mx.zeros((1, 8, 100, 64))
        values = mx.zeros((1, 8, 100, 64))
        self.wrapper.update_and_fetch_attention(layer_idx=1, keys=keys, values=values)

        # Retrieve
        retrieved = self.wrapper.retrieve_attention(1)

        self.assertIsNotNone(retrieved)
        retrieved_k, retrieved_v = retrieved
        self.assertIsNotNone(retrieved_k)
        self.assertIsNotNone(retrieved_v)

    def test_clear_specific_layer(self):
        """Test clearing specific layer"""
        # Update multiple layers
        state = mx.zeros((10, 64))
        self.wrapper.update_and_fetch_ssm(layer_idx=0, state=state)
        self.wrapper.update_and_fetch_ssm(layer_idx=2, state=state)

        # Verify layers are cached
        self.assertIsNotNone(self.wrapper.retrieve_ssm(0))
        self.assertIsNotNone(self.wrapper.retrieve_ssm(2))

        # Clear layer 0
        self.wrapper.clear(layer_idx=0)

        # Layer 0 should be gone from local cache, layer 2 should remain
        # Note: clear() only clears local cache, not managed tiers
        self.assertNotIn(0, self.wrapper.ssm_cache._local_cache)
        self.assertIn(2, self.wrapper.ssm_cache._local_cache)

    def test_clear_all(self):
        """Test clearing all layers"""
        # Update multiple layers
        state = mx.zeros((10, 64))
        self.wrapper.update_and_fetch_ssm(layer_idx=0, state=state)
        self.wrapper.update_and_fetch_ssm(layer_idx=2, state=state)

        # Clear all
        self.wrapper.clear()

        # All should be gone
        self.assertIsNone(self.wrapper.retrieve_ssm(0))
        self.assertIsNone(self.wrapper.retrieve_ssm(2))

    def test_get_statistics(self):
        """Test statistics retrieval"""
        stats = self.wrapper.get_statistics()

        self.assertIn("ssm", stats)
        self.assertIn("attention", stats)
        self.assertIn("scheduler", stats)

    def test_get_layer_type(self):
        """Test layer type retrieval"""
        self.assertEqual(self.wrapper.get_layer_type(0), LayerType.SSM)
        self.assertEqual(self.wrapper.get_layer_type(1), LayerType.ATTENTION)
        self.assertEqual(self.wrapper.get_layer_type(2), LayerType.SSM)

    def test_repr(self):
        """Test __repr__ method"""
        repr_str = repr(self.wrapper)

        self.assertIn("HybridCacheWrapper", repr_str)
        self.assertIn("SSM", repr_str)
        self.assertIn("Attention", repr_str)


class TestInjection(unittest.TestCase):
    """Tests for inject_hybrid_cache_manager"""

    def test_inject_with_auto_inject(self):
        """Test injection with auto_inject=True"""
        model = MockModel(num_layers=5)

        layer_types = {
            0: LayerType.SSM,
            1: LayerType.ATTENTION,
            2: LayerType.SSM,
            3: LayerType.ATTENTION,
            4: LayerType.SSM
        }

        config = HybridCacheConfig(total_budget_bytes=64 * 1024)

        wrapper = inject_hybrid_cache_manager(
            model, config, layer_types, auto_inject=True
        )

        # Model cache should be replaced
        self.assertEqual(model.cache, wrapper)
        self.assertIsInstance(model.cache, HybridCacheWrapper)

    def test_inject_without_auto_inject(self):
        """Test injection with auto_inject=False"""
        model = MockModel(num_layers=5)

        layer_types = {
            0: LayerType.SSM,
            1: LayerType.ATTENTION,
            2: LayerType.SSM,
            3: LayerType.ATTENTION,
            4: LayerType.SSM
        }

        config = HybridCacheConfig(total_budget_bytes=64 * 1024)

        wrapper = inject_hybrid_cache_manager(
            model, config, layer_types, auto_inject=False
        )

        # Model cache should NOT be replaced
        self.assertIsNone(model.cache)
        self.assertIsInstance(wrapper, HybridCacheWrapper)

    def test_inject_stores_original_cache(self):
        """Test injection stores original cache"""
        model = MockModel(num_layers=5)
        original_cache = "original"
        model.cache = original_cache

        layer_types = {
            0: LayerType.SSM,
            1: LayerType.ATTENTION,
            2: LayerType.SSM,
            3: LayerType.ATTENTION,
            4: LayerType.SSM
        }

        config = HybridCacheConfig(total_budget_bytes=64 * 1024)

        wrapper = inject_hybrid_cache_manager(
            model, config, layer_types, auto_inject=True
        )

        # Original cache should be stored
        self.assertTrue(hasattr(wrapper, '_original_cache'))
        self.assertEqual(wrapper._original_cache, original_cache)


class TestRestoreOriginalCache(unittest.TestCase):
    """Tests for restore_original_cache"""

    def test_restore_with_original_cache(self):
        """Test restoring original cache"""
        model = MockModel(num_layers=5)
        original_cache = "original"
        model.cache = original_cache

        layer_types = {i: LayerType.SSM for i in range(5)}
        config = HybridCacheConfig(total_budget_bytes=64 * 1024)

        wrapper = inject_hybrid_cache_manager(
            model, config, layer_types, auto_inject=True
        )

        # Restore
        restore_original_cache(model, wrapper)

        # Should be restored
        self.assertEqual(model.cache, original_cache)

    def test_restore_without_original_cache(self):
        """Test restoring when no original cache"""
        model = MockModel(num_layers=5)

        layer_types = {i: LayerType.SSM for i in range(5)}
        config = HybridCacheConfig(total_budget_bytes=64 * 1024)

        wrapper = inject_hybrid_cache_manager(
            model, config, layer_types, auto_inject=True
        )

        # Restore
        restore_original_cache(model, wrapper)

        # Should be None
        self.assertIsNone(model.cache)


class TestCreateLayerTypesFromModel(unittest.TestCase):
    """Tests for create_layer_types_from_model"""

    def test_create_from_explicit_indices(self):
        """Test creation from explicit indices"""
        model = MockModel(num_layers=10)

        attention_indices = [1, 3, 5, 7, 9]
        layer_types = create_layer_types_from_model(
            model, attention_layer_indices=attention_indices
        )

        # Check all layers
        for i in range(10):
            expected = LayerType.ATTENTION if i in attention_indices else LayerType.SSM
            self.assertEqual(layer_types[i], expected)

    def test_create_from_every_nth_pattern(self):
        """Test creation from 'every Nth' pattern"""
        model = MockModel(num_layers=12)

        layer_types = create_layer_types_from_model(
            model, attention_layer_pattern="every 4th"
        )

        # Layers 3, 7, 11 should be Attention (indices 3, 7, 11)
        expected_attention = {3, 7, 11}
        for i in range(12):
            expected = LayerType.ATTENTION if i in expected_attention else LayerType.SSM
            self.assertEqual(layer_types[i], expected)

    def test_create_from_last_n_pattern(self):
        """Test creation from 'last N' pattern"""
        model = MockModel(num_layers=10)

        layer_types = create_layer_types_from_model(
            model, attention_layer_pattern="last 3"
        )

        # Last 3 layers (7, 8, 9) should be Attention
        expected_attention = {7, 8, 9}
        for i in range(10):
            expected = LayerType.ATTENTION if i in expected_attention else LayerType.SSM
            self.assertEqual(layer_types[i], expected)

    def test_create_from_model_structure(self):
        """Test auto-detection from model structure"""
        # Model with specific Attention layers
        model_layer_types = {
            1: True,   # has self_attn
            3: True,   # has self_attn
            0: False,  # SSM
            2: False,  # SSM
            4: False   # SSM
        }

        model = MockModel(num_layers=5, layer_types=model_layer_types)

        layer_types = create_layer_types_from_model(model)

        # Check detection
        self.assertEqual(layer_types[0], LayerType.SSM)
        self.assertEqual(layer_types[1], LayerType.ATTENTION)
        self.assertEqual(layer_types[2], LayerType.SSM)
        self.assertEqual(layer_types[3], LayerType.ATTENTION)
        self.assertEqual(layer_types[4], LayerType.SSM)

    def test_create_with_no_layers_raises_error(self):
        """Test error when model has no layers"""
        model = MockModel(num_layers=0)

        with self.assertRaises(ValueError) as ctx:
            create_layer_types_from_model(model)

        self.assertIn("Could not determine number of layers", str(ctx.exception))

    def test_create_with_invalid_pattern_raises_error(self):
        """Test error with invalid pattern"""
        model = MockModel(num_layers=10)

        with self.assertRaises(ValueError) as ctx:
            create_layer_types_from_model(model, attention_layer_pattern="invalid pattern")

        self.assertIn("Unsupported pattern", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
