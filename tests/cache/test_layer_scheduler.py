"""
Unit tests for LayerScheduler (Task #73)

Tests automatic layer routing for hybrid cache management.
"""

import unittest
import mlx.core as mx

from flashmlx.cache.layer_scheduler import LayerScheduler
from flashmlx.cache.hybrid_cache_manager import (
    HybridCacheManager,
    HybridCacheConfig,
    LayerType
)


class TestLayerSchedulerBasic(unittest.TestCase):
    """Basic functionality tests"""

    def setUp(self):
        """Set up test scheduler"""
        config = HybridCacheConfig(total_budget_bytes=64 * 1024)  # 64KB

        # Mixed architecture: 3 SSM layers + 2 Attention layers
        layer_types = {
            0: LayerType.SSM,
            1: LayerType.ATTENTION,
            2: LayerType.SSM,
            3: LayerType.ATTENTION,
            4: LayerType.SSM
        }

        manager = HybridCacheManager(config=config, layer_types=layer_types)
        self.scheduler = LayerScheduler(manager)

    def test_initialization(self):
        """Test scheduler initialization"""
        self.assertIsNotNone(self.scheduler.hybrid_manager)

    def test_get_layer_type(self):
        """Test layer type retrieval"""
        self.assertEqual(self.scheduler.get_layer_type(0), LayerType.SSM)
        self.assertEqual(self.scheduler.get_layer_type(1), LayerType.ATTENTION)
        self.assertEqual(self.scheduler.get_layer_type(2), LayerType.SSM)
        self.assertIsNone(self.scheduler.get_layer_type(999))

    def test_repr(self):
        """Test __repr__ method"""
        repr_str = repr(self.scheduler)
        self.assertIn("LayerScheduler", repr_str)


class TestLayerSchedulerStoreRouting(unittest.TestCase):
    """Tests for store() routing logic"""

    def setUp(self):
        """Set up test scheduler"""
        config = HybridCacheConfig(total_budget_bytes=64 * 1024)

        layer_types = {
            0: LayerType.SSM,
            1: LayerType.ATTENTION,
            2: LayerType.SSM
        }

        manager = HybridCacheManager(config=config, layer_types=layer_types)
        self.scheduler = LayerScheduler(manager)

    def test_store_ssm_success(self):
        """Test successful SSM storage via scheduler"""
        data = mx.zeros((10, 64))
        result = self.scheduler.store(layer_idx=0, data=data, size_bytes=2560)

        self.assertTrue(result)
        self.assertEqual(self.scheduler.hybrid_manager.total_stores, 1)

    def test_store_ssm_wrong_data_format(self):
        """Test SSM layer with tuple data raises error"""
        data = (mx.zeros((1, 8, 100, 64)), mx.zeros((1, 8, 100, 64)))

        with self.assertRaises(ValueError) as ctx:
            self.scheduler.store(layer_idx=0, data=data, size_bytes=2560)

        self.assertIn("expects mx.array, got tuple", str(ctx.exception))
        self.assertIn("Attention layer", str(ctx.exception))

    def test_store_attention_success(self):
        """Test successful Attention storage via scheduler"""
        keys = mx.zeros((1, 8, 100, 64))
        values = mx.zeros((1, 8, 100, 64))
        query = mx.zeros((1, 8, 1, 64))

        result = self.scheduler.store(
            layer_idx=1,
            data=(keys, values),
            query=query
        )

        self.assertIsInstance(result, tuple)
        compressed_keys, compressed_values = result
        self.assertIsNotNone(compressed_keys)
        self.assertIsNotNone(compressed_values)
        self.assertEqual(self.scheduler.hybrid_manager.total_stores, 1)

    def test_store_attention_wrong_data_format(self):
        """Test Attention layer with single array raises error"""
        data = mx.zeros((10, 64))

        with self.assertRaises(ValueError) as ctx:
            self.scheduler.store(layer_idx=1, data=data)

        self.assertIn("expects (keys, values) tuple", str(ctx.exception))
        self.assertIn("SSM layer", str(ctx.exception))

    def test_store_attention_wrong_tuple_length(self):
        """Test Attention layer with wrong tuple length raises error"""
        data = (mx.zeros((1, 8, 100, 64)),)  # Only 1 element

        with self.assertRaises(ValueError) as ctx:
            self.scheduler.store(layer_idx=1, data=data)

        self.assertIn("tuple of length 2", str(ctx.exception))
        self.assertIn("got tuple of length 1", str(ctx.exception))

    def test_store_unknown_layer(self):
        """Test storing to unknown layer raises error"""
        data = mx.zeros((10, 64))

        with self.assertRaises(ValueError) as ctx:
            self.scheduler.store(layer_idx=999, data=data, size_bytes=2560)

        self.assertIn("not found in layer_types mapping", str(ctx.exception))


class TestLayerSchedulerRetrieveRouting(unittest.TestCase):
    """Tests for retrieve() routing logic"""

    def setUp(self):
        """Set up test scheduler"""
        config = HybridCacheConfig(total_budget_bytes=64 * 1024)

        layer_types = {
            0: LayerType.SSM,
            1: LayerType.ATTENTION,
            2: LayerType.SSM
        }

        manager = HybridCacheManager(config=config, layer_types=layer_types)
        self.scheduler = LayerScheduler(manager)

    def test_retrieve_ssm_success(self):
        """Test successful SSM retrieval via scheduler"""
        # Store data first
        data = mx.zeros((10, 64))
        self.scheduler.store(layer_idx=0, data=data, size_bytes=2560)

        # Retrieve
        retrieved = self.scheduler.retrieve(0)

        self.assertIsNotNone(retrieved)
        self.assertTrue(mx.allclose(retrieved, data))
        self.assertEqual(self.scheduler.hybrid_manager.total_retrievals, 1)

    def test_retrieve_ssm_miss(self):
        """Test SSM retrieval miss via scheduler"""
        retrieved = self.scheduler.retrieve(0)

        self.assertIsNone(retrieved)
        self.assertEqual(self.scheduler.hybrid_manager.total_retrievals, 1)

    def test_retrieve_attention_returns_none(self):
        """Test Attention retrieval returns None (no retrieval support)"""
        # Store Attention data first
        keys = mx.zeros((1, 8, 100, 64))
        values = mx.zeros((1, 8, 100, 64))
        self.scheduler.store(layer_idx=1, data=(keys, values))

        # Retrieve should return None
        retrieved = self.scheduler.retrieve(1)

        self.assertIsNone(retrieved)

    def test_retrieve_unknown_layer(self):
        """Test retrieving from unknown layer raises error"""
        with self.assertRaises(ValueError) as ctx:
            self.scheduler.retrieve(999)

        self.assertIn("not found in layer_types mapping", str(ctx.exception))


class TestLayerSchedulerMixedWorkload(unittest.TestCase):
    """Tests for mixed SSM + Attention workload"""

    def setUp(self):
        """Set up test scheduler with realistic architecture"""
        config = HybridCacheConfig(total_budget_bytes=128 * 1024)  # 128KB

        # Qwen3.5-like: 30 SSM + 10 Attention (every 4th is Attention)
        layer_types = {}
        for i in range(40):
            if (i + 1) % 4 == 0:
                layer_types[i] = LayerType.ATTENTION
            else:
                layer_types[i] = LayerType.SSM

        manager = HybridCacheManager(config=config, layer_types=layer_types)
        self.scheduler = LayerScheduler(manager)

    def test_mixed_storage(self):
        """Test storing both SSM and Attention layers"""
        # Store SSM layers
        for i in [0, 1, 2]:
            data = mx.zeros((10, 64))
            result = self.scheduler.store(layer_idx=i, data=data, size_bytes=1024)
            self.assertTrue(result)

        # Store Attention layer
        keys = mx.zeros((1, 8, 100, 64))
        values = mx.zeros((1, 8, 100, 64))
        query = mx.zeros((1, 8, 1, 64))

        result = self.scheduler.store(layer_idx=3, data=(keys, values), query=query)

        self.assertIsInstance(result, tuple)
        self.assertEqual(self.scheduler.hybrid_manager.total_stores, 4)

    def test_mixed_retrieval(self):
        """Test retrieving both SSM and Attention layers"""
        # Store SSM
        ssm_data = mx.zeros((10, 64))
        self.scheduler.store(layer_idx=0, data=ssm_data, size_bytes=1024)

        # Store Attention
        keys = mx.zeros((1, 8, 100, 64))
        values = mx.zeros((1, 8, 100, 64))
        self.scheduler.store(layer_idx=3, data=(keys, values))

        # Retrieve SSM - should succeed
        retrieved_ssm = self.scheduler.retrieve(0)
        self.assertIsNotNone(retrieved_ssm)

        # Retrieve Attention - should return None
        retrieved_attention = self.scheduler.retrieve(3)
        self.assertIsNone(retrieved_attention)

    def test_architecture_counts(self):
        """Test architecture layer counting via scheduler"""
        stats = self.scheduler.get_statistics()

        # 40 layers: 30 SSM + 10 Attention
        self.assertEqual(stats["num_ssm_layers"], 30)
        self.assertEqual(stats["num_attention_layers"], 10)


class TestLayerSchedulerUtilities(unittest.TestCase):
    """Tests for utility methods"""

    def setUp(self):
        """Set up test scheduler"""
        config = HybridCacheConfig(total_budget_bytes=64 * 1024)

        layer_types = {
            0: LayerType.SSM,
            1: LayerType.ATTENTION
        }

        manager = HybridCacheManager(config=config, layer_types=layer_types)
        self.scheduler = LayerScheduler(manager)

    def test_get_statistics(self):
        """Test statistics retrieval"""
        # Store some data
        data = mx.zeros((10, 64))
        self.scheduler.store(layer_idx=0, data=data, size_bytes=2560)

        stats = self.scheduler.get_statistics()

        self.assertIn("total_budget_bytes", stats)
        self.assertIn("num_ssm_layers", stats)
        self.assertIn("num_attention_layers", stats)
        self.assertIn("total_stores", stats)
        self.assertEqual(stats["total_stores"], 1)

    def test_clear(self):
        """Test clearing all caches"""
        # Store data
        data = mx.zeros((10, 64))
        self.scheduler.store(layer_idx=0, data=data, size_bytes=2560)

        keys = mx.zeros((1, 8, 100, 64))
        values = mx.zeros((1, 8, 100, 64))
        self.scheduler.store(layer_idx=1, data=(keys, values))

        # Clear
        self.scheduler.clear()

        # All should be empty
        stats = self.scheduler.get_statistics()
        self.assertEqual(stats["ssm"]["hot"]["entry_count"], 0)
        self.assertEqual(stats["ssm"]["warm"]["entry_count"], 0)
        self.assertEqual(stats["ssm"]["cold"]["entry_count"], 0)


if __name__ == "__main__":
    unittest.main()
