"""
Unit tests for ManagedArraysCache (Task #75)

Tests SSM layer cache wrapper with HybridCacheManager integration.
"""

import unittest
import mlx.core as mx

from flashmlx.cache.managed_arrays_cache import ManagedArraysCache
from flashmlx.cache.layer_scheduler import LayerScheduler
from flashmlx.cache.hybrid_cache_manager import (
    HybridCacheManager,
    HybridCacheConfig,
    LayerType
)


class TestManagedArraysCacheBasic(unittest.TestCase):
    """Basic functionality tests"""

    def setUp(self):
        """Set up test cache"""
        config = HybridCacheConfig(total_budget_bytes=64 * 1024)  # 64KB

        # 5 SSM layers
        layer_types = {i: LayerType.SSM for i in range(5)}

        manager = HybridCacheManager(config=config, layer_types=layer_types)
        scheduler = LayerScheduler(manager)
        self.cache = ManagedArraysCache(scheduler)

    def test_initialization(self):
        """Test cache initialization"""
        self.assertIsNotNone(self.cache.scheduler)
        self.assertEqual(len(self.cache), 0)
        self.assertEqual(self.cache.total_updates, 0)
        self.assertEqual(self.cache.total_retrievals, 0)

    def test_update_and_fetch_success(self):
        """Test successful update and fetch"""
        state = mx.zeros((10, 64))
        result = self.cache.update_and_fetch(layer_idx=0, state=state)

        self.assertTrue(mx.allclose(result, state))
        self.assertEqual(self.cache.total_updates, 1)
        self.assertIn(0, self.cache)

    def test_update_and_fetch_with_priority(self):
        """Test update with custom priority"""
        state = mx.zeros((10, 64))
        result = self.cache.update_and_fetch(layer_idx=0, state=state, priority=5.0)

        self.assertTrue(mx.allclose(result, state))
        self.assertEqual(self.cache.total_updates, 1)

    def test_retrieve_success(self):
        """Test successful retrieval"""
        # Update first
        state = mx.zeros((10, 64))
        self.cache.update_and_fetch(layer_idx=0, state=state)

        # Retrieve
        retrieved = self.cache.retrieve(0)

        self.assertIsNotNone(retrieved)
        self.assertTrue(mx.allclose(retrieved, state))
        self.assertEqual(self.cache.total_retrievals, 1)

    def test_retrieve_miss(self):
        """Test retrieval miss"""
        retrieved = self.cache.retrieve(0)

        self.assertIsNone(retrieved)
        self.assertEqual(self.cache.total_retrievals, 1)

    def test_contains(self):
        """Test contains check"""
        # Initially not present
        self.assertFalse(self.cache.contains(0))

        # Update
        state = mx.zeros((10, 64))
        self.cache.update_and_fetch(layer_idx=0, state=state)

        # Now present
        self.assertTrue(self.cache.contains(0))


class TestManagedArraysCacheLocalCache(unittest.TestCase):
    """Tests for local cache (L0) behavior"""

    def setUp(self):
        """Set up test cache"""
        config = HybridCacheConfig(total_budget_bytes=64 * 1024)
        layer_types = {i: LayerType.SSM for i in range(5)}

        manager = HybridCacheManager(config=config, layer_types=layer_types)
        scheduler = LayerScheduler(manager)
        self.cache = ManagedArraysCache(scheduler)

    def test_local_cache_hit(self):
        """Test local cache hit"""
        # Update
        state = mx.zeros((10, 64))
        self.cache.update_and_fetch(layer_idx=0, state=state)

        # Retrieve - should hit local cache
        retrieved = self.cache.retrieve(0)

        self.assertIsNotNone(retrieved)
        self.assertEqual(self.cache.local_cache_hits, 1)

    def test_local_cache_hit_rate(self):
        """Test local cache hit rate calculation"""
        # Update and retrieve multiple times
        for i in range(3):
            state = mx.zeros((10, 64))
            self.cache.update_and_fetch(layer_idx=i, state=state)

        # All retrievals should hit local cache
        for i in range(3):
            self.cache.retrieve(i)

        stats = self.cache.get_statistics()
        self.assertEqual(stats["local_cache"]["local_cache_hits"], 3)
        self.assertEqual(stats["local_cache"]["total_retrievals"], 3)
        self.assertEqual(stats["local_cache"]["local_cache_hit_rate"], 1.0)


class TestManagedArraysCacheValidation(unittest.TestCase):
    """Tests for input validation"""

    def setUp(self):
        """Set up test cache with mixed layer types"""
        config = HybridCacheConfig(total_budget_bytes=64 * 1024)

        # Mixed: SSM and Attention layers
        layer_types = {
            0: LayerType.SSM,
            1: LayerType.ATTENTION,
            2: LayerType.SSM
        }

        manager = HybridCacheManager(config=config, layer_types=layer_types)
        scheduler = LayerScheduler(manager)
        self.cache = ManagedArraysCache(scheduler)

    def test_update_non_ssm_layer_raises_error(self):
        """Test updating non-SSM layer raises error"""
        state = mx.zeros((10, 64))

        with self.assertRaises(ValueError) as ctx:
            self.cache.update_and_fetch(layer_idx=1, state=state)  # Attention layer

        self.assertIn("not an SSM layer", str(ctx.exception))
        self.assertIn("CompressedKVCache", str(ctx.exception))

    def test_retrieve_non_ssm_layer_raises_error(self):
        """Test retrieving non-SSM layer raises error"""
        with self.assertRaises(ValueError) as ctx:
            self.cache.retrieve(1)  # Attention layer

        self.assertIn("not an SSM layer", str(ctx.exception))
        self.assertIn("CompressedKVCache", str(ctx.exception))


class TestManagedArraysCacheClear(unittest.TestCase):
    """Tests for clear operations"""

    def setUp(self):
        """Set up test cache"""
        config = HybridCacheConfig(total_budget_bytes=64 * 1024)
        layer_types = {i: LayerType.SSM for i in range(5)}

        manager = HybridCacheManager(config=config, layer_types=layer_types)
        scheduler = LayerScheduler(manager)
        self.cache = ManagedArraysCache(scheduler)

    def test_clear_specific_layer(self):
        """Test clearing specific layer"""
        # Update multiple layers
        for i in range(3):
            state = mx.zeros((10, 64))
            self.cache.update_and_fetch(layer_idx=i, state=state)

        # Clear layer 1
        self.cache.clear(layer_idx=1)

        # Layer 1 should be gone from local cache
        self.assertNotIn(1, self.cache)
        # Others should remain
        self.assertIn(0, self.cache)
        self.assertIn(2, self.cache)

    def test_clear_all(self):
        """Test clearing all layers"""
        # Update multiple layers
        for i in range(3):
            state = mx.zeros((10, 64))
            self.cache.update_and_fetch(layer_idx=i, state=state)

        # Clear all
        self.cache.clear()

        # All should be gone
        self.assertEqual(len(self.cache), 0)


class TestManagedArraysCacheStatistics(unittest.TestCase):
    """Tests for statistics"""

    def setUp(self):
        """Set up test cache"""
        config = HybridCacheConfig(total_budget_bytes=64 * 1024)
        layer_types = {i: LayerType.SSM for i in range(5)}

        manager = HybridCacheManager(config=config, layer_types=layer_types)
        scheduler = LayerScheduler(manager)
        self.cache = ManagedArraysCache(scheduler)

    def test_statistics_structure(self):
        """Test statistics structure"""
        stats = self.cache.get_statistics()

        self.assertIn("managed", stats)
        self.assertIn("local_cache", stats)

        local_stats = stats["local_cache"]
        self.assertIn("size", local_stats)
        self.assertIn("total_updates", local_stats)
        self.assertIn("total_retrievals", local_stats)
        self.assertIn("local_cache_hits", local_stats)
        self.assertIn("local_cache_hit_rate", local_stats)

    def test_statistics_counts(self):
        """Test statistics counts"""
        # Update and retrieve
        state = mx.zeros((10, 64))
        self.cache.update_and_fetch(layer_idx=0, state=state)
        self.cache.retrieve(0)

        stats = self.cache.get_statistics()

        self.assertEqual(stats["local_cache"]["size"], 1)
        self.assertEqual(stats["local_cache"]["total_updates"], 1)
        self.assertEqual(stats["local_cache"]["total_retrievals"], 1)


class TestManagedArraysCacheUtilities(unittest.TestCase):
    """Tests for utility methods"""

    def setUp(self):
        """Set up test cache"""
        config = HybridCacheConfig(total_budget_bytes=64 * 1024)
        layer_types = {i: LayerType.SSM for i in range(5)}

        manager = HybridCacheManager(config=config, layer_types=layer_types)
        scheduler = LayerScheduler(manager)
        self.cache = ManagedArraysCache(scheduler)

    def test_len(self):
        """Test __len__ method"""
        self.assertEqual(len(self.cache), 0)

        # Update layers
        for i in range(3):
            state = mx.zeros((10, 64))
            self.cache.update_and_fetch(layer_idx=i, state=state)

        self.assertEqual(len(self.cache), 3)

    def test_contains_operator(self):
        """Test __contains__ method"""
        self.assertNotIn(0, self.cache)

        # Update
        state = mx.zeros((10, 64))
        self.cache.update_and_fetch(layer_idx=0, state=state)

        self.assertIn(0, self.cache)

    def test_repr(self):
        """Test __repr__ method"""
        repr_str = repr(self.cache)

        self.assertIn("ManagedArraysCache", repr_str)
        self.assertIn("local_size=", repr_str)
        self.assertIn("updates=", repr_str)
        self.assertIn("retrievals=", repr_str)


if __name__ == "__main__":
    unittest.main()
