"""
Unit tests for CompressedKVCache (Task #76)

Tests Attention layer KV cache wrapper with compression integration.
"""

import unittest
import mlx.core as mx

from flashmlx.cache.compressed_kv_cache import CompressedKVCache
from flashmlx.cache.layer_scheduler import LayerScheduler
from flashmlx.cache.hybrid_cache_manager import (
    HybridCacheManager,
    HybridCacheConfig,
    LayerType
)


class TestCompressedKVCacheBasic(unittest.TestCase):
    """Basic functionality tests"""

    def setUp(self):
        """Set up test cache"""
        config = HybridCacheConfig(total_budget_bytes=64 * 1024)  # 64KB

        # 5 Attention layers
        layer_types = {i: LayerType.ATTENTION for i in range(5)}

        manager = HybridCacheManager(config=config, layer_types=layer_types)
        scheduler = LayerScheduler(manager)
        self.cache = CompressedKVCache(scheduler)

    def test_initialization(self):
        """Test cache initialization"""
        self.assertIsNotNone(self.cache.scheduler)
        self.assertEqual(len(self.cache), 0)
        self.assertEqual(self.cache.total_updates, 0)
        self.assertEqual(self.cache.total_retrievals, 0)

    def test_update_and_fetch_success(self):
        """Test successful update and fetch with compression"""
        keys = mx.zeros((1, 8, 100, 64))
        values = mx.zeros((1, 8, 100, 64))
        query = mx.zeros((1, 8, 1, 64))

        compressed_k, compressed_v = self.cache.update_and_fetch(
            layer_idx=0,
            keys=keys,
            values=values,
            query=query
        )

        self.assertIsNotNone(compressed_k)
        self.assertIsNotNone(compressed_v)
        self.assertEqual(self.cache.total_updates, 1)
        self.assertIn(0, self.cache)

        # Verify compression happened
        self.assertLess(compressed_k.shape[2], keys.shape[2])

    def test_update_and_fetch_without_query(self):
        """Test update without query (still applies compression)"""
        keys = mx.zeros((1, 8, 100, 64))
        values = mx.zeros((1, 8, 100, 64))

        compressed_k, compressed_v = self.cache.update_and_fetch(
            layer_idx=0,
            keys=keys,
            values=values
        )

        self.assertIsNotNone(compressed_k)
        self.assertIsNotNone(compressed_v)
        self.assertEqual(self.cache.total_updates, 1)

    def test_retrieve_success(self):
        """Test successful retrieval"""
        # Update first
        keys = mx.zeros((1, 8, 100, 64))
        values = mx.zeros((1, 8, 100, 64))
        compressed_k, compressed_v = self.cache.update_and_fetch(
            layer_idx=0, keys=keys, values=values
        )

        # Retrieve
        retrieved = self.cache.retrieve(0)

        self.assertIsNotNone(retrieved)
        retrieved_k, retrieved_v = retrieved
        self.assertTrue(mx.allclose(retrieved_k, compressed_k))
        self.assertTrue(mx.allclose(retrieved_v, compressed_v))
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
        keys = mx.zeros((1, 8, 100, 64))
        values = mx.zeros((1, 8, 100, 64))
        self.cache.update_and_fetch(layer_idx=0, keys=keys, values=values)

        # Now present
        self.assertTrue(self.cache.contains(0))


class TestCompressedKVCacheCompression(unittest.TestCase):
    """Tests for compression behavior"""

    def setUp(self):
        """Set up test cache"""
        config = HybridCacheConfig(
            total_budget_bytes=64 * 1024,
            compression_ratio=3.0  # 3x compression
        )
        layer_types = {i: LayerType.ATTENTION for i in range(5)}

        manager = HybridCacheManager(config=config, layer_types=layer_types)
        scheduler = LayerScheduler(manager)
        self.cache = CompressedKVCache(scheduler)

    def test_compression_ratio_calculated(self):
        """Test compression ratio is calculated correctly"""
        keys = mx.zeros((1, 8, 100, 64))
        values = mx.zeros((1, 8, 100, 64))

        compressed_k, compressed_v = self.cache.update_and_fetch(
            layer_idx=0, keys=keys, values=values
        )

        # Verify compression happened
        original_len = keys.shape[2]
        compressed_len = compressed_k.shape[2]
        expected_ratio = original_len / compressed_len

        avg_ratio = self.cache.get_compression_ratio()
        self.assertGreater(avg_ratio, 1.0)  # Should be compressed
        self.assertAlmostEqual(avg_ratio, expected_ratio, places=2)

    def test_average_compression_ratio(self):
        """Test average compression ratio across multiple updates"""
        # Update multiple layers
        for i in range(3):
            keys = mx.zeros((1, 8, 100, 64))
            values = mx.zeros((1, 8, 100, 64))
            self.cache.update_and_fetch(layer_idx=i, keys=keys, values=values)

        # Average compression ratio should be > 1.0
        avg_ratio = self.cache.get_compression_ratio()
        self.assertGreater(avg_ratio, 1.0)


class TestCompressedKVCacheLocalCache(unittest.TestCase):
    """Tests for local cache behavior"""

    def setUp(self):
        """Set up test cache"""
        config = HybridCacheConfig(total_budget_bytes=64 * 1024)
        layer_types = {i: LayerType.ATTENTION for i in range(5)}

        manager = HybridCacheManager(config=config, layer_types=layer_types)
        scheduler = LayerScheduler(manager)
        self.cache = CompressedKVCache(scheduler)

    def test_local_cache_hit(self):
        """Test local cache hit"""
        # Update
        keys = mx.zeros((1, 8, 100, 64))
        values = mx.zeros((1, 8, 100, 64))
        self.cache.update_and_fetch(layer_idx=0, keys=keys, values=values)

        # Retrieve - should hit local cache
        retrieved = self.cache.retrieve(0)

        self.assertIsNotNone(retrieved)
        self.assertEqual(self.cache.local_cache_hits, 1)

    def test_local_cache_hit_rate(self):
        """Test local cache hit rate calculation"""
        # Update and retrieve multiple times
        for i in range(3):
            keys = mx.zeros((1, 8, 100, 64))
            values = mx.zeros((1, 8, 100, 64))
            self.cache.update_and_fetch(layer_idx=i, keys=keys, values=values)

        # All retrievals should hit local cache
        for i in range(3):
            self.cache.retrieve(i)

        stats = self.cache.get_statistics()
        self.assertEqual(stats["local_cache"]["local_cache_hits"], 3)
        self.assertEqual(stats["local_cache"]["total_retrievals"], 3)
        self.assertEqual(stats["local_cache"]["local_cache_hit_rate"], 1.0)


class TestCompressedKVCacheValidation(unittest.TestCase):
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
        self.cache = CompressedKVCache(scheduler)

    def test_update_non_attention_layer_raises_error(self):
        """Test updating non-Attention layer raises error"""
        keys = mx.zeros((1, 8, 100, 64))
        values = mx.zeros((1, 8, 100, 64))

        with self.assertRaises(ValueError) as ctx:
            self.cache.update_and_fetch(layer_idx=0, keys=keys, values=values)  # SSM layer

        self.assertIn("not an Attention layer", str(ctx.exception))
        self.assertIn("ManagedArraysCache", str(ctx.exception))

    def test_retrieve_non_attention_layer_raises_error(self):
        """Test retrieving non-Attention layer raises error"""
        with self.assertRaises(ValueError) as ctx:
            self.cache.retrieve(0)  # SSM layer

        self.assertIn("not an Attention layer", str(ctx.exception))
        self.assertIn("ManagedArraysCache", str(ctx.exception))


class TestCompressedKVCacheClear(unittest.TestCase):
    """Tests for clear operations"""

    def setUp(self):
        """Set up test cache"""
        config = HybridCacheConfig(total_budget_bytes=64 * 1024)
        layer_types = {i: LayerType.ATTENTION for i in range(5)}

        manager = HybridCacheManager(config=config, layer_types=layer_types)
        scheduler = LayerScheduler(manager)
        self.cache = CompressedKVCache(scheduler)

    def test_clear_specific_layer(self):
        """Test clearing specific layer"""
        # Update multiple layers
        for i in range(3):
            keys = mx.zeros((1, 8, 100, 64))
            values = mx.zeros((1, 8, 100, 64))
            self.cache.update_and_fetch(layer_idx=i, keys=keys, values=values)

        # Clear layer 1
        self.cache.clear(layer_idx=1)

        # Layer 1 should be gone
        self.assertNotIn(1, self.cache)
        # Others should remain
        self.assertIn(0, self.cache)
        self.assertIn(2, self.cache)

    def test_clear_all(self):
        """Test clearing all layers"""
        # Update multiple layers
        for i in range(3):
            keys = mx.zeros((1, 8, 100, 64))
            values = mx.zeros((1, 8, 100, 64))
            self.cache.update_and_fetch(layer_idx=i, keys=keys, values=values)

        # Clear all
        self.cache.clear()

        # All should be gone
        self.assertEqual(len(self.cache), 0)


class TestCompressedKVCacheStatistics(unittest.TestCase):
    """Tests for statistics"""

    def setUp(self):
        """Set up test cache"""
        config = HybridCacheConfig(total_budget_bytes=64 * 1024)
        layer_types = {i: LayerType.ATTENTION for i in range(5)}

        manager = HybridCacheManager(config=config, layer_types=layer_types)
        scheduler = LayerScheduler(manager)
        self.cache = CompressedKVCache(scheduler)

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
        self.assertIn("avg_compression_ratio", local_stats)

    def test_statistics_counts(self):
        """Test statistics counts"""
        # Update and retrieve
        keys = mx.zeros((1, 8, 100, 64))
        values = mx.zeros((1, 8, 100, 64))
        self.cache.update_and_fetch(layer_idx=0, keys=keys, values=values)
        self.cache.retrieve(0)

        stats = self.cache.get_statistics()

        self.assertEqual(stats["local_cache"]["size"], 1)
        self.assertEqual(stats["local_cache"]["total_updates"], 1)
        self.assertEqual(stats["local_cache"]["total_retrievals"], 1)
        self.assertGreater(stats["local_cache"]["avg_compression_ratio"], 1.0)


class TestCompressedKVCacheUtilities(unittest.TestCase):
    """Tests for utility methods"""

    def setUp(self):
        """Set up test cache"""
        config = HybridCacheConfig(total_budget_bytes=64 * 1024)
        layer_types = {i: LayerType.ATTENTION for i in range(5)}

        manager = HybridCacheManager(config=config, layer_types=layer_types)
        scheduler = LayerScheduler(manager)
        self.cache = CompressedKVCache(scheduler)

    def test_len(self):
        """Test __len__ method"""
        self.assertEqual(len(self.cache), 0)

        # Update layers
        for i in range(3):
            keys = mx.zeros((1, 8, 100, 64))
            values = mx.zeros((1, 8, 100, 64))
            self.cache.update_and_fetch(layer_idx=i, keys=keys, values=values)

        self.assertEqual(len(self.cache), 3)

    def test_contains_operator(self):
        """Test __contains__ method"""
        self.assertNotIn(0, self.cache)

        # Update
        keys = mx.zeros((1, 8, 100, 64))
        values = mx.zeros((1, 8, 100, 64))
        self.cache.update_and_fetch(layer_idx=0, keys=keys, values=values)

        self.assertIn(0, self.cache)

    def test_repr(self):
        """Test __repr__ method"""
        repr_str = repr(self.cache)

        self.assertIn("CompressedKVCache", repr_str)
        self.assertIn("local_size=", repr_str)
        self.assertIn("updates=", repr_str)
        self.assertIn("avg_compression=", repr_str)


if __name__ == "__main__":
    unittest.main()
