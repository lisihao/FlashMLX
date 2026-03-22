"""
Unit tests for HotTierManager (Task #70)

Tests Hot tier cache management with LRU eviction.
"""

import unittest
import mlx.core as mx
import time

from flashmlx.cache.hot_tier_manager import HotTierManager, HotCacheEntry


class TestHotCacheEntry(unittest.TestCase):
    """Tests for HotCacheEntry dataclass"""

    def test_valid_entry(self):
        """Test valid cache entry creation"""
        data = mx.zeros((10, 64))
        entry = HotCacheEntry(
            layer_idx=0,
            data=data,
            size_bytes=2560,
            priority=0.9
        )

        self.assertEqual(entry.layer_idx, 0)
        self.assertEqual(entry.size_bytes, 2560)
        self.assertEqual(entry.priority, 0.9)
        self.assertEqual(entry.access_count, 0)
        self.assertGreater(entry.last_access_time, 0)


class TestHotTierManagerBasic(unittest.TestCase):
    """Basic functionality tests"""

    def setUp(self):
        """Set up test manager"""
        self.manager = HotTierManager(budget_bytes=16 * 1024)  # 16KB

    def test_initialization(self):
        """Test manager initialization"""
        self.assertEqual(self.manager.budget_bytes, 16 * 1024)
        self.assertEqual(self.manager.get_total_size(), 0)
        self.assertEqual(self.manager.get_entry_count(), 0)
        self.assertEqual(self.manager.total_hits, 0)
        self.assertEqual(self.manager.total_misses, 0)

    def test_invalid_budget(self):
        """Test that negative budget raises error"""
        with self.assertRaises(ValueError):
            HotTierManager(budget_bytes=-1)

    def test_store_success(self):
        """Test successful store"""
        data = mx.zeros((10, 64))
        success = self.manager.store(
            layer_idx=0,
            data=data,
            size_bytes=2560
        )

        self.assertTrue(success)
        self.assertEqual(self.manager.get_total_size(), 2560)
        self.assertEqual(self.manager.get_entry_count(), 1)
        self.assertTrue(self.manager.contains(0))

    def test_store_update_existing(self):
        """Test updating existing entry"""
        data1 = mx.zeros((10, 64))
        data2 = mx.ones((10, 64))

        # Store first version
        self.manager.store(layer_idx=0, data=data1, size_bytes=2560)

        # Update
        self.manager.store(layer_idx=0, data=data2, size_bytes=2560)

        # Should still have 1 entry
        self.assertEqual(self.manager.get_entry_count(), 1)

        # Data should be updated
        retrieved = self.manager.retrieve(0)
        self.assertTrue(mx.allclose(retrieved, data2))

    def test_retrieve_hit(self):
        """Test successful retrieval"""
        data = mx.zeros((10, 64))
        self.manager.store(layer_idx=0, data=data, size_bytes=2560)

        retrieved = self.manager.retrieve(0)

        self.assertIsNotNone(retrieved)
        self.assertTrue(mx.allclose(retrieved, data))
        self.assertEqual(self.manager.total_hits, 1)

    def test_retrieve_miss(self):
        """Test retrieval miss"""
        retrieved = self.manager.retrieve(999)

        self.assertIsNone(retrieved)
        self.assertEqual(self.manager.total_misses, 1)

    def test_evict_success(self):
        """Test successful eviction"""
        data = mx.zeros((10, 64))
        self.manager.store(layer_idx=0, data=data, size_bytes=2560)

        result = self.manager.evict(0)

        self.assertIsNotNone(result)
        evicted_data, size_bytes = result
        self.assertTrue(mx.allclose(evicted_data, data))
        self.assertEqual(size_bytes, 2560)
        self.assertEqual(self.manager.get_entry_count(), 0)

    def test_evict_nonexistent(self):
        """Test eviction of non-existent entry"""
        result = self.manager.evict(999)
        self.assertIsNone(result)


class TestHotTierManagerEviction(unittest.TestCase):
    """Tests for eviction logic"""

    def setUp(self):
        """Set up test manager"""
        self.manager = HotTierManager(budget_bytes=10 * 1024)  # 10KB

    def test_store_exceeds_budget(self):
        """Test that store fails when budget exceeded"""
        # Fill budget
        for i in range(4):
            data = mx.zeros((10, 64))
            self.manager.store(layer_idx=i, data=data, size_bytes=2560)

        # Try to store more (would exceed budget)
        data = mx.zeros((10, 64))
        success = self.manager.store(layer_idx=100, data=data, size_bytes=2560)

        # Should trigger eviction
        self.assertTrue(success)

        # Should have evicted oldest entry
        self.assertEqual(self.manager.get_entry_count(), 4)

    def test_lru_eviction_policy(self):
        """Test LRU eviction policy"""
        # Store 3 entries
        for i in range(3):
            data = mx.zeros((10, 64))
            self.manager.store(layer_idx=i, data=data, size_bytes=2560)
            time.sleep(0.01)  # Ensure different timestamps

        # Access entry 0 (make it most recently used)
        self.manager.retrieve(0)

        # Get LRU candidate (should be entry 1, oldest unretrieved)
        lru = self.manager.get_lru_candidate()
        self.assertEqual(lru, 1)

    def test_priority_affects_eviction(self):
        """Test that priority affects eviction order"""
        # Store low priority entry
        data1 = mx.zeros((10, 64))
        self.manager.store(layer_idx=0, data=data1, size_bytes=2560, priority=0.3)

        time.sleep(0.01)

        # Store high priority entry
        data2 = mx.ones((10, 64))
        self.manager.store(layer_idx=1, data=data2, size_bytes=2560, priority=0.9)

        # Get LRU candidate (should be low priority entry 0)
        lru = self.manager.get_lru_candidate()
        self.assertEqual(lru, 0)


class TestHotTierManagerStatistics(unittest.TestCase):
    """Tests for statistics tracking"""

    def setUp(self):
        """Set up test manager"""
        self.manager = HotTierManager(budget_bytes=16 * 1024)

    def test_utilization(self):
        """Test utilization calculation"""
        # Empty
        self.assertEqual(self.manager.get_utilization(), 0.0)

        # Half full
        data = mx.zeros((10, 64))
        self.manager.store(layer_idx=0, data=data, size_bytes=8 * 1024)
        self.assertAlmostEqual(self.manager.get_utilization(), 0.5)

        # Full
        self.manager.store(layer_idx=1, data=data, size_bytes=8 * 1024)
        self.assertAlmostEqual(self.manager.get_utilization(), 1.0)

    def test_hit_rate(self):
        """Test hit rate calculation"""
        data = mx.zeros((10, 64))
        self.manager.store(layer_idx=0, data=data, size_bytes=2560)

        # 2 hits, 1 miss
        self.manager.retrieve(0)
        self.manager.retrieve(0)
        self.manager.retrieve(999)

        self.assertAlmostEqual(self.manager.get_hit_rate(), 2.0 / 3.0)

    def test_statistics_structure(self):
        """Test statistics dictionary structure"""
        stats = self.manager.get_statistics()

        required_keys = [
            "tier", "budget_bytes", "total_size", "utilization",
            "entry_count", "total_hits", "total_misses", "hit_rate",
            "total_evictions", "total_stores"
        ]

        for key in required_keys:
            self.assertIn(key, stats)

        self.assertEqual(stats["tier"], "hot")

    def test_clear(self):
        """Test clearing cache"""
        # Store some data
        for i in range(3):
            data = mx.zeros((10, 64))
            self.manager.store(layer_idx=i, data=data, size_bytes=2560)

        self.assertEqual(self.manager.get_entry_count(), 3)

        # Clear
        self.manager.clear()

        self.assertEqual(self.manager.get_entry_count(), 0)
        self.assertEqual(self.manager.get_total_size(), 0)


class TestHotTierManagerRepr(unittest.TestCase):
    """Tests for string representation"""

    def test_repr(self):
        """Test __repr__ method"""
        manager = HotTierManager(budget_bytes=16 * 1024)

        # Store some data
        data = mx.zeros((10, 64))
        manager.store(layer_idx=0, data=data, size_bytes=4096)
        manager.retrieve(0)

        repr_str = repr(manager)

        self.assertIn("HotTierManager", repr_str)
        self.assertIn("entries=", repr_str)
        self.assertIn("size=", repr_str)
        self.assertIn("hit_rate=", repr_str)


if __name__ == "__main__":
    unittest.main()
