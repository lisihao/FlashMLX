"""
Unit tests for WarmTierManager (Task #70)

Tests Warm tier cache management with promotion/demotion logic.
"""

import unittest
import mlx.core as mx
import time

from flashmlx.cache.warm_tier_manager import WarmTierManager, WarmCacheEntry


class TestWarmCacheEntry(unittest.TestCase):
    """Tests for WarmCacheEntry dataclass"""

    def test_valid_entry(self):
        """Test valid cache entry creation"""
        data = mx.zeros((10, 64))
        entry = WarmCacheEntry(
            layer_idx=0,
            data=data,
            size_bytes=2560
        )

        self.assertEqual(entry.layer_idx, 0)
        self.assertEqual(entry.size_bytes, 2560)
        self.assertEqual(entry.access_count, 0)
        self.assertGreater(entry.last_access_time, 0)

    def test_update_scores(self):
        """Test score updates"""
        data = mx.zeros((10, 64))
        entry = WarmCacheEntry(
            layer_idx=0,
            data=data,
            size_bytes=2560,
            access_count=10
        )

        entry.update_scores()

        # High access count should give positive promotion score
        self.assertGreater(entry.promotion_score, 0)
        # Low time since access should give low demotion score
        self.assertGreaterEqual(entry.demotion_score, 0)


class TestWarmTierManagerBasic(unittest.TestCase):
    """Basic functionality tests"""

    def setUp(self):
        """Set up test manager"""
        self.manager = WarmTierManager(budget_bytes=64 * 1024)  # 64KB

    def test_initialization(self):
        """Test manager initialization"""
        self.assertEqual(self.manager.budget_bytes, 64 * 1024)
        self.assertEqual(self.manager.get_total_size(), 0)
        self.assertEqual(self.manager.get_entry_count(), 0)

    def test_invalid_budget(self):
        """Test that negative budget raises error"""
        with self.assertRaises(ValueError):
            WarmTierManager(budget_bytes=-1)

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


class TestWarmTierManagerPromotionDemotion(unittest.TestCase):
    """Tests for promotion/demotion logic"""

    def setUp(self):
        """Set up test manager"""
        self.manager = WarmTierManager(
            budget_bytes=64 * 1024,
            promotion_threshold=5.0
        )

    def test_get_promotion_candidates_empty(self):
        """Test promotion candidates when empty"""
        candidates = self.manager.get_promotion_candidates()
        self.assertEqual(len(candidates), 0)

    def test_get_promotion_candidates(self):
        """Test getting promotion candidates"""
        # Store multiple entries
        for i in range(5):
            data = mx.zeros((10, 64))
            self.manager.store(layer_idx=i, data=data, size_bytes=2560)
            time.sleep(0.01)

        # Access some entries multiple times
        for _ in range(10):
            self.manager.retrieve(0)  # High access count
        for _ in range(5):
            self.manager.retrieve(1)  # Medium access count

        # Get promotion candidates
        candidates = self.manager.get_promotion_candidates(count=3)

        # Should return entries with high access counts
        self.assertGreater(len(candidates), 0)
        # Entry 0 should be top candidate (highest access count)
        self.assertIn(0, candidates)

    def test_get_demotion_candidates_empty(self):
        """Test demotion candidates when empty"""
        candidates = self.manager.get_demotion_candidates()
        self.assertEqual(len(candidates), 0)

    def test_get_demotion_candidates(self):
        """Test getting demotion candidates"""
        # Store multiple entries
        for i in range(5):
            data = mx.zeros((10, 64))
            self.manager.store(layer_idx=i, data=data, size_bytes=2560)
            time.sleep(0.1)  # Create time gaps

        # Access recent entries
        self.manager.retrieve(4)  # Most recent

        # Get demotion candidates
        candidates = self.manager.get_demotion_candidates(count=3)

        # Should return oldest, least accessed entries
        self.assertGreater(len(candidates), 0)
        # Entry 0 should be candidate (oldest, not accessed)
        self.assertIn(0, candidates)

    def test_promotion_threshold(self):
        """Test that promotion threshold filters candidates"""
        # Store entry with low access count
        data = mx.zeros((10, 64))
        self.manager.store(layer_idx=0, data=data, size_bytes=2560)

        # Single access (won't meet threshold)
        self.manager.retrieve(0)

        # Should not be promotion candidate
        candidates = self.manager.get_promotion_candidates()
        self.assertEqual(len(candidates), 0)


class TestWarmTierManagerEviction(unittest.TestCase):
    """Tests for eviction logic"""

    def setUp(self):
        """Set up test manager"""
        self.manager = WarmTierManager(budget_bytes=10 * 1024)  # 10KB

    def test_eviction_on_budget_exceeded(self):
        """Test automatic eviction when budget exceeded"""
        # Fill budget
        for i in range(4):
            data = mx.zeros((10, 64))
            self.manager.store(layer_idx=i, data=data, size_bytes=2560)

        # Try to store more
        data = mx.zeros((10, 64))
        success = self.manager.store(layer_idx=100, data=data, size_bytes=2560)

        # Should succeed by evicting old entry
        self.assertTrue(success)
        self.assertEqual(self.manager.get_entry_count(), 4)
        self.assertGreater(self.manager.total_evictions, 0)

    def test_eviction_uses_demotion_score(self):
        """Test that eviction prioritizes high demotion score"""
        # Store multiple entries
        for i in range(3):
            data = mx.zeros((10, 64))
            self.manager.store(layer_idx=i, data=data, size_bytes=2560)
            time.sleep(0.1)

        # Access recent entries to lower their demotion score
        self.manager.retrieve(2)

        # Fill budget to trigger eviction
        for i in range(3, 10):
            data = mx.zeros((10, 64))
            self.manager.store(layer_idx=i, data=data, size_bytes=2560)

        # Oldest entries (0, 1) should have been evicted
        self.assertFalse(self.manager.contains(0))


class TestWarmTierManagerStatistics(unittest.TestCase):
    """Tests for statistics tracking"""

    def setUp(self):
        """Set up test manager"""
        self.manager = WarmTierManager(budget_bytes=64 * 1024)

    def test_statistics_structure(self):
        """Test statistics dictionary structure"""
        stats = self.manager.get_statistics()

        required_keys = [
            "tier", "budget_bytes", "total_size", "utilization",
            "entry_count", "total_hits", "total_misses", "hit_rate",
            "total_evictions", "total_promotions", "total_demotions",
            "total_stores"
        ]

        for key in required_keys:
            self.assertIn(key, stats)

        self.assertEqual(stats["tier"], "warm")

    def test_hit_rate(self):
        """Test hit rate calculation"""
        data = mx.zeros((10, 64))
        self.manager.store(layer_idx=0, data=data, size_bytes=2560)

        # 3 hits, 2 misses
        self.manager.retrieve(0)
        self.manager.retrieve(0)
        self.manager.retrieve(0)
        self.manager.retrieve(999)
        self.manager.retrieve(998)

        self.assertAlmostEqual(self.manager.get_hit_rate(), 3.0 / 5.0)

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


class TestWarmTierManagerRepr(unittest.TestCase):
    """Tests for string representation"""

    def test_repr(self):
        """Test __repr__ method"""
        manager = WarmTierManager(budget_bytes=64 * 1024)

        data = mx.zeros((10, 64))
        manager.store(layer_idx=0, data=data, size_bytes=4096)

        repr_str = repr(manager)

        self.assertIn("WarmTierManager", repr_str)
        self.assertIn("entries=", repr_str)


if __name__ == "__main__":
    unittest.main()
