"""
Unit tests for ColdArchive (Task #70)

Tests Cold tier long-term storage with FIFO eviction and revival candidates.
"""

import unittest
import mlx.core as mx
import time

from flashmlx.cache.cold_archive import ColdArchive, ColdCacheEntry


class TestColdCacheEntry(unittest.TestCase):
    """Tests for ColdCacheEntry dataclass"""

    def test_valid_entry(self):
        """Test valid cache entry creation"""
        data = mx.zeros((10, 64))
        entry = ColdCacheEntry(
            layer_idx=0,
            data=data,
            size_bytes=2560
        )

        self.assertEqual(entry.layer_idx, 0)
        self.assertEqual(entry.size_bytes, 2560)
        self.assertEqual(entry.access_count, 0)
        self.assertGreater(entry.archived_time, 0)
        self.assertGreater(entry.last_access_time, 0)
        self.assertFalse(entry.is_compressed)

    def test_timestamps_initialization(self):
        """Test automatic timestamp initialization"""
        data = mx.zeros((10, 64))
        entry = ColdCacheEntry(
            layer_idx=0,
            data=data,
            size_bytes=2560
        )

        # archived_time and last_access_time should be initialized
        self.assertGreater(entry.archived_time, 0)
        self.assertEqual(entry.last_access_time, entry.archived_time)


class TestColdArchiveBasic(unittest.TestCase):
    """Basic functionality tests"""

    def setUp(self):
        """Set up test archive"""
        self.archive = ColdArchive(budget_bytes=256 * 1024)  # 256KB

    def test_initialization(self):
        """Test archive initialization"""
        self.assertEqual(self.archive.budget_bytes, 256 * 1024)
        self.assertEqual(self.archive.get_total_size(), 0)
        self.assertEqual(self.archive.get_entry_count(), 0)
        self.assertFalse(self.archive.enable_compression)

    def test_initialization_with_compression(self):
        """Test archive initialization with compression enabled"""
        archive = ColdArchive(budget_bytes=256 * 1024, enable_compression=True)
        self.assertTrue(archive.enable_compression)

    def test_invalid_budget(self):
        """Test that negative budget raises error"""
        with self.assertRaises(ValueError):
            ColdArchive(budget_bytes=-1)

    def test_store_success(self):
        """Test successful store"""
        data = mx.zeros((10, 64))
        success = self.archive.store(
            layer_idx=0,
            data=data,
            size_bytes=2560
        )

        self.assertTrue(success)
        self.assertEqual(self.archive.get_total_size(), 2560)
        self.assertEqual(self.archive.get_entry_count(), 1)
        self.assertTrue(self.archive.contains(0))

    def test_store_update_existing(self):
        """Test updating existing entry preserves access count"""
        data1 = mx.zeros((10, 64))
        data2 = mx.ones((10, 64))

        # Store first version
        self.archive.store(layer_idx=0, data=data1, size_bytes=2560)

        # Access it to increase access count
        self.archive.retrieve(0)
        self.archive.retrieve(0)

        # Update
        self.archive.store(layer_idx=0, data=data2, size_bytes=2560)

        # Should still have 1 entry
        self.assertEqual(self.archive.get_entry_count(), 1)

        # Access count should be preserved
        entry = self.archive.cache[0]
        self.assertEqual(entry.access_count, 2)

        # Data should be updated
        retrieved = self.archive.retrieve(0)
        self.assertTrue(mx.allclose(retrieved, data2))

    def test_retrieve_hit(self):
        """Test successful retrieval"""
        data = mx.zeros((10, 64))
        self.archive.store(layer_idx=0, data=data, size_bytes=2560)

        retrieved = self.archive.retrieve(0)

        self.assertIsNotNone(retrieved)
        self.assertTrue(mx.allclose(retrieved, data))
        self.assertEqual(self.archive.total_hits, 1)

    def test_retrieve_miss(self):
        """Test retrieval miss"""
        retrieved = self.archive.retrieve(999)

        self.assertIsNone(retrieved)
        self.assertEqual(self.archive.total_misses, 1)

    def test_retrieve_updates_access_count(self):
        """Test that retrieval updates access count"""
        data = mx.zeros((10, 64))
        self.archive.store(layer_idx=0, data=data, size_bytes=2560)

        # Initial access count should be 0
        entry = self.archive.cache[0]
        self.assertEqual(entry.access_count, 0)

        # Retrieve multiple times
        self.archive.retrieve(0)
        self.archive.retrieve(0)
        self.archive.retrieve(0)

        # Access count should increase
        self.assertEqual(entry.access_count, 3)

    def test_evict_success(self):
        """Test successful eviction"""
        data = mx.zeros((10, 64))
        self.archive.store(layer_idx=0, data=data, size_bytes=2560)

        result = self.archive.evict(0)

        self.assertIsNotNone(result)
        evicted_data, size_bytes = result
        self.assertTrue(mx.allclose(evicted_data, data))
        self.assertEqual(size_bytes, 2560)
        self.assertEqual(self.archive.get_entry_count(), 0)

    def test_evict_nonexistent(self):
        """Test eviction of non-existent entry"""
        result = self.archive.evict(999)
        self.assertIsNone(result)


class TestColdArchiveEviction(unittest.TestCase):
    """Tests for FIFO eviction logic"""

    def setUp(self):
        """Set up test archive"""
        self.archive = ColdArchive(budget_bytes=10 * 1024)  # 10KB

    def test_store_exceeds_budget(self):
        """Test automatic eviction when budget exceeded"""
        # Fill budget
        for i in range(4):
            data = mx.zeros((10, 64))
            self.archive.store(layer_idx=i, data=data, size_bytes=2560)

        # Try to store more (would exceed budget)
        data = mx.zeros((10, 64))
        success = self.archive.store(layer_idx=100, data=data, size_bytes=2560)

        # Should succeed by evicting old entries
        self.assertTrue(success)
        self.assertEqual(self.archive.get_entry_count(), 4)
        self.assertGreater(self.archive.total_evictions, 0)

    def test_fifo_eviction_policy(self):
        """Test FIFO eviction policy (oldest first)"""
        # Store 3 entries with time gaps
        for i in range(3):
            data = mx.zeros((10, 64))
            self.archive.store(layer_idx=i, data=data, size_bytes=2560)
            time.sleep(0.1)

        # Fill budget to trigger eviction
        for i in range(3, 10):
            data = mx.zeros((10, 64))
            self.archive.store(layer_idx=i, data=data, size_bytes=2560)

        # Oldest entries (0, 1, 2) should have been evicted
        self.assertFalse(self.archive.contains(0))
        self.assertFalse(self.archive.contains(1))

    def test_eviction_uses_archived_time(self):
        """Test that eviction is based on archived_time not access_time"""
        # Store multiple entries
        for i in range(3):
            data = mx.zeros((10, 64))
            self.archive.store(layer_idx=i, data=data, size_bytes=2560)
            time.sleep(0.1)

        # Access oldest entry (shouldn't prevent eviction)
        self.archive.retrieve(0)

        # Fill budget to trigger eviction
        for i in range(3, 10):
            data = mx.zeros((10, 64))
            self.archive.store(layer_idx=i, data=data, size_bytes=2560)

        # Entry 0 should still be evicted despite recent access
        # because FIFO is based on archived_time
        self.assertFalse(self.archive.contains(0))


class TestColdArchiveRevivalCandidates(unittest.TestCase):
    """Tests for revival candidate selection"""

    def setUp(self):
        """Set up test archive"""
        self.archive = ColdArchive(budget_bytes=256 * 1024)

    def test_get_revival_candidates_empty(self):
        """Test revival candidates when empty"""
        candidates = self.archive.get_revival_candidates()
        self.assertEqual(len(candidates), 0)

    def test_get_revival_candidates(self):
        """Test getting revival candidates"""
        # Store multiple entries
        for i in range(5):
            data = mx.zeros((10, 64))
            self.archive.store(layer_idx=i, data=data, size_bytes=2560)

        # Access some entries multiple times
        for _ in range(10):
            self.archive.retrieve(0)  # High access count
        for _ in range(5):
            self.archive.retrieve(1)  # Medium access count
        # Entry 2, 3, 4 have 0 access count

        # Get revival candidates
        candidates = self.archive.get_revival_candidates(count=3)

        # Should return entries with access count > 0
        self.assertGreater(len(candidates), 0)
        # Entry 0 should be top candidate (highest access count)
        self.assertEqual(candidates[0], 0)
        # Entry 1 should be second
        self.assertEqual(candidates[1], 1)

    def test_revival_candidates_filters_zero_access(self):
        """Test that revival candidates excludes zero-access entries"""
        # Store multiple entries
        for i in range(5):
            data = mx.zeros((10, 64))
            self.archive.store(layer_idx=i, data=data, size_bytes=2560)

        # Only access entry 0
        self.archive.retrieve(0)

        # Get revival candidates
        candidates = self.archive.get_revival_candidates(count=10)

        # Should only return entry 0
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0], 0)

    def test_revival_candidates_sorted_by_access_count(self):
        """Test that revival candidates are sorted by access count"""
        # Store entries
        for i in range(5):
            data = mx.zeros((10, 64))
            self.archive.store(layer_idx=i, data=data, size_bytes=2560)

        # Create different access counts
        for _ in range(15):
            self.archive.retrieve(2)  # Highest
        for _ in range(10):
            self.archive.retrieve(4)  # Second
        for _ in range(5):
            self.archive.retrieve(1)  # Third

        candidates = self.archive.get_revival_candidates(count=5)

        # Should be sorted by access count
        self.assertEqual(candidates[0], 2)
        self.assertEqual(candidates[1], 4)
        self.assertEqual(candidates[2], 1)


class TestColdArchiveOldestEntries(unittest.TestCase):
    """Tests for oldest entries tracking"""

    def setUp(self):
        """Set up test archive"""
        self.archive = ColdArchive(budget_bytes=256 * 1024)

    def test_get_oldest_entries_empty(self):
        """Test oldest entries when empty"""
        oldest = self.archive.get_oldest_entries()
        self.assertEqual(len(oldest), 0)

    def test_get_oldest_entries(self):
        """Test getting oldest entries"""
        # Store entries with time gaps
        for i in range(5):
            data = mx.zeros((10, 64))
            self.archive.store(layer_idx=i, data=data, size_bytes=2560)
            time.sleep(0.1)

        # Get oldest entries
        oldest = self.archive.get_oldest_entries(count=3)

        # Should return oldest first
        self.assertEqual(len(oldest), 3)
        self.assertEqual(oldest[0], 0)
        self.assertEqual(oldest[1], 1)
        self.assertEqual(oldest[2], 2)


class TestColdArchiveStatistics(unittest.TestCase):
    """Tests for statistics tracking"""

    def setUp(self):
        """Set up test archive"""
        self.archive = ColdArchive(budget_bytes=256 * 1024)

    def test_utilization(self):
        """Test utilization calculation"""
        # Empty
        self.assertEqual(self.archive.get_utilization(), 0.0)

        # Half full
        data = mx.zeros((10, 64))
        self.archive.store(layer_idx=0, data=data, size_bytes=128 * 1024)
        self.assertAlmostEqual(self.archive.get_utilization(), 0.5)

        # Full
        self.archive.store(layer_idx=1, data=data, size_bytes=128 * 1024)
        self.assertAlmostEqual(self.archive.get_utilization(), 1.0)

    def test_hit_rate(self):
        """Test hit rate calculation"""
        data = mx.zeros((10, 64))
        self.archive.store(layer_idx=0, data=data, size_bytes=2560)

        # 2 hits, 1 miss
        self.archive.retrieve(0)
        self.archive.retrieve(0)
        self.archive.retrieve(999)

        self.assertAlmostEqual(self.archive.get_hit_rate(), 2.0 / 3.0)

    def test_average_age(self):
        """Test average age calculation"""
        # Empty archive has 0 average age
        self.assertEqual(self.archive.get_average_age(), 0.0)

        # Store entries with time gaps
        for i in range(3):
            data = mx.zeros((10, 64))
            self.archive.store(layer_idx=i, data=data, size_bytes=2560)
            time.sleep(0.1)

        # Average age should be > 0
        avg_age = self.archive.get_average_age()
        self.assertGreater(avg_age, 0.0)

    def test_statistics_structure(self):
        """Test statistics dictionary structure"""
        stats = self.archive.get_statistics()

        required_keys = [
            "tier", "budget_bytes", "total_size", "utilization",
            "entry_count", "total_hits", "total_misses", "hit_rate",
            "total_evictions", "total_revivals", "total_stores",
            "average_age", "compression_enabled"
        ]

        for key in required_keys:
            self.assertIn(key, stats)

        self.assertEqual(stats["tier"], "cold")

    def test_clear(self):
        """Test clearing cache"""
        # Store some data
        for i in range(3):
            data = mx.zeros((10, 64))
            self.archive.store(layer_idx=i, data=data, size_bytes=2560)

        self.assertEqual(self.archive.get_entry_count(), 3)

        # Clear
        self.archive.clear()

        self.assertEqual(self.archive.get_entry_count(), 0)
        self.assertEqual(self.archive.get_total_size(), 0)


class TestColdArchiveRepr(unittest.TestCase):
    """Tests for string representation"""

    def test_repr(self):
        """Test __repr__ method"""
        archive = ColdArchive(budget_bytes=256 * 1024)

        # Store some data
        data = mx.zeros((10, 64))
        archive.store(layer_idx=0, data=data, size_bytes=4096)

        repr_str = repr(archive)

        self.assertIn("ColdArchive", repr_str)
        self.assertIn("entries=", repr_str)
        self.assertIn("size=", repr_str)
        self.assertIn("avg_age=", repr_str)


if __name__ == "__main__":
    unittest.main()
