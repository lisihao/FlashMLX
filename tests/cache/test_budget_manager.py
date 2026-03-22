"""
Unit tests for BudgetManager (Task #69)

Tests byte-based memory budget management across Hot/Warm/Cold/Pinned tiers.
"""

import unittest

from flashmlx.cache.budget_manager import BudgetManager, BudgetConfig, TierType


class TestBudgetConfig(unittest.TestCase):
    """Tests for BudgetConfig dataclass"""

    def test_valid_config(self):
        """Test valid budget configuration"""
        config = BudgetConfig(
            hot_budget=16 * 1024,    # 16KB
            warm_budget=64 * 1024,   # 64KB
            cold_budget=256 * 1024,  # 256KB
            pinned_budget=4 * 1024   # 4KB
        )

        self.assertEqual(config.hot_budget, 16 * 1024)
        self.assertEqual(config.warm_budget, 64 * 1024)
        self.assertEqual(config.cold_budget, 256 * 1024)
        self.assertEqual(config.pinned_budget, 4 * 1024)

    def test_total_budget(self):
        """Test total budget calculation"""
        config = BudgetConfig(
            hot_budget=16 * 1024,
            warm_budget=64 * 1024,
            cold_budget=256 * 1024,
            pinned_budget=4 * 1024
        )

        expected_total = (16 + 64 + 256 + 4) * 1024
        self.assertEqual(config.total_budget, expected_total)

    def test_invalid_hot_budget(self):
        """Test that negative hot_budget raises error"""
        with self.assertRaises(ValueError):
            BudgetConfig(
                hot_budget=-1,
                warm_budget=64 * 1024,
                cold_budget=256 * 1024,
                pinned_budget=4 * 1024
            )

    def test_invalid_warm_budget(self):
        """Test that zero warm_budget raises error"""
        with self.assertRaises(ValueError):
            BudgetConfig(
                hot_budget=16 * 1024,
                warm_budget=0,
                cold_budget=256 * 1024,
                pinned_budget=4 * 1024
            )

    def test_invalid_pinned_budget(self):
        """Test that negative pinned_budget raises error"""
        with self.assertRaises(ValueError):
            BudgetConfig(
                hot_budget=16 * 1024,
                warm_budget=64 * 1024,
                cold_budget=256 * 1024,
                pinned_budget=-1
            )


class TestBudgetManagerBasic(unittest.TestCase):
    """Basic functionality tests for BudgetManager"""

    def setUp(self):
        """Set up test configuration"""
        self.config = BudgetConfig(
            hot_budget=16 * 1024,
            warm_budget=64 * 1024,
            cold_budget=256 * 1024,
            pinned_budget=4 * 1024
        )
        self.manager = BudgetManager(self.config)

    def test_initialization(self):
        """Test BudgetManager initialization"""
        self.assertEqual(self.manager.config, self.config)
        self.assertEqual(self.manager.migration_count, 0)
        self.assertEqual(self.manager.total_allocations, 0)
        self.assertEqual(self.manager.total_deallocations, 0)

        # All tiers should be empty
        for tier in TierType:
            self.assertEqual(self.manager.get_tier_usage(tier), 0)
            self.assertEqual(self.manager.get_utilization(tier), 0.0)

    def test_allocate_success(self):
        """Test successful allocation"""
        success = self.manager.allocate(TierType.HOT, layer_idx=0, size=8192)

        self.assertTrue(success)
        self.assertEqual(self.manager.get_tier_usage(TierType.HOT), 8192)
        self.assertEqual(self.manager.total_allocations, 1)

    def test_allocate_exceeds_budget(self):
        """Test allocation that exceeds budget"""
        # Allocate full budget
        self.manager.allocate(TierType.HOT, layer_idx=0, size=16 * 1024)

        # Try to allocate more
        success = self.manager.allocate(TierType.HOT, layer_idx=1, size=1024)

        self.assertFalse(success)
        self.assertEqual(self.manager.total_allocations, 1)  # Only first succeeded

    def test_allocate_negative_size(self):
        """Test that negative size raises error"""
        with self.assertRaises(ValueError):
            self.manager.allocate(TierType.HOT, layer_idx=0, size=-100)

    def test_deallocate_success(self):
        """Test successful deallocation"""
        # Allocate first
        self.manager.allocate(TierType.HOT, layer_idx=0, size=8192)

        # Deallocate
        deallocated_size = self.manager.deallocate(TierType.HOT, layer_idx=0)

        self.assertEqual(deallocated_size, 8192)
        self.assertEqual(self.manager.get_tier_usage(TierType.HOT), 0)
        self.assertEqual(self.manager.total_deallocations, 1)

    def test_deallocate_nonexistent(self):
        """Test deallocation of non-existent layer"""
        deallocated_size = self.manager.deallocate(TierType.HOT, layer_idx=999)

        self.assertIsNone(deallocated_size)
        self.assertEqual(self.manager.total_deallocations, 0)

    def test_multiple_allocations_same_tier(self):
        """Test multiple allocations in same tier"""
        self.manager.allocate(TierType.HOT, layer_idx=0, size=4096)
        self.manager.allocate(TierType.HOT, layer_idx=1, size=4096)
        self.manager.allocate(TierType.HOT, layer_idx=2, size=4096)

        self.assertEqual(self.manager.get_tier_usage(TierType.HOT), 12288)
        self.assertEqual(self.manager.total_allocations, 3)


class TestBudgetManagerMigration(unittest.TestCase):
    """Tests for tier migration"""

    def setUp(self):
        """Set up test configuration"""
        self.config = BudgetConfig(
            hot_budget=16 * 1024,
            warm_budget=64 * 1024,
            cold_budget=256 * 1024,
            pinned_budget=4 * 1024
        )
        self.manager = BudgetManager(self.config)

    def test_move_success(self):
        """Test successful move between tiers"""
        # Allocate in HOT
        self.manager.allocate(TierType.HOT, layer_idx=0, size=8192)

        # Move to WARM
        success = self.manager.move(
            layer_idx=0,
            from_tier=TierType.HOT,
            to_tier=TierType.WARM
        )

        self.assertTrue(success)
        self.assertEqual(self.manager.get_tier_usage(TierType.HOT), 0)
        self.assertEqual(self.manager.get_tier_usage(TierType.WARM), 8192)
        self.assertEqual(self.manager.migration_count, 1)

    def test_move_with_size_change(self):
        """Test move with size change (e.g., compression)"""
        # Allocate 8KB in HOT
        self.manager.allocate(TierType.HOT, layer_idx=0, size=8192)

        # Move to WARM with compression to 4KB
        success = self.manager.move(
            layer_idx=0,
            from_tier=TierType.HOT,
            to_tier=TierType.WARM,
            new_size=4096
        )

        self.assertTrue(success)
        self.assertEqual(self.manager.get_tier_usage(TierType.HOT), 0)
        self.assertEqual(self.manager.get_tier_usage(TierType.WARM), 4096)

    def test_move_budget_exceeded(self):
        """Test move fails when target tier budget exceeded"""
        # Fill WARM tier
        for i in range(8):
            self.manager.allocate(TierType.WARM, layer_idx=i, size=8 * 1024)

        # Allocate in HOT
        self.manager.allocate(TierType.HOT, layer_idx=100, size=8192)

        # Try to move to full WARM tier
        success = self.manager.move(
            layer_idx=100,
            from_tier=TierType.HOT,
            to_tier=TierType.WARM
        )

        self.assertFalse(success)
        # Should remain in HOT
        self.assertEqual(self.manager.get_tier_usage(TierType.HOT), 8192)

    def test_move_nonexistent_layer(self):
        """Test move of non-existent layer"""
        success = self.manager.move(
            layer_idx=999,
            from_tier=TierType.HOT,
            to_tier=TierType.WARM
        )

        self.assertFalse(success)
        self.assertEqual(self.manager.migration_count, 0)


class TestBudgetManagerUtilization(unittest.TestCase):
    """Tests for utilization tracking"""

    def setUp(self):
        """Set up test configuration"""
        self.config = BudgetConfig(
            hot_budget=16 * 1024,
            warm_budget=64 * 1024,
            cold_budget=256 * 1024,
            pinned_budget=4 * 1024
        )
        self.manager = BudgetManager(self.config)

    def test_utilization_empty(self):
        """Test utilization when tier is empty"""
        utilization = self.manager.get_utilization(TierType.HOT)
        self.assertEqual(utilization, 0.0)

    def test_utilization_half_full(self):
        """Test utilization at 50%"""
        self.manager.allocate(TierType.HOT, layer_idx=0, size=8 * 1024)

        utilization = self.manager.get_utilization(TierType.HOT)
        self.assertAlmostEqual(utilization, 0.5)

    def test_utilization_full(self):
        """Test utilization at 100%"""
        self.manager.allocate(TierType.HOT, layer_idx=0, size=16 * 1024)

        utilization = self.manager.get_utilization(TierType.HOT)
        self.assertAlmostEqual(utilization, 1.0)

    def test_is_over_budget_false(self):
        """Test is_over_budget when under threshold"""
        self.manager.allocate(TierType.HOT, layer_idx=0, size=8 * 1024)

        self.assertFalse(self.manager.is_over_budget(TierType.HOT, threshold=0.9))

    def test_is_over_budget_true(self):
        """Test is_over_budget when over threshold"""
        self.manager.allocate(TierType.HOT, layer_idx=0, size=15 * 1024)

        self.assertTrue(self.manager.is_over_budget(TierType.HOT, threshold=0.9))

    def test_get_available_space(self):
        """Test available space calculation"""
        self.manager.allocate(TierType.HOT, layer_idx=0, size=10 * 1024)

        available = self.manager.get_available_space(TierType.HOT)
        self.assertEqual(available, 6 * 1024)

    def test_get_available_space_full(self):
        """Test available space when tier is full"""
        self.manager.allocate(TierType.HOT, layer_idx=0, size=16 * 1024)

        available = self.manager.get_available_space(TierType.HOT)
        self.assertEqual(available, 0)


class TestBudgetManagerStatistics(unittest.TestCase):
    """Tests for statistics tracking"""

    def setUp(self):
        """Set up test configuration"""
        self.config = BudgetConfig(
            hot_budget=16 * 1024,
            warm_budget=64 * 1024,
            cold_budget=256 * 1024,
            pinned_budget=4 * 1024
        )
        self.manager = BudgetManager(self.config)

    def test_statistics_structure(self):
        """Test statistics dictionary structure"""
        stats = self.manager.get_statistics()

        # Check all required keys
        required_keys = [
            "hot_usage", "hot_budget", "hot_utilization",
            "warm_usage", "warm_budget", "warm_utilization",
            "cold_usage", "cold_budget", "cold_utilization",
            "pinned_usage", "pinned_budget", "pinned_utilization",
            "total_budget", "total_usage",
            "migration_count", "total_allocations", "total_deallocations"
        ]

        for key in required_keys:
            self.assertIn(key, stats)

    def test_statistics_after_operations(self):
        """Test statistics reflect actual operations"""
        # Perform operations
        self.manager.allocate(TierType.HOT, layer_idx=0, size=8192)
        self.manager.allocate(TierType.WARM, layer_idx=1, size=16384)
        self.manager.move(layer_idx=0, from_tier=TierType.HOT, to_tier=TierType.COLD)
        self.manager.deallocate(TierType.WARM, layer_idx=1)

        stats = self.manager.get_statistics()

        self.assertEqual(stats["hot_usage"], 0)  # Moved away
        self.assertEqual(stats["warm_usage"], 0)  # Deallocated
        self.assertEqual(stats["cold_usage"], 8192)  # Moved here
        self.assertEqual(stats["total_usage"], 8192)
        self.assertEqual(stats["migration_count"], 1)
        # total_allocations = 3 because move() internally calls allocate()
        self.assertEqual(stats["total_allocations"], 3)
        # total_deallocations = 2 because move() deallocates from source + explicit deallocate
        self.assertEqual(stats["total_deallocations"], 2)

    def test_recommend_eviction_candidate(self):
        """Test eviction candidate recommendation"""
        # Allocate multiple layers
        self.manager.allocate(TierType.HOT, layer_idx=5, size=4096)
        self.manager.allocate(TierType.HOT, layer_idx=2, size=4096)
        self.manager.allocate(TierType.HOT, layer_idx=8, size=4096)

        # Should recommend the first allocated (FIFO heuristic)
        candidate = self.manager.recommend_eviction_candidate(TierType.HOT)

        # Should return the minimum layer index (simplest FIFO)
        self.assertEqual(candidate, 2)

    def test_recommend_eviction_empty_tier(self):
        """Test eviction recommendation on empty tier"""
        candidate = self.manager.recommend_eviction_candidate(TierType.HOT)
        self.assertIsNone(candidate)


class TestBudgetManagerRepr(unittest.TestCase):
    """Tests for string representation"""

    def test_repr(self):
        """Test __repr__ method"""
        config = BudgetConfig(
            hot_budget=16 * 1024,
            warm_budget=64 * 1024,
            cold_budget=256 * 1024,
            pinned_budget=4 * 1024
        )
        manager = BudgetManager(config)

        # Allocate some memory
        manager.allocate(TierType.HOT, layer_idx=0, size=8192)

        repr_str = repr(manager)

        # Should contain key information
        self.assertIn("BudgetManager", repr_str)
        self.assertIn("hot=", repr_str)
        self.assertIn("warm=", repr_str)
        self.assertIn("cold=", repr_str)


if __name__ == "__main__":
    unittest.main()
