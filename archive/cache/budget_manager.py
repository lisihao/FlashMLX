"""
Budget Manager for Hybrid Memory Manager v3

Manages memory budgets across Hot/Warm/Cold tiers using byte-based allocation.
Triggers migration when budgets are exceeded.
"""

from dataclasses import dataclass
from typing import Dict, Optional
from enum import Enum


class TierType(Enum):
    """Memory tier types"""
    HOT = "hot"      # Active states, highest priority
    WARM = "warm"    # Staging area, medium priority
    COLD = "cold"    # Archive, lowest priority
    PINNED = "pinned"  # Control channels, never evicted


@dataclass
class BudgetConfig:
    """Budget configuration for memory tiers"""
    hot_budget: int     # Hot tier budget in bytes
    warm_budget: int    # Warm tier budget in bytes
    cold_budget: int    # Cold tier budget in bytes
    pinned_budget: int  # Pinned tier budget in bytes

    def __post_init__(self):
        """Validate budget configuration"""
        if self.hot_budget <= 0:
            raise ValueError(f"hot_budget must be positive, got {self.hot_budget}")
        if self.warm_budget <= 0:
            raise ValueError(f"warm_budget must be positive, got {self.warm_budget}")
        if self.cold_budget <= 0:
            raise ValueError(f"cold_budget must be positive, got {self.cold_budget}")
        if self.pinned_budget < 0:
            raise ValueError(f"pinned_budget must be non-negative, got {self.pinned_budget}")

    @property
    def total_budget(self) -> int:
        """Total budget across all tiers"""
        return self.hot_budget + self.warm_budget + self.cold_budget + self.pinned_budget


class BudgetManager:
    """
    Manages memory budgets across Hot/Warm/Cold/Pinned tiers.

    Responsibilities:
    - Track memory usage per tier
    - Detect budget overruns
    - Recommend migration actions
    - Provide budget utilization statistics

    Example:
        >>> config = BudgetConfig(
        ...     hot_budget=16 * 1024,    # 16KB
        ...     warm_budget=64 * 1024,   # 64KB
        ...     cold_budget=256 * 1024,  # 256KB
        ...     pinned_budget=4 * 1024   # 4KB
        ... )
        >>> manager = BudgetManager(config)
        >>> manager.allocate(TierType.HOT, layer_idx=0, size=8192)
        >>> manager.get_utilization(TierType.HOT)
        0.5
    """

    def __init__(self, config: BudgetConfig):
        """
        Initialize Budget Manager.

        Args:
            config: Budget configuration
        """
        self.config = config

        # Track allocated bytes per tier per layer
        # Format: {tier: {layer_idx: bytes}}
        self.allocations: Dict[TierType, Dict[int, int]] = {
            TierType.HOT: {},
            TierType.WARM: {},
            TierType.COLD: {},
            TierType.PINNED: {},
        }

        # Statistics
        self.migration_count = 0
        self.total_allocations = 0
        self.total_deallocations = 0

    def allocate(self, tier: TierType, layer_idx: int, size: int) -> bool:
        """
        Allocate memory in a tier for a specific layer.

        Args:
            tier: Target tier
            layer_idx: Layer index
            size: Size in bytes

        Returns:
            True if allocation succeeded, False if budget exceeded
        """
        if size < 0:
            raise ValueError(f"size must be non-negative, got {size}")

        current_usage = self.get_tier_usage(tier)
        budget = self._get_tier_budget(tier)

        # Check if allocation would exceed budget
        if current_usage + size > budget:
            return False

        # Allocate
        self.allocations[tier][layer_idx] = size
        self.total_allocations += 1

        return True

    def deallocate(self, tier: TierType, layer_idx: int) -> Optional[int]:
        """
        Deallocate memory from a tier for a specific layer.

        Args:
            tier: Source tier
            layer_idx: Layer index

        Returns:
            Size that was deallocated, or None if layer not found
        """
        if layer_idx in self.allocations[tier]:
            size = self.allocations[tier].pop(layer_idx)
            self.total_deallocations += 1
            return size

        return None

    def move(
        self,
        layer_idx: int,
        from_tier: TierType,
        to_tier: TierType,
        new_size: Optional[int] = None
    ) -> bool:
        """
        Move allocation from one tier to another.

        Args:
            layer_idx: Layer index
            from_tier: Source tier
            to_tier: Target tier
            new_size: New size after migration (if None, keeps original size)

        Returns:
            True if move succeeded, False if budget constraints prevent it
        """
        # Get current size
        if layer_idx not in self.allocations[from_tier]:
            return False

        current_size = self.allocations[from_tier][layer_idx]
        target_size = new_size if new_size is not None else current_size

        # Check if target tier has budget
        if not self.allocate(to_tier, layer_idx, target_size):
            return False

        # Deallocate from source tier
        self.deallocate(from_tier, layer_idx)

        self.migration_count += 1

        return True

    def get_tier_usage(self, tier: TierType) -> int:
        """
        Get current memory usage for a tier.

        Args:
            tier: Tier type

        Returns:
            Total bytes allocated in this tier
        """
        return sum(self.allocations[tier].values())

    def get_utilization(self, tier: TierType) -> float:
        """
        Get utilization percentage for a tier.

        Args:
            tier: Tier type

        Returns:
            Utilization ratio (0.0 - 1.0)
        """
        usage = self.get_tier_usage(tier)
        budget = self._get_tier_budget(tier)

        return usage / budget if budget > 0 else 0.0

    def is_over_budget(self, tier: TierType, threshold: float = 1.0) -> bool:
        """
        Check if a tier is over budget.

        Args:
            tier: Tier type
            threshold: Utilization threshold (default 1.0 = 100%)

        Returns:
            True if utilization exceeds threshold
        """
        return self.get_utilization(tier) > threshold

    def get_available_space(self, tier: TierType) -> int:
        """
        Get available space in a tier.

        Args:
            tier: Tier type

        Returns:
            Available bytes
        """
        budget = self._get_tier_budget(tier)
        usage = self.get_tier_usage(tier)

        return max(0, budget - usage)

    def recommend_eviction_candidate(self, tier: TierType) -> Optional[int]:
        """
        Recommend a layer to evict from a tier (simple LRU heuristic).

        Args:
            tier: Tier type

        Returns:
            Layer index to evict, or None if tier is empty
        """
        if not self.allocations[tier]:
            return None

        # Simple heuristic: evict the first allocated layer (FIFO)
        # In a real implementation, this would consider access patterns
        return min(self.allocations[tier].keys())

    def get_statistics(self) -> Dict[str, any]:
        """
        Get budget manager statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "hot_usage": self.get_tier_usage(TierType.HOT),
            "hot_budget": self.config.hot_budget,
            "hot_utilization": self.get_utilization(TierType.HOT),
            "warm_usage": self.get_tier_usage(TierType.WARM),
            "warm_budget": self.config.warm_budget,
            "warm_utilization": self.get_utilization(TierType.WARM),
            "cold_usage": self.get_tier_usage(TierType.COLD),
            "cold_budget": self.config.cold_budget,
            "cold_utilization": self.get_utilization(TierType.COLD),
            "pinned_usage": self.get_tier_usage(TierType.PINNED),
            "pinned_budget": self.config.pinned_budget,
            "pinned_utilization": self.get_utilization(TierType.PINNED),
            "total_budget": self.config.total_budget,
            "total_usage": sum([
                self.get_tier_usage(TierType.HOT),
                self.get_tier_usage(TierType.WARM),
                self.get_tier_usage(TierType.COLD),
                self.get_tier_usage(TierType.PINNED),
            ]),
            "migration_count": self.migration_count,
            "total_allocations": self.total_allocations,
            "total_deallocations": self.total_deallocations,
        }

    def _get_tier_budget(self, tier: TierType) -> int:
        """Get budget for a specific tier"""
        if tier == TierType.HOT:
            return self.config.hot_budget
        elif tier == TierType.WARM:
            return self.config.warm_budget
        elif tier == TierType.COLD:
            return self.config.cold_budget
        elif tier == TierType.PINNED:
            return self.config.pinned_budget
        else:
            raise ValueError(f"Unknown tier type: {tier}")

    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (
            f"BudgetManager("
            f"hot={stats['hot_usage']}/{stats['hot_budget']} "
            f"({stats['hot_utilization']:.1%}), "
            f"warm={stats['warm_usage']}/{stats['warm_budget']} "
            f"({stats['warm_utilization']:.1%}), "
            f"cold={stats['cold_usage']}/{stats['cold_budget']} "
            f"({stats['cold_utilization']:.1%}))"
        )
