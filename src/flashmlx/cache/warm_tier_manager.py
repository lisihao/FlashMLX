"""
Warm Tier Manager

⚠️  DEPRECATED: 2026-03-22 - SSM cache sealed. See SSM_CACHE_DEPRECATION.md


Manages the Warm tier - staging area for moderately accessed cache entries.
Balances between Hot and Cold tiers.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import mlx.core as mx
import time


@dataclass
class WarmCacheEntry:
    """
    Entry in Warm tier cache.

    Args:
        layer_idx: Layer index
        data: Cached MLX array
        size_bytes: Size in bytes
        access_count: Number of accesses
        last_access_time: Last access timestamp
        promotion_score: Score for promotion to Hot tier
        demotion_score: Score for demotion to Cold tier
    """
    layer_idx: int
    data: mx.array
    size_bytes: int
    access_count: int = 0
    last_access_time: float = 0.0
    promotion_score: float = 0.0
    demotion_score: float = 0.0

    def __post_init__(self):
        """Initialize timestamps"""
        if self.last_access_time == 0.0:
            self.last_access_time = time.time()

    def update_scores(self):
        """
        Update promotion and demotion scores.

        Promotion score: based on access frequency and recency
        Demotion score: based on inactivity and low access count
        """
        current_time = time.time()
        time_since_access = current_time - self.last_access_time

        # Promotion score: high access count + recent access
        self.promotion_score = self.access_count / (1.0 + time_since_access / 3600.0)

        # Demotion score: low access count + old access
        self.demotion_score = time_since_access / (1.0 + self.access_count)


class WarmTierManager:
    """
    Warm Tier Manager - staging area between Hot and Cold tiers.

    Warm tier characteristics:
    - Medium capacity
    - Medium latency
    - Promotion to Hot on frequent access
    - Demotion to Cold on infrequent access

    Example:
        >>> manager = WarmTierManager(budget_bytes=64 * 1024 * 1024)  # 64MB
        >>> manager.store(layer_idx=0, data=cache_array, size_bytes=16384)
        >>> data = manager.retrieve(layer_idx=0)
        >>> candidates = manager.get_promotion_candidates(count=5)
    """

    def __init__(self, budget_bytes: int, promotion_threshold: float = 10.0):
        """
        Initialize Warm Tier Manager.

        Args:
            budget_bytes: Budget in bytes for Warm tier
            promotion_threshold: Threshold for promotion to Hot tier
        """
        if budget_bytes <= 0:
            raise ValueError(f"budget_bytes must be positive, got {budget_bytes}")

        self.budget_bytes = budget_bytes
        self.promotion_threshold = promotion_threshold

        # Cache storage: {layer_idx: WarmCacheEntry}
        self.cache: Dict[int, WarmCacheEntry] = {}

        # Statistics
        self.total_hits = 0
        self.total_misses = 0
        self.total_evictions = 0
        self.total_promotions = 0
        self.total_demotions = 0
        self.total_stores = 0

    def store(
        self,
        layer_idx: int,
        data: mx.array,
        size_bytes: int
    ) -> bool:
        """
        Store data in Warm tier.

        Args:
            layer_idx: Layer index
            data: Data to store
            size_bytes: Size in bytes

        Returns:
            True if stored successfully, False if budget exceeded
        """
        # Check if layer already exists
        if layer_idx in self.cache:
            # Update existing entry
            old_entry = self.cache[layer_idx]
            entry = WarmCacheEntry(
                layer_idx=layer_idx,
                data=data,
                size_bytes=size_bytes,
                access_count=old_entry.access_count + 1,
                last_access_time=time.time()
            )
            entry.update_scores()
            self.cache[layer_idx] = entry
            self.total_stores += 1
            return True

        # Check budget
        current_usage = self.get_total_size()
        if current_usage + size_bytes > self.budget_bytes:
            # Try to evict to make space
            if not self._evict_to_make_space(size_bytes):
                return False

        # Store new entry
        entry = WarmCacheEntry(
            layer_idx=layer_idx,
            data=data,
            size_bytes=size_bytes,
            access_count=1,
            last_access_time=time.time()
        )
        entry.update_scores()
        self.cache[layer_idx] = entry
        self.total_stores += 1
        return True

    def retrieve(self, layer_idx: int) -> Optional[mx.array]:
        """
        Retrieve data from Warm tier.

        Args:
            layer_idx: Layer index

        Returns:
            Data if found, None otherwise
        """
        if layer_idx in self.cache:
            entry = self.cache[layer_idx]
            # Update access metadata
            entry.access_count += 1
            entry.last_access_time = time.time()
            entry.update_scores()

            self.total_hits += 1
            return entry.data
        else:
            self.total_misses += 1
            return None

    def evict(self, layer_idx: int) -> Optional[Tuple[mx.array, int]]:
        """
        Evict a specific entry from Warm tier.

        Args:
            layer_idx: Layer index to evict

        Returns:
            (data, size_bytes) if evicted, None if not found
        """
        if layer_idx in self.cache:
            entry = self.cache.pop(layer_idx)
            self.total_evictions += 1
            return (entry.data, entry.size_bytes)
        return None

    def _evict_to_make_space(self, required_bytes: int) -> bool:
        """
        Evict entries to make space for new data.

        Uses demotion score to prioritize eviction.

        Args:
            required_bytes: Required space in bytes

        Returns:
            True if enough space freed, False otherwise
        """
        if not self.cache:
            return False

        # Update all scores
        for entry in self.cache.values():
            entry.update_scores()

        # Sort by demotion score (high score = good candidate for eviction)
        sorted_entries = sorted(
            self.cache.values(),
            key=lambda e: e.demotion_score,
            reverse=True
        )

        freed_bytes = 0
        evicted_layers = []

        for entry in sorted_entries:
            if freed_bytes >= required_bytes:
                break

            evicted_layers.append(entry.layer_idx)
            freed_bytes += entry.size_bytes

        # Evict selected entries
        for layer_idx in evicted_layers:
            self.evict(layer_idx)
            self.total_demotions += 1

        current_usage = self.get_total_size()
        return (current_usage + required_bytes) <= self.budget_bytes

    def get_promotion_candidates(self, count: int = 5) -> List[int]:
        """
        Get candidates for promotion to Hot tier.

        Args:
            count: Number of candidates to return

        Returns:
            List of layer indices (sorted by promotion score)
        """
        if not self.cache:
            return []

        # Update all scores
        for entry in self.cache.values():
            entry.update_scores()

        # Filter by promotion threshold and sort by score
        candidates = [
            entry for entry in self.cache.values()
            if entry.promotion_score >= self.promotion_threshold
        ]

        candidates.sort(key=lambda e: e.promotion_score, reverse=True)

        return [entry.layer_idx for entry in candidates[:count]]

    def get_demotion_candidates(self, count: int = 5) -> List[int]:
        """
        Get candidates for demotion to Cold tier.

        Args:
            count: Number of candidates to return

        Returns:
            List of layer indices (sorted by demotion score)
        """
        if not self.cache:
            return []

        # Update all scores
        for entry in self.cache.values():
            entry.update_scores()

        # Sort by demotion score (high = good candidate for demotion)
        sorted_entries = sorted(
            self.cache.values(),
            key=lambda e: e.demotion_score,
            reverse=True
        )

        return [entry.layer_idx for entry in sorted_entries[:count]]

    def contains(self, layer_idx: int) -> bool:
        """Check if layer exists in Warm tier"""
        return layer_idx in self.cache

    def get_total_size(self) -> int:
        """Get total size of cached data in bytes"""
        return sum(entry.size_bytes for entry in self.cache.values())

    def get_entry_count(self) -> int:
        """Get number of cached entries"""
        return len(self.cache)

    def get_utilization(self) -> float:
        """Get cache utilization ratio (0.0 - 1.0)"""
        return self.get_total_size() / self.budget_bytes if self.budget_bytes > 0 else 0.0

    def get_hit_rate(self) -> float:
        """Get cache hit rate"""
        total_accesses = self.total_hits + self.total_misses
        return self.total_hits / total_accesses if total_accesses > 0 else 0.0

    def get_statistics(self) -> Dict[str, any]:
        """
        Get Warm tier statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "tier": "warm",
            "budget_bytes": self.budget_bytes,
            "total_size": self.get_total_size(),
            "utilization": self.get_utilization(),
            "entry_count": self.get_entry_count(),
            "total_hits": self.total_hits,
            "total_misses": self.total_misses,
            "hit_rate": self.get_hit_rate(),
            "total_evictions": self.total_evictions,
            "total_promotions": self.total_promotions,
            "total_demotions": self.total_demotions,
            "total_stores": self.total_stores,
        }

    def clear(self):
        """Clear all cached entries"""
        self.cache.clear()

    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (
            f"WarmTierManager("
            f"entries={stats['entry_count']}, "
            f"size={stats['total_size']}/{stats['budget_bytes']} "
            f"({stats['utilization']:.1%}), "
            f"hit_rate={stats['hit_rate']:.1%})"
        )
