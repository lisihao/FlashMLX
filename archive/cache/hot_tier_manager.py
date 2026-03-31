"""
Hot Tier Manager

⚠️  DEPRECATED: 2026-03-22 - SSM cache sealed. See SSM_CACHE_DEPRECATION.md


Manages the Hot tier - active, frequently accessed cache entries.
Optimized for low-latency access and small working set.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import mlx.core as mx
import time


@dataclass
class HotCacheEntry:
    """
    Entry in Hot tier cache.

    Args:
        layer_idx: Layer index
        data: Cached MLX array
        size_bytes: Size in bytes
        access_count: Number of accesses
        last_access_time: Last access timestamp
        priority: Priority score (higher = more important)
    """
    layer_idx: int
    data: mx.array
    size_bytes: int
    access_count: int = 0
    last_access_time: float = 0.0
    priority: float = 1.0

    def __post_init__(self):
        """Initialize timestamps"""
        if self.last_access_time == 0.0:
            self.last_access_time = time.time()


class HotTierManager:
    """
    Hot Tier Manager - manages active, frequently accessed cache entries.

    Hot tier characteristics:
    - Small capacity (limited budget)
    - Low latency access
    - High access frequency
    - LRU eviction policy

    Example:
        >>> manager = HotTierManager(budget_bytes=16 * 1024 * 1024)  # 16MB
        >>> manager.store(layer_idx=0, data=cache_array, size_bytes=4096)
        >>> data = manager.retrieve(layer_idx=0)
        >>> manager.get_utilization()
        0.25
    """

    def __init__(self, budget_bytes: int):
        """
        Initialize Hot Tier Manager.

        Args:
            budget_bytes: Budget in bytes for Hot tier
        """
        if budget_bytes <= 0:
            raise ValueError(f"budget_bytes must be positive, got {budget_bytes}")

        self.budget_bytes = budget_bytes

        # Cache storage: {layer_idx: HotCacheEntry}
        self.cache: Dict[int, HotCacheEntry] = {}

        # Statistics
        self.total_hits = 0
        self.total_misses = 0
        self.total_evictions = 0
        self.total_stores = 0

    def store(
        self,
        layer_idx: int,
        data: mx.array,
        size_bytes: int,
        priority: float = 1.0
    ) -> bool:
        """
        Store data in Hot tier.

        Args:
            layer_idx: Layer index
            data: Data to store
            size_bytes: Size in bytes
            priority: Priority score (higher = more important)

        Returns:
            True if stored successfully, False if budget exceeded
        """
        # Check if layer already exists
        if layer_idx in self.cache:
            # Update existing entry
            old_entry = self.cache[layer_idx]
            self.cache[layer_idx] = HotCacheEntry(
                layer_idx=layer_idx,
                data=data,
                size_bytes=size_bytes,
                access_count=old_entry.access_count + 1,
                last_access_time=time.time(),
                priority=priority
            )
            self.total_stores += 1
            return True

        # Check budget
        current_usage = self.get_total_size()
        if current_usage + size_bytes > self.budget_bytes:
            # Try to evict to make space
            if not self._evict_to_make_space(size_bytes):
                return False

        # Store new entry
        self.cache[layer_idx] = HotCacheEntry(
            layer_idx=layer_idx,
            data=data,
            size_bytes=size_bytes,
            access_count=1,
            last_access_time=time.time(),
            priority=priority
        )
        self.total_stores += 1
        return True

    def retrieve(self, layer_idx: int) -> Optional[mx.array]:
        """
        Retrieve data from Hot tier.

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

            self.total_hits += 1
            return entry.data
        else:
            self.total_misses += 1
            return None

    def evict(self, layer_idx: int) -> Optional[Tuple[mx.array, int]]:
        """
        Evict a specific entry from Hot tier.

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

        Uses LRU (Least Recently Used) eviction policy.

        Args:
            required_bytes: Required space in bytes

        Returns:
            True if enough space freed, False otherwise
        """
        if not self.cache:
            return False

        # Sort by last access time (LRU)
        sorted_entries = sorted(
            self.cache.values(),
            key=lambda e: (e.priority, e.last_access_time)  # Low priority first, then old first
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

        current_usage = self.get_total_size()
        return (current_usage + required_bytes) <= self.budget_bytes

    def get_lru_candidate(self) -> Optional[int]:
        """
        Get LRU eviction candidate.

        Returns:
            Layer index of LRU candidate, or None if cache is empty
        """
        if not self.cache:
            return None

        # Find entry with lowest priority and oldest access time
        lru_entry = min(
            self.cache.values(),
            key=lambda e: (e.priority, e.last_access_time)
        )
        return lru_entry.layer_idx

    def contains(self, layer_idx: int) -> bool:
        """Check if layer exists in Hot tier"""
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
        Get Hot tier statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "tier": "hot",
            "budget_bytes": self.budget_bytes,
            "total_size": self.get_total_size(),
            "utilization": self.get_utilization(),
            "entry_count": self.get_entry_count(),
            "total_hits": self.total_hits,
            "total_misses": self.total_misses,
            "hit_rate": self.get_hit_rate(),
            "total_evictions": self.total_evictions,
            "total_stores": self.total_stores,
        }

    def clear(self):
        """Clear all cached entries"""
        self.cache.clear()

    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (
            f"HotTierManager("
            f"entries={stats['entry_count']}, "
            f"size={stats['total_size']}/{stats['budget_bytes']} "
            f"({stats['utilization']:.1%}), "
            f"hit_rate={stats['hit_rate']:.1%})"
        )
