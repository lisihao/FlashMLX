"""
Cold Archive

⚠️  DEPRECATED: 2026-03-22 - SSM cache sealed. See SSM_CACHE_DEPRECATION.md


Manages the Cold tier - long-term storage for infrequently accessed cache entries.
Optimized for capacity over latency.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import mlx.core as mx
import time


@dataclass
class ColdCacheEntry:
    """
    Entry in Cold tier cache.

    Args:
        layer_idx: Layer index
        data: Cached MLX array (may be compressed)
        size_bytes: Size in bytes
        access_count: Number of accesses (since archived)
        archived_time: When this entry was archived
        last_access_time: Last access timestamp
        is_compressed: Whether data is compressed
    """
    layer_idx: int
    data: mx.array
    size_bytes: int
    access_count: int = 0
    archived_time: float = 0.0
    last_access_time: float = 0.0
    is_compressed: bool = False

    def __post_init__(self):
        """Initialize timestamps"""
        if self.archived_time == 0.0:
            self.archived_time = time.time()
        if self.last_access_time == 0.0:
            self.last_access_time = self.archived_time


class ColdArchive:
    """
    Cold Archive - long-term storage for infrequently accessed cache entries.

    Cold tier characteristics:
    - Large capacity
    - High latency tolerable
    - Low access frequency
    - Optional compression
    - FIFO eviction policy (archive time)

    Example:
        >>> archive = ColdArchive(budget_bytes=256 * 1024 * 1024)  # 256MB
        >>> archive.store(layer_idx=0, data=cache_array, size_bytes=32768)
        >>> data = archive.retrieve(layer_idx=0)
        >>> candidates = archive.get_revival_candidates(count=10)
    """

    def __init__(self, budget_bytes: int, enable_compression: bool = False):
        """
        Initialize Cold Archive.

        Args:
            budget_bytes: Budget in bytes for Cold tier
            enable_compression: Whether to enable compression (not implemented yet)
        """
        if budget_bytes <= 0:
            raise ValueError(f"budget_bytes must be positive, got {budget_bytes}")

        self.budget_bytes = budget_bytes
        self.enable_compression = enable_compression

        # Cache storage: {layer_idx: ColdCacheEntry}
        self.cache: Dict[int, ColdCacheEntry] = {}

        # Statistics
        self.total_hits = 0
        self.total_misses = 0
        self.total_evictions = 0
        self.total_revivals = 0  # Promoted to Warm
        self.total_stores = 0

    def store(
        self,
        layer_idx: int,
        data: mx.array,
        size_bytes: int
    ) -> bool:
        """
        Store data in Cold archive.

        Args:
            layer_idx: Layer index
            data: Data to store
            size_bytes: Size in bytes

        Returns:
            True if stored successfully, False if budget exceeded
        """
        # Check if layer already exists
        if layer_idx in self.cache:
            # Update existing entry (rare in Cold tier)
            old_entry = self.cache[layer_idx]
            self.cache[layer_idx] = ColdCacheEntry(
                layer_idx=layer_idx,
                data=data,
                size_bytes=size_bytes,
                access_count=old_entry.access_count,
                archived_time=old_entry.archived_time,
                last_access_time=time.time(),
                is_compressed=self.enable_compression
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
        self.cache[layer_idx] = ColdCacheEntry(
            layer_idx=layer_idx,
            data=data,
            size_bytes=size_bytes,
            access_count=0,
            archived_time=time.time(),
            is_compressed=self.enable_compression
        )
        self.total_stores += 1
        return True

    def retrieve(self, layer_idx: int) -> Optional[mx.array]:
        """
        Retrieve data from Cold archive.

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

            # Decompress if needed (not implemented yet)
            if entry.is_compressed:
                # TODO: Implement decompression
                pass

            return entry.data
        else:
            self.total_misses += 1
            return None

    def evict(self, layer_idx: int) -> Optional[Tuple[mx.array, int]]:
        """
        Evict a specific entry from Cold archive.

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

        Uses FIFO (First In First Out) eviction policy based on archived_time.

        Args:
            required_bytes: Required space in bytes

        Returns:
            True if enough space freed, False otherwise
        """
        if not self.cache:
            return False

        # Sort by archived time (oldest first)
        sorted_entries = sorted(
            self.cache.values(),
            key=lambda e: e.archived_time
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

    def get_revival_candidates(self, count: int = 10) -> List[int]:
        """
        Get candidates for revival to Warm tier.

        Revival candidates are entries that have been accessed recently
        despite being in Cold tier.

        Args:
            count: Number of candidates to return

        Returns:
            List of layer indices (sorted by access count)
        """
        if not self.cache:
            return []

        # Sort by access count (high = good candidate for revival)
        sorted_entries = sorted(
            self.cache.values(),
            key=lambda e: e.access_count,
            reverse=True
        )

        # Filter entries with at least 1 access
        candidates = [entry for entry in sorted_entries if entry.access_count > 0]

        return [entry.layer_idx for entry in candidates[:count]]

    def get_oldest_entries(self, count: int = 10) -> List[int]:
        """
        Get oldest entries in Cold archive.

        Args:
            count: Number of entries to return

        Returns:
            List of layer indices (sorted by archived time, oldest first)
        """
        if not self.cache:
            return []

        # Sort by archived time (oldest first)
        sorted_entries = sorted(
            self.cache.values(),
            key=lambda e: e.archived_time
        )

        return [entry.layer_idx for entry in sorted_entries[:count]]

    def contains(self, layer_idx: int) -> bool:
        """Check if layer exists in Cold archive"""
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

    def get_average_age(self) -> float:
        """
        Get average age of entries in seconds.

        Returns:
            Average age in seconds
        """
        if not self.cache:
            return 0.0

        current_time = time.time()
        total_age = sum(current_time - entry.archived_time for entry in self.cache.values())
        return total_age / len(self.cache)

    def get_statistics(self) -> Dict[str, any]:
        """
        Get Cold archive statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "tier": "cold",
            "budget_bytes": self.budget_bytes,
            "total_size": self.get_total_size(),
            "utilization": self.get_utilization(),
            "entry_count": self.get_entry_count(),
            "total_hits": self.total_hits,
            "total_misses": self.total_misses,
            "hit_rate": self.get_hit_rate(),
            "total_evictions": self.total_evictions,
            "total_revivals": self.total_revivals,
            "total_stores": self.total_stores,
            "average_age": self.get_average_age(),
            "compression_enabled": self.enable_compression,
        }

    def clear(self):
        """Clear all cached entries"""
        self.cache.clear()

    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (
            f"ColdArchive("
            f"entries={stats['entry_count']}, "
            f"size={stats['total_size']}/{stats['budget_bytes']} "
            f"({stats['utilization']:.1%}), "
            f"avg_age={stats['average_age']:.1f}s)"
        )
