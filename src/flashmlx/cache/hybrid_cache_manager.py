"""
Hybrid Cache Manager

Unified cache management system for mixed-architecture LLMs (SSM + Attention).
Routes cache operations to appropriate strategies based on layer type.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import mlx.core as mx

from .attention_matching_compressor import AttentionMatchingCompressor
from .hot_tier_manager import HotTierManager
from .warm_tier_manager import WarmTierManager
from .cold_archive import ColdArchive
from .pinned_control_state import PinnedControlState
from .budget_manager import BudgetManager, BudgetConfig, TierType
from .migration_trigger import MigrationTrigger, MigrationType


class LayerType(Enum):
    """Type of layer in mixed architecture"""
    SSM = "ssm"              # State-Space Model layer (GatedDeltaNet, Mamba, etc.)
    ATTENTION = "attention"  # Full Attention layer


@dataclass
class HybridCacheConfig:
    """
    Configuration for HybridCacheManager.

    Args:
        total_budget_bytes: Total memory budget in bytes
        hot_budget_ratio: Ratio of budget for Hot tier (default 0.15 = 15%)
        warm_budget_ratio: Ratio of budget for Warm tier (default 0.25 = 25%)
        cold_budget_ratio: Ratio of budget for Cold tier (default 0.55 = 55%)
        pinned_budget_ratio: Ratio of budget for Pinned tier (default 0.05 = 5%)
        compression_ratio: Compression ratio for Attention layers (default 3.0)
        beta_calibration: Enable β calibration for Attention compression
        hot_high_waterline: Hot tier demotion threshold (default 0.80)
        warm_high_waterline: Warm tier demotion threshold (default 0.80)
        warm_low_waterline: Warm tier promotion threshold (default 0.30)
    """
    total_budget_bytes: int
    hot_budget_ratio: float = 0.15
    warm_budget_ratio: float = 0.25
    cold_budget_ratio: float = 0.55
    pinned_budget_ratio: float = 0.05
    compression_ratio: float = 3.0
    beta_calibration: bool = True
    hot_high_waterline: float = 0.80
    warm_high_waterline: float = 0.80
    warm_low_waterline: float = 0.30

    def __post_init__(self):
        """Validate configuration"""
        if self.total_budget_bytes <= 0:
            raise ValueError(f"total_budget_bytes must be positive, got {self.total_budget_bytes}")

        # Check ratios sum to ~1.0
        total_ratio = (
            self.hot_budget_ratio +
            self.warm_budget_ratio +
            self.cold_budget_ratio +
            self.pinned_budget_ratio
        )
        if not 0.99 <= total_ratio <= 1.01:
            raise ValueError(
                f"Budget ratios must sum to 1.0, got {total_ratio:.3f} "
                f"(hot={self.hot_budget_ratio}, warm={self.warm_budget_ratio}, "
                f"cold={self.cold_budget_ratio}, pinned={self.pinned_budget_ratio})"
            )

        if self.compression_ratio < 1.0:
            raise ValueError(f"compression_ratio must be >= 1.0, got {self.compression_ratio}")


class HybridCacheManager:
    """
    Unified cache manager for mixed-architecture LLMs.

    Manages two cache strategies:
    1. SSM layers: Hybrid Memory Manager v3 (Hot/Warm/Cold/Pinned tiers)
    2. Attention layers: Attention Matching compression

    Example:
        >>> config = HybridCacheConfig(total_budget_bytes=256 * 1024 * 1024)
        >>> manager = HybridCacheManager(
        ...     config=config,
        ...     layer_types={0: LayerType.SSM, 1: LayerType.ATTENTION, ...}
        ... )
        >>> # Store SSM layer cache
        >>> manager.store_ssm(layer_idx=0, data=mx.zeros((10, 64)), size_bytes=2560)
        >>> # Store Attention layer KV cache
        >>> manager.store_attention(
        ...     layer_idx=1,
        ...     keys=mx.zeros((1, 8, 100, 64)),
        ...     values=mx.zeros((1, 8, 100, 64)),
        ...     query=mx.zeros((1, 8, 1, 64))
        ... )
    """

    def __init__(
        self,
        config: HybridCacheConfig,
        layer_types: Dict[int, LayerType]
    ):
        """
        Initialize Hybrid Cache Manager.

        Args:
            config: Cache configuration
            layer_types: Mapping from layer index to layer type
        """
        self.config = config
        self.layer_types = layer_types

        # Count layer types
        self.num_ssm_layers = sum(1 for t in layer_types.values() if t == LayerType.SSM)
        self.num_attention_layers = sum(1 for t in layer_types.values() if t == LayerType.ATTENTION)

        # Initialize SSM cache components (Hybrid Memory Manager v3)
        self._init_ssm_cache()

        # Initialize Attention cache components
        self._init_attention_cache()

        # Statistics
        self.total_stores = 0
        self.total_retrievals = 0
        self.total_migrations = 0

    def _init_ssm_cache(self):
        """Initialize SSM cache components (Hybrid Memory Manager v3)"""
        # Calculate tier budgets
        budget_config = BudgetConfig(
            hot_budget=int(self.config.total_budget_bytes * self.config.hot_budget_ratio),
            warm_budget=int(self.config.total_budget_bytes * self.config.warm_budget_ratio),
            cold_budget=int(self.config.total_budget_bytes * self.config.cold_budget_ratio),
            pinned_budget=int(self.config.total_budget_bytes * self.config.pinned_budget_ratio)
        )

        # Initialize tier managers
        self.hot_tier = HotTierManager(budget_bytes=budget_config.hot_budget)
        self.warm_tier = WarmTierManager(budget_bytes=budget_config.warm_budget)
        self.cold_tier = ColdArchive(budget_bytes=budget_config.cold_budget)

        # Initialize control components
        self.pinned_state = PinnedControlState(max_pinned_positions=100)
        self.budget_manager = BudgetManager(config=budget_config)
        self.migration_trigger = MigrationTrigger(
            hot_high_waterline=self.config.hot_high_waterline,
            warm_high_waterline=self.config.warm_high_waterline,
            warm_low_waterline=self.config.warm_low_waterline
        )

    def _init_attention_cache(self):
        """Initialize Attention cache components"""
        self.attention_compressor = AttentionMatchingCompressor(
            compression_ratio=self.config.compression_ratio,
            beta_calibration=self.config.beta_calibration
        )

    def store_ssm(
        self,
        layer_idx: int,
        data: mx.array,
        size_bytes: int,
        priority: float = 1.0
    ) -> bool:
        """
        Store SSM layer cache.

        Args:
            layer_idx: Layer index
            data: SSM state array
            size_bytes: Size in bytes
            priority: Priority score (higher = more important)

        Returns:
            True if stored successfully
        """
        if self.layer_types.get(layer_idx) != LayerType.SSM:
            raise ValueError(f"Layer {layer_idx} is not an SSM layer")

        # Try Hot tier first
        success = self.hot_tier.store(
            layer_idx=layer_idx,
            data=data,
            size_bytes=size_bytes,
            priority=priority
        )

        if success:
            self.budget_manager.allocate(TierType.HOT, layer_idx, size_bytes)
            self.total_stores += 1

            # Check for migration triggers
            self._check_migrations()

        return success

    def store_attention(
        self,
        layer_idx: int,
        keys: mx.array,
        values: mx.array,
        query: Optional[mx.array] = None,
        size_bytes: Optional[int] = None
    ) -> Tuple[mx.array, mx.array]:
        """
        Store Attention layer KV cache with compression.

        Args:
            layer_idx: Layer index
            keys: Key array (batch, num_heads, seq_len, head_dim)
            values: Value array (batch, num_heads, seq_len, head_dim)
            query: Query array for attention matching (optional)
            size_bytes: Size in bytes (optional, calculated if not provided)

        Returns:
            (compressed_keys, compressed_values)
        """
        if self.layer_types.get(layer_idx) != LayerType.ATTENTION:
            raise ValueError(f"Layer {layer_idx} is not an Attention layer")

        # Compress using Attention Matching
        compressed_keys, compressed_values = self.attention_compressor.compress_kv_cache(
            layer_idx=layer_idx,
            kv_cache=(keys, values)
        )

        self.total_stores += 1

        return compressed_keys, compressed_values

    def retrieve_ssm(self, layer_idx: int) -> Optional[mx.array]:
        """
        Retrieve SSM layer cache.

        Args:
            layer_idx: Layer index

        Returns:
            Cached data if found, None otherwise
        """
        if self.layer_types.get(layer_idx) != LayerType.SSM:
            raise ValueError(f"Layer {layer_idx} is not an SSM layer")

        self.total_retrievals += 1

        # Try Hot → Warm → Cold
        data = self.hot_tier.retrieve(layer_idx)
        if data is not None:
            return data

        data = self.warm_tier.retrieve(layer_idx)
        if data is not None:
            # Consider promoting to Hot
            self._consider_promotion(layer_idx)
            return data

        data = self.cold_tier.retrieve(layer_idx)
        if data is not None:
            # Mark for potential revival
            return data

        return None

    def get_layer_type(self, layer_idx: int) -> Optional[LayerType]:
        """Get layer type for a given index"""
        return self.layer_types.get(layer_idx)

    def _check_migrations(self):
        """Check and execute migrations based on triggers"""
        # Get tier utilizations
        hot_util = self.hot_tier.get_utilization()
        warm_util = self.warm_tier.get_utilization()
        cold_util = self.cold_tier.get_utilization()

        # Get candidates
        hot_demotion = [self.hot_tier.get_lru_candidate()] if self.hot_tier.get_lru_candidate() else []
        warm_demotion = self.warm_tier.get_demotion_candidates(count=3)
        warm_promotion = self.warm_tier.get_promotion_candidates(count=3)
        cold_revival = self.cold_tier.get_revival_candidates(count=3)

        # Evaluate migrations
        decisions = self.migration_trigger.evaluate(
            hot_utilization=hot_util,
            warm_utilization=warm_util,
            cold_utilization=cold_util,
            hot_demotion_candidates=hot_demotion,
            warm_demotion_candidates=warm_demotion,
            warm_promotion_candidates=warm_promotion,
            cold_revival_candidates=cold_revival
        )

        # Execute migrations (highest urgency first)
        for decision in decisions:
            if decision.urgency < 0.5:
                # Skip low-urgency migrations
                continue

            for layer_idx in decision.layer_indices[:1]:  # Execute one at a time
                self._execute_migration(decision.migration_type, layer_idx)

    def _execute_migration(self, migration_type: MigrationType, layer_idx: int):
        """Execute a single migration"""
        if migration_type == MigrationType.HOT_TO_WARM:
            result = self.hot_tier.evict(layer_idx)
            if result:
                data, size_bytes = result
                self.warm_tier.store(layer_idx, data, size_bytes)
                self.budget_manager.move(layer_idx, TierType.HOT, TierType.WARM, size_bytes)
                self.total_migrations += 1

        elif migration_type == MigrationType.WARM_TO_COLD:
            result = self.warm_tier.evict(layer_idx)
            if result:
                data, size_bytes = result
                self.cold_tier.store(layer_idx, data, size_bytes)
                self.budget_manager.move(layer_idx, TierType.WARM, TierType.COLD, size_bytes)
                self.total_migrations += 1

        elif migration_type == MigrationType.WARM_TO_HOT:
            result = self.warm_tier.evict(layer_idx)
            if result:
                data, size_bytes = result
                self.hot_tier.store(layer_idx, data, size_bytes)
                self.budget_manager.move(layer_idx, TierType.WARM, TierType.HOT, size_bytes)
                self.total_migrations += 1

        elif migration_type == MigrationType.COLD_TO_WARM:
            result = self.cold_tier.evict(layer_idx)
            if result:
                data, size_bytes = result
                self.warm_tier.store(layer_idx, data, size_bytes)
                self.budget_manager.move(layer_idx, TierType.COLD, TierType.WARM, size_bytes)
                self.total_migrations += 1

    def _consider_promotion(self, layer_idx: int):
        """Consider promoting a layer from Warm to Hot"""
        # Check if Hot has space
        if self.hot_tier.get_utilization() >= self.config.hot_high_waterline:
            return

        # Check promotion score
        candidates = self.warm_tier.get_promotion_candidates(count=5)
        if layer_idx in candidates:
            self._execute_migration(MigrationType.WARM_TO_HOT, layer_idx)

    def get_statistics(self) -> Dict[str, any]:
        """
        Get unified cache statistics.

        Returns:
            Dictionary with comprehensive statistics
        """
        stats = {
            "total_budget_bytes": self.config.total_budget_bytes,
            "num_ssm_layers": self.num_ssm_layers,
            "num_attention_layers": self.num_attention_layers,
            "total_stores": self.total_stores,
            "total_retrievals": self.total_retrievals,
            "total_migrations": self.total_migrations,
        }

        # SSM cache stats
        stats["ssm"] = {
            "hot": self.hot_tier.get_statistics(),
            "warm": self.warm_tier.get_statistics(),
            "cold": self.cold_tier.get_statistics(),
            "budget": self.budget_manager.get_statistics(),
        }

        # Attention cache stats
        stats["attention"] = self.attention_compressor.get_compression_stats()

        # Overall utilization
        total_used = (
            self.hot_tier.get_total_size() +
            self.warm_tier.get_total_size() +
            self.cold_tier.get_total_size()
        )
        stats["overall_utilization"] = total_used / self.config.total_budget_bytes if self.config.total_budget_bytes > 0 else 0.0

        return stats

    def clear(self):
        """Clear all caches"""
        self.hot_tier.clear()
        self.warm_tier.clear()
        self.cold_tier.clear()
        self.pinned_state.clear()
        self.attention_compressor.reset_history()

    def __repr__(self) -> str:
        return (
            f"HybridCacheManager("
            f"ssm={self.num_ssm_layers}, "
            f"attention={self.num_attention_layers}, "
            f"budget={self.config.total_budget_bytes // (1024*1024)}MB)"
        )
