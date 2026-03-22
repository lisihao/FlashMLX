"""
FlashMLX Cache Management Module

Provides advanced caching strategies for mixed-architecture LLMs:
- AttentionMatchingCompressor: KV cache compression for Attention layers
- HybridMemoryManager: Tiered memory management for SSM layers
"""

from .attention_matching_compressor import AttentionMatchingCompressor
from .budget_manager import BudgetManager, BudgetConfig, TierType
from .pinned_control_state import PinnedControlState, ControlChannel, ControlChannelType
from .hot_tier_manager import HotTierManager, HotCacheEntry
from .warm_tier_manager import WarmTierManager, WarmCacheEntry
from .cold_archive import ColdArchive, ColdCacheEntry
from .migration_trigger import (
    MigrationTrigger,
    MigrationType,
    MigrationDecision,
    SemanticBoundaryDetector,
    ChunkPredictor,
    WaterlineMonitor
)
from .hybrid_cache_manager import (
    HybridCacheManager,
    HybridCacheConfig,
    LayerType
)
from .layer_scheduler import LayerScheduler
from .managed_arrays_cache import ManagedArraysCache
from .compressed_kv_cache import CompressedKVCache
from .per_layer_ssm_cache import PerLayerSSMCache
from .per_layer_attention_cache import PerLayerAttentionCache
from .injection import (
    inject_hybrid_cache_manager,
    restore_original_cache,
    get_cache_statistics,
    create_layer_types_from_model,
    # Deprecated - kept for backward compatibility
    HybridCacheWrapper,
    LayerCacheProxy
)

__all__ = [
    # Core compression and memory management
    "AttentionMatchingCompressor",
    "BudgetManager",
    "BudgetConfig",
    "TierType",
    "PinnedControlState",
    "ControlChannel",
    "ControlChannelType",
    "HotTierManager",
    "HotCacheEntry",
    "WarmTierManager",
    "WarmCacheEntry",
    "ColdArchive",
    "ColdCacheEntry",
    "MigrationTrigger",
    "MigrationType",
    "MigrationDecision",
    "SemanticBoundaryDetector",
    "ChunkPredictor",
    "WaterlineMonitor",

    # Hybrid cache manager
    "HybridCacheManager",
    "HybridCacheConfig",
    "LayerType",
    "LayerScheduler",

    # Per-layer cache implementations (NEW)
    "PerLayerSSMCache",
    "PerLayerAttentionCache",

    # Legacy global cache managers (deprecated but kept for compatibility)
    "ManagedArraysCache",
    "CompressedKVCache",

    # Injection utilities
    "inject_hybrid_cache_manager",
    "restore_original_cache",
    "get_cache_statistics",
    "create_layer_types_from_model",

    # Deprecated wrappers (kept for backward compatibility)
    "HybridCacheWrapper",
    "LayerCacheProxy",
]
