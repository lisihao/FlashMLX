"""
FlashMLX Cache Management Module

Provides advanced caching strategies for Attention layers:
- AttentionMatchingCompressor: KV cache compression for Attention layers
- CompressedKVCache: Compressed KV cache wrapper

DEPRECATED (2026-03-22):
- SSM cache components (HybridCacheManager, SimplifiedSSMCacheManager, etc.)
- See SSM_CACHE_DEPRECATION.md for details
"""

# ============================================================================
# ACTIVE: Attention Matching & KV Cache Compression
# ============================================================================
from .attention_matching_compressor import AttentionMatchingCompressor
from .budget_manager import BudgetManager, BudgetConfig, TierType
from .pinned_control_state import PinnedControlState, ControlChannel, ControlChannelType
from .compressed_kv_cache import CompressedKVCache
from .per_layer_attention_cache import PerLayerAttentionCache

# Simple Injection (Attention-only models)
from .simple_injection import (
    inject_attention_matching,
    get_compression_stats,
    CompressedArraysCache
)

# Compacted KV Cache (Attention Matching v2)
from .compacted_kv_cache import (
    CompactedKVCache,
    CompactedKVCacheLayer,
    create_compacted_cache_list,
)
from .attention_patcher import (
    repeat_kv,
    patch_attention_for_compacted_cache,
)
from .compaction_algorithm import (
    HighestAttentionKeysCompaction,
    create_compaction_algorithm,
)

# ============================================================================
# DEPRECATED: SSM Cache Components (封存但保留代码)
# ============================================================================
# 以下组件已废弃，入口已封闭，代码保留供未来可能的需求
# Deprecated on: 2026-03-22
# Reason: 场景与 ThunderLLAMA prefix caching 重叠 + GPU 稳定性问题
# See: SSM_CACHE_DEPRECATION.md
#
# from .hot_tier_manager import HotTierManager, HotCacheEntry
# from .warm_tier_manager import WarmTierManager, WarmCacheEntry
# from .cold_archive import ColdArchive, ColdCacheEntry
# from .migration_trigger import (...)
# from .hybrid_cache_manager import (...)
# from .layer_scheduler import LayerScheduler
# from .managed_arrays_cache import ManagedArraysCache
# from .per_layer_ssm_cache import PerLayerSSMCache
# from .simplified_ssm_cache import (...)
# from .injection import inject_hybrid_cache_manager, ...
# ============================================================================

__all__ = [
    # ========================================================================
    # ACTIVE: Attention Matching & KV Cache Compression
    # ========================================================================
    "AttentionMatchingCompressor",
    "BudgetManager",
    "BudgetConfig",
    "TierType",
    "PinnedControlState",
    "ControlChannel",
    "ControlChannelType",
    "CompressedKVCache",
    "PerLayerAttentionCache",

    # Simple Injection (Attention-only models)
    "inject_attention_matching",
    "get_compression_stats",
    "CompressedArraysCache",

    # Compacted KV Cache (Attention Matching v2)
    "CompactedKVCache",
    "CompactedKVCacheLayer",
    "create_compacted_cache_list",
    "repeat_kv",
    "patch_attention_for_compacted_cache",
    "HighestAttentionKeysCompaction",
    "create_compaction_algorithm",

    # ========================================================================
    # DEPRECATED: SSM Cache Components (已封闭入口)
    # ========================================================================
    # 以下接口已移除，不再对外暴露
    # 代码保留在源文件中但无法通过 import 访问
    # See: SSM_CACHE_DEPRECATION.md
    # ========================================================================
]
