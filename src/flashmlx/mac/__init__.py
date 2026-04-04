"""
FlashMLX MAC-Attention — Route 6: Decode acceleration via computation reuse.

Ported from MAC-Attention (arXiv 2604.00235) CUDA to Metal/MLX.

Three-stage decode pipeline:
  1. Match:  L2 distance search in ring cache (Metal kernel)
  2. Amend:  Partial attention + merge with cached (pure MLX)
  3. Complete: Rectification + ring cache update (pure MLX)

Usage:
    from flashmlx.mac import MACDecodeWrapper

    mac = MACDecodeWrapper(
        max_requests=64, capacity=1024,
        num_heads=32, num_kv_heads=8, head_dim=128,
    )
    output = mac(queries, keys, values, req_ids)
"""

from .attention import (
    downdate_attention,
    mac_fused_partial_attention,
    mac_partial_attention,
    mac_rectify_and_update,
    mac_windowed_attention,
    merge_attention_states,
)
from .match import mac_ring_match, mac_ring_match_reference
from .ring_cache import MACRingCache
from .wrapper import MACDecodeStats, MACDecodeWrapper

__all__ = [
    # Core wrapper
    "MACDecodeWrapper",
    "MACDecodeStats",
    # Ring cache
    "MACRingCache",
    # Match (Metal kernel + reference)
    "mac_ring_match",
    "mac_ring_match_reference",
    # Attention operations
    "mac_fused_partial_attention",
    "mac_partial_attention",
    "mac_windowed_attention",
    "merge_attention_states",
    "downdate_attention",
    "mac_rectify_and_update",
]
