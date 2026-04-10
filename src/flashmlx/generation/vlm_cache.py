"""
VLM-specific Cache Configuration

Provides optimized cache strategies for Vision-Language Models.
"""

# CRITICAL: Add mlx-lm-source to path BEFORE any imports
import sys
from pathlib import Path
mlx_lm_path = Path(__file__).parent.parent.parent.parent / "mlx-lm-source"
# Remove any existing mlx_lm from sys.modules to force reimport
if 'mlx_lm' in sys.modules:
    # Remove all mlx_lm submodules
    modules_to_remove = [k for k in sys.modules.keys() if k.startswith('mlx_lm')]
    for mod in modules_to_remove:
        del sys.modules[mod]
# Add our local mlx-lm-source to the front of sys.path
sys.path.insert(0, str(mlx_lm_path))

from typing import Optional
import mlx.nn as nn

from mlx_lm.models.cache import make_prompt_cache


def create_vlm_cache(
    model,
    kv_cache: str = "standard",
    **kwargs
):
    """Create optimized cache for VLM generation.

    Args:
        model: VLM model (Qwen2VLModel)
        kv_cache: Cache strategy name
            - "standard": No compression (baseline)
            - "scored_pq": Route 5 (81% memory savings, recommended)
            - "scored_kv_direct": Route 5 + h^(0) archive (lossless recall)
            - "triple_pq": Triple-layer + PolarQuant (72% savings)
        **kwargs: Additional cache parameters (e.g., kv_warm_bits, kv_recent_size)

    Returns:
        Optimized cache instance

    Examples:
        >>> cache = create_vlm_cache(model, kv_cache="scored_pq")
        >>> response = generator.generate(prompt, cache=cache)

    Recommended strategies for VLM:
        - Development/Testing: "standard" (no compression)
        - Production (long contexts): "scored_pq" (fast, 81% savings)
        - Production (recall needed): "scored_kv_direct" (lossless reconstruction)
    """
    # Get language model layers (VLM wraps language model)
    if hasattr(model, 'language_model'):
        # Qwen2VLModel structure
        target_model = model.language_model
    else:
        # Fallback: assume model is already the language model
        target_model = model

    # Use make_prompt_cache from mlx-lm
    cache = make_prompt_cache(
        target_model,
        kv_cache=kv_cache,
        **kwargs
    )

    return cache


def get_vlm_cache_info(cache) -> dict:
    """Get cache statistics and configuration.

    Args:
        cache: Cache instance

    Returns:
        Dictionary with cache info:
            - cache_type: Type of cache
            - tokens_cached: Current number of cached tokens (if available)
    """
    if isinstance(cache, list) and len(cache) > 0:
        cache_type = type(cache[0]).__name__
        tokens = getattr(cache[0], "offset", 0)
        return {
            "cache_type": cache_type,
            "tokens_cached": tokens,
        }
    return {"cache_type": "unknown"}


# Recommended presets for common VLM scenarios
VLM_CACHE_PRESETS = {
    "fast_dev": {
        "kv_cache": "standard",
        "description": "No compression, fastest for development"
    },
    "balanced": {
        "kv_cache": "scored_pq",
        "description": "Good balance of speed and memory (81% savings)"
    },
    "ultra_long": {
        "kv_cache": "scored_kv_direct",
        "description": "For very long contexts with recall capability"
    },
    "memory_constrained": {
        "kv_cache": "triple_pq",
        "kv_warm_bits": 3,
        "description": "Maximum memory savings (76% with 3-bit)"
    },
}


def create_vlm_cache_from_preset(model, preset: str = "balanced", **overrides):
    """Create cache using a named preset.

    Args:
        model: VLM model
        preset: Preset name ("fast_dev", "balanced", "ultra_long", "memory_constrained")
        **overrides: Override preset parameters

    Returns:
        Optimized cache instance

    Examples:
        >>> cache = create_vlm_cache_from_preset(model, "balanced")
        >>> cache = create_vlm_cache_from_preset(model, "ultra_long", max_context=16384)
    """
    if preset not in VLM_CACHE_PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Valid presets: {list(VLM_CACHE_PRESETS.keys())}")

    preset_config = VLM_CACHE_PRESETS[preset].copy()
    preset_config.pop("description", None)  # Remove description field

    # Merge preset with overrides
    config = {**preset_config, **overrides}

    return create_vlm_cache(model, **config)
