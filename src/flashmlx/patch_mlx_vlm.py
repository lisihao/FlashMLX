"""
Patch mlx-vlm with FlashMLX Cache Optimization

Monkey-patches mlx-vlm to use FlashMLX's optimized KV cache.
No system files modified - works at runtime.

Usage:
    >>> from flashmlx.patch_mlx_vlm import patch_mlx_vlm_cache
    >>> patch_mlx_vlm_cache()  # Patch once
    >>>
    >>> # Now mlx-vlm uses FlashMLX cache
    >>> from mlx_vlm import load
    >>> model, processor = load("gemma-4")
    >>>
    >>> # Create optimized cache
    >>> from mlx_vlm.models.cache import make_prompt_cache
    >>> cache = make_prompt_cache(model, kv_cache="scored_pq")  # FlashMLX!
"""

import sys
from pathlib import Path

# Ensure FlashMLX mlx-lm-source is accessible
mlx_lm_path = Path(__file__).parent.parent.parent / "mlx-lm-source"
if str(mlx_lm_path) not in sys.path:
    sys.path.insert(0, str(mlx_lm_path))


def patch_mlx_vlm_cache():
    """Patch mlx-vlm to use FlashMLX cache.

    Replaces mlx-vlm's cache module with FlashMLX's optimized version.

    This is safe because:
    1. FlashMLX interface is backward compatible
    2. Only affects current Python session
    3. No system files modified

    Examples:
        >>> patch_mlx_vlm_cache()
        >>> from mlx_vlm import load
        >>> model, processor = load("gemma-4")
        >>>
        >>> # Create FlashMLX optimized cache
        >>> from mlx_vlm.models.cache import make_prompt_cache
        >>> cache = make_prompt_cache(model, kv_cache="scored_pq")
    """
    print("Patching mlx-vlm with FlashMLX cache...")

    # Import FlashMLX cache modules
    from mlx_lm.models import cache as flashmlx_cache
    from mlx_lm.models import cache_factory as flashmlx_cache_factory

    # Import mlx-vlm
    try:
        import mlx_vlm.models.cache as mlx_vlm_cache
    except ImportError:
        print("❌ mlx-vlm not installed")
        return False

    # Backup original (optional, for unpatch)
    if not hasattr(mlx_vlm_cache, '_original_make_prompt_cache'):
        mlx_vlm_cache._original_make_prompt_cache = mlx_vlm_cache.make_prompt_cache

    # Replace with FlashMLX version
    mlx_vlm_cache.make_prompt_cache = flashmlx_cache.make_prompt_cache

    # Also add cache factory functions
    mlx_vlm_cache.make_optimized_cache = flashmlx_cache_factory.make_optimized_cache
    mlx_vlm_cache.VALID_STRATEGIES = flashmlx_cache_factory.VALID_STRATEGIES

    print("✅ mlx-vlm patched with FlashMLX cache")
    print("   Available strategies:", flashmlx_cache_factory.VALID_STRATEGIES)

    return True


def unpatch_mlx_vlm_cache():
    """Restore mlx-vlm's original cache.

    Examples:
        >>> unpatch_mlx_vlm_cache()  # Restore original
    """
    try:
        import mlx_vlm.models.cache as mlx_vlm_cache

        if hasattr(mlx_vlm_cache, '_original_make_prompt_cache'):
            mlx_vlm_cache.make_prompt_cache = mlx_vlm_cache._original_make_prompt_cache
            del mlx_vlm_cache._original_make_prompt_cache
            print("✅ mlx-vlm cache restored to original")
        else:
            print("⚠️  No backup found, already original?")

    except ImportError:
        print("❌ mlx-vlm not installed")


def check_patch_status():
    """Check if mlx-vlm is currently patched.

    Returns:
        bool: True if patched, False otherwise
    """
    try:
        import mlx_vlm.models.cache as mlx_vlm_cache

        # Check if it's the FlashMLX version
        import inspect
        source_file = inspect.getfile(mlx_vlm_cache.make_prompt_cache)

        is_patched = "mlx-lm-source" in source_file or "flashmlx" in source_file

        if is_patched:
            print(f"✅ mlx-vlm is patched (using {source_file})")
        else:
            print(f"❌ mlx-vlm is NOT patched (using {source_file})")

        return is_patched

    except ImportError:
        print("❌ mlx-vlm not installed")
        return False


__all__ = [
    "patch_mlx_vlm_cache",
    "unpatch_mlx_vlm_cache",
    "check_patch_status",
]
