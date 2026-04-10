"""
FlashMLX VLM API - Simplified Vision-Language Model Interface

Wrapper around existing test_real_weights.py functionality.
Provides one-line API for VLM usage.

Quick Start:
    >>> from flashmlx.vlm import load_vlm_components
    >>> from flashmlx.generation import VLMGenerator, create_vlm_cache
    >>>
    >>> # Load components
    >>> model, tokenizer, processor, config = load_vlm_components(
    ...     "mlx-community/Qwen2-VL-2B-Instruct-bf16"
    ... )
    >>>
    >>> # Create generator with cache
    >>> cache = create_vlm_cache(model, kv_cache="standard")
    >>> generator = VLMGenerator(model, tokenizer, config.image_token_id)
    >>>
    >>> # Generate
    >>> response = generator.generate("What is MLX?", cache=cache)

Note:
    Full high-level API (load_vlm, VLM class) coming soon.
    For now, use the component-based approach shown above.
    See examples/demo_vlm_simple.py for complete usage.
"""

# Re-export key components for easy access
from .generation import VLMGenerator, create_vlm_cache, create_vlm_cache_from_preset, VLM_CACHE_PRESETS


def load_vlm_components(model_path: str, use_4bit: bool = False):
    """Load VLM components (model, tokenizer, processor, config).

    This is a wrapper around the working test_real_weights.py loading code.

    Args:
        model_path: HuggingFace model path
        use_4bit: Use 4-bit quantized model

    Returns:
        (model, tokenizer, processor, config) tuple

    Examples:
        >>> model, tokenizer, processor, config = load_vlm_components(
        ...     "mlx-community/Qwen2-VL-2B-Instruct-bf16"
        ... )
    """
    # Import here to avoid circular dependencies
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root / "examples"))

    from test_real_weights import download_and_load_model

    return download_and_load_model(use_4bit=use_4bit)


__all__ = [
    "load_vlm_components",
    "VLMGenerator",
    "create_vlm_cache",
    "create_vlm_cache_from_preset",
    "VLM_CACHE_PRESETS",
]
