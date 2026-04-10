"""
FlashMLX Text Generation

Text generation utilities for Language Models and Vision-Language Models.
"""

from .vlm_generator import VLMGenerator
from .vlm_cache import (
    create_vlm_cache,
    create_vlm_cache_from_preset,
    get_vlm_cache_info,
    VLM_CACHE_PRESETS,
)

__all__ = [
    "VLMGenerator",
    "create_vlm_cache",
    "create_vlm_cache_from_preset",
    "get_vlm_cache_info",
    "VLM_CACHE_PRESETS",
]
