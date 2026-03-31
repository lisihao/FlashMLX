"""
FlashMLX Cache — re-exports from enhanced mlx-lm.

Single source of truth: all cache implementation lives in
mlx-lm-source/mlx_lm/models/. This module provides convenient imports.
"""

# Core cache types
from mlx_lm.models.cache import (
    KVCache,
    RotatingKVCache,
    make_prompt_cache,
)

# Cache factory (smart strategy selection)
from mlx_lm.models.cache_factory import (
    VALID_STRATEGIES,
    get_cache_info,
    make_optimized_cache,
)

# Triple-layer cache (the workhorse)
from mlx_lm.models.triple_layer_cache import TripleLayerKVCache

# Quantization strategies
from mlx_lm.models.quantization_strategies import (
    Q4_0Quantizer,
    Q8_0Quantizer,
    PolarQuantizer,
    TurboQuantizer,
    NoOpQuantizer,
    QuantizationStrategy,
    get_quantizer,
)

# AM calibrator
from mlx_lm.models.am_calibrator import auto_calibrate

__all__ = [
    # Cache types
    "KVCache",
    "RotatingKVCache",
    "TripleLayerKVCache",
    "make_prompt_cache",
    # Factory
    "make_optimized_cache",
    "VALID_STRATEGIES",
    "get_cache_info",
    # Quantization
    "QuantizationStrategy",
    "get_quantizer",
    "Q4_0Quantizer",
    "Q8_0Quantizer",
    "PolarQuantizer",
    "TurboQuantizer",
    "NoOpQuantizer",
    # Calibration
    "auto_calibrate",
]
