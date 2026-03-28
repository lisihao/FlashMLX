"""
Smart KV Cache Factory for mlx-lm.

Provides automatic cache strategy selection and parameter optimization.

Strategies:
    - "standard": Default KVCache (unbounded bf16)
    - "triple": TripleLayerKVCache with Q4_0 quantization (~48% memory savings at 16K)
    - "triple_am": Triple + AM compression (~50% memory savings, faster TG at long contexts)
    - "triple_pq": Triple + PolarQuant warm quantization (~72% savings, data-oblivious)
    - "triple_pq_am": Triple + PolarQuant warm + AM cold compression
    - "triple_tq": Triple + TurboQuant (PQ3 + damped QJL). ~73% savings, correct output.
    - "triple_tq_am": Triple + TurboQuant + AM
    - "scored_pq": Architecture D — AM-scored differential PQ4/PQ2 (~81% savings, no token deletion)
    - "auto": Auto-select based on calibration availability

Usage:
    from mlx_lm import load, generate

    model, tokenizer = load("model_path")

    # Standard (default, backward compatible)
    text = generate(model, tokenizer, prompt)

    # Triple with Q4_0 quantization
    text = generate(model, tokenizer, prompt, kv_cache="triple")

    # Triple + PolarQuant warm quantization (no calibration needed)
    text = generate(model, tokenizer, prompt, kv_cache="triple_pq")

    # Triple + PolarQuant 3-bit (more aggressive compression)
    text = generate(model, tokenizer, prompt,
                    kv_cache="triple_pq",
                    kv_warm_bits=3)

    # Triple + AM compression (needs calibration file)
    text = generate(model, tokenizer, prompt,
                    kv_cache="triple_am",
                    kv_calibration="/path/to/am_calibration.pkl")

    # Triple + PolarQuant warm + AM cold (best combo, needs calibration)
    text = generate(model, tokenizer, prompt,
                    kv_cache="triple_pq_am",
                    kv_calibration="/path/to/am_calibration.pkl")

    # Auto-detect best strategy
    text = generate(model, tokenizer, prompt, kv_cache="auto")

    # Adaptive compression ratio (auto-selects 2.0x or 3.0x based on context)
    text = generate(model, tokenizer, prompt,
                    kv_cache="triple_am",
                    kv_calibration="/path/to/am_calibration.pkl",
                    kv_compression_ratio=0)  # 0 = adaptive

Performance characteristics (Qwen3-8B, measured at 16K context):
    | Strategy     | TG Speed | Prefill Mem Savings | Quality           |
    |--------------|----------|---------------------|-------------------|
    | standard     | baseline | 0%                  | baseline          |
    | triple       | -7%      | ~48%                | lossless          |
    | triple_am    | +4-19%   | ~50%                | key facts OK      |
    | triple_pq    | -14%     | ~72% (4b), ~76% (3b)| 4b: perfect, 3b: degrades |
    | triple_tq    | -15%     | ~73%                | correct (damped QJL α=0.1)  |
    | scored_pq    | TBD      | ~81%                | AM on clean bf16, no deletion |
"""

from typing import Optional, Dict, List, Any
import os
import mlx.nn as nn

# Default parameters (tuned across Qwen3-8B, validated at 2K-16K context)
DEFAULT_RECENT_SIZE = 512
DEFAULT_WARM_SIZE = 2048
DEFAULT_COMPRESSION_RATIO = 2.0
DEFAULT_WARM_OVERFLOW_THRESHOLD = 64

# Valid strategies
VALID_STRATEGIES = ("standard", "triple", "triple_am", "triple_pq", "triple_pq_am", "triple_tq", "triple_tq_am", "scored_pq", "auto")

# Adaptive ratio: pass compression_ratio=0 to enable context-aware ratio selection
# Data-driven mapping (Qwen3-8B benchmarks):
#   <= 8K tokens:  3.0x — better TG (+5-12%), +59-64% memory savings
#   > 8K tokens:   2.0x — stable TG, avoids long-context regression
ADAPTIVE_RATIO = 0.0


def make_optimized_cache(
    model: nn.Module,
    strategy: str = "standard",
    calibration_file: Optional[str] = None,
    compression_ratio: float = DEFAULT_COMPRESSION_RATIO,
    recent_size: int = DEFAULT_RECENT_SIZE,
    warm_size: int = DEFAULT_WARM_SIZE,
    max_kv_size: Optional[int] = None,
    warm_quantizer: Optional[str] = None,
    warm_bits: int = 4,
) -> List[Any]:
    """
    Create optimized KV cache list for model.

    Args:
        model: The language model.
        strategy: Cache strategy ("standard", "triple", "triple_am",
                  "triple_pq", "triple_pq_am", "auto").
        calibration_file: Path to AM calibration .pkl file (for triple_am/triple_pq_am).
        compression_ratio: AM compression ratio (default: 2.0).
        recent_size: Recent layer size in tokens (default: 512).
        warm_size: Warm layer size in tokens (default: 2048).
        max_kv_size: If set, use RotatingKVCache (overrides strategy).
        warm_quantizer: Warm layer quantizer name ("q4_0", "polarquant", "noop").
                       None = strategy default. Overrides strategy-implied quantizer.
        warm_bits: Bits for PolarQuant (2, 3, or 4). Default: 4.

    Returns:
        List of cache objects, one per layer.
    """
    from mlx_lm.models.cache import KVCache, RotatingKVCache

    if strategy not in VALID_STRATEGIES:
        raise ValueError(
            f"Unknown cache strategy: {strategy!r}. "
            f"Use one of: {', '.join(VALID_STRATEGIES)}"
        )

    num_layers = len(model.layers)

    # max_kv_size takes precedence (existing behavior)
    if max_kv_size is not None:
        return [
            RotatingKVCache(max_size=max_kv_size, keep=4)
            for _ in range(num_layers)
        ]

    # Resolve "auto" strategy
    if strategy == "auto":
        strategy = _auto_detect_strategy(calibration_file)

    if strategy == "standard":
        return [KVCache() for _ in range(num_layers)]

    # Triple or Triple+AM or Triple+PQ or Triple+PQ+AM
    from mlx_lm.models.triple_layer_cache import TripleLayerKVCache

    is_scored = strategy == "scored_pq"
    enable_am = strategy in ("triple_am", "triple_pq_am", "triple_tq_am") or is_scored

    # Validate calibration for AM
    if enable_am and calibration_file is None:
        print("[CacheFactory] Warning: AM strategies require calibration_file. "
              "Falling back to triple (no AM).")
        enable_am = False

    if enable_am and not os.path.exists(calibration_file):
        print(f"[CacheFactory] Warning: calibration file not found: {calibration_file}. "
              "Falling back to triple (no AM).")
        enable_am = False

    # Resolve warm quantizer
    quantizer_obj = None
    if is_scored:
        pass  # P2 Scored: no custom quantizer needed, uses default Q4_0 for warm aging
    elif warm_quantizer is not None:
        from mlx_lm.models.quantization_strategies import get_quantizer
        if warm_quantizer in ("polarquant", "turboquant"):
            quantizer_obj = get_quantizer(warm_quantizer, bits=warm_bits)
        else:
            quantizer_obj = get_quantizer(warm_quantizer)
    elif strategy in ("triple_pq", "triple_pq_am"):
        from mlx_lm.models.quantization_strategies import PolarQuantizer
        quantizer_obj = PolarQuantizer(bits=warm_bits)
    elif strategy in ("triple_tq", "triple_tq_am"):
        from mlx_lm.models.quantization_strategies import TurboQuantizer
        quantizer_obj = TurboQuantizer(bits=warm_bits)

    # Build cache list
    cache_kwargs = dict(
        memory_budget_mb=100.0,
        recent_size=recent_size,
        warm_size=warm_size,
        enable_warm_quant=True,
        enable_cold_am=enable_am,
        enable_cold_quant=True,
        calibration_file=calibration_file if enable_am else None,
        compression_ratio=compression_ratio,
        warm_overflow_threshold=DEFAULT_WARM_OVERFLOW_THRESHOLD,
        scored_mode=is_scored,
    )
    if is_scored:
        # P2 Scored: skip all warm quantization during prefill.
        # Everything stays bf16 in Recent → faster prefill (matches standard speed).
        cache_kwargs["lazy_prefill_threshold"] = 65536
        # Adaptive ratio: 3.0x at <=16K (better TG + quality), 2.0x at >16K (safe).
        # Data: 3.0x gives +30% TG, -65% memory, AND +25% quality vs 2.0x at 16K.
        if compression_ratio == DEFAULT_COMPRESSION_RATIO:
            cache_kwargs["compression_ratio"] = ADAPTIVE_RATIO  # 0 = adaptive
    if quantizer_obj is not None:
        cache_kwargs["warm_quantizer"] = quantizer_obj

    return [
        TripleLayerKVCache(
            **cache_kwargs,
            layer_idx=i,
        )
        for i in range(num_layers)
    ]


def _auto_detect_strategy(calibration_file: Optional[str]) -> str:
    """Auto-detect best cache strategy based on available resources."""
    if calibration_file and os.path.exists(calibration_file):
        return "triple_am"

    # Default to triple (Q4_0) — always safe, ~40% prefill savings, ~0% TG overhead
    return "triple"


def get_cache_info(cache_list: List[Any]) -> Dict[str, Any]:
    """
    Get diagnostic info about cache configuration.

    Useful for logging and debugging.

    Returns:
        dict with cache strategy, type, and configuration details.
    """
    from mlx_lm.models.cache import KVCache, RotatingKVCache

    if not cache_list:
        return {"strategy": "empty", "num_layers": 0}

    c0 = cache_list[0]
    info = {
        "num_layers": len(cache_list),
        "type": type(c0).__name__,
    }

    try:
        from mlx_lm.models.triple_layer_cache import TripleLayerKVCache
        if isinstance(c0, TripleLayerKVCache):
            if c0.scored_mode:
                info["strategy"] = "scored_pq"
            elif c0.enable_cold_am:
                info["strategy"] = "triple_am"
            else:
                info["strategy"] = "triple"
            info["recent_size"] = c0.recent_size
            info["warm_size"] = c0.warm_size
            info["compression_ratio"] = c0.compression_ratio
            info["flat_mode"] = c0._flat_mode
            info["scored_mode"] = c0.scored_mode
            if c0._scored_active:
                info["scored_active"] = True
                info["flat_tokens"] = c0._flat_offset if c0._flat_mode else 0
            if c0._flat_mode:
                info["flat_tokens"] = c0._flat_offset
                info["true_offset"] = c0._true_offset
            return info
    except ImportError:
        pass

    if isinstance(c0, KVCache):
        info["strategy"] = "standard"
    elif isinstance(c0, RotatingKVCache):
        info["strategy"] = "rotating"
        info["max_size"] = c0.max_size
    else:
        info["strategy"] = "unknown"

    return info
