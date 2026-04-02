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
    - "scored_kv_direct": Route 5 — scored_pq + h^(0) archive (lossless reconstruction capability)
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
VALID_STRATEGIES = ("standard", "triple", "triple_am", "triple_pq", "triple_pq_am", "triple_tq", "triple_tq_am", "scored_pq", "scored_kv_direct", "kv_direct", "auto")

# Adaptive ratio: pass compression_ratio=0 to enable context-aware ratio selection
# Data-driven mapping (Qwen3-8B benchmarks):
#   <= 8K tokens:  3.0x — better TG (+5-12%), +59-64% memory savings
#   > 8K tokens:   2.0x — stable TG, avoids long-context regression
ADAPTIVE_RATIO = 0.0


def _detect_architecture(model: nn.Module):
    """
    Detect model architecture type for cache routing.

    Returns:
        (is_hybrid, attention_layer_indices, native_caches):
        is_hybrid=True for mixed SSM+Attention models (Qwen3.5, PLaMo2, Jamba, etc.).
        attention_layer_indices contains indices of layers that use standard KV cache.
        native_caches is the list from model.make_cache() (reused by hybrid branch to
        avoid double allocation), or None for pure Transformers.
    """
    from mlx_lm.models.cache import KVCache

    if not hasattr(model, "make_cache"):
        # No custom cache → pure Transformer
        num_layers = len(model.layers)
        return False, list(range(num_layers)), None

    try:
        native_caches = model.make_cache()
    except Exception as e:
        print(f"[CacheFactory] Warning: model.make_cache() failed: {e}. "
              "Treating as pure Transformer.")
        num_layers = len(model.layers)
        return False, list(range(num_layers)), None

    attention_indices = []
    for i, cache in enumerate(native_caches):
        if isinstance(cache, KVCache):
            attention_indices.append(i)

    is_hybrid = len(attention_indices) < len(native_caches)
    if is_hybrid:
        print(f"[CacheFactory] Hybrid architecture detected: "
              f"{len(attention_indices)}/{len(native_caches)} attention layers "
              f"(indices: {attention_indices[:5]}{'...' if len(attention_indices) > 5 else ''})")
    return is_hybrid, attention_indices, native_caches if is_hybrid else None


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
    flat_quant: Optional[str] = None,
    scored_max_cache: int = 2048,
    kv_direct_budget: int = 512,
    h0_quant: Optional[str] = None,
    pinned_tokens: int = 0,
    density_mode: Optional[str] = None,
    density_scale: float = 0.0,
    probe_layers: int = 0,
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
        flat_quant: Flat buffer quantization: None (bf16), "q8_0" (int8 + scales),
                   "q4_0" (nibble-packed + per-group scales), or "turboquant"
                   (PolarQuant packed uint32 + per-token norms, ~3.8x compression).
                   Reduces steady-state KV memory. Dequantizes on every TG step.
                   Note: "turboquant" requires head_dim >= 128 for usable attention
                   quality. Models with smaller head_dim auto-downgrade to "q4_0".
        scored_max_cache: Maximum tokens retained in flat buffer during scored_pq
                         chunked prefill eviction. Default: 2048.
        pinned_tokens: First N tokens are never evicted by AM scoring/compression.
                      Typical use: system prompt protection for multi-agent reuse.
                      Default: 0 (no pinning).

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
        # Check for hybrid architecture (SSM+Attention models like Qwen3.5)
        is_hybrid, _, native_caches = _detect_architecture(model)
        if is_hybrid:
            return native_caches
        return [KVCache() for _ in range(num_layers)]

    # KV-Direct v2: model-level h^(0) checkpointing (paper 2603.19664)
    if strategy == "kv_direct":
        from mlx_lm.models.kv_direct_cache import (
            KVDirectCache, H0Store, apply_h0_capture,
        )

        is_hybrid, attn_indices, native_caches = _detect_architecture(model)
        h0_store = H0Store()

        if is_hybrid:
            attn_set = set(attn_indices)
            caches = []
            for i in range(num_layers):
                if i in attn_set:
                    caches.append(KVDirectCache(
                        budget=kv_direct_budget, h0_store=h0_store,
                    ))
                else:
                    caches.append(native_caches[i])
        else:
            caches = [
                KVDirectCache(budget=kv_direct_budget, h0_store=h0_store)
                for _ in range(num_layers)
            ]

        # Install model-level h^(0) capture + reconstruction
        apply_h0_capture(model, caches, h0_store)

        n_patched = len([c for c in caches if isinstance(c, KVDirectCache)])
        print(f"[CacheFactory] KV-Direct v2: budget={kv_direct_budget}, "
              f"{n_patched} layers, h^(0) checkpointing")
        return caches

    # Triple or Triple+AM or Triple+PQ or Triple+PQ+AM or Scored KV-Direct
    from mlx_lm.models.triple_layer_cache import TripleLayerKVCache

    is_scored_kv_direct = strategy == "scored_kv_direct"
    is_scored = strategy == "scored_pq" or is_scored_kv_direct
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
        if warm_quantizer in ("polarquant", "turboquant", "turboquant_qjl"):
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
        flat_quant=flat_quant,
        pinned_tokens=pinned_tokens,
        density_mode=density_mode,
        density_scale=density_scale,
    )
    if is_scored:
        # P2 Scored: skip all warm quantization during prefill.
        # Everything stays bf16 in Recent → faster prefill (matches standard speed).
        cache_kwargs["lazy_prefill_threshold"] = 65536
        # Adaptive ratio: 3.0x at <=16K (better TG + quality), 2.0x at >16K (safe).
        # Data: 3.0x gives +30% TG, -65% memory, AND +25% quality vs 2.0x at 16K.
        if compression_ratio == DEFAULT_COMPRESSION_RATIO:
            cache_kwargs["compression_ratio"] = ADAPTIVE_RATIO  # 0 = adaptive
        # Chunked prefill eviction: bound PP memory by AM scoring during prefill
        cache_kwargs["scored_prefill_chunk_evict"] = True
        cache_kwargs["scored_prefill_max_cache"] = scored_max_cache
    if quantizer_obj is not None:
        cache_kwargs["warm_quantizer"] = quantizer_obj

    # Detect hybrid architecture (SSM + Attention)
    is_hybrid, attn_indices, native_caches = _detect_architecture(model)

    if is_hybrid and strategy == "scored_pq":
        # Auto-disable plain scored_pq for hybrid SSM+Attention models.
        # Benchmark data (Qwen3.5-35B-A3B): Q8/Q4 flat_quant causes TG regression.
        # scored_kv_direct is NOT auto-disabled — h^(0) capture is valuable on hybrid.
        attn_ratio = len(attn_indices) / num_layers
        print(f"[CacheFactory] scored_pq auto-disabled for hybrid model "
              f"({len(attn_indices)}/{num_layers} attention layers, {attn_ratio:.0%}). "
              f"Use scored_kv_direct or expert offloading instead.")
        return native_caches

    if is_hybrid:
        # Hybrid path: SSM layers keep native cache, Attention layers get triple cache
        # native_caches already allocated by _detect_architecture (no double allocation)
        attn_set = set(attn_indices)
        caches = []
        for i in range(num_layers):
            if i in attn_set:
                caches.append(TripleLayerKVCache(**cache_kwargs, layer_idx=i))
            else:
                caches.append(native_caches[i])
    else:
        # Pure Transformer path (existing logic, unchanged)
        caches = [
            TripleLayerKVCache(
                **cache_kwargs,
                layer_idx=i,
            )
            for i in range(num_layers)
        ]

    # Route 5: Scored KV-Direct — install h^(0) capture on top of scored_pq
    if is_scored_kv_direct:
        from mlx_lm.models.kv_direct_cache import (
            H0Store, ReconstructionBudget, apply_h0_capture_only,
        )

        # Route 0↔5 coupling: ReconstructionBudget controls how aggressively
        # Route 0 can compress, knowing Route 5 can reconstruct.
        recon_budget = ReconstructionBudget()
        h0_store = H0Store(quant=h0_quant, recon_budget=recon_budget)
        for c in caches:
            if isinstance(c, TripleLayerKVCache):
                c._h0_store = h0_store

        apply_h0_capture_only(model, h0_store)
        n_layers = sum(1 for c in caches if isinstance(c, TripleLayerKVCache))
        quant_label = h0_quant or "bf16"
        print(f"[CacheFactory] Scored KV-Direct: h^(0) capture installed, "
              f"{n_layers} layers, h^(0) archive active, h0_quant={quant_label}, "
              f"recon_budget=({recon_budget.max_recall_per_turn}/turn, "
              f"cd={recon_budget.cooldown_turns})")

        # H0Probe: attention-based eviction scoring (replaces key-norm surprise)
        if probe_layers > 0:
            from mlx_lm.models.h0_probe import H0Probe
            from mlx_lm.models.kv_direct_cache import _find_inner_model
            inner = _find_inner_model(model)
            probe = H0Probe(inner, n_probe_layers=probe_layers)
            TripleLayerKVCache._shared_probe = probe
            for c in caches:
                if isinstance(c, TripleLayerKVCache):
                    c._probe_eviction_enabled = True
            print(f"[CacheFactory] H0Probe installed: {probe_layers} layers, "
                  f"attention-based eviction enabled")

    return caches


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

    # Find representative cache — on hybrid models, c0 may be ArraysCache (SSM layer).
    # Look for the first TripleLayerKVCache or KVDirectCache for accurate info.
    c0 = cache_list[0]
    triple_cache = None
    try:
        from mlx_lm.models.triple_layer_cache import TripleLayerKVCache
        triple_cache = next((c for c in cache_list if isinstance(c, TripleLayerKVCache)), None)
    except ImportError:
        pass

    info = {
        "num_layers": len(cache_list),
        "type": type(c0).__name__,
    }

    try:
        from mlx_lm.models.triple_layer_cache import TripleLayerKVCache
        tc = triple_cache or c0
        if isinstance(tc, TripleLayerKVCache):
            if tc.scored_mode and tc._h0_store is not None:
                info["strategy"] = "scored_kv_direct"
                info["h0_count"] = tc._h0_store.count
                info["h0_bytes"] = tc._h0_store.nbytes
                if tc._h0_store.recon_budget:
                    info["recon_budget"] = tc._h0_store.recon_budget.stats
            elif tc.scored_mode:
                info["strategy"] = "scored_pq"
            elif tc.enable_cold_am:
                info["strategy"] = "triple_am"
            else:
                info["strategy"] = "triple"
            info["recent_size"] = tc.recent_size
            info["warm_size"] = tc.warm_size
            info["compression_ratio"] = tc.compression_ratio
            info["flat_mode"] = tc._flat_mode
            info["scored_mode"] = tc.scored_mode
            if tc._scored_active:
                info["scored_active"] = True
                info["flat_tokens"] = tc._flat_offset if tc._flat_mode else 0
            if tc._flat_mode:
                info["flat_tokens"] = tc._flat_offset
                info["true_offset"] = tc._true_offset
            return info
    except ImportError:
        pass

    try:
        from mlx_lm.models.kv_direct_cache import KVDirectCache
        if isinstance(c0, KVDirectCache):
            info["strategy"] = "kv_direct"
            info["budget"] = c0._budget
            info["offset"] = c0.offset
            info["recent_count"] = c0._recent_count
            if c0._h0_store is not None:
                info["h0_tokens"] = c0._h0_store.count
                info["h0_bytes"] = c0._h0_store.nbytes
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
