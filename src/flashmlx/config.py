"""
FlashMLX configuration — Pydantic models for KV cache and expert offloading.

Designed for direct embedding in ThunderOMLX's settings_v2.py:

    class GlobalSettingsV2(BaseSettings):
        flashmlx: FlashMLXConfig = Field(default_factory=FlashMLXConfig)

All configs are serializable to JSON for persistence and remote configuration.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Route 0: Density Router — discrete compression levels
# Paper 2603.25926 insight: continuous ratios collapse, discrete ones are stable.
# ---------------------------------------------------------------------------

class DensityLevel(Enum):
    """Discrete compression levels for Route 0 Density Router.

    Each level defines a keep-ratio and its corresponding compression ratio.
    The log2 of the compression ratio is used for scale-space arithmetic.
    """

    keep_80 = (0.80, 1.25)   # light compression
    keep_50 = (0.50, 2.0)    # current default (scored_pq adaptive)
    keep_33 = (0.33, 3.0)    # moderate compression
    keep_20 = (0.20, 5.0)    # aggressive (needs Route 5 backup)
    keep_10 = (0.10, 10.0)   # extreme (Route 5 required)

    def __init__(self, keep_ratio: float, compression_ratio: float):
        self.keep_ratio = keep_ratio
        self.compression_ratio = compression_ratio
        self.log2_ratio = math.log2(compression_ratio)


# Sorted by log2_ratio for snap_to_nearest binary-style lookup
_LEVELS_BY_LOG2 = sorted(DensityLevel, key=lambda d: d.log2_ratio)


def snap_to_nearest(
    log2_target: float,
    scale: float = 0.0,
    levels: list[DensityLevel] | None = None,
) -> DensityLevel:
    """Snap a continuous log2 compression target to the nearest discrete level.

    Operates in log2 space: scale=+1 doubles compression, scale=-1 halves it.

    Args:
        log2_target: Raw log2(compression_ratio) from density signal.
        scale: Additive bias in log2 space (user/mode knob).
        levels: Override level set (default: all 5 DensityLevel values).

    Returns:
        The closest DensityLevel.
    """
    if levels is None:
        levels = _LEVELS_BY_LOG2

    adjusted = log2_target + scale
    best = levels[0]
    best_dist = abs(adjusted - best.log2_ratio)
    for lvl in levels[1:]:
        dist = abs(adjusted - lvl.log2_ratio)
        if dist < best_dist:
            best = lvl
            best_dist = dist
    return best


class CacheConfig(BaseModel):
    """KV Cache optimization configuration.

    Maps to make_prompt_cache() / make_optimized_cache() kwargs.

    Strategies:
        - "standard": Default KVCache (unbounded bf16)
        - "triple": TripleLayerKVCache with Q4_0 warm quantization
        - "triple_am": Triple + AM compression (needs calibration)
        - "triple_pq": Triple + PolarQuant warm quantization
        - "triple_pq_am": Triple + PolarQuant + AM
        - "scored_pq": AM-scored differential compression (recommended)
        - "scored_kv_direct": Route 5 — h^(0) capture + prefix exact reconstruction
        - "auto": Auto-select based on calibration availability

    Flat quantization (flat_quant):
        Applied to the flat buffer in scored_pq or triple modes.
        - None: bf16 (fastest TG, most memory)
        - "q8_0": int8 + per-group scales (-49% KV, minimal TG cost)
        - "q4_0": nibble-packed (-72% KV, moderate TG cost)
        - "turboquant": PolarQuant PQ4 (-74% KV, requires head_dim>=128)
    """

    strategy: str = Field(
        default="standard",
        description="Cache strategy name",
    )
    flat_quant: Optional[str] = Field(
        default=None,
        description="Flat buffer quantization: None, 'q8_0', 'q4_0', 'turboquant'",
    )
    compression_ratio: float = Field(
        default=2.0,
        description="AM compression ratio (0 = adaptive)",
    )
    scored_max_cache: int = Field(
        default=2048,
        description="Max tokens in flat buffer during scored_pq chunked prefill",
    )
    calibration_file: Optional[str] = Field(
        default=None,
        description="Path to AM calibration .pkl file",
    )
    warm_bits: int = Field(
        default=4,
        description="Bits for warm layer quantization (2, 3, or 4)",
    )
    recent_size: int = Field(
        default=512,
        description="Recent layer size in tokens",
    )
    warm_size: int = Field(
        default=2048,
        description="Warm layer size in tokens",
    )
    kv_direct_budget: int = Field(
        default=512,
        description="KV-Direct: number of recent tokens to keep as full K/V",
    )
    h0_quant: Optional[str] = Field(
        default=None,
        description="h^(0) quantization for scored_kv_direct: None (bf16), 'q8', 'q4'",
    )
    pinned_tokens: int = Field(
        default=0,
        description="First N tokens are never evicted (system prompt protection)",
    )

    # Route 0: Density Router
    density_mode: str = Field(
        default="off",
        description="Density router mode: 'off' | 'balanced' | 'ultra_long' | 'recall_first'",
    )
    density_scale: float = Field(
        default=0.0,
        description="Additive bias in log2 space. +1 = double compression, -1 = halve.",
    )

    # H0Probe: attention-based eviction
    probe_layers: int = Field(
        default=0,
        description="Attention probe depth for eviction (0=disabled, 3=recommended)",
    )

    # Auto-reconstruction: automatically reconstruct from h^(0) after prefill
    auto_reconstruct: bool = Field(
        default=False,
        description="Auto-trigger h^(0) reconstruction after prefill completes",
    )

    # --- ThunderOMLX SSD Cache Bridge (Tiers 1-3) ---
    enable_compressed_ssd: bool = Field(
        default=True,
        description="Tier 1: Store compressed flat buffer blocks to SSD (50-75% space savings)",
    )
    enable_h0_ssd: bool = Field(
        default=True,
        description="Tier 2: Store H0 blocks to SSD alongside KV blocks",
    )
    h0_ssd_quant: Optional[str] = Field(
        default="q8",
        description="Tier 2: H0 block quantization for SSD: None (bf16), 'q8', 'q4'",
    )
    enable_cold_restoration: bool = Field(
        default=True,
        description="Tier 3: Enable 3PIR cold cache restoration from H0-only SSD blocks",
    )

    @field_validator("density_mode")
    @classmethod
    def validate_density_mode(cls, v: str) -> str:
        valid = ("off", "balanced", "ultra_long", "recall_first")
        if v not in valid:
            raise ValueError(f"Unknown density_mode: {v!r}. Use one of: {valid}")
        return v

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        valid = (
            "standard", "triple", "triple_am", "triple_pq",
            "triple_pq_am", "triple_tq", "triple_tq_am",
            "scored_pq", "scored_kv_direct", "kv_direct", "auto",
        )
        if v not in valid:
            raise ValueError(f"Unknown strategy: {v!r}. Use one of: {valid}")
        return v

    @field_validator("flat_quant")
    @classmethod
    def validate_flat_quant(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in ("q8_0", "q4_0", "turboquant"):
            raise ValueError(f"Unknown flat_quant: {v!r}. Use: None, 'q8_0', 'q4_0', 'turboquant'")
        return v

    @field_validator("h0_quant", "h0_ssd_quant")
    @classmethod
    def validate_h0_quant(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in ("q8", "q4"):
            raise ValueError(f"Unknown h0_quant: {v!r}. Use: None, 'q8', 'q4'")
        return v

    @field_validator("warm_bits")
    @classmethod
    def validate_warm_bits(cls, v: int) -> int:
        if v not in (2, 3, 4):
            raise ValueError(f"warm_bits must be 2, 3, or 4, got {v}")
        return v

    def to_cache_kwargs(self) -> dict[str, Any]:
        """Convert to make_prompt_cache() keyword arguments.

        Returns:
            dict suitable for ``make_prompt_cache(model, **kwargs)``
        """
        kwargs: dict[str, Any] = {}
        if self.strategy != "standard":
            kwargs["kv_cache"] = self.strategy
        if self.flat_quant is not None:
            kwargs["kv_flat_quant"] = self.flat_quant
        if self.calibration_file is not None:
            kwargs["kv_calibration"] = self.calibration_file
        if self.compression_ratio != 2.0:
            kwargs["kv_compression_ratio"] = self.compression_ratio
        if self.warm_bits != 4:
            kwargs["kv_warm_bits"] = self.warm_bits
        if self.scored_max_cache != 2048:
            kwargs["kv_scored_max_cache"] = self.scored_max_cache
        if self.h0_quant is not None:
            kwargs["h0_quant"] = self.h0_quant
        if self.pinned_tokens > 0:
            kwargs["pinned_tokens"] = self.pinned_tokens
        if self.density_mode != "off":
            kwargs["density_mode"] = self.density_mode
            kwargs["density_scale"] = self.density_scale
        if self.probe_layers > 0:
            kwargs["probe_layers"] = self.probe_layers
        if self.auto_reconstruct:
            kwargs["auto_reconstruct"] = self.auto_reconstruct
        return kwargs

    def to_factory_kwargs(self) -> dict[str, Any]:
        """Convert to make_optimized_cache() keyword arguments.

        Returns:
            dict suitable for ``make_optimized_cache(model, **kwargs)``
        """
        kwargs: dict[str, Any] = {
            "strategy": self.strategy,
            "recent_size": self.recent_size,
            "warm_size": self.warm_size,
            "warm_bits": self.warm_bits,
        }
        if self.flat_quant is not None:
            kwargs["flat_quant"] = self.flat_quant
        if self.calibration_file is not None:
            kwargs["calibration_file"] = self.calibration_file
        if self.compression_ratio != 2.0:
            kwargs["compression_ratio"] = self.compression_ratio
        if self.scored_max_cache != 2048:
            kwargs["scored_max_cache"] = self.scored_max_cache
        if self.strategy in ("kv_direct", "scored_kv_direct"):
            kwargs["kv_direct_budget"] = self.kv_direct_budget
        if self.h0_quant is not None:
            kwargs["h0_quant"] = self.h0_quant
        if self.pinned_tokens > 0:
            kwargs["pinned_tokens"] = self.pinned_tokens
        if self.density_mode != "off":
            kwargs["density_mode"] = self.density_mode
            kwargs["density_scale"] = self.density_scale
        if self.probe_layers > 0:
            kwargs["probe_layers"] = self.probe_layers
        if self.auto_reconstruct:
            kwargs["auto_reconstruct"] = self.auto_reconstruct
        return kwargs

    def effective_density_scale(self) -> float:
        """Resolve density_scale from density_mode preset or manual override."""
        _MODE_SCALES = {
            "off": 0.0,
            "balanced": 0.0,
            "ultra_long": 1.5,
            "recall_first": 2.5,
        }
        if self.density_mode in _MODE_SCALES and self.density_scale == 0.0:
            return _MODE_SCALES[self.density_mode]
        return self.density_scale


class OffloadConfig(BaseModel):
    """Expert offloading configuration (MoE models only).

    Controls FlashMLX Route 1: three-tier expert management
    (GPU pool → CPU warm → SSD cold).
    """

    enabled: bool = Field(
        default=False,
        description="Enable expert offloading",
    )
    pool_size: int = Field(
        default=0,
        description="GPU expert pool size (0 = auto-detect)",
    )
    max_workers: int = Field(
        default=4,
        description="Max SSD loader threads",
    )
    cpu_cache_gb: Optional[float] = Field(
        default=None,
        description="CPU warm cache size in GB (None = auto)",
    )


class FlashMLXConfig(BaseModel):
    """Complete FlashMLX configuration, serializable to JSON.

    Combines cache optimization and expert offloading settings.

    Example:
        config = FlashMLXConfig(
            cache=CacheConfig(strategy="scored_pq", flat_quant="q8_0"),
            offload=OffloadConfig(enabled=True),
        )
        # Serialize
        json_str = config.model_dump_json(indent=2)
        # Deserialize
        config2 = FlashMLXConfig.model_validate_json(json_str)
    """

    cache: CacheConfig = Field(default_factory=CacheConfig)
    offload: OffloadConfig = Field(default_factory=OffloadConfig)

    def apply(self, model, model_path: Optional[str] = None):
        """Apply this config to a model.

        Creates optimized cache and optionally patches for expert offloading.

        Args:
            model: The loaded MLX language model.
            model_path: Path to model weights (required for offloading).

        Returns:
            tuple: (cache_list, offload_context_or_None)
        """
        from mlx_lm.models.cache import make_prompt_cache

        cache_list = make_prompt_cache(model, **self.cache.to_cache_kwargs())

        offload_ctx = None
        if self.offload.enabled and model_path is not None:
            from mlx_lm.models.expert_offload import patch_model_for_offload
            offload_ctx = patch_model_for_offload(
                model=model,
                model_path=model_path,
                pool_size=self.offload.pool_size,
                max_workers=self.offload.max_workers,
            )

        return cache_list, offload_ctx
