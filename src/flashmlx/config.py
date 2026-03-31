"""
FlashMLX configuration — Pydantic models for KV cache and expert offloading.

Designed for direct embedding in ThunderOMLX's settings_v2.py:

    class GlobalSettingsV2(BaseSettings):
        flashmlx: FlashMLXConfig = Field(default_factory=FlashMLXConfig)

All configs are serializable to JSON for persistence and remote configuration.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


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

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        valid = (
            "standard", "triple", "triple_am", "triple_pq",
            "triple_pq_am", "triple_tq", "triple_tq_am",
            "scored_pq", "auto",
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
        return kwargs


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
