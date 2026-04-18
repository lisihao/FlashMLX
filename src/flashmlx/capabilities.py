"""
Model capability detection and configuration recommendation.

Inspects a loaded MLX model to determine which FlashMLX optimizations
are safe and beneficial, then generates a recommended FlashMLXConfig.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import mlx.nn as nn


@dataclass
class ModelCapabilities:
    """Detected capabilities of a loaded model.

    Attributes:
        model_type: Architecture class ("transformer", "hybrid", "moe").
        num_layers: Total number of model layers.
        num_attention_layers: Number of layers using standard KV cache.
        head_dim: Attention head dimension.
        num_kv_heads: Number of KV heads.
        is_moe: True if model has Mixture-of-Experts layers.
        is_hybrid: True if model mixes SSM + Attention (e.g. Qwen3.5, Jamba).
        supports_scored_pq: Whether scored_pq is recommended.
        supports_turboquant: Whether turboquant flat_quant is usable (head_dim >= 128).
        supports_expert_offload: Whether expert offloading is applicable.
        has_calibration: Whether an AM calibration file was found.
        calibration_path: Path to calibration file, if found.
        recommended_strategy: Best cache strategy for this model.
        recommended_flat_quant: Best flat_quant for this model.
        warnings: Any compatibility warnings.
    """

    model_type: str = "transformer"
    num_layers: int = 0
    num_attention_layers: int = 0
    head_dim: int = 128
    num_kv_heads: int = 8
    is_moe: bool = False
    is_hybrid: bool = False
    supports_scored_pq: bool = True
    supports_turboquant: bool = True
    supports_expert_offload: bool = False
    has_calibration: bool = False
    calibration_path: Optional[str] = None
    recommended_strategy: str = "scored_pq"
    recommended_flat_quant: Optional[str] = "q8_0"
    warnings: list[str] = field(default_factory=list)


def detect_capabilities(
    model: nn.Module,
    model_path: Optional[str] = None,
) -> ModelCapabilities:
    """Detect which FlashMLX optimizations a model supports.

    Args:
        model: A loaded MLX language model.
        model_path: Optional path to model directory (used for calibration lookup).

    Returns:
        ModelCapabilities with detected features and recommendations.
    """
    caps = ModelCapabilities()
    caps.num_layers = len(model.layers)

    # Detect head_dim and num_kv_heads from first attention layer
    _detect_attention_params(model, caps)

    # Detect architecture type (hybrid / MoE / pure transformer)
    _detect_architecture_type(model, caps)

    # Check calibration availability
    if model_path:
        _check_calibration(model, model_path, caps)

    # Determine what's supported
    caps.supports_turboquant = caps.head_dim >= 128
    caps.supports_scored_pq = not caps.is_hybrid

    if caps.is_hybrid:
        caps.warnings.append(
            f"Hybrid model ({caps.num_attention_layers}/{caps.num_layers} attention layers). "
            "scored_pq disabled — use expert offloading for MoE memory optimization."
        )

    if not caps.supports_turboquant:
        caps.warnings.append(
            f"head_dim={caps.head_dim} < 128. turboquant unavailable, using q4_0 fallback."
        )

    # Set recommendations
    if caps.is_hybrid:
        caps.model_type = "hybrid"
        caps.recommended_strategy = "standard"
        caps.recommended_flat_quant = None
    elif caps.is_moe:
        caps.model_type = "moe"
        caps.recommended_strategy = "scored_pq" if caps.has_calibration else "auto"
        caps.recommended_flat_quant = "q8_0"
    else:
        caps.model_type = "transformer"
        caps.recommended_strategy = "scored_pq" if caps.has_calibration else "auto"
        caps.recommended_flat_quant = "q8_0"

    return caps


def recommend_config(
    model: nn.Module,
    model_path: Optional[str] = None,
    memory_budget_gb: Optional[float] = None,
) -> "FlashMLXConfig":
    """Generate a recommended FlashMLXConfig for the given model.

    Decision tree:
      1. Hybrid (SSM+Attention) → standard cache, no flat_quant
      2. MoE → scored_pq + q8_0 + offload enabled
      3. Transformer + head_dim>=128 → scored_pq + q8_0
      4. Transformer + head_dim<128 → scored_pq + q8_0 (turboquant unavailable)

    If memory_budget_gb is set and tight, upgrades to q4_0 or turboquant.

    Args:
        model: A loaded MLX language model.
        model_path: Optional path to model directory.
        memory_budget_gb: Optional memory constraint in GB.

    Returns:
        FlashMLXConfig with recommended settings.
    """
    from .config import FlashMLXConfig, CacheConfig, OffloadConfig

    caps = detect_capabilities(model, model_path)

    cache = CacheConfig(
        strategy=caps.recommended_strategy,
        flat_quant=caps.recommended_flat_quant,
    )

    if caps.calibration_path:
        cache.calibration_file = caps.calibration_path

    # Tighten compression under memory pressure
    if memory_budget_gb is not None and memory_budget_gb < 16.0:
        if caps.supports_turboquant:
            cache.flat_quant = "turboquant"
        else:
            cache.flat_quant = "q4_0"

    offload = OffloadConfig(enabled=caps.is_moe and caps.supports_expert_offload)

    return FlashMLXConfig(cache=cache, offload=offload)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _detect_attention_params(model: nn.Module, caps: ModelCapabilities) -> None:
    """Extract head_dim and num_kv_heads from model layers."""
    if not hasattr(model, "layers") or len(model.layers) == 0:
        return

    layer0 = model.layers[0]

    # Try common attribute paths
    for attr in ("self_attn", "attention", "attn"):
        attn = getattr(layer0, attr, None)
        if attn is not None:
            # head_dim
            if hasattr(attn, "head_dim"):
                caps.head_dim = attn.head_dim
            elif hasattr(attn, "dims_per_head"):
                caps.head_dim = attn.dims_per_head

            # num_kv_heads
            if hasattr(attn, "n_kv_heads"):
                caps.num_kv_heads = attn.n_kv_heads
            elif hasattr(attn, "num_kv_heads"):
                caps.num_kv_heads = attn.num_kv_heads
            break


def _detect_architecture_type(model: nn.Module, caps: ModelCapabilities) -> None:
    """Detect hybrid (SSM+Attention) and MoE architectures."""
    from mlx_lm.models.cache import KVCache

    # Check hybrid via make_cache
    if hasattr(model, "make_cache"):
        try:
            native_caches = model.make_cache()
            attn_count = sum(1 for c in native_caches if isinstance(c, KVCache))
            caps.num_attention_layers = attn_count
            caps.is_hybrid = attn_count < len(native_caches)
        except Exception:
            caps.num_attention_layers = caps.num_layers
    else:
        caps.num_attention_layers = caps.num_layers

    # Check MoE: look for experts attribute in layers
    if hasattr(model, "layers") and len(model.layers) > 0:
        layer0 = model.layers[0]
        mlp = getattr(layer0, "mlp", None)
        has_experts = (
            hasattr(layer0, "experts")
            or hasattr(layer0, "block_sparse_moe")
            or hasattr(mlp, "experts")
            or hasattr(mlp, "switch_mlp")
        )
        caps.is_moe = has_experts
        caps.supports_expert_offload = has_experts


def _check_calibration(
    model: nn.Module,
    model_path: str,
    caps: ModelCapabilities,
) -> None:
    """Check for existing AM calibration files."""
    # Check standard calibration directories
    search_dirs = [
        Path(model_path).parent / "calibrations",
        Path(model_path) / "calibrations",
        Path.home() / ".cache" / "flashmlx" / "calibrations",
    ]

    for d in search_dirs:
        if d.is_dir():
            for f in d.glob("am_calibration_*.pkl"):
                caps.has_calibration = True
                caps.calibration_path = str(f)
                return
