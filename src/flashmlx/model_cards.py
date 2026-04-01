"""
Model Cards — per-model configuration as single source of truth.

Each model has one JSON card file containing:
  - Architecture info (layers, heads, hybrid/MoE/transformer)
  - Optimal CacheConfig (THE definitive parameters)
  - Benchmark results (verified data)
  - Strategy guidance (what to avoid and why)

Usage:
    from flashmlx.model_cards import load_card

    card = load_card("/path/to/model")
    cache = make_prompt_cache(model, **card.to_cache_kwargs())

Card files live in:
    1. FlashMLX/model_cards/<model_id>.json  (primary)
    2. <model_path>/flashmlx_card.json       (embedded fallback)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from .config import CacheConfig, OffloadConfig, FlashMLXConfig


# ---------------------------------------------------------------------------
# Card directory — sibling to src/flashmlx/
# ---------------------------------------------------------------------------
CARD_DIR = Path(__file__).resolve().parent.parent.parent / "model_cards"


class ArchitectureInfo(BaseModel):
    """Model architecture description."""

    type: str = Field(
        description="'pure_transformer' | 'hybrid_ssm_attention' | 'moe'",
    )
    num_layers: int = Field(description="Total number of model layers")
    attention_layers: int = Field(description="Layers with standard KV attention")
    hidden_size: int = Field(description="Model hidden dimension")
    head_dim: int = Field(default=128, description="Per-head dimension")
    num_kv_heads: int = Field(default=8, description="Number of KV heads")


class BenchmarkResult(BaseModel):
    """Single benchmark measurement at a specific context length."""

    context: int = Field(description="Prompt length in tokens")
    pp_toks: float = Field(description="Prefill throughput (tok/s)")
    tg_toks: float = Field(description="Token generation throughput (tok/s)")
    ttft_ms: float = Field(description="Time to first token (ms)")
    tg_mem_mb: float = Field(description="Steady-state TG memory delta (MB)")
    pp_peak_mb: float = Field(description="Peak memory during prefill (MB)")


class ModeConfig(BaseModel):
    """Route 0 product mode configuration."""

    density_scale: float = Field(description="Additive bias in log2 space")
    strategy: str = Field(default="scored_pq", description="Cache strategy for this mode")
    description: str = Field(default="", description="Human-readable description")


class ModelCard(BaseModel):
    """Per-model configuration card — single source of truth.

    Contains everything needed to run a model with optimal parameters:
    architecture info, cache config, benchmark data, and guidance.
    """

    # Identity
    model_id: str = Field(description="Unique identifier, e.g. 'qwen3-8b-mlx-4bit'")
    model_name: str = Field(description="Human-readable name")
    model_path: str = Field(description="Default path to model weights")

    # Architecture
    architecture: ArchitectureInfo

    # THE definitive config — single source of truth
    optimal: CacheConfig = Field(
        default_factory=CacheConfig,
        description="Optimal cache configuration for this model",
    )
    offload: OffloadConfig = Field(
        default_factory=OffloadConfig,
        description="Expert offloading configuration",
    )

    # Benchmark data (verified, optimal config)
    benchmarks: dict[str, BenchmarkResult] = Field(
        default_factory=dict,
        description="Verified benchmarks keyed by context label ('4k', '8k', '16k', '32k')",
    )
    # Standard path baselines for comparison
    standard_baselines: dict[str, BenchmarkResult] = Field(
        default_factory=dict,
        description="Standard (unoptimized) baselines for the same context lengths",
    )
    platform: str = Field(default="", description="Hardware platform, e.g. 'M4 Max 64GB'")

    # Strategy guidance
    not_recommended: list[str] = Field(
        default_factory=list,
        description="Strategies to avoid for this model (with reasons in notes)",
    )
    notes: str = Field(default="", description="Human-readable notes and warnings")

    # Route 0: Product modes
    modes: dict[str, ModeConfig] = Field(
        default_factory=dict,
        description="Route 0 product modes: 'balanced', 'ultra_long', 'recall_first'",
    )

    def to_config(self) -> FlashMLXConfig:
        """Build a complete FlashMLXConfig from this card."""
        return FlashMLXConfig(cache=self.optimal, offload=self.offload)

    def to_cache_kwargs(self, mode: str | None = None) -> dict[str, Any]:
        """Convert optimal config to make_prompt_cache() kwargs.

        Args:
            mode: Optional Route 0 mode name ('balanced', 'ultra_long',
                  'recall_first'). If provided and defined in card.modes,
                  overrides strategy and adds density_scale.
        """
        kwargs = self.optimal.to_cache_kwargs()
        if mode and mode in self.modes:
            mc = self.modes[mode]
            kwargs["density_mode"] = mode
            kwargs["density_scale"] = mc.density_scale
            if mc.strategy != self.optimal.strategy:
                kwargs["kv_cache"] = mc.strategy
        return kwargs

    def is_hybrid(self) -> bool:
        return self.architecture.type == "hybrid_ssm_attention"

    def is_moe(self) -> bool:
        return self.architecture.type == "moe"


# ---------------------------------------------------------------------------
# Card loading
# ---------------------------------------------------------------------------

def load_card(model_path: str) -> Optional[ModelCard]:
    """Load a model card by matching model_path.

    Search order:
        1. model_cards/<name>.json — match by directory name or model_id
        2. <model_path>/flashmlx_card.json — embedded in model directory

    Args:
        model_path: Path to the model weights directory.

    Returns:
        ModelCard if found, None otherwise.
    """
    model_dir_name = os.path.basename(model_path.rstrip("/")).lower()

    # Search card directory
    if CARD_DIR.is_dir():
        for card_file in sorted(CARD_DIR.glob("*.json")):
            if card_file.name.startswith("_"):
                continue
            try:
                card = ModelCard.model_validate_json(card_file.read_text())
            except Exception:
                continue
            card_dir = os.path.basename(card.model_path.rstrip("/")).lower()
            if card_dir == model_dir_name or card.model_id in model_dir_name:
                return card

    # Fallback: embedded card in model directory
    embedded = Path(model_path) / "flashmlx_card.json"
    if embedded.exists():
        try:
            return ModelCard.model_validate_json(embedded.read_text())
        except Exception:
            pass

    return None


def load_card_or_detect(model, model_path: str) -> ModelCard:
    """Load card from file, or auto-detect and generate one.

    Falls back to detect_capabilities() + recommend_config() when
    no card file exists. The auto-generated card is marked with a note.

    Args:
        model: A loaded MLX language model.
        model_path: Path to model weights directory.

    Returns:
        ModelCard (from file or auto-detected).
    """
    card = load_card(model_path)
    if card is not None:
        return card

    from .capabilities import detect_capabilities, recommend_config

    caps = detect_capabilities(model, model_path)
    config = recommend_config(model, model_path)

    return ModelCard(
        model_id=os.path.basename(model_path).lower().replace(" ", "-"),
        model_name=os.path.basename(model_path),
        model_path=model_path,
        architecture=ArchitectureInfo(
            type=caps.model_type,
            num_layers=caps.num_layers,
            attention_layers=caps.num_attention_layers,
            hidden_size=caps.head_dim * caps.num_kv_heads,
            head_dim=caps.head_dim,
            num_kv_heads=caps.num_kv_heads,
        ),
        optimal=config.cache,
        offload=config.offload,
        notes="Auto-detected. Run benchmark to create a verified card.",
    )


def save_card(card: ModelCard, path: Optional[Path] = None) -> Path:
    """Save a model card to JSON.

    Args:
        card: The ModelCard to save.
        path: Explicit output path. Defaults to model_cards/<model_id>.json.

    Returns:
        Path to the saved file.
    """
    if path is None:
        CARD_DIR.mkdir(parents=True, exist_ok=True)
        path = CARD_DIR / f"{card.model_id}.json"
    path.write_text(card.model_dump_json(indent=2) + "\n")
    return path


def list_cards() -> list[ModelCard]:
    """List all available model cards."""
    cards = []
    if not CARD_DIR.is_dir():
        return cards
    for card_file in sorted(CARD_DIR.glob("*.json")):
        if card_file.name.startswith("_"):
            continue
        try:
            cards.append(ModelCard.model_validate_json(card_file.read_text()))
        except Exception:
            continue
    return cards
