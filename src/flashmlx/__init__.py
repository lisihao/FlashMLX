"""
FlashMLX — Enhanced MLX-LM inference engine for Apple Silicon.

Three optimization routes:
  Route 1: Expert Offloading (MoE models — Qwen3.5, Mixtral, etc.)
  Route 2: Chunked Prefill + Streaming Eviction (long context)
  Route 3: Scored P2 + Pluggable Flat Buffer Quantization (KV cache compression)

Quick start:
    import flashmlx

    model, tokenizer = flashmlx.load("model_path")

    # Auto-detect best config
    config = flashmlx.recommend_config(model, model_path="model_path")
    cache = flashmlx.make_prompt_cache(model, **config.cache.to_cache_kwargs())

    # Or use directly with generate()
    text = flashmlx.generate(model, tokenizer, prompt, kv_cache="scored_pq")
"""

__version__ = "1.0.0"

# ---------------------------------------------------------------------------
# Model loading (re-export from enhanced mlx-lm)
# ---------------------------------------------------------------------------
from mlx_lm.utils import load
from mlx_lm.generate import generate, stream_generate

# ---------------------------------------------------------------------------
# FlashMLX configuration
# ---------------------------------------------------------------------------
from .config import FlashMLXConfig, CacheConfig, OffloadConfig, DensityLevel, snap_to_nearest

# ---------------------------------------------------------------------------
# Model capability detection + recommended config
# ---------------------------------------------------------------------------
from .capabilities import detect_capabilities, recommend_config, ModelCapabilities

# ---------------------------------------------------------------------------
# Cache creation (re-export from enhanced mlx-lm)
# ---------------------------------------------------------------------------
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.cache_factory import make_optimized_cache, VALID_STRATEGIES

# ---------------------------------------------------------------------------
# Model Cards (per-model config — single source of truth)
# ---------------------------------------------------------------------------
from .model_cards import ModelCard, ModeConfig, load_card, load_card_or_detect, save_card, list_cards

# ---------------------------------------------------------------------------
# Quantization strategies (re-export)
# ---------------------------------------------------------------------------
from mlx_lm.models.quantization_strategies import get_quantizer, QuantizationStrategy

# ---------------------------------------------------------------------------
# Reconstruction Controller (programmatic h^(0) → K/V reconstruction API)
# ---------------------------------------------------------------------------
from .reconstruction import (
    ReconstructionController,
    NullReconstructionController,
    ReconState,
    ReconStrategy,
    ReconResult,
    ReconCostEstimate,
    ReconStats,
)

# ---------------------------------------------------------------------------
# 3PIR: RC Engine (chunk-level non-blocking reconstruction)
# ---------------------------------------------------------------------------
from .rc_engine import RCEngine, RCSequenceState, RCChunkResult

# ---------------------------------------------------------------------------
# MAC-Attention: Monkey Patch for mlx-lm (Route 6)
# ---------------------------------------------------------------------------
from .patch import patch_mlx_lm, unpatch_mlx_lm, enable_profiling, disable_profiling, get_profiling_stats

__all__ = [
    "__version__",
    # Model loading
    "load",
    "generate",
    "stream_generate",
    # Config
    "FlashMLXConfig",
    "CacheConfig",
    "OffloadConfig",
    "DensityLevel",
    "snap_to_nearest",
    # Capabilities
    "detect_capabilities",
    "recommend_config",
    "ModelCapabilities",
    # Cache
    "make_prompt_cache",
    "make_optimized_cache",
    "VALID_STRATEGIES",
    # Model Cards
    "ModelCard",
    "ModeConfig",
    "load_card",
    "load_card_or_detect",
    "save_card",
    "list_cards",
    # Quantization
    "get_quantizer",
    "QuantizationStrategy",
    # Reconstruction Controller
    "ReconstructionController",
    "NullReconstructionController",
    "ReconState",
    "ReconStrategy",
    "ReconResult",
    "ReconCostEstimate",
    "ReconStats",
    # 3PIR RC Engine
    "RCEngine",
    "RCSequenceState",
    "RCChunkResult",
    # MAC-Attention Patch
    "patch_mlx_lm",
    "unpatch_mlx_lm",
    "enable_profiling",
    "disable_profiling",
    "get_profiling_stats",
]
