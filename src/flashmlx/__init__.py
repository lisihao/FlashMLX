"""
FlashMLX — Enhanced MLX-LM inference engine for Apple Silicon.

Three optimization routes:
  Route 1: Expert Offloading (MoE models — Qwen3.5, Mixtral, etc.)
  Route 2: Chunked Prefill + Streaming Eviction (long context)
  Route 3: Scored P2 + Pluggable Flat Buffer Quantization (KV cache compression)

Quick start (Text Models):
    import flashmlx

    model, tokenizer = flashmlx.load("model_path")

    # Auto-detect best config
    config = flashmlx.recommend_config(model, model_path="model_path")
    cache = flashmlx.make_prompt_cache(model, **config.cache.to_cache_kwargs())

    # Or use directly with generate()
    text = flashmlx.generate(model, tokenizer, prompt, kv_cache="scored_pq")

Quick start (Vision-Language Models):
    from flashmlx import load_vlm

    # One-line loading
    vlm = load_vlm("mlx-community/Qwen2-VL-2B-Instruct-bf16")

    # Text generation
    response = vlm.generate("What is MLX?")

    # Vision+text generation
    response = vlm.generate("What's in this image?", image="cat.jpg")
"""

__version__ = "1.0.0"

# ---------------------------------------------------------------------------
# Ensure local mlx-lm-source is prioritized
# ---------------------------------------------------------------------------
import sys
from pathlib import Path as _Path
_mlx_lm_path = _Path(__file__).parent.parent.parent / "mlx-lm-source"
if str(_mlx_lm_path) not in sys.path:
    sys.path.insert(0, str(_mlx_lm_path))

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
from mlx_lm.models.quantization_strategies import get_quantizer, QuantizationStrategy, TurboAngleQuantizerWrapper
from mlx_lm.models.turboangle import TurboAngleQuantizer

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
# ⚠️  EXPERIMENTAL - NOT RECOMMENDED FOR PRODUCTION USE
#
# MAC-Attention 在 MLX/Apple Silicon 上无法达到预期加速效果。
# 实测：Hit 82%, Skip 66% → 加速比仅 0.92×-1.02×
# 原因：MLX decode attention 对 partial 输入没有线性加速。
#
# 详见：MAC_ATTENTION_EXPERIMENTAL.md
# ---------------------------------------------------------------------------
from .patch import patch_mlx_lm, unpatch_mlx_lm, enable_profiling, disable_profiling, get_profiling_stats

# ---------------------------------------------------------------------------
# VLM: Vision-Language Model API (Qwen2-VL, LLaVA, etc.)
# ---------------------------------------------------------------------------
from .vlm import load_vlm_components

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
    "TurboAngleQuantizer",
    "TurboAngleQuantizerWrapper",
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
    # VLM API
    "load_vlm_components",
]
