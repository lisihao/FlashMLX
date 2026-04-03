"""
ThunderOMLX integration adapter — concrete FlashMLXProvider implementation.

Provides setup_flashmlx() as the primary integration entry point.

Usage in ThunderOMLX engine_core.py:
    from flashmlx.integration import setup_flashmlx
    from flashmlx import FlashMLXConfig, CacheConfig

    config = FlashMLXConfig(
        cache=CacheConfig(strategy="scored_pq", flat_quant="q8_0"),
    )
    cache_list, offload_ctx = setup_flashmlx(model, tokenizer, model_path, config)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import mlx.nn as nn


def setup_flashmlx(
    model: nn.Module,
    tokenizer: Any,
    model_path: str,
    config: Optional[Any] = None,
) -> tuple[list, Any]:
    """One-call FlashMLX integration entry point.

    Steps:
      1. detect_capabilities → check model compatibility
      2. auto_calibrate → if scored_pq and no calibration file
      3. make_prompt_cache → create optimized cache
      4. patch_model_for_offload → if MoE and offload.enabled

    Args:
        model: The loaded MLX language model.
        tokenizer: The model's tokenizer.
        model_path: Path to model directory on disk.
        config: FlashMLXConfig instance. If None, uses recommend_config().

    Returns:
        (cache_list, offload_context_or_None)
    """
    from ..capabilities import detect_capabilities, recommend_config
    from ..config import FlashMLXConfig

    if config is None:
        config = recommend_config(model, model_path)
    elif not isinstance(config, FlashMLXConfig):
        raise TypeError(f"Expected FlashMLXConfig, got {type(config).__name__}")

    caps = detect_capabilities(model, model_path)

    # Auto-calibrate if needed
    if (
        config.cache.strategy in ("scored_pq", "triple_am", "triple_pq_am", "auto")
        and config.cache.calibration_file is None
        and not caps.has_calibration
    ):
        from mlx_lm.models.am_calibrator import auto_calibrate

        cal_path = auto_calibrate(
            model,
            tokenizer,
            compression_ratio=config.cache.compression_ratio,
        )
        if cal_path:
            config.cache.calibration_file = str(cal_path)
    elif caps.calibration_path and config.cache.calibration_file is None:
        config.cache.calibration_file = caps.calibration_path

    # Warn about incompatible configurations
    for warning in caps.warnings:
        print(f"[FlashMLX] {warning}")

    # Override strategy for hybrid models
    if caps.is_hybrid and config.cache.strategy == "scored_pq":
        print("[FlashMLX] scored_pq auto-disabled for hybrid model. Using standard.")
        config.cache.strategy = "standard"
        config.cache.flat_quant = None

    return config.apply(model, model_path)


def flashmlx_settings_schema() -> dict:
    """Return JSON Schema for FlashMLXConfig.

    ThunderOMLX can use this for settings validation and documentation.

    Returns:
        dict: JSON Schema compatible with Pydantic v2.
    """
    from ..config import FlashMLXConfig
    return FlashMLXConfig.model_json_schema()


class ThunderOMLXAdapter:
    """Concrete FlashMLXProvider implementation for ThunderOMLX.

    Implements the FlashMLXProvider protocol so ThunderOMLX can
    consume FlashMLX capabilities through a stable interface.
    """

    def create_cache(
        self,
        model: nn.Module,
        strategy: str = "scored_pq",
        flat_quant: Optional[str] = "q8_0",
        calibration_file: Optional[str] = None,
        **kwargs: Any,
    ) -> list:
        from mlx_lm.models.cache_factory import make_optimized_cache
        return make_optimized_cache(
            model,
            strategy=strategy,
            flat_quant=flat_quant,
            calibration_file=calibration_file,
            **kwargs,
        )

    def apply_expert_offload(
        self,
        model: nn.Module,
        model_path: str,
        pool_size: int = 0,
        max_workers: int = 4,
    ) -> Any:
        from mlx_lm.models.expert_offload import patch_model_for_offload
        return patch_model_for_offload(
            model=model,
            model_path=model_path,
            pool_size=pool_size,
            max_workers=max_workers,
        )

    def detect_capabilities(
        self,
        model: nn.Module,
        model_path: Optional[str] = None,
    ) -> Any:
        from ..capabilities import detect_capabilities
        return detect_capabilities(model, model_path)

    def get_cache_info(self, cache_list: list) -> dict:
        from mlx_lm.models.cache_factory import get_cache_info
        return get_cache_info(cache_list)

    def auto_calibrate(
        self,
        model: nn.Module,
        tokenizer: Any,
        compression_ratio: float = 2.0,
    ) -> Optional[str]:
        from mlx_lm.models.am_calibrator import auto_calibrate
        path = auto_calibrate(model, tokenizer, compression_ratio=compression_ratio)
        return str(path) if path else None

    # ------------------------------------------------------------------
    # Cache bridge (Tier 1-2)
    # ------------------------------------------------------------------

    def export_compressed_cache_state(self, cache_list: list) -> Optional[Dict]:
        """Export all layers' compressed flat state + H0Store metadata."""
        try:
            from mlx_lm.models.triple_layer_cache import TripleLayerKVCache
        except ImportError:
            return None

        result = {'layers': [], 'h0_store': None, 'format': 'flashmlx_compressed_v1'}
        h0_store = None

        for i, cache in enumerate(cache_list):
            if isinstance(cache, TripleLayerKVCache):
                flat_state = cache.export_flat_state()
                result['layers'].append({
                    'layer_idx': i,
                    'type': 'TripleLayerKVCache',
                    'flat_state': flat_state,
                    'meta_state': cache.meta_state,
                })
                if h0_store is None:
                    h0_store = getattr(cache, '_h0_store', None)
            else:
                result['layers'].append({
                    'layer_idx': i,
                    'type': type(cache).__name__,
                    'state': cache.state if hasattr(cache, 'state') else None,
                    'meta_state': getattr(cache, 'meta_state', None),
                })

        if h0_store is not None and h0_store.count > 0:
            result['h0_store'] = {
                'count': h0_store.count,
                'quant': h0_store._quant or 'bf16',
                'nbytes': h0_store.nbytes,
            }

        return result

    def import_compressed_cache_state(
        self, cache_list: list, state: Dict
    ) -> bool:
        """Import compressed cache state into target cache objects."""
        try:
            from mlx_lm.models.triple_layer_cache import TripleLayerKVCache
        except ImportError:
            return False

        if state.get('format') != 'flashmlx_compressed_v1':
            return False

        for layer_info in state.get('layers', []):
            idx = layer_info['layer_idx']
            if idx >= len(cache_list):
                continue

            cache = cache_list[idx]
            if isinstance(cache, TripleLayerKVCache) and layer_info.get('flat_state'):
                cache.import_flat_state(layer_info['flat_state'])
                if layer_info.get('meta_state'):
                    cache.meta_state = layer_info['meta_state']
            elif hasattr(cache, 'state') and layer_info.get('state'):
                cache.state = layer_info['state']

        return True

    def export_h0_blocks(
        self, cache_list: list, block_size: int = 64
    ) -> Optional[Dict]:
        """Export H0Store as block-aligned chunks for ThunderOMLX SSD."""
        h0_store = None
        for c in cache_list:
            h0_store = getattr(c, '_h0_store', None)
            if h0_store is not None and h0_store.count > 0:
                break

        if h0_store is None:
            return None

        blocks = h0_store.export_blocks(block_size=block_size)
        return {
            'blocks': blocks,
            'total_tokens': h0_store.count,
            'quant': h0_store._quant or 'bf16',
            'nbytes_per_token': h0_store.nbytes / max(h0_store.count, 1),
            'format': 'h0_blocks_v1',
        }
