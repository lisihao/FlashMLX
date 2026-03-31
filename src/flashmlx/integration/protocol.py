"""
FlashMLX Provider Protocol — abstract interface for consumers.

ThunderOMLX (or any other consumer) programs against this protocol,
enabling optional FlashMLX dependency with graceful fallback.

Usage in ThunderOMLX:
    try:
        from flashmlx.integration import FlashMLXProvider
        provider: FlashMLXProvider = ...
    except ImportError:
        provider = None  # fallback to vanilla mlx-lm
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable

import mlx.nn as nn


@runtime_checkable
class FlashMLXProvider(Protocol):
    """Protocol interface for FlashMLX capabilities.

    All methods are typed against plain Python / mlx types,
    avoiding hard coupling to FlashMLX internals.
    """

    def create_cache(
        self,
        model: nn.Module,
        strategy: str = "scored_pq",
        flat_quant: Optional[str] = "q8_0",
        calibration_file: Optional[str] = None,
        **kwargs: Any,
    ) -> list:
        """Create an optimized KV cache list for the model.

        Args:
            model: The loaded MLX language model.
            strategy: Cache strategy name.
            flat_quant: Flat buffer quantization method.
            calibration_file: Path to AM calibration .pkl.
            **kwargs: Additional make_optimized_cache kwargs.

        Returns:
            List of cache objects, one per model layer.
        """
        ...

    def apply_expert_offload(
        self,
        model: nn.Module,
        model_path: str,
        pool_size: int = 0,
        max_workers: int = 4,
    ) -> Any:
        """Patch model for expert offloading (MoE models).

        Args:
            model: The loaded MLX language model.
            model_path: Path to model weights on disk.
            pool_size: GPU expert pool size (0 = auto).
            max_workers: SSD loader thread count.

        Returns:
            OffloadContext that must be kept alive during generation.
        """
        ...

    def detect_capabilities(
        self,
        model: nn.Module,
        model_path: Optional[str] = None,
    ) -> Any:
        """Detect model capabilities and recommended configuration.

        Returns:
            ModelCapabilities dataclass.
        """
        ...

    def get_cache_info(self, cache_list: list) -> dict:
        """Get diagnostic info about a cache configuration.

        Returns:
            dict with strategy, type, memory usage, etc.
        """
        ...

    def auto_calibrate(
        self,
        model: nn.Module,
        tokenizer: Any,
        compression_ratio: float = 2.0,
    ) -> Optional[str]:
        """Generate AM calibration data for a model.

        Returns:
            Path to the generated calibration .pkl file.
        """
        ...
