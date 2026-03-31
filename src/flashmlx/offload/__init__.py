"""
FlashMLX Expert Offloading — re-exports from enhanced mlx-lm.

Route 1: Three-tier intelligent expert management for MoE models.
  Tier 0: GPU Pool (mx.take + gather_qmm, zero Python sync)
  Tier 1: CPU Warm (numpy arrays, ~273 GB/s on UMA)
  Tier 2: SSD Cold (pread from safetensors via OS page cache)
"""

from mlx_lm.models.expert_offload import (
    FlashBatchGenerator,
    FlashMoeSwitchGLU,
    OffloadContext,
    ThunderOMLXBridge,
    patch_model_for_offload,
)

__all__ = [
    "patch_model_for_offload",
    "OffloadContext",
    "FlashBatchGenerator",
    "FlashMoeSwitchGLU",
    "ThunderOMLXBridge",
]
