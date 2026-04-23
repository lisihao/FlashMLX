"""FlashMLX custom Metal kernels for MoE inference on Apple Silicon."""

from .fused_moe import fused_gate_up_swiglu
