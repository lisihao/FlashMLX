# Copyright © 2026 FlashMLX
# Adapted from MLX-VLM (https://github.com/Blaizzy/mlx-vlm)
#
# Vision Encoder for Vision-Language Models (VLMs)
# Supports Qwen2-VL, Qwen3-VL and similar architectures

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


@dataclass
class VisionConfig:
    """Configuration for Vision Encoder"""
    model_type: str = "qwen2_vl"
    hidden_size: int = 3584  # Output dimension (to Language Model)
    embed_dim: int = 1152    # Internal embedding dimension
    depth: int = 32          # Number of transformer blocks
    num_heads: int = 16      # Attention heads
    mlp_ratio: float = 4.0   # MLP hidden_dim = embed_dim * mlp_ratio

    # Patch parameters
    patch_size: int = 14
    temporal_patch_size: int = 2
    in_channels: int = 3     # RGB
    spatial_merge_size: int = 2  # Merge 2x2 patches


# ============================================================================
# Rotary Position Embedding for Vision
# ============================================================================

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb_vision(tensor, freqs) -> mx.array:
    """Apply rotary position embedding to vision tensor.

    Args:
        tensor: Input tensor [batch, heads, seq_len, head_dim]
        freqs: Position frequencies [seq_len, head_dim/2]

    Returns:
        Tensor with rotary position embedding applied
    """
    orig_dtype = tensor.dtype

    cos = mx.cos(freqs)
    sin = mx.sin(freqs)

    # Expand dimensions for broadcasting
    cos = mx.expand_dims(cos, axis=1)  # [seq_len, 1, head_dim/2]
    cos = mx.tile(cos, (1, 1, 2))      # [seq_len, 1, head_dim]
    cos = mx.expand_dims(cos, axis=0)  # [1, seq_len, 1, head_dim]

    sin = mx.expand_dims(sin, axis=1)
    sin = mx.tile(sin, (1, 1, 2))
    sin = mx.expand_dims(sin, axis=0)

    output = (tensor * cos) + (rotate_half(tensor) * sin)
    return output.astype(orig_dtype)


class VisionRotaryEmbedding(nn.Module):
    """2D Rotary Position Embedding for Vision Tokens.

    Unlike text (1D), vision uses 2D position encoding for (height, width).
    """
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta

    def __call__(self, seqlen: int) -> mx.array:
        """Generate position frequencies.

        Args:
            seqlen: Sequence length (max of height or width), can be int or mx.array

        Returns:
            Position frequencies [seqlen, dim]
        """
        # Convert seqlen to Python int if it's an MLX array
        if isinstance(seqlen, mx.array):
            seqlen = seqlen.item()

        inv_freq = 1.0 / (
            self.theta ** (mx.arange(0, self.dim, 2, dtype=mx.float32) / self.dim)
        )
        seq = mx.arange(seqlen, dtype=inv_freq.dtype)
        freqs = mx.outer(seq, inv_freq)
        return freqs


# ============================================================================
# Patch Embedding
# ============================================================================

def check_array_shape(arr):
    """Check if array needs transpose for Conv3d weight."""
    shape = arr.shape

    if len(shape) not in [4, 5]:
        return False

    B, out_channels, kH, KW, t = shape

    if t == 3:
        return True

    # Check if out_channels is the largest, and kH and KW are the same
    if (out_channels >= kH) and (out_channels >= KW) and (kH == KW):
        return True
    else:
        return False


class PatchEmbed(nn.Module):
    """Convert image to patches using 3D convolution.

    Supports both images and videos:
    - Image: [B, C, 1, H, W] → patches
    - Video: [B, C, T, H, W] → patches

    Args:
        patch_size: Size of each patch (default: 14)
        temporal_patch_size: Temporal dimension patch size (default: 2)
        in_channels: Input channels (default: 3 for RGB)
        embed_dim: Output embedding dimension (default: 1152)
    """
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=False,
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """
        Args:
            hidden_states: [B*T, C, H, W] or [B, C, T, H, W]

        Returns:
            Patches: [num_patches, embed_dim]
        """
        hidden_states = hidden_states.reshape(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        ).moveaxis(1, 4)  # Move channels to last dimension for Conv3d

        hidden_states = self.proj(hidden_states)
        hidden_states = hidden_states.reshape(-1, self.embed_dim)
        return hidden_states


class PatchMerger(nn.Module):
    """Merge spatial patches to reduce token count.

    Merges spatial_merge_size x spatial_merge_size patches into one token.
    For example, with spatial_merge_size=2, 32x32 patches → 16x16 tokens.

    Args:
        dim: Output dimension (to language model)
        context_dim: Input dimension (from vision transformer)
        spatial_merge_size: Number of patches to merge in each dimension
    """
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = nn.LayerNorm(context_dim, eps=1e-6)
        self.mlp = [
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: [num_patches, context_dim]

        Returns:
            Merged tokens: [num_patches // (spatial_merge_size^2), dim]
        """
        x = self.ln_q(x).reshape(-1, self.hidden_size)
        for layer in self.mlp:
            x = layer(x)
        return x


# ============================================================================
# Vision Transformer Blocks
# ============================================================================

class Attention(nn.Module):
    """Multi-head self-attention for vision tokens.

    Uses 2D rotary position embedding and supports batched sequences.
    """
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def __call__(
        self, x: mx.array, cu_seqlens: mx.array, rotary_pos_emb: mx.array = None
    ) -> mx.array:
        """
        Args:
            x: [seq_length, dim]
            cu_seqlens: Cumulative sequence lengths for batching
            rotary_pos_emb: Rotary position embeddings

        Returns:
            Attention output: [seq_length, dim]
        """
        seq_length = x.shape[0]
        qkv = (
            self.qkv(x).reshape(seq_length, 3, self.num_heads, -1).transpose(1, 0, 2, 3)
        )
        q, k, v = mx.split(qkv, 3)

        # Apply rotary position embedding
        q = apply_rotary_pos_emb_vision(mx.expand_dims(q, 0), rotary_pos_emb)[0]
        k = apply_rotary_pos_emb_vision(mx.expand_dims(k, 0), rotary_pos_emb)[0]

        # Transpose for attention: [heads, seq_len, head_dim]
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Split by sequence for batched attention
        splits = [
            mx.split(tensor, cu_seqlens[1:-1].tolist(), axis=2) for tensor in (q, k, v)
        ]

        attn_outputs = []
        for q, k, v in zip(*splits):
            output = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=self.scale, mask=None
            )
            attn_outputs.append(output)
        output = mx.concatenate(attn_outputs, axis=2)
        output = output.transpose(0, 2, 1, 3)
        output = output.reshape(seq_length, -1)
        return self.proj(output)


class MLP(nn.Module):
    """Feed-forward network for vision transformer."""
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.activation_fn = nn.GELU(approx="fast")
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.activation_fn(self.fc1(x))
        x = self.fc2(x)
        return x


class Qwen2VLVisionBlock(nn.Module):
    """Single transformer block for vision encoder.

    Architecture: Pre-norm Transformer
    - LayerNorm → Attention → Residual
    - LayerNorm → MLP → Residual
    """
    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.embed_dim, eps=1e-6)
        mlp_hidden_dim = int(config.embed_dim * config.mlp_ratio)

        self.attn = Attention(dim=config.embed_dim, num_heads=config.num_heads)
        self.mlp = MLP(dim=config.embed_dim, hidden_dim=mlp_hidden_dim)

    def __call__(self, hidden_states, cu_seqlens, rotary_pos_emb) -> mx.array:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


# ============================================================================
# Main Vision Model
# ============================================================================

class VisionModel(nn.Module):
    """Complete Vision Encoder for VLMs.

    Converts images/videos into vision token embeddings that can be
    consumed by the language model.

    Pipeline:
        Image [H, W, 3] → PatchEmbed → [num_patches, embed_dim]
                        → VisionBlocks → [num_patches, embed_dim]
                        → PatchMerger → [num_tokens, hidden_size]

    Example:
        448x448 image → 1024 patches (32x32 grid)
                      → 256 tokens (16x16 grid after 2x2 merge)

    Args:
        config: VisionConfig with model parameters
    """
    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        if self.model_type != "qwen2_vl":
            raise ValueError(f"Unsupported model type: {self.model_type}")
        self.spatial_merge_size = config.spatial_merge_size

        self.patch_embed = PatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
        )

        head_dim = config.embed_dim // config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        self.blocks = [Qwen2VLVisionBlock(config) for _ in range(config.depth)]
        self.merger = PatchMerger(dim=config.hidden_size, context_dim=config.embed_dim)

    def rot_pos_emb(self, grid_thw):
        """Generate 2D rotary position embeddings for vision grid.

        Args:
            grid_thw: [batch_size, 3] array of (temporal, height, width)

        Returns:
            Position embeddings: [total_patches, rotary_dim]
        """
        pos_ids = []

        for t, h, w in grid_thw:
            h, w = int(h), int(w)  # Ensure h and w are integers

            # Height position IDs
            hpos_ids = mx.expand_dims(mx.arange(h), 1)
            hpos_ids = mx.repeat(hpos_ids, w, axis=1)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = mx.transpose(hpos_ids, (0, 2, 1, 3))
            hpos_ids = hpos_ids.flatten()

            # Width position IDs
            wpos_ids = mx.expand_dims(mx.arange(w), 0)
            wpos_ids = mx.repeat(wpos_ids, h, axis=0)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = mx.transpose(wpos_ids, (0, 2, 1, 3))
            wpos_ids = wpos_ids.flatten()

            # Stack (h, w) positions
            stacked_pos_ids = mx.stack([hpos_ids, wpos_ids], axis=-1)
            pos_ids.append(mx.tile(stacked_pos_ids, (t, 1)))

        pos_ids = mx.concatenate(pos_ids, axis=0)
        max_grid_size = mx.max(grid_thw[:, 1:])
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)

        rotary_pos_emb_full = rotary_pos_emb_full[pos_ids]

        return rotary_pos_emb_full.reshape(pos_ids.shape[0], -1)

    def __call__(
        self,
        hidden_states: mx.array,
        grid_thw: mx.array,
        output_hidden_states: Optional[bool] = None,
    ) -> mx.array:
        """Encode vision input to token embeddings.

        Args:
            hidden_states: Pixel values [B*T, C, H, W]
            grid_thw: Grid dimensions [batch_size, 3] (temporal, height, width)
            output_hidden_states: Whether to return all hidden states

        Returns:
            Vision token embeddings: [num_tokens, hidden_size]

        Example:
            Input: 448x448 RGB image
            grid_thw: [[1, 32, 32]] (1 frame, 32x32 patches)
            Output: [256, 3584] (256 vision tokens after 2x2 merge)
        """
        # 1. Patch Embedding: Image → Patches
        hidden_states = self.patch_embed(hidden_states)

        # 2. Generate 2D position embeddings
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        # 3. Calculate cumulative sequence lengths for batching
        batch_size = grid_thw.shape[0]
        cu_seqlens = []
        for i in range(batch_size):
            seq_len = grid_thw[i, 1] * grid_thw[i, 2]
            cu_seqlens.append(mx.repeat(seq_len, grid_thw[i, 0]))
        cu_seqlens = mx.concatenate(cu_seqlens)
        cu_seqlens = mx.cumsum(cu_seqlens.astype(mx.int32), axis=0)
        cu_seqlens = mx.pad(cu_seqlens, (1, 0), mode="constant", constant_values=0)

        # 4. Pass through transformer blocks
        encoder_states = (hidden_states,) if output_hidden_states else None
        for blk in self.blocks:
            hidden_states = blk(
                hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb
            )
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

        # 5. Merge patches: 32x32 → 16x16 (4x reduction)
        return self.merger(hidden_states)

    def sanitize(self, weights):
        """Sanitize vision weights loaded from Hugging Face.

        Handles weight format conversion:
        - PyTorch Conv3d: [out, in, kH, KW, kT]
        - MLX Conv3d: [out, kH, KW, kT, in]
        - PyTorch Linear: [out, in]
        - MLX Linear: [in, out]
        """
        sanitized_weights = {}
        for k, v in weights.items():
            if "position_ids" in k:
                # Remove unused position_ids
                continue
            elif "patch_embed.proj.weight" in k:
                # Transpose Conv3d weights if needed
                if check_array_shape(v):
                    sanitized_weights[k] = v
                else:
                    sanitized_weights[k] = v.transpose(0, 2, 3, 4, 1)
            elif ".weight" in k and len(v.shape) == 2:
                # Transpose Linear layer weights (PyTorch [out, in] → MLX [in, out])
                # Skip quantization params (.biases, .scales) - they are 2D but shouldn't be transposed
                if not (k.endswith('.biases') or k.endswith('.scales')):
                    sanitized_weights[k] = v.T
                else:
                    sanitized_weights[k] = v
            else:
                sanitized_weights[k] = v

        return sanitized_weights
