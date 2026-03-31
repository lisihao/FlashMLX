"""
Attention Patcher for CompactedKVCache support

Monkey patches MLX-LM attention layers to apply beta bias terms
from CompactedKVCache before attention computation.
"""
import types
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import scaled_dot_product_attention


def repeat_kv(x: mx.array, n_rep: int) -> mx.array:
    """
    Repeat KV heads for Grouped Query Attention (GQA)

    Args:
        x: Input array of shape (B, n_kv_heads, seq_len, head_dim)
           or (B, n_kv_heads, seq_len) for beta
        n_rep: Number of repetitions (num_attention_heads // num_key_value_heads)

    Returns:
        Output array with repeated heads:
        - (B, n_heads, seq_len, head_dim) for KV
        - (B, n_heads, seq_len) for beta
    """
    if n_rep == 1:
        return x

    # x shape: (B, n_kv_heads, seq_len, ...) → (B, n_kv_heads, 1, seq_len, ...)
    # repeat along new axis → (B, n_kv_heads, n_rep, seq_len, ...)
    # reshape → (B, n_heads, seq_len, ...)

    B, n_kv_heads, *rest = x.shape

    # Add dimension for repetition
    x_expanded = mx.expand_dims(x, axis=2)  # (B, n_kv_heads, 1, seq_len, ...)

    # Repeat along the new axis
    x_repeated = mx.repeat(x_expanded, n_rep, axis=2)  # (B, n_kv_heads, n_rep, seq_len, ...)

    # Reshape to merge n_kv_heads and n_rep
    n_heads = n_kv_heads * n_rep
    output_shape = [B, n_heads] + rest
    x_output = x_repeated.reshape(output_shape)

    return x_output


def patch_attention_for_compacted_cache(model, verbose: bool = True):
    """
    Monkey patch model's attention layers to support CompactedKVCache

    Modifies each attention layer's __call__ method to:
    1. Detect CompactedKVCache instances
    2. Extract beta bias terms for the current layer
    3. Apply beta to attention mask before softmax

    Args:
        model: MLX-LM model instance (e.g., Qwen3, Llama, etc.)
        verbose: Whether to print patching information

    Returns:
        None (patches model in-place)
    """
    from flashmlx.cache.compacted_kv_cache import CompactedKVCacheLayer

    # Get model layers
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    elif hasattr(model, 'layers'):
        layers = model.layers
    else:
        raise ValueError("Cannot find model layers")

    num_layers = len(layers)

    if verbose:
        print(f"✓ Patching {num_layers} attention layers for CompactedKVCache support")

    # Patch each layer's attention
    for layer_idx, layer in enumerate(layers):
        # Save reference to original attention module
        attention_module = layer.self_attn

        # Create patched __call__ method
        def make_patched_call(layer_idx, attn):
            """Create patched __call__ with closure over layer_idx and attn"""

            def patched_call(self, x: mx.array, mask: Optional[mx.array] = None, cache: Optional[any] = None) -> mx.array:
                """
                Patched attention forward pass with CompactedKVCache support

                Identical to original except for beta bias application when
                cache is a CompactedKVCache instance.
                """
                B, L, D = x.shape

                # Q, K, V projection (same as original)
                queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

                # Reshape and transpose (same as original)
                queries = self.q_norm(queries.reshape(B, L, self.n_heads, -1)).transpose(0, 2, 1, 3)
                keys = self.k_norm(keys.reshape(B, L, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
                values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

                # Apply RoPE and update cache (same as original)
                if cache is not None:
                    queries = self.rope(queries, offset=cache.offset)
                    keys = self.rope(keys, offset=cache.offset)
                    keys, values = cache.update_and_fetch(keys, values)
                else:
                    queries = self.rope(queries)
                    keys = self.rope(keys)

                # ✅ NEW: Detect CompactedKVCacheLayer and apply beta
                modified_mask = mask
                if cache is not None and isinstance(cache, CompactedKVCacheLayer):
                    beta = cache.get_beta()

                    if beta is not None:
                        # Calculate dimensions
                        n_rep = self.n_heads // self.n_kv_heads

                        # Repeat beta for GQA: (B, n_kv_heads, t) → (B, n_heads, t)
                        beta_heads = repeat_kv(beta, n_rep)

                        prefix_length = beta_heads.shape[-1]
                        query_length = queries.shape[2]
                        kv_length = keys.shape[2]

                        # Initialize or copy mask
                        if modified_mask is None:
                            modified_mask = mx.zeros((B, self.n_heads, query_length, kv_length), dtype=queries.dtype)
                        else:
                            # Copy to avoid modifying shared tensor
                            modified_mask = mx.array(modified_mask)

                        # Apply beta to prefix positions
                        # Shape: (B, n_heads, Q, KV) + (B, n_heads, 1, t)
                        # We need to add beta to [:, :, :, :prefix_length]
                        beta_expanded = mx.expand_dims(beta_heads, axis=2)  # (B, n_heads, 1, t)

                        # Create updated mask by adding beta to prefix region
                        prefix_mask = modified_mask[:, :, :, :prefix_length] + beta_expanded

                        # Reconstruct full mask
                        if kv_length > prefix_length:
                            suffix_mask = modified_mask[:, :, :, prefix_length:]
                            modified_mask = mx.concatenate([prefix_mask, suffix_mask], axis=-1)
                        else:
                            modified_mask = prefix_mask

                # Attention (using modified mask)
                output = scaled_dot_product_attention(
                    queries, keys, values, cache=cache, scale=self.scale, mask=modified_mask
                )
                output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
                return self.o_proj(output)

            return patched_call

        # Apply the patch using types.MethodType for proper binding
        layer.self_attn.__call__ = types.MethodType(
            make_patched_call(layer_idx, attention_module),
            attention_module
        )

    if verbose:
        print(f"✓ Successfully patched {num_layers} layers")
