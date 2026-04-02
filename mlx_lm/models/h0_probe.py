"""
H0Probe: Lightweight attention probe for importance-based KV eviction.

Runs h^(0) through only the first N transformer layers (~N/36 compute for
Qwen3-8B) and extracts per-token attention importance via cumulative
column-sum scoring (H2O-style).

Technical approach:
  - Temporarily monkey-patches scaled_dot_product_attention to capture
    attention weights during probe execution (restored after).
  - Computes attention in q_chunk batches (64 queries at a time) to limit
    peak memory to ~64 MB at 8K context.
  - Returns per-token importance: sum of attention received across all
    query positions, heads, and probe layers.

Usage:
    probe = H0Probe(inner_model, n_probe_layers=3)
    scores = probe.score_tokens(h0_store)  # np.ndarray (n_tokens,)
"""

import numpy as np
import mlx.core as mx


class H0Probe:
    """3-layer forward probe for attention-based importance scoring."""

    def __init__(self, inner_model, n_probe_layers=3):
        self._inner_model = inner_model
        self._n_layers = n_probe_layers

    def score_tokens(self, h0_store, start=0, end=None, q_chunk=64):
        """Run probe and return per-token importance scores.

        Feeds h^(0)[start:end] through the first N layers with SDPA
        monkey-patched to capture cumulative attention column-sums.

        Args:
            h0_store: H0Store with archived embeddings.
            start: Start token index (default 0).
            end: End token index (default h0_store.count).
            q_chunk: Query chunk size for attention computation.
                Controls peak memory: (1, n_heads, q_chunk, kv_len) * 4B.
                Default 64 = ~64 MB peak at 8K context with 32 heads.

        Returns:
            np.ndarray of shape (n_tokens,) with cumulative attention importance.
        """
        end = end or h0_store.count
        h0_range = h0_store.get_range(start, end)
        n_tokens = end - start

        from mlx_lm.models import base as base_mod
        original_sdpa = base_mod.scaled_dot_product_attention

        # Accumulator for column-sum importance
        importance = [mx.zeros((n_tokens,))]

        def _sdpa_with_capture(queries, keys, values, cache, scale, mask, sinks=None):
            B, n_heads, q_len, head_dim = queries.shape
            kv_len = keys.shape[2]
            n_kv_heads = keys.shape[1]

            # GQA expansion for score computation
            if n_heads != n_kv_heads:
                n_rep = n_heads // n_kv_heads
                keys_exp = mx.repeat(keys, n_rep, axis=1)
            else:
                keys_exp = keys

            # Compute attention column-sums in chunks to limit memory
            col_sums = mx.zeros((B, n_heads, kv_len))
            for qi in range(0, q_len, q_chunk):
                qe = min(qi + q_chunk, q_len)
                q_slice = queries[:, :, qi:qe, :]
                scores = mx.matmul(q_slice, keys_exp.transpose(0, 1, 3, 2)) / scale
                if mask is not None:
                    if mask.ndim == 4:
                        m_slice = mask[:, :, qi:qe, :]
                    else:
                        m_slice = mask
                    scores = scores + m_slice
                attn = mx.softmax(scores, axis=-1)
                col_sums = col_sums + attn.sum(axis=2)

            # Accumulate: sum over batch and heads → (kv_len,)
            token_importance = col_sums.sum(axis=(0, 1))
            # Pad to full n_tokens if kv_len < n_tokens (early layers)
            if token_importance.shape[0] < n_tokens:
                pad = mx.zeros((n_tokens - token_importance.shape[0],))
                token_importance = mx.concatenate([token_importance, pad])
            importance[0] = importance[0] + token_importance[:n_tokens]

            # Use fast path for actual output (correct forward pass)
            return mx.fast.scaled_dot_product_attention(
                queries, keys, values, scale=scale, mask=mask, sinks=sinks
            )

        try:
            base_mod.scaled_dot_product_attention = _sdpa_with_capture

            from mlx_lm.models.cache import KVCache
            from mlx_lm.models.base import create_attention_mask

            temp_caches = [KVCache() for _ in range(self._n_layers)]
            mask = create_attention_mask(h0_range, temp_caches[0])

            h = h0_range
            layers = self._inner_model.layers[:self._n_layers]
            for layer, tc in zip(layers, temp_caches):
                h = layer(h, mask, tc)
            mx.eval(h)

        finally:
            base_mod.scaled_dot_product_attention = original_sdpa

        result = importance[0]
        mx.eval(result)
        return np.array(result.astype(mx.float32))
