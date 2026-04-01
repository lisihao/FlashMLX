"""
KV-Direct v2: Model-Level h^(0) Checkpointing for KV Cache

Implements the KV-Direct paper's (arxiv 2603.19664) core compression:
store ONE h^(0) per token (embed_tokens output), reconstruct K/V for
evicted tokens via forward pass through transformer layers.

Memory savings (Qwen3-8B): 18x for evicted tokens.
  Standard KV per token: 36 layers * 2 * 8 * 128 * 2B = 147,456 bytes
  h^(0) per token:       4096 * 2B = 8,192 bytes

Technique: __class__ swapping on inner model (zero model code changes).
"""

import mlx.core as mx

from .base import create_attention_mask
from .cache import _BaseCache, KVCache
from .cache import create_attention_mask as cache_create_attention_mask


# ---------------------------------------------------------------------------
# H0Store: Shared h^(0) storage (one instance per generation)
# ---------------------------------------------------------------------------

class H0Store:
    """Shared storage for h^(0) = embed_tokens(x), used by all layers.

    Stores the initial embedding output for every token processed.
    Memory: T_total * d_hidden * dtype_bytes (shared across all L layers).

    Quantization modes:
      None:  bf16 exact storage (default)
      'q8':  int8 absmax per-token — 2x compression, near-lossless
      'q4':  int4 absmax per-token packed uint8 — 4x compression, lossy
    """

    def __init__(self, quant=None):
        self._quant = quant
        self._h0 = None       # bf16: (B, T, d) | q8: int8 (B, T, d) | q4: uint8 (B, T, d//2)
        self._scales = None   # q8/q4: float16 (B, T, 1)
        self._count = 0

    def append(self, h0):
        """Append new tokens' h^(0). h0 shape: (B, L, d_hidden)."""
        if self._quant == 'q8':
            qdata, scales = self._q8_encode(h0)
            if self._h0 is None:
                self._h0 = qdata
                self._scales = scales
            else:
                self._h0 = mx.concatenate([self._h0, qdata], axis=1)
                self._scales = mx.concatenate([self._scales, scales], axis=1)
        elif self._quant == 'q4':
            qdata, scales = self._q4_encode(h0)
            if self._h0 is None:
                self._h0 = qdata
                self._scales = scales
            else:
                self._h0 = mx.concatenate([self._h0, qdata], axis=1)
                self._scales = mx.concatenate([self._scales, scales], axis=1)
        else:
            if self._h0 is None:
                self._h0 = h0
            else:
                self._h0 = mx.concatenate([self._h0, h0], axis=1)
        self._count = self._h0.shape[1]

    def get_range(self, start, end):
        """Return dequantized h^(0) for tokens [start:end].

        Returns: (B, end-start, d_hidden) in bfloat16.
        """
        if self._quant == 'q8':
            return self._q8_decode(
                self._h0[:, start:end, :], self._scales[:, start:end, :]
            )
        elif self._quant == 'q4':
            return self._q4_decode(
                self._h0[:, start:end, :], self._scales[:, start:end, :]
            )
        return self._h0[:, start:end, :]

    def get_evicted(self, n_evicted):
        """Return h^(0) for the first n_evicted tokens (oldest = evicted).

        Returns: (B, n_evicted, d_hidden) in bfloat16.
        """
        return self.get_range(0, n_evicted)

    @property
    def count(self):
        return self._count

    @property
    def nbytes(self):
        if self._h0 is None:
            return 0
        total = self._h0.nbytes
        if self._scales is not None:
            total += self._scales.nbytes
        return total

    # --- Q8: int8 absmax per-token ---

    @staticmethod
    def _q8_encode(h0):
        """Quantize bf16 h^(0) to int8 with per-token absmax scaling.

        Args: h0 (B, L, d) bf16
        Returns: (qdata (B, L, d) int8, scales (B, L, 1) float16)
        """
        scales = mx.abs(h0).max(axis=-1, keepdims=True).astype(mx.float16)
        scales = mx.maximum(scales, 1e-8)  # avoid div-by-zero
        qdata = mx.round(h0 / scales * 127.0).astype(mx.int8)
        return qdata, scales

    @staticmethod
    def _q8_decode(qdata, scales):
        """Dequantize int8 back to bf16.

        Args: qdata (B, L, d) int8, scales (B, L, 1) float16
        Returns: (B, L, d) bfloat16
        """
        return (qdata.astype(mx.bfloat16) * scales.astype(mx.bfloat16)) / 127.0

    # --- Q4: int4 absmax per-token, packed into uint8 ---

    @staticmethod
    def _q4_encode(h0):
        """Quantize bf16 h^(0) to int4 (packed uint8) with per-token absmax.

        Args: h0 (B, L, d) bf16 — d must be even
        Returns: (qdata (B, L, d//2) uint8, scales (B, L, 1) float16)

        Packing: two int4 values [-8..7] per uint8 byte.
          high nibble = h0[..., 2i], low nibble = h0[..., 2i+1]
          stored as (val + 8) to map [-8..7] → [0..15]
        """
        B, L, d = h0.shape
        scales = mx.abs(h0).max(axis=-1, keepdims=True).astype(mx.float16)
        scales = mx.maximum(scales, 1e-8)
        # Quantize to [-8, 7] range
        normalized = h0 / scales * 7.0
        clipped = mx.clip(mx.round(normalized), -8.0, 7.0)
        # Offset to [0, 15] for unsigned packing
        offset = (clipped + 8.0).astype(mx.uint8)
        # Pack pairs: high nibble | low nibble
        even = offset[..., 0::2]   # (B, L, d//2)
        odd = offset[..., 1::2]    # (B, L, d//2)
        packed = (even << 4) | odd  # (B, L, d//2) uint8
        return packed, scales

    @staticmethod
    def _q4_decode(packed, scales):
        """Dequantize packed int4 back to bf16.

        Args: packed (B, L, d//2) uint8, scales (B, L, 1) float16
        Returns: (B, L, d) bfloat16
        """
        B, L, half_d = packed.shape
        # Unpack high/low nibbles
        high = (packed >> 4).astype(mx.float16) - 8.0   # [-8, 7]
        low = (packed & 0x0F).astype(mx.float16) - 8.0  # [-8, 7]
        # Interleave: stack [high, low] on last axis → (B, L, half_d, 2) → reshape (B, L, d)
        result = mx.stack([high, low], axis=-1).reshape(B, L, half_d * 2)
        return (result * scales / 7.0).astype(mx.bfloat16)


# ---------------------------------------------------------------------------
# KVDirectCache: Per-layer cache with reconstruction injection
# ---------------------------------------------------------------------------

class KVDirectCache(_BaseCache):
    """Per-layer KV-Direct cache (v2: model-level h^(0) checkpointing).

    Two-region design:
      Recent Region: last ``budget`` tokens -> full K/V stored
      Evicted Region: older tokens -> K/V reconstructed on-the-fly from
                      shared h^(0) via forward pass (transient, freed after use)
    """

    def __init__(self, budget=512, h0_store=None):
        self._budget = budget

        # Recent K/V window (B, n_kv_heads, <=budget, head_dim)
        self._recent_keys = None
        self._recent_values = None
        self._recent_count = 0

        # Sequence tracking
        self.offset = 0

        # Shared h^(0) store (set by factory, shared across all layers)
        self._h0_store = h0_store

        # Reconstructed K/V injection (set by model patch, consumed once)
        self._recon_keys = None
        self._recon_values = None

    def update_and_fetch(self, keys, values):
        """Update cache with new K/V and return full-sequence K/V.

        Args:
            keys: (B, n_kv_heads, L, head_dim) with RoPE applied
            values: (B, n_kv_heads, L, head_dim)

        Returns:
            (all_keys, all_values) covering the full sequence.
        """
        L = keys.shape[2]

        # 1. Track total sequence position
        self.offset += L

        # 2. Append to recent window
        if self._recent_keys is None:
            self._recent_keys = keys
            self._recent_values = values
        else:
            self._recent_keys = mx.concatenate(
                [self._recent_keys, keys], axis=2
            )
            self._recent_values = mx.concatenate(
                [self._recent_values, values], axis=2
            )
        self._recent_count = self._recent_keys.shape[2]

        # 3. Within budget — standard behavior, no eviction
        if self.offset <= self._budget:
            self._recon_keys = None
            self._recon_values = None
            return self._recent_keys, self._recent_values

        # 4. Trim recent window to budget
        if self._recent_count > self._budget:
            trim_n = self._recent_count - self._budget
            self._recent_keys = self._recent_keys[..., trim_n:, :]
            self._recent_values = self._recent_values[..., trim_n:, :]
            self._recent_count = self._budget

        # 5. Prepend reconstructed K/V (injected by model-level patch)
        if self._recon_keys is not None:
            all_k = mx.concatenate([self._recon_keys, self._recent_keys], axis=2)
            all_v = mx.concatenate([self._recon_values, self._recent_values], axis=2)
            self._recon_keys = None
            self._recon_values = None
            return all_k, all_v

        # 6. Fallback: no reconstruction (shouldn't happen in steady state)
        return self._recent_keys, self._recent_values

    # --- _BaseCache interface ---

    def size(self):
        return self.offset

    @property
    def nbytes(self):
        """Per-layer recent K/V only. h^(0) is shared — reported separately."""
        kv_bytes = 0
        if self._recent_keys is not None:
            kv_bytes = self._recent_keys.nbytes + self._recent_values.nbytes
        return kv_bytes

    def empty(self):
        return self._recent_keys is None

    @property
    def state(self):
        if self._recent_keys is not None:
            return self._recent_keys, self._recent_values
        return []

    @state.setter
    def state(self, v):
        if v is not None and v:
            self._recent_keys, self._recent_values = v
            self._recent_count = self._recent_keys.shape[2]
            self.offset = self._recent_count

    def make_mask(self, *args, **kwargs):
        return cache_create_attention_mask(*args, offset=self.offset, **kwargs)


# ---------------------------------------------------------------------------
# Model-level monkey patch: h^(0) capture + reconstruction
# ---------------------------------------------------------------------------

def apply_h0_capture(model, caches, h0_store):
    """Install h^(0) capture and reconstruction via model-level __class__ swap.

    Patches the inner model (e.g. Qwen3Model) to:
      1. Capture h^(0) = embed_tokens(inputs) on every forward call
      2. Before the layer loop: if eviction needed, run reconstruction
         forward pass with temp KVCaches to produce evicted K/V
      3. Inject reconstructed K/V into each KVDirectCache
      4. Run the normal layer loop

    Args:
        model: Outer Model (has model.model as inner model with layers).
        caches: List of cache objects (KVDirectCache for attention layers).
        h0_store: H0Store instance for shared h^(0) storage.
    """
    inner_model = model.model
    base_cls = type(inner_model)
    parent_call = base_cls.__call__

    kv_direct_indices = [
        i for i, c in enumerate(caches) if isinstance(c, KVDirectCache)
    ]

    def _make_patched_call(h0s, kv_indices, orig_call):
        def __call__(self, inputs, cache=None, input_embeddings=None):
            # Phase 0: Compute h^(0)
            if input_embeddings is not None:
                h = input_embeddings
            else:
                h = self.embed_tokens(inputs)

            h0s.append(h)

            # Phase 1: Reconstruction (when eviction is needed)
            if cache is not None:
                # Find the first KVDirectCache in the actual cache list
                c0_kvd = None
                active_indices = []
                for idx in kv_indices:
                    if idx < len(cache) and isinstance(cache[idx], KVDirectCache):
                        if c0_kvd is None:
                            c0_kvd = cache[idx]
                        active_indices.append(idx)

                if c0_kvd is not None:
                    future_offset = c0_kvd.offset + h.shape[1]
                    if future_offset > c0_kvd._budget:
                        n_evicted = future_offset - c0_kvd._budget
                        if n_evicted > 0 and h0s.count >= n_evicted:
                            _run_reconstruction(
                                self, cache, h0s, n_evicted, active_indices
                            )

            # Phase 2: Normal forward pass (reuse already-computed h)
            if cache is None:
                cache = [None] * len(self.layers)

            mask = create_attention_mask(h, cache[0])

            for layer, c in zip(self.layers, cache):
                h = layer(h, mask, c)

            return self.norm(h)

        return __call__

    inner_model.__class__ = type(
        f"_KVDirect_{base_cls.__name__}",
        (base_cls,),
        {"__call__": _make_patched_call(h0_store, kv_direct_indices, parent_call)},
    )


# ---------------------------------------------------------------------------
# Route 5: h^(0) capture only (for scored_pq fusion)
# ---------------------------------------------------------------------------

def apply_h0_capture_only(model, h0_store):
    """Install h^(0) capture WITHOUT reconstruction — for scored_pq fusion.

    Patches the inner model to capture embed_tokens output into h0_store
    on every forward call, then delegates to the original __call__ with
    input_embeddings to avoid recomputation.

    Unlike apply_h0_capture, this does NOT run reconstruction — the
    scored_pq cache handles attention normally via its flat buffer.
    """
    inner_model = model.model
    base_cls = type(inner_model)
    parent_call = base_cls.__call__

    def _make_capture_call(h0s, orig_call):
        def __call__(self, inputs, cache=None, input_embeddings=None):
            # Compute h^(0) once
            if input_embeddings is not None:
                h = input_embeddings
            else:
                h = self.embed_tokens(inputs)

            # Store in h^(0) archive
            h0s.append(h)

            # Delegate to original forward with pre-computed embeddings
            return orig_call(self, inputs, cache, input_embeddings=h)

        return __call__

    inner_model.__class__ = type(
        f"_H0Capture_{base_cls.__name__}",
        (base_cls,),
        {"__call__": _make_capture_call(h0_store, parent_call)},
    )


# ---------------------------------------------------------------------------
# Route 5: On-demand reconstruction from h^(0)
# ---------------------------------------------------------------------------

def reconstruct_kv(inner_model, h0_store, start, end):
    """Reconstruct K/V for tokens [start:end] from h^(0).

    Runs a forward pass through all layers using h^(0) as input,
    producing the exact K/V that would have been computed originally.

    Args:
        inner_model: The inner model (e.g., model.model) with .layers.
        h0_store: H0Store containing archived embeddings.
        start: Start token index (inclusive). Must be 0 for now (prefix-only).
        end: End token index (exclusive).

    Returns:
        List of (keys, values) tuples per layer, each (B, n_kv_heads, N, head_dim).
    """
    if start != 0:
        raise NotImplementedError(
            "reconstruct_kv currently only supports prefix reconstruction (start=0). "
            "Arbitrary range reconstruction requires KVCache initial_offset support."
        )

    h_range = h0_store.get_range(start, end)

    num_layers = len(inner_model.layers)
    temp_caches = [KVCache() for _ in range(num_layers)]

    mask = create_attention_mask(h_range, temp_caches[0])

    h = h_range
    for layer, tc in zip(inner_model.layers, temp_caches):
        h = layer(h, mask, tc)

    return [tc.state for tc in temp_caches]


def _run_reconstruction(inner_model, caches, h0_store, n_evicted, kv_direct_indices):
    """Run reconstruction forward pass to produce evicted K/V.

    Creates FRESH temp KVCaches each call, feeds h^(0)[0:n_evicted] through
    all layers, extracts K/V and injects into real caches. Temp caches are
    freed after use — only h^(0) persists (paper-faithful, memory-efficient).

    Correctness:
      - Evicted tokens are always the oldest (positions 0..n_evicted-1)
      - They only attend to each other (causal mask) — no need for recent K/V
      - Temp caches start at offset=0 → RoPE positions 0..n_evicted-1 (correct)
    """
    h_e = h0_store.get_evicted(n_evicted)

    num_layers = len(inner_model.layers)
    temp_caches = [KVCache() for _ in range(num_layers)]

    mask_e = create_attention_mask(h_e, temp_caches[0])

    h = h_e
    for layer, tc in zip(inner_model.layers, temp_caches):
        h = layer(h, mask_e, tc)

    kv_direct_set = set(kv_direct_indices)
    for i in range(num_layers):
        if i in kv_direct_set:
            tc = temp_caches[i]
            k, v = tc.state
            caches[i]._recon_keys = k
            caches[i]._recon_values = v
