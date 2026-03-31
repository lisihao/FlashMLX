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
    """

    def __init__(self):
        self._h0 = None   # (B, T_total, d_hidden) or None
        self._count = 0
        # Persistent reconstruction caches (incremental optimization)
        self._recon_caches = None    # List[KVCache], one per layer
        self._recon_count = 0        # Evicted tokens already in recon_caches

    def append(self, h0):
        """Append new tokens' h^(0). h0 shape: (B, L, d_hidden)."""
        if self._h0 is None:
            self._h0 = h0
        else:
            self._h0 = mx.concatenate([self._h0, h0], axis=1)
        self._count = self._h0.shape[1]

    def get_evicted(self, n_evicted):
        """Return h^(0) for the first n_evicted tokens (oldest = evicted).

        Returns: (B, n_evicted, d_hidden)
        """
        return self._h0[:, :n_evicted, :]

    def get_newly_evicted(self, n_evicted):
        """Return h^(0) for tokens [_recon_count : n_evicted].

        These are tokens evicted since the last reconstruction pass.
        Returns: (B, delta, d_hidden)
        """
        return self._h0[:, self._recon_count:n_evicted, :]

    def init_recon_caches(self, num_layers):
        """Initialize persistent reconstruction KVCaches (called once)."""
        self._recon_caches = [KVCache() for _ in range(num_layers)]
        self._recon_count = 0

    @property
    def count(self):
        return self._count

    @property
    def nbytes(self):
        base = 0
        if self._h0 is not None:
            base = self._h0.nbytes
        if self._recon_caches is not None:
            base += sum(rc.nbytes for rc in self._recon_caches)
        return base


# ---------------------------------------------------------------------------
# KVDirectCache: Per-layer cache with reconstruction injection
# ---------------------------------------------------------------------------

class KVDirectCache(_BaseCache):
    """Per-layer KV-Direct cache (v2: model-level h^(0) checkpointing).

    Two-region design:
      Recent Region: last ``budget`` tokens -> full K/V stored
      Evicted Region: older tokens -> K/V reconstructed from shared h^(0)
                      via forward pass, injected as _recon_keys/_recon_values
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
        """Per-layer KV bytes only. H0Store/recon_caches are shared — report separately."""
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


def _run_reconstruction(inner_model, caches, h0_store, n_evicted, kv_direct_indices):
    """Run reconstruction forward pass to produce evicted K/V.

    INCREMENTAL OPTIMIZATION:
      - First call: batch-process all n_evicted tokens (O(n_evicted² × L)).
      - Subsequent calls: only process newly evicted tokens (O(n_evicted × L)).
        Previous tokens' K/V are already cached in persistent recon_caches.

    Correctness:
      - Evicted tokens are always the oldest (positions 0..n_evicted-1)
      - They only attend to each other (causal mask) — no need for recent K/V
      - KVCache.offset auto-increments → RoPE positions stay correct
      - Incremental append ≡ batch processing (causal, deterministic)
    """
    num_layers = len(inner_model.layers)

    # Initialize persistent recon_caches on first call
    if h0_store._recon_caches is None:
        h0_store.init_recon_caches(num_layers)

    recon_caches = h0_store._recon_caches
    prev_count = h0_store._recon_count

    if prev_count == 0:
        # Batch mode: first reconstruction, process all evicted tokens
        h_e = h0_store.get_evicted(n_evicted)
        mask_e = create_attention_mask(h_e, recon_caches[0])
        h = h_e
        for layer, rc in zip(inner_model.layers, recon_caches):
            h = layer(h, mask_e, rc)
    else:
        # Incremental mode: only process newly evicted token(s)
        h_new = h0_store.get_newly_evicted(n_evicted)
        mask_new = create_attention_mask(h_new, recon_caches[0])
        h = h_new
        for layer, rc in zip(inner_model.layers, recon_caches):
            h = layer(h, mask_new, rc)

    h0_store._recon_count = n_evicted

    # Extract K/V from persistent recon_caches and inject into real caches
    kv_direct_set = set(kv_direct_indices)
    for i in range(num_layers):
        if i in kv_direct_set:
            k, v = recon_caches[i].state
            caches[i]._recon_keys = k
            caches[i]._recon_values = v
