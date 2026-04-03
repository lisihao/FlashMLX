"""
KV-Direct v2: Model-Level h^(0) Checkpointing for KV Cache

Implements the KV-Direct paper's (arxiv 2603.19664) core compression:
store ONE h^(0) per token (embed_tokens output), reconstruct K/V for
evicted tokens via forward pass through transformer layers.

Memory savings (Qwen3-8B): 18x for evicted tokens.
  Standard KV per token: 36 layers * 2 * 8 * 128 * 2B = 147,456 bytes
  h^(0) per token:       4096 * 2B = 8,192 bytes

Reconstruction mode:
  - Prefix Exact: reconstruct_prefix_kv(start=0, end=N) — continuous prefix
    replay from position 0. Produces bit-identical K/V when h^(0) is bf16.
    This is the only mode the paper validates.
  - Sparse reconstruction is NOT supported (incorrect RoPE + causal mask).

h^(0) quantization:
  - bf16: exact storage (paper-faithful)
  - q8/q4: compressed archive, NOT exact — introduces quantization error.
    Useful for memory savings but outputs may diverge from standard.

Paper validation scope: 135M–4B models. 8B+ is empirically validated
by FlashMLX benchmarks but outside paper's verification range.

Technique: __class__ swapping on inner model (zero model code changes).
"""

import json
import logging
import os

import mlx.core as mx
import numpy as np

from .base import create_attention_mask
from .cache import _BaseCache, KVCache
from .cache import create_attention_mask as cache_create_attention_mask

logger = logging.getLogger(__name__)

# Sentinel to detect double-patching
_KV_DIRECT_PATCHED = "_kv_direct_patched"


# ---------------------------------------------------------------------------
# Route 0: Reconstruction Budget (controls Route 0↔5 coupling)
# ---------------------------------------------------------------------------

class ReconstructionBudget:
    """Controls how aggressively Route 0 can rely on Route 5 reconstruction.

    Prevents over-dependence on the reconstruction path, which would cause:
    - Tail latency spikes (user feels "usually fast, sudden stutter on details")
    - Excessive GPU utilization from frequent h^(0) replays

    Usage:
        budget = ReconstructionBudget()
        if budget.can_reconstruct(estimated_tokens=1024):
            # Route 0 allows aggressive compression
            ...
        budget.record_reconstruction(actual_tokens=1024, latency_ms=350)
    """

    def __init__(
        self,
        max_recall_per_turn: int = 1,
        max_recon_tokens: int = 2048,
        max_recon_latency_ms: float = 500.0,
        cooldown_turns: int = 2,
    ):
        self.max_recall_per_turn = max_recall_per_turn
        self.max_recon_tokens = max_recon_tokens
        self.max_recon_latency_ms = max_recon_latency_ms
        self.cooldown_turns = cooldown_turns

        # Runtime state
        self._recalls_this_turn = 0
        self._turns_since_last_recall = cooldown_turns  # Start with budget available
        self._total_reconstructions = 0
        self._total_recon_tokens = 0
        self._total_recon_ms = 0.0

    def can_reconstruct(self, estimated_tokens: int = 0) -> bool:
        """Check if reconstruction budget allows another recall.

        Args:
            estimated_tokens: Expected number of tokens to reconstruct.

        Returns:
            True if reconstruction is allowed within budget.
        """
        if self._recalls_this_turn >= self.max_recall_per_turn:
            return False
        if estimated_tokens > self.max_recon_tokens:
            return False
        if self._turns_since_last_recall < self.cooldown_turns:
            return False
        return True

    def record_reconstruction(self, actual_tokens: int, latency_ms: float):
        """Record that a reconstruction happened this turn."""
        self._recalls_this_turn += 1
        self._turns_since_last_recall = 0
        self._total_reconstructions += 1
        self._total_recon_tokens += actual_tokens
        self._total_recon_ms += latency_ms

    def advance_turn(self):
        """Call at the start of each generation turn to reset per-turn state."""
        self._recalls_this_turn = 0
        self._turns_since_last_recall += 1

    @property
    def max_allowed_scale(self) -> float:
        """Maximum Route 0 density_scale when reconstruction is available."""
        if self.can_reconstruct():
            return 3.0  # Allow aggressive compression
        return 1.0  # Conservative — no budget for rescue

    @property
    def stats(self) -> dict:
        return {
            "total_reconstructions": self._total_reconstructions,
            "total_recon_tokens": self._total_recon_tokens,
            "total_recon_ms": self._total_recon_ms,
            "recalls_this_turn": self._recalls_this_turn,
            "turns_since_last_recall": self._turns_since_last_recall,
            "can_reconstruct": self.can_reconstruct(),
        }


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

    def __init__(self, quant=None, recon_budget=None):
        self._quant = quant
        self._h0 = None       # bf16: (B, T, d) | q8: int8 (B, T, d) | q4: uint8 (B, T, d//2)
        self._scales = None   # q8/q4: float16 (B, T, 1)
        self._count = 0
        self.recon_budget = recon_budget or ReconstructionBudget()

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

    # --- Block-aligned serialization (ThunderOMLX SSD integration) ---

    def export_blocks(self, block_size: int = 64) -> list:
        """Export H0Store as block-aligned chunks for paged SSD storage.

        Each block contains raw h^(0) data (preserving quantization) for
        exactly `block_size` tokens, matching ThunderOMLX's paged cache
        block granularity.  The last block may be shorter.

        Args:
            block_size: Tokens per block (match ThunderOMLX block_size).

        Returns:
            List of dicts, each representing one block::

                {
                    'h0': mx.array,          # (B, <=block_size, dim)
                    'scales': mx.array|None,  # (B, <=block_size, 1) if quantized
                    'block_idx': int,
                    'token_start': int,
                    'token_end': int,
                    'quant': str,             # 'bf16' | 'q8' | 'q4'
                }
        """
        if self._h0 is None or self._count == 0:
            return []

        blocks = []
        n_blocks = (self._count + block_size - 1) // block_size

        for i in range(n_blocks):
            start = i * block_size
            end = min(start + block_size, self._count)

            block = {
                'h0': self._h0[:, start:end, :],
                'block_idx': i,
                'token_start': start,
                'token_end': end,
                'quant': self._quant or 'bf16',
            }
            if self._scales is not None:
                block['scales'] = self._scales[:, start:end, :]
            else:
                block['scales'] = None

            blocks.append(block)

        return blocks

    def import_blocks(self, blocks: list) -> int:
        """Import H0Store from block-aligned chunks.

        Restores internal state from a list of blocks produced by
        ``export_blocks()``.  Blocks are sorted by ``block_idx`` before
        concatenation so order in the input list does not matter.

        Args:
            blocks: List of block dicts from export_blocks().

        Returns:
            Total token count restored.
        """
        if not blocks:
            return 0

        sorted_blocks = sorted(blocks, key=lambda b: b['block_idx'])

        quant = sorted_blocks[0].get('quant', 'bf16')
        self._quant = None if quant == 'bf16' else quant

        h0_parts = []
        scales_parts = []
        has_scales = sorted_blocks[0].get('scales') is not None

        for block in sorted_blocks:
            h0_parts.append(block['h0'])
            if has_scales and block.get('scales') is not None:
                scales_parts.append(block['scales'])

        self._h0 = mx.concatenate(h0_parts, axis=1)
        if has_scales and scales_parts:
            self._scales = mx.concatenate(scales_parts, axis=1)
        else:
            self._scales = None

        self._count = self._h0.shape[1]
        return self._count

    @staticmethod
    def block_hash_key(parent_hash: bytes, block_idx: int) -> bytes:
        """Compute SHA-256 hash key for an H0 block on SSD.

        Uses a ``h0:`` prefix to ensure H0 block hashes never collide
        with KV block hashes in the same SSD directory.

        Args:
            parent_hash: Hash of the parent KV block (or sequence prefix).
            block_idx: Block index within this sequence's H0 data.

        Returns:
            32-byte SHA-256 digest.
        """
        import hashlib
        h = hashlib.sha256()
        h.update(b"h0:")
        h.update(parent_hash)
        h.update(block_idx.to_bytes(8, 'little'))
        return h.digest()

    # --- Disk persistence ---

    def save(self, path, metadata=None):
        """Save h^(0) archive to disk as NPZ with self-describing metadata.

        Args:
            path: File path (e.g. '/Volumes/toshiba/flashmlx/h0_cache/run1.npz').
            metadata: Optional dict with model_name, session_id, etc.
        """
        from datetime import datetime

        os.makedirs(os.path.dirname(path), exist_ok=True)

        if self._h0 is not None:
            mx.eval(self._h0)
        if self._scales is not None:
            mx.eval(self._scales)

        meta = {
            "quant": self._quant or "bf16",
            "count": self._count,
            "format_version": 1,
            "created_at": datetime.now().isoformat(),
            **(metadata or {}),
        }

        save_dict = {"meta": np.array(json.dumps(meta))}
        if self._h0 is not None:
            save_dict["h0"] = np.array(self._h0)
        if self._scales is not None:
            save_dict["scales"] = np.array(self._scales)

        np.savez(path, **save_dict)
        size_mb = os.path.getsize(path + ".npz" if not path.endswith(".npz") else path) / (1024 * 1024)
        logger.info(f"H0Store saved: {path} ({size_mb:.1f} MB, {self._count} tokens, quant={meta['quant']})")

    @classmethod
    def load(cls, path):
        """Load h^(0) archive from disk.

        Returns:
            (H0Store, metadata_dict) tuple.
        """
        if not path.endswith(".npz"):
            path = path + ".npz"
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data["meta"]))

        quant = meta.get("quant", "bf16")
        store = cls(quant=quant if quant != "bf16" else None)

        dtype_map = {"bf16": mx.bfloat16, "q8": mx.int8, "q4": mx.uint8}

        if "h0" in data:
            target_dtype = dtype_map.get(quant, mx.bfloat16)
            store._h0 = mx.array(data["h0"]).astype(target_dtype)
            store._count = store._h0.shape[1]
        if "scales" in data:
            store._scales = mx.array(data["scales"]).astype(mx.float16)

        logger.info(f"H0Store loaded: {path} ({store._count} tokens, quant={quant})")
        return store, meta


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

        # Reconstructed K/V injection
        self._recon_keys = None
        self._recon_values = None
        self._recon_persistent = True  # persist across TG steps

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
            if not self._recon_persistent:
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
            if not self._recon_persistent:
                self._recon_keys = None
                self._recon_values = None
            return all_k, all_v

        # 6. Fallback: no reconstruction (shouldn't happen in steady state)
        return self._recent_keys, self._recent_values

    def clear_reconstruction(self):
        """Explicitly clear persistent reconstruction data."""
        self._recon_keys = None
        self._recon_values = None

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

    Raises:
        RuntimeError: If already patched (double-patch prevention).
    """
    inner_model = _find_inner_model(model)

    # Guard: prevent double-patching
    if getattr(inner_model, _KV_DIRECT_PATCHED, False):
        raise RuntimeError(
            "apply_h0_capture: model already patched. "
            "Call unpatch_model() first or reuse existing patch."
        )

    base_cls = type(inner_model)
    parent_call = base_cls.__call__

    kv_direct_indices = [
        i for i, c in enumerate(caches) if isinstance(c, KVDirectCache)
    ]

    def _make_patched_call(h0s, kv_indices, orig_call):
        def __call__(self, inputs, cache=None, input_embeddings=None):
            # Guard: batch_size > 1 is unsupported for h^(0) CAPTURE.
            # Exception: _batched_rc_mode bypasses this for reconstruction-only
            # passes where h^(0) is sourced via BatchedH0View (B>1 safe).
            B = inputs.shape[0] if input_embeddings is None else input_embeddings.shape[0]
            if B > 1 and not getattr(self, "_batched_rc_mode", False):
                raise RuntimeError(
                    f"KV-Direct h^(0) capture requires batch_size=1, got {B}. "
                    f"h^(0) storage is sequence-global — multi-batch would corrupt "
                    f"token ordering. Use standard KV cache for batched inference, "
                    f"or set model.model._batched_rc_mode=True for RC batch passes."
                )

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
    inner_model._kv_direct_patched = True
    inner_model._kv_direct_base_cls = base_cls


# ---------------------------------------------------------------------------
# Route 5: h^(0) capture only (for scored_pq fusion)
# ---------------------------------------------------------------------------

def _find_inner_model(model):
    """Find the innermost model with embed_tokens.

    Handles different model hierarchies:
      - Qwen3:   Model.model (Qwen3Model)
      - Qwen3.5: Model.language_model.model (Qwen3_5TextModel)
    """
    # Qwen3 pattern: model.model.embed_tokens
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        return model.model
    # Qwen3.5 / VLM pattern: model.language_model.model.embed_tokens
    if hasattr(model, 'language_model'):
        lm = model.language_model
        if hasattr(lm, 'model') and hasattr(lm.model, 'embed_tokens'):
            return lm.model
    raise ValueError(
        f"Cannot find inner model with embed_tokens in {type(model).__name__}. "
        f"Available attributes: {[k for k in dir(model) if not k.startswith('_')]}"
    )


def apply_h0_capture_only(model, h0_store):
    """Install h^(0) capture WITHOUT reconstruction — for scored_pq fusion.

    EXPERIMENTAL: This patch captures embed_tokens output into h0_store
    on every forward call, then delegates to the original __call__ with
    input_embeddings to avoid recomputation.

    Unlike apply_h0_capture, this does NOT run reconstruction — the
    scored_pq cache handles attention normally via its flat buffer.
    h^(0) is stored for potential future prefix reconstruction.

    Raises:
        RuntimeError: If already patched (double-patch prevention).
    """
    inner_model = _find_inner_model(model)

    # Guard: prevent double-patching
    if getattr(inner_model, _KV_DIRECT_PATCHED, False):
        raise RuntimeError(
            "apply_h0_capture_only: model already patched. "
            "Call unpatch_model() first or reuse existing patch."
        )

    base_cls = type(inner_model)
    parent_call = base_cls.__call__

    def _make_capture_call(h0s, orig_call):
        def __call__(self, inputs, cache=None, input_embeddings=None):
            # Guard: batch_size > 1 is unsupported for h^(0) CAPTURE.
            # Exception: _batched_rc_mode bypasses this for reconstruction-only
            # passes where h^(0) is sourced via BatchedH0View (B>1 safe).
            B = inputs.shape[0] if input_embeddings is None else input_embeddings.shape[0]
            if B > 1 and not getattr(self, "_batched_rc_mode", False):
                raise RuntimeError(
                    f"KV-Direct h^(0) capture requires batch_size=1, got {B}. "
                    f"h^(0) storage is sequence-global — multi-batch would corrupt "
                    f"token ordering. Use standard KV cache for batched inference, "
                    f"or set model.model._batched_rc_mode=True for RC batch passes."
                )

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
    inner_model._kv_direct_patched = True
    inner_model._kv_direct_base_cls = base_cls


# ---------------------------------------------------------------------------
# Route 5: On-demand reconstruction from h^(0)
# ---------------------------------------------------------------------------

def reconstruct_prefix_kv(inner_model, h0_store, start, end,
                           chunk_size=0, eval_every=4):
    """Reconstruct K/V for a continuous prefix [0:end] from h^(0).

    Runs a forward pass through all layers using h^(0) as input,
    producing the exact K/V that would have been computed originally.

    IMPORTANT: Only continuous prefix reconstruction is supported.
    h^(0) replays tokens causally from position 0, so arbitrary sparse
    ranges (e.g. [50:100] without [0:50]) produce incorrect RoPE positions
    and causal masks. See paper (arxiv 2603.19664) Section 3.2.

    Args:
        inner_model: The inner model (e.g., model.model) with .layers.
        h0_store: H0Store containing archived embeddings.
        start: Must be 0 (prefix reconstruction only).
        end: End token index (exclusive).
        chunk_size: If > 0, process h^(0) in chunks of this size.
            Default: 0 (no chunking, process all at once).
        eval_every: GPU sync every N chunks instead of every chunk.
            Reduces sync overhead by batching computation. Default: 4.

    Returns:
        List of (keys, values) tuples per layer, each (B, n_kv_heads, N, head_dim).

    Raises:
        NotImplementedError: If start != 0 (sparse reconstruction not supported).
    """
    import time as _time

    if start != 0:
        raise NotImplementedError(
            "Only prefix reconstruction (start=0) is supported. "
            "Sparse hole-filling requires offset-aware KVCache (not implemented)."
        )

    n_tokens = end - start
    num_layers = len(inner_model.layers)

    # Non-chunked path: process all tokens at once (original behavior)
    if chunk_size <= 0 or n_tokens <= chunk_size:
        h_range = h0_store.get_range(start, end)
        temp_caches = [KVCache() for _ in range(num_layers)]
        mask = create_attention_mask(h_range, temp_caches[0])
        h = h_range
        for layer, tc in zip(inner_model.layers, temp_caches):
            h = layer(h, mask, tc)
        return [tc.state for tc in temp_caches]

    # Dequant-once: fetch full h^(0) range upfront, then slice per chunk.
    # Avoids per-chunk dequantization overhead (128→1 dequant for q8/q4).
    t0 = _time.perf_counter()
    h0_full = h0_store.get_range(start, end)
    mx.eval(h0_full)
    t_dequant = (_time.perf_counter() - t0) * 1000

    # Chunked path with batched eval.
    temp_caches = [KVCache() for _ in range(num_layers)]
    n_chunks = 0
    n_evals = 0

    for chunk_start in range(start, end, chunk_size):
        chunk_end = min(chunk_start + chunk_size, end)
        h_chunk = h0_full[:, chunk_start:chunk_end, :]  # slice, no dequant
        mask = create_attention_mask(h_chunk, temp_caches[0])

        h = h_chunk
        for layer, tc in zip(inner_model.layers, temp_caches):
            h = layer(h, mask, tc)

        n_chunks += 1
        # Batch eval: sync every eval_every chunks to reduce GPU overhead.
        if n_chunks % eval_every == 0:
            mx.eval(h)
            n_evals += 1

    # Final eval for any remaining un-evaluated chunks.
    if n_chunks % eval_every != 0:
        mx.eval(h)
        n_evals += 1

    t_total = (_time.perf_counter() - t0) * 1000
    logger.info(f"reconstruct_prefix_kv: {n_tokens} tokens, {n_chunks} chunks, "
                f"{n_evals} evals, dequant={t_dequant:.0f}ms, total={t_total:.0f}ms")

    return [tc.state for tc in temp_caches]


# Default chunk size for reconstruction (512 tokens).
# Benchmarked: chunk=512/eval_every=8 is ~5% faster than chunk=256/eval_every=4.
# Larger chunks = fewer mx.eval() syncs. Set to 0 for no chunking (fastest, OOM risk).
RECON_CHUNK_SIZE = 512


# ---------------------------------------------------------------------------
# 3PIR: Stateful single-chunk reconstruction primitive
# ---------------------------------------------------------------------------

def reconstruct_prefix_kv_stateful(inner_model, h0_chunk, temp_caches):
    """Process one chunk of h^(0) through all layers with persistent temp_caches.

    This is the core primitive for 3PIR (Three-Phase Interleaved Reconstruction).
    Unlike reconstruct_prefix_kv() which creates fresh temp_caches and processes
    everything in one call, this function:

    1. Accepts persistent temp_caches that accumulate K/V across multiple calls
    2. Processes exactly one chunk per call (~512 tokens, ~1.3ms on M4 Max)
    3. Returns immediately — caller controls scheduling and GPU budget

    Causal correctness: temp_caches carry forward all K/V from previous chunks,
    so chunk N's tokens correctly attend to tokens 0..N*chunk_size-1 via the
    attention mask computed from temp_caches[0].offset.

    Bit-exact guarantee: when called sequentially on consecutive chunks
    [0:512], [512:1024], ..., the final temp_caches contain identical K/V
    to a single reconstruct_prefix_kv(0, total_tokens) call, provided
    h^(0) is bf16 (no quantization error).

    Args:
        inner_model: Inner model with .layers (e.g., model.model).
        h0_chunk: (B, chunk_len, d_hidden) — one chunk of h^(0) embeddings.
            B can be >1 for cross-sequence batched reconstruction.
        temp_caches: List[KVCache] — one per layer, persistent across calls.
            On first call, pass freshly created [KVCache() for _ in layers].
            On subsequent calls, pass the same list (mutated in place by layer()).

    Returns:
        int — number of tokens processed in this chunk (= h0_chunk.shape[1]).
             temp_caches are mutated in place with new K/V appended.

    Example (3PIR chunk loop):
        temp_caches = [KVCache() for _ in range(num_layers)]
        h0_full = h0_store.get_range(0, 8192)
        mx.eval(h0_full)

        for offset in range(0, 8192, 512):
            chunk = h0_full[:, offset:offset+512, :]
            reconstruct_prefix_kv_stateful(inner_model, chunk, temp_caches)
            mx.eval(temp_caches[-1].keys)  # sync this chunk
            # ... yield to TG scheduling ...

        kv_list = [tc.state for tc in temp_caches]  # final result
    """
    mask = create_attention_mask(h0_chunk, temp_caches[0])
    h = h0_chunk
    for layer, tc in zip(inner_model.layers, temp_caches):
        h = layer(h, mask, tc)
    return h0_chunk.shape[1]


def extract_kv_from_temp_caches(temp_caches):
    """Extract (keys, values) tuples from temp_caches after reconstruction.

    Convenience function to convert completed temp_caches into the same
    format as reconstruct_prefix_kv() returns.

    Args:
        temp_caches: List[KVCache] after all chunks have been processed.

    Returns:
        List of (keys, values) tuples, one per layer.
        Each tuple is (B, n_kv_heads, N_total, head_dim).
    """
    return [tc.state for tc in temp_caches]


# ---------------------------------------------------------------------------
# 3PIR: BatchedH0View — cross-sequence B>1 virtual view
# ---------------------------------------------------------------------------

class BatchedH0View:
    """Create a B>1 virtual view from multiple H0Stores for batched RC.

    Each H0Store maintains B=1 independently. BatchedH0View pads and stacks
    them into a temporary (B, T_max, d) tensor for a single batched forward
    pass through the model layers.

    This enables the KV-Direct paper's key insight: at medium batch sizes,
    recomputing from h^(0) is 5x faster than reading full KV from memory,
    because recomputation is compute-bound while KV read is bandwidth-bound.

    Usage:
        view = BatchedH0View(
            stores=[agent_a.h0_store, agent_b.h0_store],
            ranges=[(0, 512), (0, 512)]
        )
        batched_h0, lengths = view.get_batched_h0()
        # batched_h0: (2, 512, d), lengths: [512, 512]

        # After forward pass through layers with temp_caches:
        per_seq_kv = view.split_kv_results(kv_list, lengths)
    """

    def __init__(self, stores, ranges):
        """
        Args:
            stores: List[H0Store] — one per sequence in the batch.
            ranges: List[Tuple[int, int]] — (start, end) per sequence.
        """
        self.stores = stores
        self.ranges = ranges

    def get_batched_h0(self):
        """Return padded+stacked h^(0) for batch forward pass.

        Returns:
            (batched_h0, lengths):
            - batched_h0: (B, T_max, d) mx.array
            - lengths: List[int] — actual token count per sequence
        """
        chunks = []
        lengths = []
        max_len = 0

        for store, (start, end) in zip(self.stores, self.ranges):
            h0 = store.get_range(start, end)   # (1, L_i, d)
            chunks.append(h0)
            length = end - start
            lengths.append(length)
            max_len = max(max_len, length)

        # Pad to max_len and stack
        padded = []
        for h0, length in zip(chunks, lengths):
            if length < max_len:
                pad = mx.zeros(
                    (1, max_len - length, h0.shape[2]), dtype=h0.dtype
                )
                h0 = mx.concatenate([h0, pad], axis=1)
            padded.append(h0)

        batched = mx.concatenate(padded, axis=0)   # (B, T_max, d)
        return batched, lengths

    @staticmethod
    def split_kv_results(kv_list, actual_lengths):
        """Split batched K/V results back to per-sequence.

        After a batched forward pass produces (B, H, T_max, D) K/V,
        split into per-sequence results trimmed to actual lengths.

        Args:
            kv_list: List of (keys, values) per layer.
                Each keys/values is (B, n_kv_heads, T_max, head_dim).
            actual_lengths: List[int] — actual length per sequence.

        Returns:
            List[List[Tuple]] — results[seq_idx] = [(k, v), ...] per layer.
        """
        results = []
        for i, length in enumerate(actual_lengths):
            per_seq = []
            for k, v in kv_list:
                per_seq.append((
                    k[i:i+1, :, :length, :],
                    v[i:i+1, :, :length, :],
                ))
            results.append(per_seq)
        return results


def reconstruct_targeted(inner_model, h0_store, max_end,
                         importance_scores=None,
                         min_coverage=0.95, chunk_size=512, eval_every=8):
    """Importance-guided depth-reduced reconstruction.

    Uses probe attention scores to find the minimal prefix [0:actual_end]
    that covers min_coverage of total importance. If all important tokens
    are in [0:M] where M < max_end, only reconstructs [0:M].

    Args:
        inner_model: Inner model with .layers.
        h0_store: H0Store with archived embeddings.
        max_end: Maximum reconstruction depth.
        importance_scores: np.ndarray (max_end,) from H0Probe.score_tokens().
        min_coverage: Fraction of total importance to cover (default 0.95).
        chunk_size: Chunk size for reconstruction (default 512 for speed).
        eval_every: GPU sync every N chunks (default 4).

    Returns:
        (kv_list, actual_end) — kv_list as from reconstruct_prefix_kv,
        actual_end = tokens actually reconstructed.
    """
    actual_end = max_end

    if importance_scores is not None and len(importance_scores) > 0:
        scores = importance_scores[:max_end]
        cumsum = np.cumsum(scores)
        total = cumsum[-1]
        if total > 0:
            threshold = total * min_coverage
            cut = int(np.searchsorted(cumsum, threshold)) + 1
            actual_end = min(max_end, cut + 256)   # safety margin
            actual_end = max(actual_end, max_end // 2)  # minimum 50%
            logger.info(f"reconstruct_targeted: {max_end} → {actual_end} "
                        f"({actual_end / max_end * 100:.0f}% depth, "
                        f"coverage={min_coverage:.0%})")

    kv_list = reconstruct_prefix_kv(
        inner_model, h0_store, 0, actual_end,
        chunk_size=chunk_size, eval_every=eval_every
    )
    return kv_list, actual_end


def _run_reconstruction(inner_model, caches, h0_store, n_evicted, kv_direct_indices,
                        chunk_size=RECON_CHUNK_SIZE, importance_scores=None,
                        eval_every=8):
    """Run reconstruction forward pass to produce evicted K/V.

    Creates FRESH temp KVCaches each call, feeds h^(0)[0:n_evicted] through
    all layers, extracts K/V and injects into real caches. Temp caches are
    freed after use — only h^(0) persists (paper-faithful, memory-efficient).

    Args:
        chunk_size: If > 0, process in chunks to yield GPU for concurrent agents.
        importance_scores: If provided, uses reconstruct_targeted for depth
            reduction (faster reconstruction of important regions only).
        eval_every: GPU sync every N chunks (default 4).
    """
    if importance_scores is not None:
        kv_list, actual_end = reconstruct_targeted(
            inner_model, h0_store, n_evicted,
            importance_scores=importance_scores,
            chunk_size=max(chunk_size, 512),
            eval_every=eval_every,
        )
    else:
        kv_list = reconstruct_prefix_kv(
            inner_model, h0_store, 0, n_evicted,
            chunk_size=chunk_size, eval_every=eval_every
        )

    kv_direct_set = set(kv_direct_indices)
    num_layers = len(inner_model.layers)
    for i in range(num_layers):
        if i in kv_direct_set and i < len(kv_list):
            k, v = kv_list[i]
            caches[i]._recon_keys = k
            caches[i]._recon_values = v


# ---------------------------------------------------------------------------
# Unpatch: restore original model class
# ---------------------------------------------------------------------------

def unpatch_model(model):
    """Remove KV-Direct monkey-patch, restoring original __call__.

    Safe to call on unpatched models (no-op).

    Args:
        model: Outer Model previously patched by apply_h0_capture[_only].

    Returns:
        True if unpatch occurred, False if model was not patched.
    """
    try:
        inner_model = _find_inner_model(model)
    except ValueError:
        return False

    if not getattr(inner_model, _KV_DIRECT_PATCHED, False):
        return False

    base_cls = getattr(inner_model, "_kv_direct_base_cls", None)
    if base_cls is None:
        logger.warning("unpatch_model: _kv_direct_base_cls missing, cannot restore")
        return False

    inner_model.__class__ = base_cls
    inner_model._kv_direct_patched = False
    del inner_model._kv_direct_base_cls
    return True
