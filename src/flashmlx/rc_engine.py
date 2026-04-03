"""
RCEngine: Chunk-level non-blocking reconstruction engine for 3PIR.

The core innovation of Three-Phase Interleaved Reconstruction (3PIR):
instead of blocking ~20s for full reconstruction, process h^(0) in 512-token
chunks (~1.3ms each), yielding to TG scheduling between chunks.

Architecture:
    ThunderOMLX Scheduler
        ↓ try_rc_step()
    RCScheduler (budget + queue)
        ↓ process_chunk() / process_batched_chunk()
    RCEngine (this module)
        ↓ reconstruct_prefix_kv_stateful()
    kv_direct_cache.py (single-chunk primitive)

Key design decisions:
    1. temp_caches are PERSISTENT across chunks — causal correctness
    2. h^(0) is dequantized ONCE at registration, not per-chunk
    3. B>1 batched RC via BatchedH0View when multiple sequences align
    4. Injection is atomic: all layers' K/V injected + mx.eval() in one call

Usage:
    engine = RCEngine(chunk_size=512)

    # Register a sequence needing reconstruction
    state = engine.register_sequence(
        seq_id="agent_a",
        h0_store=h0_store,
        inner_model=model.model,
        target_cache_list=cache_list,
    )

    # Process chunks one at a time (called by scheduler)
    while not state.is_complete:
        result = engine.process_chunk(state)
        mx.eval(state.temp_caches[-1].keys)  # GPU sync
        # ... yield to TG ...

    # Inject completed reconstruction into target caches
    engine.inject_completed(state)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RCSequenceState:
    """Per-sequence reconstruction progress, persistent across chunks.

    The temp_caches list is the critical piece: it accumulates K/V
    from all processed chunks, ensuring causal correctness. Each layer
    gets one KVCache that grows with each process_chunk() call.
    """

    sequence_id: str                    # Agent/Request identifier
    h0_store: Any                       # H0Store reference
    inner_model: Any                    # model.model with .layers
    target_cache_list: List[Any]        # Target caches for injection

    # Reconstruction range
    total_tokens: int                   # Total tokens to reconstruct
    reconstructed_tokens: int = 0       # Tokens completed so far

    # Pre-dequantized h^(0) — fetched once at registration
    h0_full: Optional[mx.array] = None  # (1, total_tokens, d_hidden)

    # Persistent temp KVCaches (grow across chunks)
    temp_caches: List[Any] = field(default_factory=list)

    # Optional: importance-guided cutoff
    importance_scores: Optional[Any] = None
    effective_end: Optional[int] = None  # After importance cutoff

    # Status lifecycle: pending → active → completed → injected
    status: str = "pending"
    created_at: float = 0.0

    # Chunk processing stats
    chunks_processed: int = 0
    total_time_ms: float = 0.0

    @property
    def is_complete(self) -> bool:
        """Whether all chunks have been processed."""
        target = self.effective_end if self.effective_end is not None else self.total_tokens
        return self.reconstructed_tokens >= target

    @property
    def progress(self) -> float:
        """Reconstruction progress as fraction [0, 1]."""
        target = self.effective_end if self.effective_end is not None else self.total_tokens
        if target <= 0:
            return 1.0
        return min(1.0, self.reconstructed_tokens / target)

    @property
    def remaining_tokens(self) -> int:
        target = self.effective_end if self.effective_end is not None else self.total_tokens
        return max(0, target - self.reconstructed_tokens)

    @property
    def remaining_chunks(self) -> int:
        """Estimated remaining chunks (assuming default chunk_size=512)."""
        return (self.remaining_tokens + 511) // 512


@dataclass(frozen=True)
class RCChunkResult:
    """Result from processing a single RC chunk."""

    sequence_id: str
    chunk_start: int                    # Start token of this chunk
    chunk_end: int                      # End token (exclusive)
    tokens_processed: int               # = chunk_end - chunk_start
    time_ms: float                      # Wall time for this chunk
    cumulative_tokens: int              # Total tokens processed so far
    is_final: bool                      # Whether this was the last chunk


# ---------------------------------------------------------------------------
# RCEngine
# ---------------------------------------------------------------------------

class RCEngine:
    """Chunk-level non-blocking reconstruction engine.

    Provides the FlashMLX SDK layer for 3PIR. Does NOT manage scheduling
    or budgets — that's RCScheduler's job. This engine focuses on:

    1. Sequence registration (h^(0) dequant, temp_cache creation)
    2. Single-chunk processing (delegating to reconstruct_prefix_kv_stateful)
    3. Cross-sequence batched processing (BatchedH0View + B>1 forward)
    4. K/V injection into target caches
    """

    def __init__(self, chunk_size: int = 512):
        self.chunk_size = chunk_size
        self._active: Dict[str, RCSequenceState] = {}

    def register_sequence(
        self,
        seq_id: str,
        h0_store: Any,
        inner_model: Any,
        target_cache_list: List[Any],
        importance_scores: Optional[Any] = None,
        min_coverage: float = 0.95,
    ) -> RCSequenceState:
        """Register a new sequence for reconstruction.

        This performs one-time setup:
        1. Compute effective_end via importance scores (if available)
        2. Dequantize full h^(0) range once (avoids per-chunk dequant)
        3. Create persistent temp_caches (one per layer)

        Args:
            seq_id: Unique identifier for this reconstruction.
            h0_store: H0Store with archived h^(0) embeddings.
            inner_model: Inner model with .layers.
            target_cache_list: Caches to inject K/V into when complete.
            importance_scores: Optional importance scores for depth reduction.
            min_coverage: Importance coverage threshold (default 0.95).

        Returns:
            RCSequenceState ready for process_chunk() calls.
        """
        import numpy as np
        from mlx_lm.models.cache import KVCache

        total_tokens = h0_store.count
        effective_end = total_tokens

        # Importance-guided depth reduction
        if importance_scores is not None and len(importance_scores) > 0:
            scores = importance_scores[:total_tokens]
            cumsum = np.cumsum(scores)
            total_imp = cumsum[-1]
            if total_imp > 0:
                threshold = total_imp * min_coverage
                cut = int(np.searchsorted(cumsum, threshold)) + 1
                effective_end = min(total_tokens, cut + 256)    # safety margin
                effective_end = max(effective_end, total_tokens // 2)  # min 50%
                logger.info(
                    f"[RCEngine] {seq_id}: importance cutoff {total_tokens} → {effective_end} "
                    f"({effective_end / total_tokens * 100:.0f}%, coverage={min_coverage:.0%})"
                )

        # Dequantize h^(0) once upfront
        h0_full = h0_store.get_range(0, effective_end)
        mx.eval(h0_full)

        # Create persistent temp caches
        num_layers = len(inner_model.layers)
        temp_caches = [KVCache() for _ in range(num_layers)]

        state = RCSequenceState(
            sequence_id=seq_id,
            h0_store=h0_store,
            inner_model=inner_model,
            target_cache_list=target_cache_list,
            total_tokens=total_tokens,
            h0_full=h0_full,
            temp_caches=temp_caches,
            importance_scores=importance_scores,
            effective_end=effective_end,
            status="active",
            created_at=time.perf_counter(),
        )

        self._active[seq_id] = state

        logger.info(
            f"[RCEngine] Registered {seq_id}: {effective_end} tokens, "
            f"{(effective_end + self.chunk_size - 1) // self.chunk_size} chunks, "
            f"{num_layers} layers"
        )

        return state

    def register_from_h0_blocks(
        self,
        seq_id: str,
        h0_blocks: list,
        inner_model: Any,
        target_cache_list: List[Any],
        h0_quant: Optional[str] = None,
        importance_scores: Optional[Any] = None,
        min_coverage: float = 0.95,
    ) -> RCSequenceState:
        """Register a sequence for reconstruction from SSD-loaded H0 blocks.

        This is the Tier 3 (3PIR Cold Cache Restoration) entry point:
        ThunderOMLX loads H0 blocks from SSD on a cold prefix hit,
        then calls this to begin non-blocking KV reconstruction.

        Args:
            seq_id: Unique identifier for this reconstruction.
            h0_blocks: List of block dicts from H0Store.export_blocks().
            inner_model: Inner model with .layers.
            target_cache_list: Caches to inject K/V into when complete.
            h0_quant: Override quantization mode ('q8'|'q4'|None for bf16).
                If None, inferred from the first block's 'quant' field.
            importance_scores: Optional importance scores for depth reduction.
            min_coverage: Importance coverage threshold.

        Returns:
            RCSequenceState ready for process_chunk() calls.
        """
        from mlx_lm.models.kv_direct_cache import H0Store

        if h0_quant is None and h0_blocks:
            raw_quant = h0_blocks[0].get('quant', 'bf16')
            h0_quant = None if raw_quant == 'bf16' else raw_quant

        store = H0Store(quant=h0_quant)
        n_tokens = store.import_blocks(h0_blocks)

        logger.info(
            f"[RCEngine] register_from_h0_blocks {seq_id}: "
            f"{n_tokens} tokens from {len(h0_blocks)} blocks, "
            f"quant={h0_quant or 'bf16'}"
        )

        return self.register_sequence(
            seq_id=seq_id,
            h0_store=store,
            inner_model=inner_model,
            target_cache_list=target_cache_list,
            importance_scores=importance_scores,
            min_coverage=min_coverage,
        )

    def process_chunk(self, state: RCSequenceState) -> RCChunkResult:
        """Process one chunk of reconstruction for a sequence.

        This is the hot path — called once per scheduler step per active sequence.
        Delegates to reconstruct_prefix_kv_stateful() which runs one forward pass
        through all layers for chunk_size tokens.

        Typical latency: ~1.3ms for 512 tokens on M4 Max (Qwen3-8B, 36 layers).

        Args:
            state: RCSequenceState from register_sequence().

        Returns:
            RCChunkResult with timing and progress info.
        """
        from mlx_lm.models.kv_direct_cache import reconstruct_prefix_kv_stateful

        target = state.effective_end if state.effective_end is not None else state.total_tokens
        chunk_start = state.reconstructed_tokens
        chunk_end = min(chunk_start + self.chunk_size, target)

        if chunk_start >= target:
            # Already complete — return no-op result
            return RCChunkResult(
                sequence_id=state.sequence_id,
                chunk_start=chunk_start,
                chunk_end=chunk_start,
                tokens_processed=0,
                time_ms=0.0,
                cumulative_tokens=state.reconstructed_tokens,
                is_final=True,
            )

        # Slice pre-dequantized h^(0)
        h0_chunk = state.h0_full[:, chunk_start:chunk_end, :]

        t0 = time.perf_counter()

        # Core: single-chunk stateful reconstruction
        tokens = reconstruct_prefix_kv_stateful(
            state.inner_model, h0_chunk, state.temp_caches
        )

        # GPU sync for this chunk
        mx.eval(state.temp_caches[-1].keys)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Update state
        state.reconstructed_tokens = chunk_end
        state.chunks_processed += 1
        state.total_time_ms += elapsed_ms

        is_final = chunk_end >= target
        if is_final:
            state.status = "completed"

        return RCChunkResult(
            sequence_id=state.sequence_id,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            tokens_processed=chunk_end - chunk_start,
            time_ms=elapsed_ms,
            cumulative_tokens=state.reconstructed_tokens,
            is_final=is_final,
        )

    def process_batched_chunk(
        self, states: List[RCSequenceState]
    ) -> List[RCChunkResult]:
        """Process one chunk for multiple sequences in a single B>1 forward pass.

        Batched reconstruction exploits KV-Direct's core insight: at batch>1,
        recomputation from h^(0) is compute-bound while KV reading is
        bandwidth-bound. On Apple Silicon UMA, the compute path wins at B>=2.

        Requirements:
            - All states must share the same inner_model
            - All states must have the same chunk offset (for aligned causal masks)
            - inner_model._batched_rc_mode must be True

        Args:
            states: List of RCSequenceState at the same chunk offset.

        Returns:
            List of RCChunkResult, one per sequence.
        """
        from mlx_lm.models.kv_direct_cache import (
            BatchedH0View,
            extract_kv_from_temp_caches,
            reconstruct_prefix_kv_stateful,
        )
        from mlx_lm.models.cache import KVCache

        if not states:
            return []

        # Single sequence — fall back to non-batched path
        if len(states) == 1:
            return [self.process_chunk(states[0])]

        inner_model = states[0].inner_model
        num_layers = len(inner_model.layers)

        # Compute chunk ranges for each sequence
        ranges = []
        for s in states:
            target = s.effective_end if s.effective_end is not None else s.total_tokens
            start = s.reconstructed_tokens
            end = min(start + self.chunk_size, target)
            ranges.append((start, end))

        # Build batched h^(0) from pre-dequantized data
        chunks = []
        lengths = []
        max_len = 0
        for s, (start, end) in zip(states, ranges):
            h0_chunk = s.h0_full[:, start:end, :]  # (1, L_i, d)
            chunks.append(h0_chunk)
            length = end - start
            lengths.append(length)
            max_len = max(max_len, length)

        # Pad and stack → (B, T_max, d)
        padded = []
        for h0, length in zip(chunks, lengths):
            if length < max_len:
                pad = mx.zeros(
                    (1, max_len - length, h0.shape[2]), dtype=h0.dtype
                )
                h0 = mx.concatenate([h0, pad], axis=1)
            padded.append(h0)
        batched_h0 = mx.concatenate(padded, axis=0)  # (B, T_max, d)

        # Create batched temp caches for this forward pass
        # Note: we need SEPARATE batched temp_caches because each sequence
        # has its own persistent temp_caches at potentially different offsets.
        # For aligned offsets (all same), we can batch; otherwise use round-robin.
        offsets = [s.reconstructed_tokens for s in states]
        if len(set(offsets)) > 1:
            # Offsets not aligned — fall back to sequential
            logger.debug(
                f"[RCEngine] Batch abort: offsets not aligned {offsets}, "
                f"falling back to sequential"
            )
            return [self.process_chunk(s) for s in states]

        # All offsets aligned — create shared batched temp_caches
        # We use fresh batched temp_caches and then split results
        batched_temp_caches = [KVCache() for _ in range(num_layers)]

        # If offset > 0, we need to pre-fill batched temp_caches with prior K/V
        # from each sequence's individual temp_caches, stacked to B>1
        offset = offsets[0]
        if offset > 0:
            for layer_idx in range(num_layers):
                # Stack each sequence's temp_cache K/V for this layer
                k_list = []
                v_list = []
                for s in states:
                    tc = s.temp_caches[layer_idx]
                    k, v = tc.state
                    k_list.append(k)
                    v_list.append(v)

                # Pad to same length and stack
                max_kv_len = max(k.shape[2] for k in k_list)
                padded_k = []
                padded_v = []
                for k, v in zip(k_list, v_list):
                    if k.shape[2] < max_kv_len:
                        kpad = mx.zeros(
                            (1, k.shape[1], max_kv_len - k.shape[2], k.shape[3]),
                            dtype=k.dtype,
                        )
                        vpad = mx.zeros(
                            (1, v.shape[1], max_kv_len - v.shape[2], v.shape[3]),
                            dtype=v.dtype,
                        )
                        k = mx.concatenate([k, kpad], axis=2)
                        v = mx.concatenate([v, vpad], axis=2)
                    padded_k.append(k)
                    padded_v.append(v)

                stacked_k = mx.concatenate(padded_k, axis=0)  # (B, H, T, D)
                stacked_v = mx.concatenate(padded_v, axis=0)  # (B, H, T, D)

                # Inject pre-existing K/V into batched temp cache
                # KVCache uses update_and_fetch, so we set internal state directly
                batched_temp_caches[layer_idx].keys = stacked_k
                batched_temp_caches[layer_idx].values = stacked_v
                batched_temp_caches[layer_idx].offset = max_kv_len

        # Enable B>1 mode on inner model
        prev_mode = getattr(inner_model, "_batched_rc_mode", False)
        inner_model._batched_rc_mode = True

        t0 = time.perf_counter()

        try:
            # Single batched forward pass
            reconstruct_prefix_kv_stateful(
                inner_model, batched_h0, batched_temp_caches
            )
            mx.eval(batched_temp_caches[-1].keys)
        finally:
            inner_model._batched_rc_mode = prev_mode

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Extract batched K/V and split back to per-sequence
        batched_kv = extract_kv_from_temp_caches(batched_temp_caches)

        results = []
        for i, (s, (start, end), length) in enumerate(
            zip(states, ranges, lengths)
        ):
            # Extract this sequence's K/V slice from batched result
            for layer_idx in range(num_layers):
                bk, bv = batched_kv[layer_idx]
                # Slice this sequence and trim to actual length
                seq_k = bk[i:i+1, :, :s.reconstructed_tokens + length, :]
                seq_v = bv[i:i+1, :, :s.reconstructed_tokens + length, :]

                # Update this sequence's persistent temp_caches
                s.temp_caches[layer_idx].keys = seq_k
                s.temp_caches[layer_idx].values = seq_v
                s.temp_caches[layer_idx].offset = s.reconstructed_tokens + length

            # Update state
            s.reconstructed_tokens = end
            s.chunks_processed += 1
            s.total_time_ms += elapsed_ms / len(states)  # amortized

            target = s.effective_end if s.effective_end is not None else s.total_tokens
            is_final = end >= target
            if is_final:
                s.status = "completed"

            results.append(RCChunkResult(
                sequence_id=s.sequence_id,
                chunk_start=start,
                chunk_end=end,
                tokens_processed=length,
                time_ms=elapsed_ms / len(states),
                cumulative_tokens=s.reconstructed_tokens,
                is_final=is_final,
            ))

        logger.debug(
            f"[RCEngine] Batched chunk: B={len(states)}, "
            f"{max_len} tokens, {elapsed_ms:.1f}ms total"
        )

        return results

    def inject_completed(self, state: RCSequenceState) -> Tuple[int, float]:
        """Inject completed reconstruction K/V into target caches.

        This is the final step: extract accumulated K/V from temp_caches
        and inject into the target TripleLayerKVCaches (or KVDirectCaches).

        Injection is atomic: all layers injected, then single mx.eval().

        Args:
            state: Completed RCSequenceState (status == "completed").

        Returns:
            (layers_injected, memory_mb) — number of layers and memory delta.
        """
        from mlx_lm.models.kv_direct_cache import extract_kv_from_temp_caches

        if state.status != "completed":
            logger.warning(
                f"[RCEngine] inject_completed called on {state.sequence_id} "
                f"with status={state.status}"
            )
            return 0, 0.0

        kv_list = extract_kv_from_temp_caches(state.temp_caches)

        layers_injected = 0
        recon_arrays = []

        for i, cache in enumerate(state.target_cache_list):
            if i < len(kv_list) and kv_list[i] is not None:
                k, v = kv_list[i]
                if hasattr(cache, "inject_reconstruction"):
                    cache.inject_reconstruction(k, v)
                else:
                    cache._recon_keys = k
                    cache._recon_values = v
                recon_arrays.extend([k, v])
                layers_injected += 1

        if recon_arrays:
            mx.eval(*recon_arrays)

        # Update prefix counts for dedup
        actual_tokens = state.reconstructed_tokens
        for cache in state.target_cache_list:
            if hasattr(cache, "_flat_prefix_token_count"):
                cache._flat_prefix_token_count = max(
                    getattr(cache, "_flat_prefix_token_count", 0),
                    actual_tokens,
                )

        memory_mb = sum(a.nbytes for a in recon_arrays) / (1024 * 1024)
        state.status = "injected"

        # Clean up temp state to free memory
        state.temp_caches = []
        state.h0_full = None

        # Remove from active tracking
        self._active.pop(state.sequence_id, None)

        logger.info(
            f"[RCEngine] Injected {state.sequence_id}: "
            f"{actual_tokens} tokens, {layers_injected} layers, "
            f"{state.chunks_processed} chunks, {state.total_time_ms:.0f}ms total, "
            f"{memory_mb:.1f}MB"
        )

        return layers_injected, memory_mb

    def abort(self, seq_id: str) -> bool:
        """Abort an in-progress reconstruction, freeing all resources.

        Safe to call at any time. If the sequence is not found, returns False.
        """
        state = self._active.pop(seq_id, None)
        if state is None:
            return False

        # Free temp state
        state.temp_caches = []
        state.h0_full = None
        state.status = "aborted"

        logger.info(f"[RCEngine] Aborted {seq_id} at {state.reconstructed_tokens}/{state.total_tokens}")
        return True

    @property
    def active_count(self) -> int:
        """Number of sequences currently being reconstructed."""
        return len(self._active)

    @property
    def active_sequences(self) -> Dict[str, RCSequenceState]:
        """Read-only view of active reconstructions."""
        return dict(self._active)

    def stats(self) -> dict:
        """Summary statistics for logging/monitoring."""
        states = list(self._active.values())
        return {
            "active_count": len(states),
            "total_remaining_tokens": sum(s.remaining_tokens for s in states),
            "total_remaining_chunks": sum(s.remaining_chunks for s in states),
            "sequences": {
                s.sequence_id: {
                    "progress": f"{s.progress:.0%}",
                    "reconstructed": s.reconstructed_tokens,
                    "total": s.effective_end or s.total_tokens,
                    "chunks": s.chunks_processed,
                    "time_ms": s.total_time_ms,
                }
                for s in states
            },
        }
