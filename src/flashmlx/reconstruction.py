"""
ReconstructionController: Programmatic API for KV cache reconstruction.

Provides ThunderOMLX (or any upper-level scheduler) with full control over
h^(0) → K/V reconstruction without needing to know about internal model
patching, H0Store layouts, or per-layer cache injection.

Usage:
    from flashmlx import ReconstructionController

    cache_list = make_prompt_cache(model, **cache_kwargs)
    recon = ReconstructionController.from_cache(cache_list, model)

    if recon.available:
        cost = recon.estimate_cost(n_tokens=4096)
        result = recon.reconstruct(strategy="targeted", coverage=0.95)
        print(result.tokens_reconstructed, result.time_ms)
"""

from __future__ import annotations

import enum
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ReconState(enum.Enum):
    """Lifecycle state of the reconstruction controller."""

    IDLE = "idle"
    RECONSTRUCTING = "reconstructing"
    COMPLETED = "completed"
    FAILED = "failed"


class ReconStrategy(enum.Enum):
    """Reconstruction strategy selection."""

    FULL = "full"          # Reconstruct all h^(0) tokens
    PARTIAL = "partial"    # Reconstruct up to max_tokens
    TARGETED = "targeted"  # Probe-guided, coverage-based


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReconStats:
    """Current snapshot of the reconstruction system state."""

    h0_tokens: int
    h0_bytes: int
    h0_quant: str               # "bf16", "q8", "q4"
    probe_available: bool
    probe_layers: int
    n_cache_layers: int
    has_inner_model: bool
    reconstruction_count: int
    last_result: Optional[ReconResult]
    state: ReconState


@dataclass(frozen=True)
class ReconCostEstimate:
    """Estimated cost of a reconstruction operation."""

    n_tokens: int
    n_layers: int
    time_ms_est: float
    memory_mb_est: float
    strategy: str
    probe_available: bool


@dataclass
class ReconResult:
    """Result from a completed reconstruction."""

    success: bool
    strategy: str
    tokens_requested: int
    tokens_reconstructed: int
    layers_injected: int
    time_ms: float
    memory_delta_mb: float
    h0_tokens_available: int
    error: Optional[str] = None
    coverage: Optional[float] = None


# ---------------------------------------------------------------------------
# ReconstructionController
# ---------------------------------------------------------------------------

class ReconstructionController:
    """High-level API for KV cache reconstruction from h^(0).

    Provides ThunderOMLX with a clean SDK to:
    1. Query reconstruction availability and cost
    2. Trigger reconstruction with strategy control
    3. Monitor reconstruction state and results

    Thread Safety:
        All public methods are thread-safe. Reconstruction is serialized
        via a non-blocking lock to prevent concurrent replays.

    Idempotency:
        Calling reconstruct() when reconstruction is already injected is safe.
        The controller clears existing reconstruction before starting a new one.
    """

    def __init__(
        self,
        inner_model: Any,
        cache_list: List[Any],
        h0_store: Any,
        probe: Any = None,
        recon_budget: Any = None,
    ):
        self._inner_model = inner_model
        self._cache_list = cache_list
        self._h0_store = h0_store
        self._probe = probe
        self._recon_budget = recon_budget

        self._state = ReconState.IDLE
        self._last_result: Optional[ReconResult] = None
        self._reconstruction_count = 0
        self._lock = threading.Lock()

    # ---- Factory ----

    @classmethod
    def from_cache(
        cls, cache_list: list, model: Any = None
    ) -> "ReconstructionController":
        """Create a controller by discovering components from cache list + model.

        Returns NullReconstructionController if reconstruction is not available
        (no h^(0) store, no inner model, etc.).
        """
        if not cache_list:
            return NullReconstructionController()

        # Find TripleLayerKVCache instances
        try:
            from mlx_lm.models.triple_layer_cache import TripleLayerKVCache
            attn_caches = [
                c for c in cache_list if isinstance(c, TripleLayerKVCache)
            ]
        except ImportError:
            attn_caches = []

        if not attn_caches:
            return NullReconstructionController()

        # Check if controller is already attached
        for c in attn_caches:
            ctrl = getattr(c, "_reconstruction_controller", None)
            if ctrl is not None and isinstance(ctrl, ReconstructionController):
                return ctrl

        # Discover h0_store
        h0_store = None
        for c in attn_caches:
            h0_store = getattr(c, "_h0_store", None)
            if h0_store is not None:
                break

        if h0_store is None:
            return NullReconstructionController()

        # Discover inner model
        inner_model = None
        if model is not None:
            try:
                from mlx_lm.models.kv_direct_cache import _find_inner_model
                inner_model = _find_inner_model(model)
            except (ImportError, ValueError):
                pass

        if inner_model is None:
            try:
                from mlx_lm.models.triple_layer_cache import TripleLayerKVCache
                inner_model = getattr(TripleLayerKVCache, "_shared_inner_model", None)
            except ImportError:
                pass

        if inner_model is None:
            return NullReconstructionController()

        # Discover probe
        probe = None
        try:
            from mlx_lm.models.triple_layer_cache import TripleLayerKVCache
            probe = getattr(TripleLayerKVCache, "_shared_probe", None)
        except ImportError:
            pass

        recon_budget = getattr(h0_store, "recon_budget", None)

        return cls(
            inner_model=inner_model,
            cache_list=attn_caches,
            h0_store=h0_store,
            probe=probe,
            recon_budget=recon_budget,
        )

    # ---- Query interface (read-only, no lock) ----

    @property
    def available(self) -> bool:
        """Whether reconstruction is possible."""
        return (
            self._inner_model is not None
            and self._h0_store is not None
            and self._h0_store.count > 0
            and self._cache_list is not None
            and len(self._cache_list) > 0
        )

    @property
    def state(self) -> ReconState:
        return self._state

    @property
    def last_result(self) -> Optional[ReconResult]:
        return self._last_result

    @property
    def stats(self) -> ReconStats:
        h0_quant = "bf16"
        if self._h0_store is not None:
            h0_quant = getattr(self._h0_store, "_quant", None) or "bf16"

        return ReconStats(
            h0_tokens=self._h0_store.count if self._h0_store else 0,
            h0_bytes=self._h0_store.nbytes if self._h0_store else 0,
            h0_quant=h0_quant,
            probe_available=self._probe is not None,
            probe_layers=getattr(self._probe, "_n_layers", 0) if self._probe else 0,
            n_cache_layers=len(self._cache_list) if self._cache_list else 0,
            has_inner_model=self._inner_model is not None,
            reconstruction_count=self._reconstruction_count,
            last_result=self._last_result,
            state=self._state,
        )

    def estimate_cost(
        self,
        n_tokens: Optional[int] = None,
        strategy: str = "full",
        coverage: float = 0.95,
    ) -> ReconCostEstimate:
        """Estimate time and memory cost of reconstruction.

        Cost model (empirical, Qwen3-8B M4 Max):
          ~2.5ms/token for 36 layers, scales linearly.
        """
        if n_tokens is None:
            n_tokens = self._h0_store.count if self._h0_store else 0

        n_layers = len(self._inner_model.layers) if self._inner_model else 0
        ms_per_token = 2.5 * (n_layers / 36.0)

        effective_tokens = n_tokens
        if strategy == "targeted" and self._probe is not None:
            effective_tokens = int(n_tokens * max(0.5, coverage))

        time_est = effective_tokens * ms_per_token
        # Per-token per-layer: 2 * n_kv_heads * head_dim * 2B ≈ 4096 bytes
        bytes_per_token_per_layer = 4096
        memory_est = (
            effective_tokens * bytes_per_token_per_layer * n_layers
        ) / (1024 * 1024)

        return ReconCostEstimate(
            n_tokens=effective_tokens,
            n_layers=n_layers,
            time_ms_est=time_est,
            memory_mb_est=memory_est,
            strategy=strategy,
            probe_available=self._probe is not None,
        )

    # ---- Reconstruction operations (lock-protected) ----

    def reconstruct(
        self,
        strategy: str = "full",
        max_tokens: Optional[int] = None,
        coverage: float = 0.95,
        chunk_size: int = 512,
        eval_every: int = 8,
    ) -> ReconResult:
        """Trigger KV cache reconstruction from h^(0).

        Args:
            strategy: "full" | "partial" | "targeted"
            max_tokens: For "partial", cap reconstruction depth.
            coverage: For "targeted", minimum importance coverage (0-1).
            chunk_size: Tokens per GPU batch during reconstruction.
            eval_every: GPU sync every N chunks.

        Returns:
            ReconResult — always returns, never raises.
        """
        if not self.available:
            return ReconResult(
                success=False, strategy=strategy,
                tokens_requested=0, tokens_reconstructed=0,
                layers_injected=0, time_ms=0.0, memory_delta_mb=0.0,
                h0_tokens_available=0,
                error="Reconstruction not available",
            )

        acquired = self._lock.acquire(blocking=False)
        if not acquired:
            return ReconResult(
                success=False, strategy=strategy,
                tokens_requested=0, tokens_reconstructed=0,
                layers_injected=0, time_ms=0.0, memory_delta_mb=0.0,
                h0_tokens_available=self._h0_store.count,
                error="Reconstruction already in progress",
            )

        try:
            self._state = ReconState.RECONSTRUCTING
            return self._do_reconstruct(
                strategy, max_tokens, coverage, chunk_size, eval_every
            )
        finally:
            self._lock.release()

    def _do_reconstruct(
        self, strategy, max_tokens, coverage, chunk_size, eval_every
    ) -> ReconResult:
        """Internal implementation (runs under lock)."""
        import mlx.core as mx
        from mlx_lm.models.kv_direct_cache import (
            reconstruct_prefix_kv,
            reconstruct_targeted,
        )

        t0 = time.perf_counter()
        n_available = self._h0_store.count
        n_recon = n_available

        try:
            # Clear any existing reconstruction
            for c in self._cache_list:
                if hasattr(c, "clear_reconstruction"):
                    c.clear_reconstruction()

            # Determine reconstruction range
            if strategy == "partial" and max_tokens is not None:
                n_recon = min(n_available, max_tokens)

            # Run reconstruction
            actual_reconstructed = n_recon

            if strategy == "targeted" and self._probe is not None:
                importance_scores = self._probe.score_tokens(self._h0_store)
                kv_list, actual_reconstructed = reconstruct_targeted(
                    self._inner_model,
                    self._h0_store,
                    n_recon,
                    importance_scores=importance_scores,
                    min_coverage=coverage,
                    chunk_size=max(chunk_size, 512),
                    eval_every=eval_every,
                )
            else:
                kv_list = reconstruct_prefix_kv(
                    self._inner_model,
                    self._h0_store,
                    0,
                    n_recon,
                    chunk_size=chunk_size,
                    eval_every=eval_every,
                )

            # Inject into caches
            layers_injected = 0
            recon_arrays = []
            for i, c in enumerate(self._cache_list):
                if i < len(kv_list) and kv_list[i]:
                    k, v = kv_list[i]
                    if hasattr(c, "inject_reconstruction"):
                        c.inject_reconstruction(k, v)
                    else:
                        c._recon_keys = k
                        c._recon_values = v
                    recon_arrays.extend([k, v])
                    layers_injected += 1

            if recon_arrays:
                mx.eval(*recon_arrays)

            # Update prefix counts for dedup
            for c in self._cache_list:
                if hasattr(c, "_flat_prefix_token_count"):
                    c._flat_prefix_token_count = max(
                        getattr(c, "_flat_prefix_token_count", 0),
                        actual_reconstructed,
                    )

            elapsed_ms = (time.perf_counter() - t0) * 1000

            if self._recon_budget:
                self._recon_budget.record_reconstruction(
                    actual_reconstructed, elapsed_ms
                )

            result = ReconResult(
                success=True,
                strategy=strategy,
                tokens_requested=n_recon,
                tokens_reconstructed=actual_reconstructed,
                layers_injected=layers_injected,
                time_ms=elapsed_ms,
                memory_delta_mb=sum(a.nbytes for a in recon_arrays) / (1024 * 1024),
                h0_tokens_available=n_available,
                coverage=coverage if strategy == "targeted" else None,
            )

            self._state = ReconState.COMPLETED
            self._last_result = result
            self._reconstruction_count += 1

            logger.info(
                f"[ReconController] {strategy}: "
                f"{actual_reconstructed}/{n_available} tokens, "
                f"{layers_injected} layers, {elapsed_ms:.0f}ms"
            )
            return result

        except Exception as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            result = ReconResult(
                success=False,
                strategy=strategy,
                tokens_requested=n_recon,
                tokens_reconstructed=0,
                layers_injected=0,
                time_ms=elapsed_ms,
                memory_delta_mb=0.0,
                h0_tokens_available=n_available,
                error=str(e),
            )
            self._state = ReconState.FAILED
            self._last_result = result
            logger.error(f"[ReconController] Failed: {e}")
            return result

    def reconstruct_from_h0_blocks(
        self,
        h0_blocks: list,
        h0_quant: Optional[str] = None,
        strategy: str = "full",
        coverage: float = 0.95,
        chunk_size: int = 512,
        eval_every: int = 8,
    ) -> ReconResult:
        """Reconstruct KV cache from SSD-loaded H0 blocks (blocking).

        Tier 3 entry point for blocking cold-cache restoration.
        For non-blocking 3PIR, use RCEngine.register_from_h0_blocks()
        with the async step API instead.

        Args:
            h0_blocks: List of block dicts from H0Store.export_blocks().
            h0_quant: Override quantization ('q8'|'q4'|None for bf16).
                If None, inferred from blocks.
            strategy: "full" or "targeted".
            coverage: Importance coverage for targeted strategy.
            chunk_size: Tokens per GPU batch.
            eval_every: GPU sync every N chunks.

        Returns:
            ReconResult with reconstruction stats.
        """
        from mlx_lm.models.kv_direct_cache import H0Store

        if h0_quant is None and h0_blocks:
            raw_quant = h0_blocks[0].get('quant', 'bf16')
            h0_quant = None if raw_quant == 'bf16' else raw_quant

        store = H0Store(quant=h0_quant)
        n_tokens = store.import_blocks(h0_blocks)

        if n_tokens == 0:
            return ReconResult(
                success=False, strategy=strategy,
                tokens_requested=0, tokens_reconstructed=0,
                layers_injected=0, time_ms=0.0, memory_delta_mb=0.0,
                h0_tokens_available=0,
                error="No tokens in H0 blocks",
            )

        # Temporarily swap in the reconstructed H0Store
        original_store = self._h0_store
        self._h0_store = store
        try:
            return self.reconstruct(
                strategy=strategy, coverage=coverage,
                chunk_size=chunk_size, eval_every=eval_every,
            )
        finally:
            self._h0_store = original_store

    def clear(self):
        """Clear all reconstructed K/V from caches. Frees memory."""
        for c in self._cache_list:
            if hasattr(c, "clear_reconstruction"):
                c.clear_reconstruction()
        self._state = ReconState.IDLE

    # ---- 3PIR Async API (non-blocking, chunk-at-a-time) ----

    def reconstruct_async_start(
        self,
        strategy: str = "full",
        coverage: float = 0.95,
        chunk_size: int = 512,
        seq_id: Optional[str] = None,
    ) -> Optional[Any]:
        """Start a non-blocking reconstruction, returning an RCSequenceState.

        Unlike reconstruct() which blocks until completion, this returns
        immediately with a state object that can be advanced one chunk at
        a time via reconstruct_async_step().

        This is the entry point for 3PIR integration with ThunderOMLX's
        scheduler — RC chunks are processed in TG idle windows.

        Args:
            strategy: "full" or "targeted" (importance-guided depth reduction).
            coverage: For "targeted", importance coverage threshold.
            chunk_size: Tokens per chunk (default 512).
            seq_id: Optional sequence identifier for logging.

        Returns:
            RCSequenceState if reconstruction is available, None otherwise.
        """
        if not self.available:
            return None

        from .rc_engine import RCEngine

        # Determine importance scores for targeted strategy
        importance_scores = None
        if strategy == "targeted" and self._probe is not None:
            importance_scores = self._probe.score_tokens(self._h0_store)

        engine = RCEngine(chunk_size=chunk_size)
        state = engine.register_sequence(
            seq_id=seq_id or f"recon_{self._reconstruction_count}",
            h0_store=self._h0_store,
            inner_model=self._inner_model,
            target_cache_list=self._cache_list,
            importance_scores=importance_scores,
            min_coverage=coverage,
        )

        # Store engine reference on state for step/complete calls
        state._rc_engine = engine
        self._state = ReconState.RECONSTRUCTING

        logger.info(
            f"[ReconController] Async start: {state.total_tokens} tokens, "
            f"strategy={strategy}, chunk_size={chunk_size}"
        )
        return state

    def reconstruct_async_step(self, state: Any) -> Any:
        """Process one chunk of the async reconstruction.

        Call this repeatedly (e.g., once per scheduler step) until
        state.is_complete is True.

        Args:
            state: RCSequenceState from reconstruct_async_start().

        Returns:
            RCChunkResult with timing and progress info.
        """
        engine = getattr(state, "_rc_engine", None)
        if engine is None:
            raise ValueError("State has no _rc_engine — was it created by reconstruct_async_start()?")
        return engine.process_chunk(state)

    def reconstruct_async_complete(self, state: Any) -> ReconResult:
        """Inject completed async reconstruction into target caches.

        Call this after state.is_complete is True. Injects all accumulated
        K/V into the target caches and returns a ReconResult.

        Args:
            state: Completed RCSequenceState.

        Returns:
            ReconResult with full stats.
        """
        engine = getattr(state, "_rc_engine", None)
        if engine is None:
            raise ValueError("State has no _rc_engine — was it created by reconstruct_async_start()?")

        # Clear existing reconstruction before injecting new
        for c in self._cache_list:
            if hasattr(c, "clear_reconstruction"):
                c.clear_reconstruction()

        layers_injected, memory_mb = engine.inject_completed(state)

        if self._recon_budget:
            self._recon_budget.record_reconstruction(
                state.reconstructed_tokens, state.total_time_ms
            )

        result = ReconResult(
            success=True,
            strategy="async",
            tokens_requested=state.effective_end or state.total_tokens,
            tokens_reconstructed=state.reconstructed_tokens,
            layers_injected=layers_injected,
            time_ms=state.total_time_ms,
            memory_delta_mb=memory_mb,
            h0_tokens_available=state.total_tokens,
        )

        self._state = ReconState.COMPLETED
        self._last_result = result
        self._reconstruction_count += 1

        logger.info(
            f"[ReconController] Async complete: "
            f"{state.reconstructed_tokens} tokens, "
            f"{layers_injected} layers, "
            f"{state.chunks_processed} chunks, "
            f"{state.total_time_ms:.0f}ms total"
        )
        return result


# ---------------------------------------------------------------------------
# Null Object — safe no-op when reconstruction is not available
# ---------------------------------------------------------------------------

class NullReconstructionController(ReconstructionController):
    """No-op controller returned when h^(0) reconstruction is not available.

    All operations are safe no-ops. ThunderOMLX never needs to check for None.
    """

    def __init__(self):
        self._inner_model = None
        self._cache_list = []
        self._h0_store = None
        self._probe = None
        self._recon_budget = None
        self._state = ReconState.IDLE
        self._last_result = None
        self._reconstruction_count = 0
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return False

    def reconstruct(self, **kwargs) -> ReconResult:
        return ReconResult(
            success=False,
            strategy=kwargs.get("strategy", "none"),
            tokens_requested=0,
            tokens_reconstructed=0,
            layers_injected=0,
            time_ms=0.0,
            memory_delta_mb=0.0,
            h0_tokens_available=0,
            error="Reconstruction not available (no h^(0) store)",
        )

    def estimate_cost(self, **kwargs) -> ReconCostEstimate:
        return ReconCostEstimate(
            n_tokens=0,
            n_layers=0,
            time_ms_est=0.0,
            memory_mb_est=0.0,
            strategy="none",
            probe_available=False,
        )

    def reconstruct_async_start(self, **kwargs):
        return None

    def reconstruct_async_step(self, state):
        return None

    def reconstruct_async_complete(self, state):
        return ReconResult(
            success=False,
            strategy="none",
            tokens_requested=0,
            tokens_reconstructed=0,
            layers_injected=0,
            time_ms=0.0,
            memory_delta_mb=0.0,
            h0_tokens_available=0,
            error="Reconstruction not available (no h^(0) store)",
        )

    def clear(self):
        pass
