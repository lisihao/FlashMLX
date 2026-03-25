"""
Compacted KV Cache with automatic compression.

Integrates both Fast Path (Phase A) and Quality Path (Phase B) compression.
"""

import mlx.core as mx
from typing import Optional, Tuple

from .cache import _BaseCache, create_attention_mask
from ..compaction.fast_v2 import compact_multi_head_fast_v2
from ..compaction.quality import compact_multi_head_quality


class CompactedKVCache(_BaseCache):
    """
    KV Cache with automatic compression.

    Automatically compresses the KV cache when it exceeds a threshold,
    using either Fast Path (Phase A) or Quality Path (Phase B).

    **Hybrid Architecture Support:**
    This cache supports both pure Transformer and hybrid architectures
    (e.g., Qwen3.5 with SSM + Attention layers):
    - Attention layers use `cache.update_and_fetch()` (with compression)
    - SSM layers use `cache[0]` and `cache[1]` (without compression)

    Parameters
    ----------
    max_size : int
        Maximum cache size before triggering compression
    compression_ratio : float, default=5.0
        Target compression ratio (e.g., 5.0 = compress to 1/5 of original size)
    recent_ratio : float, default=0.5
        Ratio of recent tokens to preserve (Fast Path only)
    enable_compression : bool, default=True
        Enable/disable automatic compression
    use_quality_path : bool, default=False
        Use Quality Path instead of Fast Path
        - Fast Path: O(budget), best for structured data
        - Quality Path: O(budget²), best for random data (100% improvement)
    quality_fit_beta : bool, default=True
        Enable beta fitting in Quality Path (only if use_quality_path=True)
    quality_fit_c2 : bool, default=True
        Enable C2 fitting in Quality Path (only if use_quality_path=True)

    Notes
    -----
    **Fast Path (default):**
    - Recent (50%) + Random (50%) selection
    - O(budget) time complexity
    - Best for data with attention locality

    **Quality Path (use_quality_path=True):**
    - Attention-aware selection + Beta fitting + C2 fitting
    - O(budget²) time complexity
    - Best for random data (achieves 100% improvement over Fast Path)
    - Handles random KV cache distributions perfectly

    **Hybrid Architecture (SSM + Attention):**
    - Attention layers: KV cache compressed automatically
    - SSM layers: States stored separately (subscript access)
    - Compatible with Qwen3.5, Qwen3Next, and similar models

    Examples
    --------
    >>> # Fast Path (default) - works for both pure Transformer and hybrid
    >>> cache = CompactedKVCache(max_size=4096, compression_ratio=5.0)
    >>>
    >>> # Quality Path (for random data)
    >>> cache = CompactedKVCache(
    ...     max_size=4096,
    ...     compression_ratio=5.0,
    ...     use_quality_path=True
    ... )
    >>>
    >>> # Usage with Attention layers (automatic)
    >>> keys, values = cache.update_and_fetch(keys, values)
    >>>
    >>> # Usage with SSM layers (automatic)
    >>> conv_state = cache[0]  # Read conv_state
    >>> cache[0] = new_conv_state  # Write conv_state
    >>> ssm_state = cache[1]  # Read ssm_state
    >>> cache[1] = new_ssm_state  # Write ssm_state
    >>> cache.advance(sequence_length)  # Advance offset
    """

    step = 256

    def __init__(
        self,
        max_size: int = 4096,
        compression_ratio: float = 5.0,
        recent_ratio: float = 0.5,
        enable_compression: bool = True,
        use_quality_path: bool = False,
        quality_fit_beta: bool = True,
        quality_fit_c2: bool = True,
        calibration_file: Optional[str] = None,
        layer_idx: Optional[int] = None,
    ):
        # Attention layer attributes (compressed KV cache)
        self.keys = None
        self.values = None
        self.beta = None  # ✅ AM Fix: Store attention bias for compressed KV
        self.offset = 0  # Physical cache offset (how many slots used)
        self.sequence_position = 0  # ✅ RoPE Fix: Logical sequence position (never resets after compression)
        self.max_size = max_size
        self.compression_ratio = compression_ratio
        self.recent_ratio = recent_ratio
        self.enable_compression = enable_compression

        # Quality Path configuration
        self.use_quality_path = use_quality_path
        self.quality_fit_beta = quality_fit_beta
        self.quality_fit_c2 = quality_fit_c2

        # Statistics
        self.num_compressions = 0
        self.total_tokens_before = 0
        self.total_tokens_after = 0

        # SSM layer attributes (for hybrid architectures like Qwen3.5)
        # These are NOT compressed (SSM states are small)
        self._ssm_states = [None, None]  # [conv_state, ssm_state]
        self.lengths = None  # Optional: for SSM length tracking

        # ✅ Offline Calibration support
        self.calibration = None
        self.layer_idx = layer_idx
        if calibration_file is not None:
            self._load_calibration(calibration_file, layer_idx)

    def _load_calibration(self, calibration_file: str, layer_idx: int):
        """Load pre-fitted calibration parameters from file."""
        import pickle
        import numpy as np

        if layer_idx is None:
            raise ValueError("layer_idx must be specified when using calibration_file")

        try:
            with open(calibration_file, 'rb') as f:
                calibration_data = pickle.load(f)

            # Extract calibration for this layer
            layer_calibration = calibration_data['calibration'][layer_idx]

            # Convert numpy arrays back to MLX
            self.calibration = {
                'Ck': mx.array(layer_calibration['Ck']),
                'beta': mx.array(layer_calibration['beta']),
                'selected_indices': mx.array(layer_calibration['selected_indices']),
                'compression_ratio': layer_calibration['compression_ratio'],
                'budget': layer_calibration['budget']
            }

            print(f"[CompactedKVCache] Loaded calibration for layer {layer_idx}")
            print(f"  Budget: {self.calibration['budget']}")
            print(f"  Compression ratio: {self.calibration['compression_ratio']}")

        except Exception as e:
            print(f"[CompactedKVCache] Failed to load calibration: {e}")
            print(f"  Will fall back to runtime calibration")
            self.calibration = None

    def update_and_fetch(self, keys, values):
        """
        Update cache with new keys/values and fetch all cached data.

        If cache size exceeds max_size, automatically compress using Fast Path v2.

        Parameters
        ----------
        keys : mx.array, shape (B, n_heads, num_steps, head_dim)
        values : mx.array, shape (B, n_heads, num_steps, head_dim)

        Returns
        -------
        keys : mx.array
            All cached keys
        values : mx.array
            All cached values
        beta : mx.array or None
            Attention bias for AM compression (shape: B, n_heads, offset)
            None if compression not used or beta not available
        """
        B, n_kv_heads, num_steps, head_dim = keys.shape

        # First time: initialize
        if self.keys is None:
            # Allocate with step size
            allocated_steps = ((num_steps - 1) // self.step + 1) * self.step
            self.keys = mx.zeros((B, n_kv_heads, allocated_steps, head_dim), dtype=keys.dtype)
            self.values = mx.zeros((B, n_kv_heads, allocated_steps, head_dim), dtype=values.dtype)

        # Check if need to expand
        if (self.offset + num_steps) > self.keys.shape[2]:
            new_steps = ((self.offset + num_steps - 1) // self.step + 1) * self.step
            new_keys = mx.zeros((B, n_kv_heads, new_steps, head_dim), dtype=keys.dtype)
            new_values = mx.zeros((B, n_kv_heads, new_steps, head_dim), dtype=values.dtype)

            if self.offset > 0:
                new_keys[..., :self.offset, :] = self.keys[..., :self.offset, :]
                new_values[..., :self.offset, :] = self.values[..., :self.offset, :]

            self.keys = new_keys
            self.values = new_values

            # ✅ CRITICAL FIX: Expand beta along with keys/values
            if self.beta is not None:
                new_beta = mx.zeros((B, n_kv_heads, new_steps), dtype=self.beta.dtype)
                if self.offset > 0:
                    new_beta[..., :self.offset] = self.beta[..., :self.offset]
                self.beta = new_beta

        # Append new keys/values
        self.keys[..., self.offset:self.offset + num_steps, :] = keys
        self.values[..., self.offset:self.offset + num_steps, :] = values
        self.offset += num_steps
        self.sequence_position += num_steps  # ✅ RoPE Fix: Track sequence position separately

        # ✅ Phase 1 Fix: 移除热路径压缩检查
        # 压缩应该由 CompactionEngine 独立触发，不在 update_and_fetch() 中执行
        # Original code (removed):
        # if self.enable_compression and self.offset > self.max_size:
        #     self._compress()

        # ✅ AM Fix: Return beta for attention bias
        # beta shape: (B, n_heads, offset) - one bias per cached token
        beta_slice = self.beta[..., :self.offset] if self.beta is not None else None

        return (
            self.keys[..., :self.offset, :],
            self.values[..., :self.offset, :],
            beta_slice
        )

    def _compress(self, queries: Optional[mx.array] = None):
        """
        Compress the KV cache using Fast Path or Quality Path.

        If pre-fitted calibration is available, directly applies it (fast).
        Otherwise, falls back to runtime calibration (slower).

        Parameters
        ----------
        queries : mx.array, optional
            Reference queries (Qref) for AM compression.
            Shape: (B, n_heads, num_queries, head_dim)
            Only used if calibration is not available.

        Reduces cache size from current offset to target budget.
        """
        target_budget = int(self.offset / self.compression_ratio)
        if target_budget < 10:
            # Too small to compress meaningfully
            return

        B, n_heads = self.keys.shape[:2]

        # Extract current cache
        current_keys = self.keys[..., :self.offset, :]  # (B, n_heads, offset, head_dim)
        current_values = self.values[..., :self.offset, :]

        # ✅ Check if offline calibration is available and applicable
        use_calibration = False
        if self.calibration is not None:
            # Check if ALL calibration indices are within current sequence length
            import numpy as np
            selected_indices = self.calibration['selected_indices']
            max_index = int(np.max(np.array(selected_indices)))
            current_length = self.offset

            # CRITICAL: Indices must be strictly within bounds
            if max_index < current_length:
                use_calibration = True
                calibration_length = max_index + 1
                print(f"[CompactedKVCache] Using pre-fitted calibration (fast path)")
                print(f"[CompactedKVCache]   Calibration max index: {max_index}, Current length: {current_length}")
            else:
                calibration_length = max_index + 1
                out_of_bounds = int(np.sum(np.array(selected_indices) >= current_length))
                print(f"[CompactedKVCache] Indices out of bounds: {out_of_bounds}/{len(selected_indices)}")
                print(f"[CompactedKVCache]   Max index: {max_index}, Current length: {current_length}")
                print(f"[CompactedKVCache] Falling back to runtime calibration for safety")

        # ✅ Offline Calibration Path (fast)
        if use_calibration:
            # Apply pre-fitted compression (no OMP, just indexing)
            selected_indices = self.calibration['selected_indices']
            beta = self.calibration['beta']
            budget = self.calibration['budget']

            # Adapt budget to target_budget
            if len(selected_indices) > target_budget:
                selected_indices = selected_indices[:target_budget]
                beta = beta[:target_budget]
                budget = target_budget

            # Compress each batch independently
            compressed_keys_list = []
            compressed_values_list = []
            compressed_beta_list = []

            for b in range(B):
                K_batch = current_keys[b]  # (n_heads, offset, head_dim)
                V_batch = current_values[b]

                # Apply compression for each head
                C1_heads = []
                C2_heads = []
                beta_heads = []

                for h in range(n_heads):
                    # Select keys/values using calibration indices
                    C1_h = mx.take(K_batch[h], selected_indices, axis=0)  # (budget, head_dim)
                    C2_h = mx.take(V_batch[h], selected_indices, axis=0)

                    # Apply beta weighting
                    C1_h = C1_h * beta[:, None]  # (budget, head_dim)
                    C2_h = C2_h * beta[:, None]

                    C1_heads.append(C1_h[None, ...])  # (1, budget, head_dim)
                    C2_heads.append(C2_h[None, ...])
                    beta_heads.append(beta[None, :])  # (1, budget)

                C1 = mx.concatenate(C1_heads, axis=0)  # (n_heads, budget, head_dim)
                C2 = mx.concatenate(C2_heads, axis=0)
                beta_batch = mx.concatenate(beta_heads, axis=0)  # (n_heads, budget)

                compressed_keys_list.append(C1[None, ...])
                compressed_values_list.append(C2[None, ...])
                compressed_beta_list.append(beta_batch[None, ...])

            compressed_keys = mx.concatenate(compressed_keys_list, axis=0)
            compressed_values = mx.concatenate(compressed_values_list, axis=0)
            compressed_beta = mx.concatenate(compressed_beta_list, axis=0)

        else:
            # ✅ Runtime Calibration: Fallback to runtime fitting (slower)
            print(f"[CompactedKVCache] Using runtime calibration (slow path)")

            compressed_keys_list = []
            compressed_values_list = []
            compressed_beta_list = []

            for b in range(B):
                K_batch = current_keys[b]  # (n_heads, offset, head_dim)
                V_batch = current_values[b]

                # Extract queries for this batch (if provided)
                Q_batch = queries[b] if queries is not None else None

                if self.use_quality_path:
                    # Compress using Quality Path
                    C1, beta, C2 = compact_multi_head_quality(
                        K_batch, V_batch, budget=target_budget,
                        queries=Q_batch,
                        fit_beta=self.quality_fit_beta,
                        fit_c2=self.quality_fit_c2
                    )
                else:
                    # Compress using Fast Path v2
                    C1, beta, C2 = compact_multi_head_fast_v2(
                        K_batch, V_batch, budget=target_budget, recent_ratio=self.recent_ratio
                    )

                compressed_keys_list.append(C1[None, ...])
                compressed_values_list.append(C2[None, ...])
                compressed_beta_list.append(beta[None, ...])

            compressed_keys = mx.concatenate(compressed_keys_list, axis=0)
            compressed_values = mx.concatenate(compressed_values_list, axis=0)
            compressed_beta = mx.concatenate(compressed_beta_list, axis=0)

        # Update cache with compressed data
        self.keys[..., :target_budget, :] = compressed_keys
        self.values[..., :target_budget, :] = compressed_values

        # ✅ AM Fix: Initialize beta storage if needed
        if self.beta is None:
            # Allocate beta with same structure as keys (but without head_dim)
            self.beta = mx.zeros((B, n_heads, self.keys.shape[2]), dtype=compressed_beta.dtype)

        # ✅ AM Fix: Store beta
        self.beta[..., :target_budget] = compressed_beta

        # Update statistics
        self.num_compressions += 1
        self.total_tokens_before += self.offset
        self.total_tokens_after += target_budget

        # ✅ RoPE Fix: Reset physical cache offset, but DON'T reset sequence_position
        # - offset: physical cache size (resets to budget after compression)
        # - sequence_position: logical position for RoPE (never resets)
        self.offset = target_budget
        # sequence_position remains unchanged - it tracks the true position in the sequence!

    def compact(self, queries: Optional[mx.array] = None) -> bool:
        """
        Manually trigger compression (to be called by CompactionEngine).

        This is the correct way to compress the KV cache, as opposed to
        automatic compression in update_and_fetch() which is in the hot path.

        Parameters
        ----------
        queries : mx.array, optional
            Reference queries (Qref) for AM compression.
            Shape: (B, n_heads, num_queries, head_dim)
            If None, uses keys as queries (approximation).

        Returns
        -------
        bool
            True if compression was performed, False otherwise.

        Examples
        --------
        >>> # Offline compaction (correct way)
        >>> if cache.offset > cache.max_size:
        >>>     # Sample reference queries
        >>>     queries = sample_recent_queries(cache, num_queries=128)
        >>>     # Trigger compression
        >>>     cache.compact(queries=queries)
        """
        if not self.enable_compression:
            return False

        if self.offset <= self.max_size:
            return False

        self._compress(queries=queries)
        return True

    def get_stats(self) -> dict:
        """
        Get compression statistics.

        Returns
        -------
        stats : dict
            - num_compressions: Number of times cache was compressed
            - total_tokens_before: Total tokens before all compressions
            - total_tokens_after: Total tokens after all compressions
            - avg_compression_ratio: Average compression ratio achieved
            - current_size: Current cache size
        """
        if self.num_compressions == 0:
            avg_ratio = 1.0
        else:
            avg_ratio = self.total_tokens_before / self.total_tokens_after

        return {
            'num_compressions': self.num_compressions,
            'total_tokens_before': self.total_tokens_before,
            'total_tokens_after': self.total_tokens_after,
            'avg_compression_ratio': avg_ratio,
            'current_size': self.offset,
        }

    @property
    def state(self):
        if self.offset == 0 or self.keys is None:
            return []
        return [
            self.keys[..., :self.offset, :],
            self.values[..., :self.offset, :],
        ]

    @state.setter
    def state(self, v):
        if not v:
            self.keys = None
            self.values = None
            self.offset = 0
            self.sequence_position = 0  # ✅ RoPE Fix: Reset sequence position
        else:
            self.keys, self.values = v
            self.offset = self.keys.shape[2]
            self.sequence_position = self.offset  # ✅ RoPE Fix: Restore sequence position

    @property
    def meta_state(self):
        return tuple(map(str, (
            self.offset,
            self.max_size,
            self.compression_ratio,
            self.recent_ratio,
            int(self.enable_compression),
            self.num_compressions,
            self.total_tokens_before,
            self.total_tokens_after,
            int(self.use_quality_path),
            int(self.quality_fit_beta),
            int(self.quality_fit_c2),
        )))

    @meta_state.setter
    def meta_state(self, v):
        # Handle both old format (8 values) and new format (11 values)
        if len(v) == 8:
            # Old format: no Quality Path params
            (self.offset, self.max_size, self.compression_ratio,
             self.recent_ratio, enable_int, self.num_compressions,
             self.total_tokens_before, self.total_tokens_after) = map(float, v)
            # Default Quality Path settings
            self.use_quality_path = False
            self.quality_fit_beta = True
            self.quality_fit_c2 = True
        else:
            # New format: with Quality Path params
            (self.offset, self.max_size, self.compression_ratio,
             self.recent_ratio, enable_int, self.num_compressions,
             self.total_tokens_before, self.total_tokens_after,
             use_quality_int, fit_beta_int, fit_c2_int) = map(float, v)
            self.use_quality_path = bool(int(use_quality_int))
            self.quality_fit_beta = bool(int(fit_beta_int))
            self.quality_fit_c2 = bool(int(fit_c2_int))

        self.offset = int(self.offset)
        self.max_size = int(self.max_size)
        self.enable_compression = bool(int(enable_int))
        self.num_compressions = int(self.num_compressions)
        self.total_tokens_before = int(self.total_tokens_before)
        self.total_tokens_after = int(self.total_tokens_after)

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        return n

    def make_mask(self, N, window_size=None, return_array=False):
        """Create attention mask for the cached sequence.

        Args:
            N: Sequence length
            window_size: Optional sliding window size
            return_array: If True, return actual mask array instead of "causal" string
        """
        if N == 1:
            return None
        if return_array or (window_size and N > window_size):
            from .base import create_causal_mask
            return create_causal_mask(N + self.offset, window_size=window_size)
        return "causal"

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        return self.keys.nbytes + self.values.nbytes

    def size(self):
        return self.offset

    # ========================================
    # Hybrid Architecture Support (SSM layers)
    # ========================================

    def __getitem__(self, index):
        """
        Get SSM state by index (for hybrid architectures).

        SSM layers (like GatedDeltaNet in Qwen3.5) expect subscriptable cache:
        - cache[0]: conv_state
        - cache[1]: ssm_state

        Note: Always returns None to let SSM layers initialize fresh state.
        This avoids batch size mismatch issues during generation.

        Parameters
        ----------
        index : int
            State index (0 or 1)

        Returns
        -------
        state : mx.array or None
            Always None (SSM layers will initialize fresh state)

        Raises
        ------
        IndexError
            If index is not 0 or 1

        Examples
        --------
        >>> conv_state = cache[0]  # Returns None
        >>> ssm_state = cache[1]   # Returns None
        """
        if index not in [0, 1]:
            raise IndexError(
                f"CompactedKVCache SSM state index must be 0 or 1, got {index}. "
                f"(0 = conv_state, 1 = ssm_state)"
            )
        # Always return None to let SSM layer initialize fresh state
        # This avoids batch size mismatch during generation
        result = None
        # Debug: Uncomment to see what SSM layer is asking for
        # print(f"[DEBUG] CompactedKVCache.__getitem__({index}) -> {result}")
        return result

    def __setitem__(self, index, value):
        """
        Set SSM state by index (for hybrid architectures).

        SSM layers (like GatedDeltaNet in Qwen3.5) update cache states:
        - cache[0] = conv_state
        - cache[1] = ssm_state

        Note: This is a no-op to avoid batch size mismatch issues.
        SSM states are not persisted across calls.

        Parameters
        ----------
        index : int
            State index (0 or 1)
        value : mx.array or None
            The new SSM state (ignored)

        Raises
        ------
        IndexError
            If index is not 0 or 1

        Examples
        --------
        >>> cache[0] = conv_input[:, -n_keep:, :]  # No-op
        >>> cache[1] = new_ssm_state               # No-op
        """
        if index not in [0, 1]:
            raise IndexError(
                f"CompactedKVCache SSM state index must be 0 or 1, got {index}. "
                f"(0 = conv_state, 1 = ssm_state)"
            )
        # Do not store SSM state to avoid batch size mismatch
        # SSM layers will reinitialize state on each call
        pass

    def advance(self, n: int):
        """
        Advance the sequence offset by n tokens (for SSM layers).

        This is called by SSM layers after processing n tokens.
        It updates the offset to track the current sequence position.

        Parameters
        ----------
        n : int
            Number of tokens to advance

        Examples
        --------
        >>> cache.advance(sequence_length)  # After SSM processing
        """
        self.offset += n
