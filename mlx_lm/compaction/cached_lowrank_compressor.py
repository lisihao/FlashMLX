"""
Cached Low-Rank SSM State Compressor

Optimization: Cache SVD decomposition and reuse when state structure is stable.

Author: FlashMLX Research
Date: 2026-03-21
Task: #53 - SSM State Compression Optimization
"""

import mlx.core as mx
from typing import Dict, Optional, Tuple


class CachedLowRankCompressor:
    """
    Low-Rank SSM State Compressor with SVD Caching

    Caches U/Vt decomposition and reuses when state structure is stable.
    Avoids expensive SVD recomputation for every token.
    """

    def __init__(
        self,
        rank: int = 32,
        cache_threshold: float = 0.15,
        enable_cache: bool = True
    ):
        """
        Args:
            rank: Number of singular values to keep
            cache_threshold: Max relative change to allow cache reuse (default: 0.15 = 15%)
            enable_cache: Whether to enable caching (for ablation testing)
        """
        self.rank = rank
        self.cache_threshold = cache_threshold
        self.enable_cache = enable_cache

        # Cache storage
        self.cached_U = None      # (B, Hv, Dv, rank)
        self.cached_Vt = None     # (B, Hv, rank, Dk)
        self.last_state_norm = None  # Scalar norm for change detection

        # Statistics
        self.stats = {
            'total_compress': 0,
            'cache_hit': 0,
            'cache_miss': 0,
            'svd_avoided': 0
        }

    def _compute_state_signature(self, state: mx.array) -> float:
        """
        Compute a signature for state to detect structural changes.

        Uses Frobenius norm as a simple indicator.
        """
        return float(mx.linalg.norm(state).item())

    def _should_recompute_svd(self, state: mx.array) -> bool:
        """
        Decide if we need to recompute SVD or can reuse cache.

        Returns:
            True if SVD recomputation needed, False if can use cache
        """
        if not self.enable_cache:
            return True

        if self.cached_U is None or self.cached_Vt is None:
            return True  # No cache yet

        # Compute current state signature
        current_norm = self._compute_state_signature(state)

        # Check relative change
        if self.last_state_norm is not None:
            relative_change = abs(current_norm - self.last_state_norm) / (self.last_state_norm + 1e-8)

            if relative_change < self.cache_threshold:
                self.stats['cache_hit'] += 1
                return False  # Can reuse cache

        self.stats['cache_miss'] += 1
        return True  # Need to recompute

    def _full_svd_compress(self, state: mx.array) -> Dict[str, mx.array]:
        """
        Full SVD compression (expensive, but accurate).

        Computes SVD for all heads and stores U/S/Vt.
        """
        B, Hv, Dv, Dk = state.shape
        original_dtype = state.dtype

        U_list, S_list, Vt_list = [], [], []

        for b in range(B):
            U_batch, S_batch, Vt_batch = [], [], []

            for h in range(Hv):
                # Convert to float32 for SVD
                state_slice = state[b, h].astype(mx.float32)

                # SVD on (Dv, Dk) slice (must run on CPU)
                with mx.stream(mx.cpu):
                    U, S, Vt = mx.linalg.svd(state_slice)

                # Keep top-rank components
                U_batch.append(U[:, :self.rank])
                S_batch.append(S[:self.rank])
                Vt_batch.append(Vt[:self.rank, :])

            U_list.append(mx.stack(U_batch))
            S_list.append(mx.stack(S_batch))
            Vt_list.append(mx.stack(Vt_batch))

        U = mx.stack(U_list)
        S = mx.stack(S_list)
        Vt = mx.stack(Vt_list)

        # Cache U and Vt for future reuse
        if self.enable_cache:
            self.cached_U = U
            self.cached_Vt = Vt
            self.last_state_norm = self._compute_state_signature(state)

        return {
            'U': U,
            'S': S,
            'Vt': Vt,
            'rank': self.rank,
            'original_shape': (B, Hv, Dv, Dk),
            'original_dtype': str(original_dtype),
            'cached': False
        }

    def _cached_compress(self, state: mx.array) -> Dict[str, mx.array]:
        """
        Fast compression using cached U/Vt.

        Only computes S = U^T @ state @ Vt^T (much faster than full SVD).
        """
        B, Hv, Dv, Dk = state.shape
        original_dtype = state.dtype

        # Reuse cached U and Vt
        U = self.cached_U
        Vt = self.cached_Vt

        # Compute S using cached decomposition
        # state ≈ U @ S @ Vt
        # => S = U^T @ state @ Vt^T
        S_list = []

        for b in range(B):
            S_batch = []

            for h in range(Hv):
                state_slice = state[b, h].astype(mx.float32)

                # Project: U^T @ state @ Vt^T
                # (rank, Dv) @ (Dv, Dk) @ (Dk, rank) = (rank, rank)
                proj = U[b, h].T @ state_slice @ Vt[b, h].T

                # Extract diagonal as singular values
                S_batch.append(mx.diag(proj))

            S_list.append(mx.stack(S_batch))

        S = mx.stack(S_list)

        self.stats['svd_avoided'] += 1

        return {
            'U': U,
            'S': S,
            'Vt': Vt,
            'rank': self.rank,
            'original_shape': (B, Hv, Dv, Dk),
            'original_dtype': str(original_dtype),
            'cached': True
        }

    def compress(self, state: mx.array) -> Dict[str, mx.array]:
        """
        Compress SSM state (with optional caching).

        Args:
            state: (B, Hv, Dv, Dk) SSM state tensor

        Returns:
            compressed: Dict with keys 'U', 'S', 'Vt', 'cached'
        """
        if state is None:
            return None

        self.stats['total_compress'] += 1

        # Decide: full SVD or cached compression
        if self._should_recompute_svd(state):
            return self._full_svd_compress(state)
        else:
            return self._cached_compress(state)

    def decompress(self, compressed: Dict[str, mx.array]) -> mx.array:
        """
        Reconstruct SSM state from compressed representation.

        Args:
            compressed: Dict with keys 'U', 'S', 'Vt'

        Returns:
            state: (B, Hv, Dv, Dk) reconstructed state
        """
        if compressed is None:
            return None

        U = compressed['U']
        S = compressed['S']
        Vt = compressed['Vt']

        B, Hv, Dv, rank = U.shape
        Dk = Vt.shape[-1]

        # Reconstruct per slice
        state_list = []

        for b in range(B):
            state_batch = []

            for h in range(Hv):
                # Reconstruct: state[b,h] = U @ diag(S) @ Vt
                reconstructed = U[b, h] @ mx.diag(S[b, h]) @ Vt[b, h]
                state_batch.append(reconstructed)

            state_list.append(mx.stack(state_batch))

        state = mx.stack(state_list)

        # Convert back to original dtype if specified
        if 'original_dtype' in compressed:
            dtype_str = compressed['original_dtype']
            if 'bfloat16' in dtype_str:
                state = state.astype(mx.bfloat16)
            elif 'float16' in dtype_str:
                state = state.astype(mx.float16)

        return state

    def get_compression_ratio(self, original_shape: Tuple[int, ...]) -> float:
        """Calculate compression ratio"""
        B, Hv, Dv, Dk = original_shape

        original_size = B * Hv * Dv * Dk
        compressed_size = (
            B * Hv * Dv * self.rank +  # U
            B * Hv * self.rank +        # S
            B * Hv * self.rank * Dk     # Vt
        )

        return original_size / compressed_size

    def get_cache_stats(self) -> Dict[str, float]:
        """
        Get caching statistics.

        Returns:
            stats: {
                'total': total compressions,
                'cache_hit_rate': % of cache hits,
                'svd_avoided': number of SVD computations avoided
            }
        """
        total = self.stats['total_compress']

        if total == 0:
            return {
                'total': 0,
                'cache_hit_rate': 0.0,
                'svd_avoided': 0
            }

        cache_hit_rate = self.stats['cache_hit'] / total * 100

        return {
            'total': total,
            'cache_hit_rate': cache_hit_rate,
            'cache_hit': self.stats['cache_hit'],
            'cache_miss': self.stats['cache_miss'],
            'svd_avoided': self.stats['svd_avoided']
        }

    def reset_cache(self):
        """Reset cache (for testing or when switching prompts)"""
        self.cached_U = None
        self.cached_Vt = None
        self.last_state_norm = None

        self.stats = {
            'total_compress': 0,
            'cache_hit': 0,
            'cache_miss': 0,
            'svd_avoided': 0
        }
