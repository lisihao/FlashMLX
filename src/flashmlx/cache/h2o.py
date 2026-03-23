"""
H2O: Heavy-Hitter Oracle for Efficient Generative Inference

Paper: https://arxiv.org/abs/2306.14048
Authors: Tsinghua & CMU (ICML 2023)

Core Idea:
- Track cumulative attention scores for each token
- Keep tokens with highest accumulated attention (heavy hitters)
- Also keep recent tokens for locality
- Evict low-attention tokens dynamically

Key Insight:
- A small fraction of tokens (heavy hitters) receive most attention
- These tokens are crucial for generation quality
- Can be identified via online statistics without lookahead

Advantages over StreamingLLM:
- Adaptive: Identifies important tokens automatically
- No assumption about position-based importance
- Better quality on diverse attention patterns
"""

from typing import Optional, Tuple, List
import mlx.core as mx


class H2OCache:
    """
    H2O (Heavy-Hitter Oracle) KV Cache with adaptive eviction.

    Memory layout:
        [heavy_hitters] + [recent_window]

    The cache dynamically adjusts which tokens to keep based on:
    1. Cumulative attention scores (heavy hitters)
    2. Recency (recent window)

    Parameters
    ----------
    max_capacity : int
        Maximum cache size (e.g., 256 tokens)
    recent_ratio : float
        Fraction of cache reserved for recent tokens (default: 0.25)
        Example: max_capacity=256, recent_ratio=0.25 → 64 recent, 192 heavy hitters
    accumulation_window : int
        Number of recent steps to accumulate attention over (default: 32)
        Prevents stale tokens from staying forever

    Attributes
    ----------
    k_cache, v_cache : List[mx.array]
        Key/Value cache (position-indexed)
    attention_scores : mx.array
        Cumulative attention scores for each position
    position_map : dict
        Maps cache position to original sequence position
    """

    def __init__(
        self,
        max_capacity: int = 256,
        recent_ratio: float = 0.25,
        accumulation_window: int = 32
    ):
        if not 0 < recent_ratio < 1:
            raise ValueError(f"recent_ratio must be in (0, 1), got {recent_ratio}")

        self.max_capacity = max_capacity
        self.recent_size = int(max_capacity * recent_ratio)
        self.heavy_size = max_capacity - self.recent_size
        self.accumulation_window = accumulation_window

        # Cache storage
        self.k_cache: List[mx.array] = []
        self.v_cache: List[mx.array] = []

        # Attention tracking
        self.attention_scores: mx.array = mx.array([])  # Cumulative scores
        self.step_counter = 0  # For windowed accumulation

        # Statistics
        self.total_tokens_seen = 0
        self.eviction_count = 0

    def update_attention(self, attention_weights: mx.array) -> None:
        """
        Update cumulative attention scores.

        Parameters
        ----------
        attention_weights : mx.array, shape (current_cache_size,)
            Attention weights from most recent step
            Must match current cache size
        """
        cache_size = len(self.k_cache)

        if attention_weights.shape[0] != cache_size:
            raise ValueError(
                f"attention_weights size ({attention_weights.shape[0]}) "
                f"must match cache size ({cache_size})"
            )

        # Initialize or reset accumulation
        if self.attention_scores.size == 0 or self.attention_scores.shape[0] != cache_size:
            self.attention_scores = mx.zeros((cache_size,), dtype=attention_weights.dtype)
            self.step_counter = 0

        # Accumulate attention
        self.attention_scores = self.attention_scores + attention_weights
        self.step_counter += 1

        # Windowed accumulation: decay old scores
        if self.step_counter >= self.accumulation_window:
            # Decay: keep recent portion of cumulative attention
            decay_factor = 0.9
            self.attention_scores = self.attention_scores * decay_factor
            self.step_counter = 0

    def append(
        self,
        k: mx.array,
        v: mx.array,
        attention_weights: Optional[mx.array] = None
    ) -> None:
        """
        Append a new key-value pair with optional attention update.

        If cache is full, evicts the token with lowest cumulative attention
        that is not in the recent window.

        Parameters
        ----------
        k : mx.array, shape (d,)
            Key vector
        v : mx.array, shape (d,)
            Value vector
        attention_weights : mx.array, optional, shape (current_size,)
            Attention weights from this step (for score tracking)
        """
        self.total_tokens_seen += 1

        # Update attention statistics if provided
        if attention_weights is not None and len(self.k_cache) > 0:
            self.update_attention(attention_weights)

        # Add new token
        self.k_cache.append(k)
        self.v_cache.append(v)

        # Evict if over capacity
        if len(self.k_cache) > self.max_capacity:
            self._evict_one()

    def _evict_one(self) -> None:
        """
        Evict one token using H2O policy:
        1. Identify recent window (last recent_size tokens) - never evict these
        2. Find token with lowest attention score outside recent window
        3. Evict that token
        """
        cache_size = len(self.k_cache)

        if cache_size <= self.max_capacity:
            return  # No eviction needed

        # Identify protected region (recent tokens)
        recent_start = cache_size - self.recent_size

        # Find eviction candidate: lowest attention outside recent window
        # Only consider tokens in [0, recent_start)
        if recent_start <= 0:
            # Edge case: all tokens are recent, evict oldest
            evict_idx = 0
        else:
            # Get attention scores for evictable region
            evictable_scores = self.attention_scores[:recent_start]

            # Find minimum
            evict_idx = int(mx.argmin(evictable_scores))

        # Evict token at evict_idx
        self.k_cache.pop(evict_idx)
        self.v_cache.pop(evict_idx)

        # Update attention scores (remove evicted position)
        if self.attention_scores.size > 0:
            # Remove score at evict_idx
            scores_list = list(self.attention_scores)
            scores_list.pop(evict_idx)
            self.attention_scores = mx.array(scores_list) if scores_list else mx.array([])

        self.eviction_count += 1

    def get_cache(self) -> Tuple[mx.array, mx.array]:
        """
        Get the full cache as contiguous arrays.

        Returns
        -------
        K : mx.array, shape (current_size, d)
            Keys
        V : mx.array, shape (current_size, d)
            Values
        """
        if not self.k_cache:
            return mx.array([]), mx.array([])

        K = mx.stack(self.k_cache, axis=0)
        V = mx.stack(self.v_cache, axis=0)

        return K, V

    def __len__(self) -> int:
        """Current cache size."""
        return len(self.k_cache)

    def reset(self) -> None:
        """Clear the cache."""
        self.k_cache.clear()
        self.v_cache.clear()
        self.attention_scores = mx.array([])
        self.step_counter = 0
        self.total_tokens_seen = 0
        self.eviction_count = 0

    def info(self) -> dict:
        """Get cache statistics."""
        cache_size = len(self)

        # Analyze attention distribution
        if self.attention_scores.size > 0:
            recent_start = max(0, cache_size - self.recent_size)

            heavy_scores = (
                self.attention_scores[:recent_start]
                if recent_start > 0
                else mx.array([])
            )
            recent_scores = (
                self.attention_scores[recent_start:]
                if recent_start < cache_size
                else mx.array([])
            )

            return {
                'current_size': cache_size,
                'max_capacity': self.max_capacity,
                'heavy_size': len(heavy_scores) if heavy_scores.size > 0 else 0,
                'recent_size': len(recent_scores) if recent_scores.size > 0 else 0,
                'total_tokens_seen': self.total_tokens_seen,
                'eviction_count': self.eviction_count,
                'avg_attention_heavy': float(mx.mean(heavy_scores)) if heavy_scores.size > 0 else 0.0,
                'avg_attention_recent': float(mx.mean(recent_scores)) if recent_scores.size > 0 else 0.0,
                'step_counter': self.step_counter,
                'accumulation_window': self.accumulation_window
            }
        else:
            return {
                'current_size': cache_size,
                'max_capacity': self.max_capacity,
                'heavy_size': 0,
                'recent_size': 0,
                'total_tokens_seen': self.total_tokens_seen,
                'eviction_count': self.eviction_count
            }


def h2o_compress(
    K: mx.array,
    V: mx.array,
    queries: mx.array,
    max_capacity: int = 256,
    recent_ratio: float = 0.25
) -> Tuple[mx.array, mx.array, List[int]]:
    """
    Compress KV cache using H2O policy (one-shot version).

    This simulates H2O eviction by:
    1. Computing attention for all queries
    2. Accumulating attention scores
    3. Keeping top heavy hitters + recent tokens

    Parameters
    ----------
    K : mx.array, shape (T, d)
        Original keys
    V : mx.array, shape (T, d)
        Original values
    queries : mx.array, shape (n, d)
        Query vectors for computing attention
    max_capacity : int
        Target cache size
    recent_ratio : float
        Fraction reserved for recent tokens

    Returns
    -------
    K_compressed : mx.array, shape (t, d)
        Compressed keys
    V_compressed : mx.array, shape (t, d)
        Compressed values
    indices : List[int]
        Indices of kept tokens
    """
    T, d = K.shape
    n = queries.shape[0]

    if T <= max_capacity:
        return K, V, list(range(T))

    # Compute attention for all queries and accumulate
    scale = 1.0 / mx.sqrt(mx.array(d, dtype=K.dtype))
    attention_weights = mx.softmax(queries @ K.T * scale, axis=-1)  # (n, T)

    # Accumulate attention across queries
    cumulative_attention = mx.sum(attention_weights, axis=0)  # (T,)

    # Determine sizes
    recent_size = int(max_capacity * recent_ratio)
    heavy_size = max_capacity - recent_size

    # Select heavy hitters from non-recent tokens
    recent_start = T - recent_size
    if recent_start <= 0:
        # All tokens are recent
        indices = list(range(T))
    else:
        # Identify heavy hitters in [0, recent_start)
        evictable_scores = cumulative_attention[:recent_start]
        heavy_indices = mx.argsort(-evictable_scores)[:heavy_size]  # Top-k
        heavy_indices = sorted([int(i) for i in heavy_indices])

        # Recent tokens
        recent_indices = list(range(recent_start, T))

        # Combine
        indices = heavy_indices + recent_indices

    # Extract compressed cache
    K_compressed = K[indices]
    V_compressed = V[indices]

    return K_compressed, V_compressed, indices


def test_h2o_quality(
    K: mx.array,
    V: mx.array,
    queries: mx.array,
    max_capacity: int = 256,
    recent_ratio: float = 0.25
) -> dict:
    """
    Test H2O compression quality.

    Parameters
    ----------
    K : mx.array, shape (T, d)
        Original keys
    V : mx.array, shape (T, d)
        Original values
    queries : mx.array, shape (n, d)
        Test queries
    max_capacity : int
        Target cache size
    recent_ratio : float
        Recent window ratio

    Returns
    -------
    results : dict
        Quality metrics and statistics
    """
    T, d = K.shape
    n = queries.shape[0]

    # Compress
    K_comp, V_comp, indices = h2o_compress(
        K, V, queries, max_capacity, recent_ratio
    )
    t = K_comp.shape[0]

    # Compute outputs
    scale = 1.0 / mx.sqrt(mx.array(d, dtype=K.dtype))

    # Original
    attn_orig = mx.softmax(queries @ K.T * scale, axis=-1)
    out_orig = attn_orig @ V

    # Compressed
    attn_comp = mx.softmax(queries @ K_comp.T * scale, axis=-1)
    out_comp = attn_comp @ V_comp

    # Quality
    out_orig_flat = mx.reshape(out_orig, (-1,))
    out_comp_flat = mx.reshape(out_comp, (-1,))
    cos_sim = float(
        mx.sum(out_orig_flat * out_comp_flat) /
        (mx.linalg.norm(out_orig_flat) * mx.linalg.norm(out_comp_flat))
    )
    mse = float(mx.mean((out_orig - out_comp) ** 2))

    # Analyze kept tokens
    recent_size = int(max_capacity * recent_ratio)
    heavy_size = t - recent_size

    # Attention to heavy vs recent
    attn_to_heavy = float(mx.mean(mx.sum(attn_comp[:, :heavy_size], axis=1))) if heavy_size > 0 else 0.0
    attn_to_recent = float(mx.mean(mx.sum(attn_comp[:, heavy_size:], axis=1))) if heavy_size < t else 0.0

    return {
        'compression_ratio': T / t,
        'original_size': T,
        'compressed_size': t,
        'heavy_hitters': heavy_size,
        'recent_window': recent_size,
        'cosine_similarity': cos_sim,
        'mse': mse,
        'attention_to_heavy': attn_to_heavy,
        'attention_to_recent': attn_to_recent,
        'kept_indices': indices
    }
