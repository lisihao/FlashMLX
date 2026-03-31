"""
StreamingLLM: Efficient Streaming Language Models with Attention Sinks

Paper: https://arxiv.org/abs/2309.17453
Authors: MIT & Meta AI (2023)

Core Idea:
- Keep initial tokens (attention sinks) + recent tokens
- Discard middle tokens
- Enables infinite-length generation with fixed memory

Key Finding:
- Initial tokens (especially BOS) absorb large attention weights
- Recent tokens contain most relevant context
- Middle tokens contribute little to generation quality
"""

from typing import Optional, Tuple, List
import mlx.core as mx


class StreamingLLMCache:
    """
    StreamingLLM KV Cache with attention sinks and sliding window.

    Memory layout:
        [sink_0, sink_1, ..., sink_n] + [recent_0, recent_1, ..., recent_m]
        |<------ attention sinks ----->|  |<----- sliding window -------->|

    Parameters
    ----------
    max_capacity : int
        Maximum cache size (e.g., 256 tokens)
    num_sinks : int
        Number of initial tokens to keep as attention sinks (default: 4)
        These are typically special tokens like BOS that absorb attention

    Attributes
    ----------
    sinks_k, sinks_v : List[mx.array]
        Key/Value vectors for attention sinks
    recent_k, recent_v : List[mx.array]
        Key/Value vectors for recent context
    """

    def __init__(self, max_capacity: int = 256, num_sinks: int = 4):
        if num_sinks >= max_capacity:
            raise ValueError(f"num_sinks ({num_sinks}) must be < max_capacity ({max_capacity})")

        self.max_capacity = max_capacity
        self.num_sinks = num_sinks
        self.window_size = max_capacity - num_sinks

        # Storage for attention sinks
        self.sinks_k: List[mx.array] = []
        self.sinks_v: List[mx.array] = []

        # Storage for recent tokens (sliding window)
        self.recent_k: List[mx.array] = []
        self.recent_v: List[mx.array] = []

        self.total_tokens_seen = 0

    def append(self, k: mx.array, v: mx.array) -> None:
        """
        Append a new key-value pair to the cache.

        Eviction policy:
        - First num_sinks tokens → stored as sinks (never evicted)
        - Subsequent tokens → stored in sliding window
        - When window is full → evict oldest recent token

        Parameters
        ----------
        k : mx.array, shape (d,)
            Key vector
        v : mx.array, shape (d,)
            Value vector
        """
        self.total_tokens_seen += 1

        # First num_sinks tokens become attention sinks
        if len(self.sinks_k) < self.num_sinks:
            self.sinks_k.append(k)
            self.sinks_v.append(v)
        else:
            # Add to recent window
            self.recent_k.append(k)
            self.recent_v.append(v)

            # Evict oldest if window is full
            if len(self.recent_k) > self.window_size:
                self.recent_k.pop(0)
                self.recent_v.pop(0)

    def get_cache(self) -> Tuple[mx.array, mx.array]:
        """
        Get the full cache as contiguous arrays.

        Returns
        -------
        K : mx.array, shape (current_size, d)
            Concatenated keys [sinks + recent]
        V : mx.array, shape (current_size, d)
            Concatenated values [sinks + recent]
        """
        if not self.sinks_k and not self.recent_k:
            # Empty cache
            return mx.array([]), mx.array([])

        all_k = self.sinks_k + self.recent_k
        all_v = self.sinks_v + self.recent_v

        K = mx.stack(all_k, axis=0)  # (current_size, d)
        V = mx.stack(all_v, axis=0)

        return K, V

    def __len__(self) -> int:
        """Current cache size."""
        return len(self.sinks_k) + len(self.recent_k)

    def reset(self) -> None:
        """Clear the cache."""
        self.sinks_k.clear()
        self.sinks_v.clear()
        self.recent_k.clear()
        self.recent_v.clear()
        self.total_tokens_seen = 0

    def info(self) -> dict:
        """Get cache statistics."""
        return {
            'current_size': len(self),
            'max_capacity': self.max_capacity,
            'num_sinks': self.num_sinks,
            'num_recent': len(self.recent_k),
            'window_size': self.window_size,
            'total_tokens_seen': self.total_tokens_seen,
            'evicted_tokens': max(0, self.total_tokens_seen - len(self))
        }


def streaming_llm_compress(
    K: mx.array,
    V: mx.array,
    max_capacity: int = 256,
    num_sinks: int = 4
) -> Tuple[mx.array, mx.array, List[int]]:
    """
    Compress KV cache using StreamingLLM policy.

    This is a functional API for one-shot compression of existing cache.
    For incremental/streaming usage, use StreamingLLMCache class.

    Parameters
    ----------
    K : mx.array, shape (T, d)
        Original keys
    V : mx.array, shape (T, d)
        Original values
    max_capacity : int
        Target cache size
    num_sinks : int
        Number of initial tokens to keep

    Returns
    -------
    K_compressed : mx.array, shape (t, d)
        Compressed keys
    V_compressed : mx.array, shape (t, d)
        Compressed values
    indices : List[int]
        Indices of kept tokens
    """
    T = K.shape[0]

    if T <= max_capacity:
        # No compression needed
        return K, V, list(range(T))

    # Keep first num_sinks (attention sinks)
    sink_indices = list(range(num_sinks))

    # Keep last (max_capacity - num_sinks) tokens (recent context)
    window_size = max_capacity - num_sinks
    recent_indices = list(range(T - window_size, T))

    # Combine indices
    indices = sink_indices + recent_indices

    # Extract compressed cache
    K_compressed = K[indices]
    V_compressed = V[indices]

    return K_compressed, V_compressed, indices


# ============================================================================
# Quality Testing Utilities
# ============================================================================

def test_streaming_llm_quality(
    K: mx.array,
    V: mx.array,
    queries: mx.array,
    max_capacity: int = 256,
    num_sinks: int = 4
) -> dict:
    """
    Test StreamingLLM compression quality.

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
    num_sinks : int
        Number of attention sinks

    Returns
    -------
    results : dict
        Quality metrics and statistics
    """
    T, d = K.shape
    n = queries.shape[0]

    # Compress
    K_comp, V_comp, indices = streaming_llm_compress(
        K, V, max_capacity, num_sinks
    )
    t = K_comp.shape[0]

    # Compute outputs
    scale = 1.0 / mx.sqrt(mx.array(d, dtype=K.dtype))

    # Original output
    attn_orig = mx.softmax(queries @ K.T * scale, axis=-1)  # (n, T)
    out_orig = attn_orig @ V  # (n, d)

    # Compressed output
    attn_comp = mx.softmax(queries @ K_comp.T * scale, axis=-1)  # (n, t)
    out_comp = attn_comp @ V_comp  # (n, d)

    # Cosine similarity
    out_orig_flat = mx.reshape(out_orig, (-1,))
    out_comp_flat = mx.reshape(out_comp, (-1,))
    cos_sim = float(
        mx.sum(out_orig_flat * out_comp_flat) /
        (mx.linalg.norm(out_orig_flat) * mx.linalg.norm(out_comp_flat))
    )

    # MSE
    mse = float(mx.mean((out_orig - out_comp) ** 2))

    # Attention distribution analysis
    # How much attention goes to sinks vs recent?
    attn_to_sinks = float(mx.mean(mx.sum(attn_comp[:, :num_sinks], axis=1)))
    attn_to_recent = float(mx.mean(mx.sum(attn_comp[:, num_sinks:], axis=1)))

    return {
        'compression_ratio': T / t,
        'original_size': T,
        'compressed_size': t,
        'num_sinks': num_sinks,
        'window_size': t - num_sinks,
        'cosine_similarity': cos_sim,
        'mse': mse,
        'attention_to_sinks': attn_to_sinks,
        'attention_to_recent': attn_to_recent,
        'kept_indices': indices
    }
