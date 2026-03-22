"""
Attention Matching Compressor

Based on the paper: "Fast KV Compaction via Attention Matching"
https://github.com/adamzweiger/compaction

Core idea:
1. Identify important KV pairs through attention weights
2. Retain high-weight keys, evict low-weight keys
3. Use β compensation to correct compressed attention distribution

This module implements the core compression logic (Task #66).
β calibration is implemented in Task #67.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import mlx.core as mx


class AttentionMatchingCompressor:
    """
    Attention Matching Compressor for KV Cache compression.

    Compresses KV cache for Attention layers based on attention weights,
    following the methodology from "Fast KV Compaction via Attention Matching".

    Args:
        compression_ratio: Target compression ratio (e.g., 2.0 means compress to 50% of original size)
        beta_calibration: Whether to enable β calibration (implemented in Task #67)
        eviction_policy: Key eviction strategy ("top_k" or "weighted")
        attention_history_window: Number of recent steps to average attention weights

    Example:
        >>> compressor = AttentionMatchingCompressor(compression_ratio=2.0)
        >>> keys = mx.random.normal((1, 8, 100, 64))  # (batch, heads, seq_len, head_dim)
        >>> values = mx.random.normal((1, 8, 100, 64))
        >>> compressed_keys, compressed_values = compressor.compress_kv_cache(
        ...     layer_idx=0,
        ...     kv_cache=(keys, values)
        ... )
        >>> compressed_keys.shape[2]  # 50 (100 / 2.0)
    """

    def __init__(
        self,
        compression_ratio: float = 2.0,
        beta_calibration: bool = True,
        eviction_policy: str = "top_k",
        attention_history_window: int = 10
    ):
        """Initialize the Attention Matching Compressor."""
        if compression_ratio < 1.0:
            raise ValueError(f"compression_ratio must be >= 1.0, got {compression_ratio}")

        if eviction_policy not in ["top_k", "weighted"]:
            raise ValueError(f"eviction_policy must be 'top_k' or 'weighted', got {eviction_policy}")

        self.compression_ratio = compression_ratio
        self.beta_calibration = beta_calibration
        self.eviction_policy = eviction_policy
        self.attention_history_window = attention_history_window

        # Attention weights history for each layer
        # Format: {layer_idx: [weights_step1, weights_step2, ...]}
        self.attention_history: Dict[int, List[np.ndarray]] = {}

        # β calibration parameters (Task #67)
        self.beta_params: Dict[int, float] = {}

        # Statistics
        self.compression_stats = {
            "total_compressions": 0,
            "total_keys_before": 0,
            "total_keys_after": 0,
        }

    def compress_kv_cache(
        self,
        layer_idx: int,
        kv_cache: Tuple[mx.array, mx.array]
    ) -> Tuple[mx.array, mx.array]:
        """
        Compress KV cache for a specific layer.

        Args:
            layer_idx: Index of the Attention layer
            kv_cache: Tuple of (keys, values) arrays
                      Shape: (batch_size, num_heads, seq_len, head_dim)

        Returns:
            Compressed (keys, values) tuple
            Shape: (batch_size, num_heads, compressed_seq_len, head_dim)
        """
        keys, values = kv_cache
        batch_size, num_heads, seq_len, head_dim = keys.shape

        # Validate shapes
        if values.shape != keys.shape:
            raise ValueError(f"Keys and values must have same shape, got {keys.shape} vs {values.shape}")

        # Calculate target sequence length after compression
        target_seq_len = max(1, int(seq_len / self.compression_ratio))

        # If sequence is too short to compress meaningfully, return as-is
        # (need at least compression_ratio tokens to achieve target compression)
        if seq_len < self.compression_ratio or seq_len <= target_seq_len:
            # Track no compression for statistics
            self.compression_stats['total_compressions'] += 1
            self.compression_stats['total_keys_before'] += seq_len
            self.compression_stats['total_keys_after'] += seq_len
            return keys, values

        # Step 1: Compute average attention weights (recent N steps)
        avg_attention_weights = self._compute_avg_attention_weights(
            layer_idx=layer_idx,
            keys=keys
        )

        # Step 2: Select keys to keep based on attention weights
        keep_indices = self._select_keys_to_keep(
            attention_weights=avg_attention_weights,
            target_count=target_seq_len,
            eviction_policy=self.eviction_policy
        )

        # Step 3: Compress keys and values
        # Convert numpy indices to MLX array
        keep_indices_mx = mx.array(keep_indices, dtype=mx.int32)
        compressed_keys = mx.take(keys, keep_indices_mx, axis=2)
        compressed_values = mx.take(values, keep_indices_mx, axis=2)

        # Step 4: β calibration
        if self.beta_calibration:
            beta = self._calibrate_beta(
                layer_idx=layer_idx,
                original_weights=avg_attention_weights,
                keep_indices=keep_indices
            )
            self.beta_params[layer_idx] = beta

        # Update statistics
        self.compression_stats["total_compressions"] += 1
        self.compression_stats["total_keys_before"] += seq_len
        self.compression_stats["total_keys_after"] += target_seq_len

        return compressed_keys, compressed_values

    def _compute_avg_attention_weights(
        self,
        layer_idx: int,
        keys: mx.array
    ) -> np.ndarray:
        """
        Compute average attention weights over recent steps.

        In a real implementation, attention weights would be collected during forward pass.
        For now, we use a simplified heuristic based on key norms as a proxy for importance.

        Args:
            layer_idx: Layer index
            keys: Keys array (batch, heads, seq_len, head_dim)

        Returns:
            Average attention weights (seq_len,)
        """
        # Initialize history if needed
        if layer_idx not in self.attention_history:
            self.attention_history[layer_idx] = []

        # Compute key importance heuristic: L2 norm across head_dim
        # Shape: (batch, heads, seq_len)
        key_norms = mx.sqrt(mx.sum(keys ** 2, axis=-1))

        # Average across batch and heads
        # Shape: (seq_len,)
        avg_key_norms = mx.mean(key_norms, axis=(0, 1))

        # Convert to numpy for history storage
        # Use tolist() to avoid MLX -> numpy buffer format issues
        current_weights = np.array(avg_key_norms.tolist())

        # Normalize to sum to 1 (simulating softmax attention distribution)
        current_weights = current_weights / current_weights.sum()

        # Add to history
        self.attention_history[layer_idx].append(current_weights)

        # Keep only recent N steps
        if len(self.attention_history[layer_idx]) > self.attention_history_window:
            self.attention_history[layer_idx] = self.attention_history[layer_idx][-self.attention_history_window:]

        # Compute average over recent steps with same sequence length
        # (handles cases where sequence length changes between calls)
        recent_weights = self.attention_history[layer_idx]
        current_seq_len = len(current_weights)

        # Filter history to only include weights with matching length
        matching_weights = [w for w in recent_weights if len(w) == current_seq_len]

        if matching_weights:
            avg_weights = np.mean(matching_weights, axis=0)
        else:
            # No matching history, use current weights
            avg_weights = current_weights

        return avg_weights

    def _select_keys_to_keep(
        self,
        attention_weights: np.ndarray,
        target_count: int,
        eviction_policy: str
    ) -> np.ndarray:
        """
        Select which keys to keep based on attention weights.

        Args:
            attention_weights: Attention weight distribution (seq_len,)
            target_count: Number of keys to keep
            eviction_policy: "top_k" or "weighted"

        Returns:
            Indices of keys to keep (sorted)
        """
        seq_len = len(attention_weights)

        if target_count >= seq_len:
            return np.arange(seq_len)

        if eviction_policy == "top_k":
            # Paper's default strategy: keep top-k keys with highest attention weights
            keep_indices = np.argsort(attention_weights)[-target_count:]
            # Sort to maintain original order
            return np.sort(keep_indices)

        elif eviction_policy == "weighted":
            # Weighted random sampling (exploration vs exploitation)
            # Normalize weights to probabilities
            probabilities = attention_weights / attention_weights.sum()

            # Sample without replacement
            keep_indices = np.random.choice(
                seq_len,
                size=target_count,
                replace=False,
                p=probabilities
            )

            # Sort to maintain original order
            return np.sort(keep_indices)

        else:
            raise ValueError(f"Unknown eviction policy: {eviction_policy}")

    def _calibrate_beta(
        self,
        layer_idx: int,
        original_weights: np.ndarray,
        keep_indices: np.ndarray
    ) -> float:
        """
        Calibrate β parameter to correct compressed attention distribution.

        Based on the paper's approach: minimize KL divergence between
        original and compressed attention distributions.

        The β parameter compensates for the change in the normalization constant
        when keys are removed. Mathematically:

        β ≈ log(sum(exp(original_scores)) / sum(exp(compressed_scores)))

        Args:
            layer_idx: Layer index
            original_weights: Original attention weight distribution (seq_len,)
            keep_indices: Indices of kept keys

        Returns:
            β parameter (scalar)
        """
        # Extract compressed weights
        compressed_weights = original_weights[keep_indices]

        # Normalize to probabilities
        original_probs = original_weights / original_weights.sum()
        compressed_probs = compressed_weights / compressed_weights.sum()

        # Method 1: Simple ratio-based β (fast approximation)
        # β = log(N_original / N_compressed)
        seq_len_original = len(original_weights)
        seq_len_compressed = len(compressed_weights)

        if seq_len_compressed == 0:
            return 0.0

        # Simple approximation based on sequence length ratio
        beta_simple = np.log(seq_len_original / seq_len_compressed)

        # Method 2: Distribution-aware β (more accurate)
        # Account for the actual distribution of weights
        # β = log(sum(original_weights) / sum(compressed_weights))
        original_sum = original_weights.sum()
        compressed_sum = compressed_weights.sum()

        if compressed_sum > 0:
            beta_distribution = np.log(original_sum / compressed_sum)
        else:
            beta_distribution = 0.0

        # Use a weighted combination of both methods
        # Distribution-aware is more accurate, simple provides stability
        beta = 0.7 * beta_distribution + 0.3 * beta_simple

        return float(beta)

    def apply_beta_compensation(
        self,
        layer_idx: int,
        attention_scores: mx.array
    ) -> mx.array:
        """
        Apply β compensation to attention scores.

        The β parameter is added to attention scores (logits) before softmax
        to correct for the changed normalization constant after compression.

        Mathematical justification:
        - Original: softmax(scores)
        - After compression: softmax(scores + β)
        - β compensates for removed keys

        Args:
            layer_idx: Layer index
            attention_scores: Attention scores before softmax
                              Shape: (batch, heads, query_len, key_len)

        Returns:
            Compensated attention scores (same shape)

        Example:
            >>> # In attention forward pass (after compression)
            >>> scores = query @ compressed_keys.transpose(-2, -1)
            >>> scores = compressor.apply_beta_compensation(layer_idx, scores)
            >>> attn_weights = mx.softmax(scores, axis=-1)
        """
        if layer_idx not in self.beta_params:
            # No β available for this layer (not compressed yet)
            return attention_scores

        beta = self.beta_params[layer_idx]

        # Apply β compensation: add β to all attention scores
        # This shifts the entire distribution to account for removed keys
        compensated_scores = attention_scores + beta

        return compensated_scores

    def get_compression_stats(self) -> Dict[str, float]:
        """
        Get compression statistics.

        Returns:
            Dictionary with compression metrics
        """
        total_before = self.compression_stats["total_keys_before"]
        total_after = self.compression_stats["total_keys_after"]

        avg_compression_ratio = total_before / total_after if total_after > 0 else 0.0

        return {
            "total_compressions": self.compression_stats["total_compressions"],
            "total_keys_before": total_before,
            "total_keys_after": total_after,
            "avg_compression_ratio": avg_compression_ratio,
            "configured_compression_ratio": self.compression_ratio,
        }

    def reset_history(self, layer_idx: Optional[int] = None):
        """
        Reset attention weight history.

        Args:
            layer_idx: If specified, reset only this layer. Otherwise, reset all layers.
        """
        if layer_idx is not None:
            if layer_idx in self.attention_history:
                self.attention_history[layer_idx] = []
        else:
            self.attention_history = {}

    def __repr__(self) -> str:
        return (
            f"AttentionMatchingCompressor("
            f"compression_ratio={self.compression_ratio}, "
            f"beta_calibration={self.beta_calibration}, "
            f"eviction_policy={self.eviction_policy})"
        )
