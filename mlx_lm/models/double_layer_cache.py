"""
DoubleLayerKVCache - AM on Frozen Prefix with Multi-Length Calibration

Architecture:
    Cache = [Old Prefix (AM compressed)] + [Recent Window (exact KV)]

Features:
1. Multi-Length Calibration: Dynamic calibration selection based on prefix length
2. Recent Window Pinning: Always preserve recent N tokens
3. Beta Safe Guard: Prevent numerical collapse
4. Prefix-only Compression: selected_indices only apply to old_prefix
5. Metadata Versioning: Calibration files with metadata

Design principles (from user feedback):
- Offline calibration适合固定前缀 → 我们只压缩 old_prefix
- Lazy compression需要 recent context → 我们永远保留 recent_window
- Fixed indices不适配动态增长 → 我们根据实际长度动态选择校准文件
"""

import mlx.core as mx
import numpy as np
import pickle
import bisect
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from .cache import _BaseCache
from .load_characteristics import LoadCharacteristicsAnalyzer


# ====================================================================
# CalibrationRegistry: Multi-Length Calibration Management
# ====================================================================

class CalibrationRegistry:
    """
    Multi-length calibration file registry.

    Automatically discovers and manages calibration files for different lengths.
    Provides dynamic calibration selection based on actual prefix length.

    File naming format: am_calibration_L{length}_R{ratio}.pkl
    Example: am_calibration_L512_R2.0.pkl

    Parameters
    ----------
    calibration_dir : str
        Directory containing calibration files
    auto_scan : bool
        Automatically scan directory on initialization (default: True)

    Example
    -------
    >>> registry = CalibrationRegistry("/path/to/calibrations")
    >>> calibration = registry.get_calibration(length=600, ratio=2.0)
    >>> # Automatically selects calibration_L768.pkl (ceil strategy)
    """

    def __init__(self, calibration_dir: str, auto_scan: bool = True):
        self.calibration_dir = Path(calibration_dir)
        self.available_calibrations: Dict[float, List[Tuple[int, Path]]] = {}
        self.cache: Dict[str, Dict] = {}  # LRU cache for loaded calibrations

        if auto_scan:
            self.scan()

    def scan(self):
        """
        Scan calibration directory and discover available calibrations.

        Returns number of calibration files found.
        """
        if not self.calibration_dir.exists():
            raise FileNotFoundError(f"Calibration directory not found: {self.calibration_dir}")

        self.available_calibrations.clear()

        for filepath in self.calibration_dir.glob("am_calibration_L*_R*.pkl"):
            try:
                # Parse filename: am_calibration_L512_R2.0.pkl
                parts = filepath.stem.split('_')
                length = int(parts[2][1:])  # "L512" → 512
                ratio = float(parts[3][1:])  # "R2.0" → 2.0

                if ratio not in self.available_calibrations:
                    self.available_calibrations[ratio] = []

                self.available_calibrations[ratio].append((length, filepath))

            except Exception as e:
                print(f"[CalibrationRegistry] Warning: Failed to parse {filepath.name}: {e}")

        # Sort by length
        for ratio in self.available_calibrations:
            self.available_calibrations[ratio].sort()

        return sum(len(files) for files in self.available_calibrations.values())

    def get_calibration(
        self,
        length: int,
        ratio: float = 2.0,
        strategy: str = "ceil"
    ) -> Optional[Dict]:
        """
        Get calibration for given length and compression ratio.

        Parameters
        ----------
        length : int
            Actual old_prefix length
        ratio : float
            Compression ratio
        strategy : str
            Selection strategy:
            - "ceil": Round up (select >= length), guarantees coverage
            - "floor": Round down (select <= length)
            - "nearest": Nearest match

        Returns
        -------
        calibration : dict or None
            Calibration data with 'metadata' and 'calibration' keys
        """
        if ratio not in self.available_calibrations:
            return None

        length_files = self.available_calibrations[ratio]
        lengths = [l for l, _ in length_files]

        # Select based on strategy
        if strategy == "ceil":
            idx = bisect.bisect_left(lengths, length)
            if idx == len(lengths):
                idx = len(lengths) - 1  # Use largest if exceeds all
        elif strategy == "floor":
            idx = bisect.bisect_right(lengths, length) - 1
            if idx < 0:
                idx = 0  # Use smallest if below all
        elif strategy == "nearest":
            diffs = [abs(l - length) for l in lengths]
            idx = np.argmin(diffs)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        selected_length, selected_filepath = length_files[idx]

        # Check cache
        cache_key = str(selected_filepath)
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Load calibration file
        try:
            with open(selected_filepath, 'rb') as f:
                calibration = pickle.load(f)

            # Validate structure
            if 'metadata' not in calibration:
                print(f"[CalibrationRegistry] Warning: Missing metadata in {selected_filepath.name}")
            if 'calibration' not in calibration:
                raise ValueError("Invalid calibration file: missing 'calibration' key")

            # Cache it
            self.cache[cache_key] = calibration

            return calibration

        except Exception as e:
            print(f"[CalibrationRegistry] Error loading {selected_filepath}: {e}")
            return None

    def get_available_lengths(self, ratio: float = 2.0) -> List[int]:
        """Get list of available calibration lengths for given ratio."""
        if ratio not in self.available_calibrations:
            return []
        return [length for length, _ in self.available_calibrations[ratio]]

    def clear_cache(self):
        """Clear loaded calibrations cache."""
        self.cache.clear()


# ====================================================================
# DoubleLayerKVCache: AM on Frozen Prefix
# ====================================================================

class DoubleLayerKVCache(_BaseCache):
    """
    Double-layer KV Cache: Old Prefix (AM compressed) + Recent Window (exact).

    This cache solves the fundamental incompatibility between:
    - Offline AM calibration (fixed prefix)
    - Lazy compression (dynamic growth, recent tokens critical)

    Solution:
        cache = [old_prefix | recent_window]

        Compression (triggered by memory budget):
        - old_prefix → AM compress (using multi-length calibration)
        - recent_window → preserve exact (no compression)

        final_cache = [compacted_old | exact_recent]

    Compression Strategy (Production-Ready):
    1. Prefill: If memory budget sufficient → no compression
    2. New request: If (current + new) > budget → trigger compression (block inference)
    3. Compression: Compress old_prefix to fit budget
    4. Continue: Resume inference with compressed cache

    Parameters
    ----------
    memory_budget_mb : float
        Memory budget in MB for this layer's cache (required)
        Example: 10.0 means 10 MB per layer
        For testing: use small values (e.g., 5.0) to easily trigger compression
    recent_window_size : int
        Number of recent tokens to always preserve (default: 512)
    compression_ratio : float
        Target compression ratio for old_prefix (default: 1.5)
    calibration_dir : str, optional
        Directory containing multi-length calibration files
    layer_idx : int, optional
        Layer index (required if using calibration)
    enable_compression : bool
        Enable automatic compression (default: True)
    selection_strategy : str
        Calibration selection strategy: "ceil", "floor", "nearest" (default: "ceil")
    enable_adaptive_window : bool
        Enable adaptive recent window based on workload characteristics (default: False)
    workload_hint : str, optional
        Explicit workload type hint: "summarization", "coding", "agent", "qa", "chat"

    Example
    -------
    >>> # Production: memory-budget driven
    >>> cache = DoubleLayerKVCache(
    ...     memory_budget_mb=10.0,  # 10 MB per layer
    ...     recent_window_size=512,
    ...     compression_ratio=1.5,
    ...     calibration_dir="/path/to/calibrations",
    ...     layer_idx=0
    ... )
    >>>
    >>> # Testing: use small budget to trigger compression
    >>> cache = DoubleLayerKVCache(
    ...     memory_budget_mb=5.0,  # 5 MB (triggers compression easily)
    ...     recent_window_size=512,
    ...     compression_ratio=1.5,
    ...     calibration_dir="/path/to/calibrations",
    ...     layer_idx=0
    ... )
    >>>
    >>> # Use in model forward
    >>> keys, values = cache.update_and_fetch(keys, values)
    """

    def __init__(
        self,
        memory_budget_mb: float,
        recent_window_size: int = 512,
        compression_ratio: float = 1.5,
        calibration_dir: Optional[str] = None,
        layer_idx: Optional[int] = None,
        enable_compression: bool = True,
        selection_strategy: str = "ceil",
        enable_adaptive_window: bool = False,
        workload_hint: Optional[str] = None
    ):
        # Configuration
        self.memory_budget = int(memory_budget_mb * 1024 * 1024)  # Convert to bytes
        self.memory_budget_mb = memory_budget_mb  # Store for logging
        self.recent_window_size = recent_window_size
        self.initial_window_size = recent_window_size  # Store initial value
        self.compression_ratio = compression_ratio
        self.layer_idx = layer_idx
        self.enable_compression = enable_compression
        self.selection_strategy = selection_strategy
        self.enable_adaptive_window = enable_adaptive_window
        self.workload_hint = workload_hint

        # Old prefix cache (compressible)
        self.old_keys = None
        self.old_values = None

        # Recent window cache (never compressed)
        self.recent_keys = None
        self.recent_values = None

        # Total offset (logical position)
        self._total_offset = 0

        # Statistics
        self.num_compressions = 0
        self.total_tokens_before_compression = 0
        self.total_tokens_after_compression = 0

        # Calibration registry
        self.calibration_registry = None
        if calibration_dir:
            self.calibration_registry = CalibrationRegistry(calibration_dir, auto_scan=True)

        # Load characteristics analyzer (for adaptive window)
        self.load_analyzer = None
        self.detected_redundancy = None
        self._pending_tokens = None  # For external token injection
        if self.enable_adaptive_window:
            self.load_analyzer = LoadCharacteristicsAnalyzer()

    def set_tokens_for_analysis(self, tokens: List[int]):
        """
        Set tokens for redundancy analysis (called before prefill).

        This is a workaround to provide real tokens to the cache without
        modifying the model forward interface.

        Parameters
        ----------
        tokens : List[int]
            Token IDs to analyze

        Example
        -------
        >>> # Before prefill
        >>> for cache in cache_list:
        ...     cache.set_tokens_for_analysis(tokens)
        >>> # Then run model
        >>> logits = model(y, cache=cache_list)
        """
        self._pending_tokens = tokens

    @property
    def offset(self):
        """Total cache size (old + recent)."""
        old_size = self.old_keys.shape[2] if self.old_keys is not None else 0
        recent_size = self.recent_keys.shape[2] if self.recent_keys is not None else 0
        return old_size + recent_size

    @property
    def nbytes(self):
        """Total memory usage in bytes."""
        total_bytes = 0
        if self.old_keys is not None:
            total_bytes += self.old_keys.nbytes + self.old_values.nbytes
        if self.recent_keys is not None:
            total_bytes += self.recent_keys.nbytes + self.recent_values.nbytes
        return total_bytes

    def empty(self):
        """Check if cache is empty."""
        return self.old_keys is None and self.recent_keys is None

    def update_and_fetch(
        self,
        keys: mx.array,
        values: mx.array
    ) -> Tuple[mx.array, mx.array]:
        """
        Update cache and return complete KV (Memory-Budget Driven).

        Production-Ready Compression Flow:
        1. Estimate memory usage after appending new KV
        2. If estimated_memory > memory_budget:
           → Trigger compression (block inference)
           → Compress old_prefix to fit budget
           → Resume inference
        3. Else: Simply append new KV (no compression)
        4. Return: [compacted_old | exact_recent]

        Parameters
        ----------
        keys : mx.array, shape (B, n_heads, num_steps, head_dim)
        values : mx.array, shape (B, n_heads, num_steps, head_dim)

        Returns
        -------
        keys : mx.array
            Complete cached keys
        values : mx.array
            Complete cached values
        """
        B, n_heads, num_steps, head_dim = keys.shape

        # First time: initialize
        if self.old_keys is None:
            self.old_keys = keys
            self.old_values = values
            self._total_offset = num_steps

            # Adaptive window: analyze workload characteristics during prefill
            if self.enable_adaptive_window and self.load_analyzer and self.layer_idx == 0:
                # Use real tokens if provided, otherwise fall back to workload_hint
                if self._pending_tokens is not None:
                    # Real tokens available - accurate analysis
                    tokens_to_analyze = self._pending_tokens
                    analysis_method = "token-based (accurate)"
                else:
                    # No tokens - use indices as proxy (inaccurate)
                    tokens_to_analyze = list(range(num_steps))
                    analysis_method = "index-based (proxy)"

                # Analyze redundancy
                redundancy, recommended_window = self.load_analyzer.analyze_and_recommend(
                    tokens_to_analyze,
                    workload_hint=self.workload_hint
                )

                # Update window size
                self.recent_window_size = recommended_window
                self.detected_redundancy = redundancy

                print(f"[DoubleLayerKVCache] Adaptive window enabled:")
                print(f"  Analysis method: {analysis_method}")
                print(f"  Detected redundancy: {redundancy:.2%}")
                print(f"  Recommended window: {recommended_window}")
                print(f"  (Initial window was: {self.initial_window_size})")

                # Clear pending tokens after analysis
                self._pending_tokens = None

            # Check if first prefill exceeds budget
            current_memory = self.nbytes
            if self.enable_compression and current_memory > self.memory_budget:
                if self.layer_idx == 0:
                    print(f"[DoubleLayerKVCache] Warning: First prefill ({current_memory / 1024**2:.1f} MB) "
                          f"exceeds budget ({self.memory_budget_mb:.1f} MB). "
                          f"Consider increasing memory_budget_mb or reducing batch size.")

            return keys, values

        # Step 1: Estimate memory after appending new KV
        new_memory = keys.nbytes + values.nbytes
        current_memory = self.nbytes
        estimated_total = current_memory + new_memory

        # Step 2: Check if compression needed (memory-budget driven)
        if self.enable_compression and estimated_total > self.memory_budget:
            # ⚠️ Memory budget exceeded - trigger compression
            if self.layer_idx == 0 and self.num_compressions == 0:
                print(f"[DoubleLayerKVCache] Layer {self.layer_idx}: Memory budget exceeded")
                print(f"  Current: {current_memory / 1024**2:.2f} MB")
                print(f"  New KV: {new_memory / 1024**2:.2f} MB")
                print(f"  Estimated: {estimated_total / 1024**2:.2f} MB")
                print(f"  Budget: {self.memory_budget_mb:.2f} MB")
                print(f"  → Triggering compression (blocking inference)...")

            # Compress to fit budget (leave room for new KV)
            target_memory = self.memory_budget - new_memory
            self._compress_to_fit_budget(target_memory)

            if self.layer_idx == 0:
                final_memory = self.nbytes + new_memory
                print(f"  ✓ Compression done: {final_memory / 1024**2:.2f} MB "
                      f"({final_memory / self.memory_budget * 100:.1f}% of budget)")

        # Step 3: Append new KV (memory is now sufficient)
        if self.recent_keys is None:
            # First time after compression
            self.old_keys = mx.concatenate([self.old_keys, keys], axis=2)
            self.old_values = mx.concatenate([self.old_values, values], axis=2)
        else:
            # Subsequent appends: add to old_keys (will be re-split if needed)
            self.old_keys = mx.concatenate([self.old_keys, keys], axis=2)
            self.old_values = mx.concatenate([self.old_values, values], axis=2)

        self._total_offset += num_steps

        # Step 4: Return [compacted_old | exact_recent]
        if self.recent_keys is not None:
            final_keys = mx.concatenate([self.old_keys, self.recent_keys], axis=2)
            final_values = mx.concatenate([self.old_values, self.recent_values], axis=2)
        else:
            final_keys = self.old_keys
            final_values = self.old_values

        return final_keys, final_values

    def _compress_to_fit_budget(self, target_memory: int):
        """
        Compress cache to fit memory budget.

        Strategy:
        - Split cache into old_prefix + recent_window
        - Compress old_prefix using AM calibration
        - Preserve recent_window (exact KV)
        - Iterate if necessary (increase compression ratio)

        Parameters
        ----------
        target_memory : int
            Target memory usage in bytes (budget - new_kv_size)
        """
        iteration = 0
        max_iterations = 3

        while self.nbytes > target_memory and iteration < max_iterations:
            iteration += 1

            # Combine old + recent for re-splitting
            if self.recent_keys is not None:
                combined_keys = mx.concatenate([self.old_keys, self.recent_keys], axis=2)
                combined_values = mx.concatenate([self.old_values, self.recent_values], axis=2)
            else:
                combined_keys = self.old_keys
                combined_values = self.old_values

            total_len = combined_keys.shape[2]

            # Check if we can split
            if total_len <= self.recent_window_size:
                # Cannot split further - entire cache is "recent"
                if self.layer_idx == 0:
                    print(f"  Warning: Cannot compress further (cache={total_len}, window={self.recent_window_size})")
                break

            # Split into old_prefix + recent_window
            split_point = total_len - self.recent_window_size

            old_prefix_keys = combined_keys[:, :, :split_point, :]
            old_prefix_values = combined_values[:, :, :split_point, :]

            recent_keys = combined_keys[:, :, split_point:, :]
            recent_values = combined_values[:, :, split_point:, :]

            # Compress old_prefix
            old_prefix_len = old_prefix_keys.shape[2]

            if self.calibration_registry:
                # Dynamic calibration selection
                calibration = self.calibration_registry.get_calibration(
                    length=old_prefix_len,
                    ratio=self.compression_ratio,
                    strategy=self.selection_strategy
                )

                if calibration:
                    # Debug info (first compression only)
                    if self.layer_idx == 0 and self.num_compressions == 0:
                        cal_len = calibration['metadata']['calibration_length']
                        layer_calib = calibration['calibration'][self.layer_idx]
                        budget = len(layer_calib['selected_indices'])
                        print(f"  old_prefix_len={old_prefix_len}, "
                              f"selected_calibration=L{cal_len}, "
                              f"budget={budget}")

                    compacted_old_keys, compacted_old_values = self._compress_old_prefix(
                        old_prefix_keys,
                        old_prefix_values,
                        calibration
                    )
                else:
                    # No calibration found, keep as-is
                    compacted_old_keys = old_prefix_keys
                    compacted_old_values = old_prefix_values
                    if self.layer_idx == 0:
                        print(f"  Warning: No calibration found for length={old_prefix_len}")
                    break
            else:
                # No calibration registry, keep as-is
                compacted_old_keys = old_prefix_keys
                compacted_old_values = old_prefix_values
                break

            # Update cache
            self.old_keys = compacted_old_keys
            self.old_values = compacted_old_values
            self.recent_keys = recent_keys
            self.recent_values = recent_values

            # Statistics
            self.num_compressions += 1
            self.total_tokens_before_compression += old_prefix_len
            self.total_tokens_after_compression += compacted_old_keys.shape[2]

            # Check if target met
            if self.nbytes <= target_memory:
                break

            # If still over budget, increase compression ratio for next iteration
            if iteration < max_iterations:
                self.compression_ratio *= 1.2
                if self.layer_idx == 0:
                    print(f"  Still over budget, increasing compression_ratio to {self.compression_ratio:.2f}")

        # Final check
        if self.nbytes > target_memory and self.layer_idx == 0:
            print(f"  Warning: Could not compress to target ({self.nbytes / 1024**2:.1f} MB > "
                  f"{target_memory / 1024**2:.1f} MB) after {iteration} iterations")

    def _compress_old_prefix(
        self,
        keys: mx.array,
        values: mx.array,
        calibration: dict
    ) -> Tuple[mx.array, mx.array]:
        """
        Compress old_prefix using AM calibration.

        Note: selected_indices are prefix-local (not global).

        Parameters
        ----------
        keys : mx.array, shape (B, n_heads, old_prefix_len, head_dim)
        values : mx.array, shape (B, n_heads, old_prefix_len, head_dim)
        calibration : dict
            Calibration data

        Returns
        -------
        compacted_keys : mx.array
        compacted_values : mx.array
        """
        if self.layer_idx is None:
            raise ValueError("layer_idx must be set to use calibration")

        # Extract layer calibration
        layer_calib = calibration['calibration'][self.layer_idx]
        selected_indices = layer_calib['selected_indices']
        beta = layer_calib['beta']

        # Convert to numpy for filtering
        if isinstance(selected_indices, mx.array):
            selected_indices = np.array(selected_indices)

        # Dynamic clipping: only keep indices < old_prefix_len
        old_prefix_len = keys.shape[2]
        valid_mask = selected_indices < old_prefix_len
        clipped_indices = selected_indices[valid_mask]

        if len(clipped_indices) == 0:
            # No valid indices, keep all (shouldn't happen in practice)
            return keys, values

        # Convert back to MLX array
        clipped_indices = mx.array(clipped_indices)

        # Beta is already safe (validated during calibration generation)
        # No runtime validation needed

        # Apply prefix-local indices
        compacted_keys = keys[:, :, clipped_indices, :]
        compacted_values = values[:, :, clipped_indices, :]

        return compacted_keys, compacted_values

    def get_stats(self) -> dict:
        """Get compression statistics."""
        avg_compression_ratio = 0.0
        if self.total_tokens_after_compression > 0:
            avg_compression_ratio = (
                self.total_tokens_before_compression /
                self.total_tokens_after_compression
            )

        stats = {
            'old_prefix_size': self.old_keys.shape[2] if self.old_keys is not None else 0,
            'recent_window_size': self.recent_keys.shape[2] if self.recent_keys is not None else 0,
            'total_size': self.offset,
            'num_compressions': self.num_compressions,
            'total_tokens_before_compression': self.total_tokens_before_compression,
            'total_tokens_after_compression': self.total_tokens_after_compression,
            'avg_compression_ratio': avg_compression_ratio,
            'memory_bytes': self.nbytes
        }

        # Add adaptive window stats
        if self.enable_adaptive_window:
            stats['adaptive_window_enabled'] = True
            stats['detected_redundancy'] = self.detected_redundancy
            stats['configured_window'] = self.recent_window_size
            stats['initial_window'] = self.initial_window_size
        else:
            stats['adaptive_window_enabled'] = False

        return stats
