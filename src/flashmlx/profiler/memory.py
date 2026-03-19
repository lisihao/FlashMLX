"""
Memory profiling utilities
"""

import tracemalloc
from typing import Optional, Dict, Any
import mlx.core as mx


class MemoryTracker:
    """Track memory usage during profiling"""

    def __init__(self):
        self.snapshots = []
        self._baseline_python = 0
        self._baseline_metal = 0
        self._peak_python = 0
        self._peak_metal = 0
        self._tracking = False

    def start(self):
        """Start memory tracking"""
        # Start Python memory tracking
        tracemalloc.start()

        # Record baseline
        self._baseline_python = self._get_python_memory_mb()
        self._baseline_metal = self._get_metal_memory_mb()

        self._peak_python = self._baseline_python
        self._peak_metal = self._baseline_metal

        self._tracking = True

    def stop(self):
        """Stop memory tracking"""
        if not self._tracking:
            return

        # Stop Python tracking
        tracemalloc.stop()
        self._tracking = False

    def snapshot(self, label: str = ""):
        """Take a memory snapshot"""
        if not self._tracking:
            return None

        snapshot = {
            "label": label,
            "python_mb": self._get_python_memory_mb(),
            "metal_mb": self._get_metal_memory_mb(),
            "metal_peak_mb": self._get_metal_peak_memory_mb(),
        }

        # Update peaks
        self._peak_python = max(self._peak_python, snapshot["python_mb"])
        self._peak_metal = max(self._peak_metal, snapshot["metal_peak_mb"])

        self.snapshots.append(snapshot)
        return snapshot

    def get_current_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        return {
            "python_mb": self._get_python_memory_mb(),
            "metal_mb": self._get_metal_memory_mb(),
            "metal_peak_mb": self._get_metal_peak_memory_mb(),
        }

    def get_peak_usage(self) -> Dict[str, float]:
        """Get peak memory usage"""
        return {
            "python_mb": self._peak_python,
            "metal_mb": self._peak_metal,
        }

    def get_baseline(self) -> Dict[str, float]:
        """Get baseline memory usage"""
        return {
            "python_mb": self._baseline_python,
            "metal_mb": self._baseline_metal,
        }

    def get_delta(self) -> Dict[str, float]:
        """Get memory delta from baseline"""
        current = self.get_current_usage()
        baseline = self.get_baseline()

        return {
            "python_mb": current["python_mb"] - baseline["python_mb"],
            "metal_mb": current["metal_mb"] - baseline["metal_mb"],
        }

    @staticmethod
    def _get_python_memory_mb() -> float:
        """Get current Python memory usage in MB"""
        if not tracemalloc.is_tracing():
            return 0.0

        current, peak = tracemalloc.get_traced_memory()
        return current / (1024 * 1024)  # Convert to MB

    @staticmethod
    def _get_metal_memory_mb() -> float:
        """Get current Metal memory usage in MB"""
        try:
            # MLX active memory (currently allocated)
            return mx.metal.get_active_memory() / (1024 * 1024)
        except Exception:
            return 0.0

    @staticmethod
    def _get_metal_peak_memory_mb() -> float:
        """Get peak Metal memory usage in MB"""
        try:
            # MLX peak memory since last reset
            return mx.metal.get_peak_memory() / (1024 * 1024)
        except Exception:
            return 0.0

    def reset_metal_peak(self):
        """Reset Metal peak memory counter"""
        try:
            mx.metal.reset_peak_memory()
        except Exception:
            pass

    def format_usage(self, usage: Dict[str, float]) -> str:
        """Format memory usage for display"""
        return (
            f"Python: {usage.get('python_mb', 0):.1f} MB, "
            f"Metal: {usage.get('metal_mb', 0):.1f} MB"
        )
