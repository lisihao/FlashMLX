"""
Main Profiler class
"""

import time
from pathlib import Path
from typing import Optional
from datetime import datetime
import mlx.core as mx

from .config import ProfilerConfig, InstrumentationLevel
from .logger import ProfileLogger
from .instrumentation import (
    set_active_profiler,
    instrument_module,
    restore_module,
)


class Profiler:
    """
    Main profiler context manager

    Example:
        with Profiler("my_experiment"):
            model.generate(prompt)
    """

    def __init__(
        self,
        name: str = "profile",
        config: Optional[ProfilerConfig] = None,
        **kwargs
    ):
        """
        Initialize profiler

        Args:
            name: Experiment name
            config: ProfilerConfig instance
            **kwargs: Override config parameters
        """
        self.name = name
        self.config = config or ProfilerConfig()

        # Override config with kwargs
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        # Update name
        self.config.name = name

        # Create output file path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(self.config.output_dir)
        self.output_file = output_dir / f"{name}_{timestamp}.json"

        # Create logger
        self.logger = ProfileLogger(str(self.output_file))

        # Track instrumented modules
        self._instrumented = {}
        self._start_time = None
        self._end_time = None

    def __enter__(self):
        """Enter context - start profiling"""
        # Set as active profiler
        set_active_profiler(self)

        # Record start time
        self._start_time = time.perf_counter()

        # Instrument MLX functions based on level
        self._instrument_mlx()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context - stop profiling"""
        # Record end time
        self._end_time = time.perf_counter()

        # Restore original functions
        self._restore_mlx()

        # Clear active profiler
        set_active_profiler(None)

        # Save results
        self._save_results()

        return False

    def _instrument_mlx(self):
        """Instrument MLX functions based on configuration"""
        level = self.config.level

        # Basic: core operations
        basic_functions = [
            "matmul",
            "conv2d",
            "softmax",
        ]

        # Detailed: add more operations
        detailed_functions = basic_functions + [
            "add",
            "multiply",
            "divide",
            "reshape",
            "transpose",
        ]

        # Full: all operations
        full_functions = detailed_functions + [
            "concatenate",
            "split",
            "pad",
            "sum",
            "mean",
        ]

        # Select functions based on level
        if level == InstrumentationLevel.BASIC:
            functions = basic_functions
        elif level == InstrumentationLevel.DETAILED:
            functions = detailed_functions
        else:  # FULL
            functions = full_functions

        # Instrument mx module
        self._instrumented['mx'] = instrument_module(mx, functions, prefix="mx.")

        # Instrument mx.fast if DETAILED or FULL
        if level in [InstrumentationLevel.DETAILED, InstrumentationLevel.FULL]:
            fast_functions = [
                "scaled_dot_product_attention",
                "rope",
                "rms_norm",
            ]
            self._instrumented['mx.fast'] = instrument_module(
                mx.fast, fast_functions, prefix="mx.fast."
            )

    def _restore_mlx(self):
        """Restore original MLX functions"""
        if 'mx' in self._instrumented:
            restore_module(mx, self._instrumented['mx'])
        if 'mx.fast' in self._instrumented:
            restore_module(mx.fast, self._instrumented['mx.fast'])

    def _save_results(self):
        """Save profiling results"""
        # Calculate total time
        total_time_s = self._end_time - self._start_time

        # Prepare metadata
        metadata = {
            "experiment_name": self.name,
            "timestamp": datetime.now().isoformat(),
            "config": self.config.to_dict(),
            "total_time_s": total_time_s,
            "event_count": self.logger.get_event_count(),
        }

        # Save
        self.logger.save(metadata)

        print(f"✅ Profile saved to: {self.output_file}")
        print(f"   Total time: {total_time_s:.2f}s")
        print(f"   Events: {self.logger.get_event_count()}")

    def log_function_call(self, function_name: str, duration_ms: float, **kwargs):
        """Log a function call (called by instrumentation)"""
        # Filter out fast functions
        if duration_ms < self.config.min_function_time_ms:
            return

        self.logger.log_function_call(
            function_name=function_name,
            duration_ms=duration_ms,
            **kwargs
        )

    def region(self, name: str):
        """
        Create a named region for manual profiling

        Example:
            with profiler.region("prefill"):
                model.forward(tokens)
        """
        return ProfileRegion(self, name)


class ProfileRegion:
    """Named profiling region"""

    def __init__(self, profiler: Profiler, name: str):
        self.profiler = profiler
        self.name = name
        self._start_time = None

    def __enter__(self):
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.perf_counter() - self._start_time) * 1000
        self.profiler.logger.log_event(
            event_type="region",
            name=self.name,
            duration_ms=duration_ms
        )
        return False
