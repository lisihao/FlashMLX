"""
Profiler configuration
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional


class InstrumentationLevel(Enum):
    """Instrumentation granularity level"""
    BASIC = "basic"      # Only top-level functions
    DETAILED = "detailed"  # Include sub-functions
    FULL = "full"        # All functions + kernels


@dataclass
class ProfilerConfig:
    """Profiler configuration"""

    name: str = "default"
    level: InstrumentationLevel = InstrumentationLevel.BASIC

    # Features
    capture_memory: bool = False
    capture_kernels: bool = False
    capture_args: bool = False
    capture_stack: bool = False

    # Output
    output_dir: str = "./profiling_data"
    output_format: str = "json"  # json, sqlite, parquet

    # Performance
    max_overhead_percent: float = 5.0
    min_function_time_ms: float = 0.1  # Don't log faster functions

    # Self-optimization
    auto_optimize: bool = False
    max_passes: int = 3
    min_coverage: float = 0.8

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "level": self.level.value,
            "capture_memory": self.capture_memory,
            "capture_kernels": self.capture_kernels,
            "capture_args": self.capture_args,
            "capture_stack": self.capture_stack,
            "output_dir": self.output_dir,
            "output_format": self.output_format,
            "max_overhead_percent": self.max_overhead_percent,
            "min_function_time_ms": self.min_function_time_ms,
            "auto_optimize": self.auto_optimize,
            "max_passes": self.max_passes,
            "min_coverage": self.min_coverage,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProfilerConfig":
        """Create from dictionary"""
        config = cls()
        config.name = data.get("name", "default")
        config.level = InstrumentationLevel(data.get("level", "basic"))
        config.capture_memory = data.get("capture_memory", False)
        config.capture_kernels = data.get("capture_kernels", False)
        config.capture_args = data.get("capture_args", False)
        config.capture_stack = data.get("capture_stack", False)
        config.output_dir = data.get("output_dir", "./profiling_data")
        config.output_format = data.get("output_format", "json")
        config.max_overhead_percent = data.get("max_overhead_percent", 5.0)
        config.min_function_time_ms = data.get("min_function_time_ms", 0.1)
        config.auto_optimize = data.get("auto_optimize", False)
        config.max_passes = data.get("max_passes", 3)
        config.min_coverage = data.get("min_coverage", 0.8)
        return config
