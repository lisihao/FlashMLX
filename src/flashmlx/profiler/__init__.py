"""
FlashMLX Profiler - Self-Optimizing Performance Analysis Tool
"""

from .profiler import Profiler
from .instrumentation import profile, instrument_function, instrument_module
from .logger import ProfileLogger
from .analyzer import ProfileAnalyzer
from .config import ProfilerConfig, InstrumentationLevel

__all__ = [
    "Profiler",
    "profile",
    "instrument_function",
    "instrument_module",
    "ProfileLogger",
    "ProfileAnalyzer",
    "ProfilerConfig",
    "InstrumentationLevel",
]
