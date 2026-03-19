"""
Tests for FlashMLX profiler
"""

import pytest
import mlx.core as mx
from pathlib import Path
import json
import time

from flashmlx.profiler import (
    Profiler,
    profile,
    ProfilerConfig,
    InstrumentationLevel,
    ProfileAnalyzer,
)


def test_profiler_context_manager():
    """Test basic profiler context manager"""
    with Profiler("test_basic") as p:
        # Simple MLX operations
        a = mx.ones((10, 10))
        b = mx.ones((10, 10))
        c = mx.matmul(a, b)
        mx.eval(c)

    # Check output file exists
    assert p.output_file.exists()

    # Load and verify
    with open(p.output_file, 'r') as f:
        data = json.load(f)

    assert "metadata" in data
    assert "events" in data
    assert data["metadata"]["experiment_name"] == "test_basic"


def test_profiler_levels():
    """Test different instrumentation levels"""
    # Basic level
    with Profiler("test_basic_level", level=InstrumentationLevel.BASIC):
        a = mx.ones((10, 10))
        b = mx.matmul(a, a)
        mx.eval(b)

    # Detailed level
    with Profiler("test_detailed_level", level=InstrumentationLevel.DETAILED):
        a = mx.ones((10, 10))
        b = mx.matmul(a, a)
        mx.eval(b)


def test_profile_decorator():
    """Test @profile decorator"""

    @profile("my_function")
    def my_function(x):
        return mx.matmul(x, x)

    with Profiler("test_decorator"):
        x = mx.ones((10, 10))
        result = my_function(x)
        mx.eval(result)


def test_profile_region():
    """Test manual profiling regions"""
    with Profiler("test_regions") as p:
        with p.region("region1"):
            a = mx.ones((10, 10))
            b = mx.matmul(a, a)
            mx.eval(b)

        with p.region("region2"):
            c = mx.ones((10, 10))
            d = mx.matmul(c, c)
            mx.eval(d)


def test_analyzer():
    """Test profile analyzer"""
    # Create profile
    with Profiler("test_analyzer") as p:
        for _ in range(5):
            a = mx.ones((10, 10))
            b = mx.matmul(a, a)
            mx.eval(b)

    # Analyze
    analyzer = ProfileAnalyzer(str(p.output_file))

    # Check statistics
    stats = analyzer.get_function_stats()
    assert "mx.matmul" in stats
    assert stats["mx.matmul"]["count"] == 5

    # Check hotspots
    hotspots = analyzer.get_top_hotspots(5)
    assert len(hotspots) > 0
    assert hotspots[0]["name"] == "mx.matmul"


def test_min_function_time_filter():
    """Test filtering of fast functions"""
    config = ProfilerConfig(min_function_time_ms=10.0)  # Only log > 10ms

    with Profiler("test_filter", config=config):
        # Fast operation (should be filtered)
        a = mx.ones((2, 2))
        b = mx.matmul(a, a)
        mx.eval(b)

        # Slow operation (should be logged)
        time.sleep(0.02)  # 20ms
        c = mx.ones((100, 100))
        d = mx.matmul(c, c)
        mx.eval(d)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
