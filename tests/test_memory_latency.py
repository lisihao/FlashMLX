"""
Tests for memory and latency profiling
"""

import pytest
import mlx.core as mx
import time

from flashmlx.profiler import (
    Profiler,
    MemoryTracker,
    LatencyTracker,
    GenerationLatencyTracker,
)


def test_memory_tracker():
    """Test memory tracking"""
    tracker = MemoryTracker()
    tracker.start()

    # Get baseline
    baseline = tracker.get_baseline()
    assert "python_mb" in baseline
    assert "metal_mb" in baseline

    # Allocate memory
    a = mx.random.normal((1000, 1000))
    mx.eval(a)

    # Take snapshot
    snapshot = tracker.snapshot("after_alloc")
    assert snapshot is not None
    assert snapshot["metal_mb"] > baseline["metal_mb"]

    # Get peak
    peak = tracker.get_peak_usage()
    assert peak["metal_mb"] >= baseline["metal_mb"]

    tracker.stop()


def test_profiler_with_memory():
    """Test profiler with memory tracking enabled"""
    with Profiler("test_memory", capture_memory=True) as p:
        a = mx.random.normal((1000, 1000))
        b = mx.matmul(a, a)
        mx.eval(b)

        # Take snapshot
        p.memory_tracker.snapshot("after_matmul")

    # Check snapshots
    assert len(p.memory_tracker.snapshots) > 0


def test_latency_tracker():
    """Test latency tracking"""
    tracker = LatencyTracker()

    # Record some latencies
    for i in range(10):
        tracker.start_timer("operation")
        time.sleep(0.001)  # 1ms
        tracker.stop_timer("operation")

    # Get stats
    stats = tracker.get_stats("operation")
    assert stats is not None
    assert stats["count"] == 10
    assert stats["mean_ms"] >= 1.0
    assert stats["p95_ms"] >= stats["mean_ms"]


def test_generation_latency_tracker():
    """Test generation latency tracking"""
    tracker = GenerationLatencyTracker()

    # Simulate generation
    tracker.start_generation()

    # First token (TTFT)
    time.sleep(0.01)  # 10ms
    tracker.record_token()

    # Subsequent tokens
    for _ in range(5):
        time.sleep(0.003)  # 3ms
        tracker.record_token()

    summary = tracker.end_generation()

    assert summary["num_tokens"] == 6
    assert summary["ttft_ms"] >= 10
    assert "tokens_per_second" in summary


def test_profiler_with_latency():
    """Test profiler with latency tracking"""
    with Profiler("test_latency") as p:
        # Record some latencies
        for i in range(3):
            p.latency_tracker.start_timer(f"step_{i}")
            time.sleep(0.001)
            p.latency_tracker.stop_timer(f"step_{i}")

    # Check latency stats in metadata
    latency_stats = p.latency_tracker.get_all_stats()
    assert len(latency_stats) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
