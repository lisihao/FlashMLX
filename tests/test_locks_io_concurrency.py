"""
Tests for lock, IO, and concurrency tracking
"""

import pytest
import threading
import time
import tempfile
import os

from flashmlx.profiler import (
    Profiler,
    LockTracker,
    IOTracker,
    ConcurrencyTracker,
)


def test_lock_tracker():
    """Test lock tracking"""
    tracker = LockTracker()
    tracker.start()

    # Create tracked locks
    lock1 = tracker.tracked_lock("lock1")
    lock2 = tracker.tracked_lock("lock2")

    # Acquire and release
    with lock1:
        time.sleep(0.001)  # Hold for 1ms

    with lock2:
        time.sleep(0.002)  # Hold for 2ms

    # Get stats
    stats = tracker.get_stats()
    assert stats["total_acquisitions"] >= 2
    assert "lock1" in stats["locks"]
    assert "lock2" in stats["locks"]
    assert stats["locks"]["lock1"]["acquisitions"] == 1
    assert stats["locks"]["lock2"]["acquisitions"] == 1

    tracker.stop()


def test_lock_contention():
    """Test lock contention detection"""
    tracker = LockTracker()
    tracker.start()

    lock = tracker.tracked_lock("contended_lock")

    results = []

    def worker():
        with lock:
            time.sleep(0.01)  # Hold lock for 10ms
            results.append(1)

    # Create threads that will contend for lock
    threads = []
    for _ in range(3):
        t = threading.Thread(target=worker)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # Check stats
    stats = tracker.get_stats()
    assert stats["total_acquisitions"] == 3
    assert stats["total_contention_ms"] > 0  # Should have some contention

    tracker.stop()


def test_io_tracker():
    """Test IO tracking"""
    tracker = IOTracker()
    tracker.start()

    # Create temp file
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        temp_path = f.name
        f.write("Hello, World!\n" * 100)

    try:
        # Read file
        with open(temp_path, 'r') as f:
            content = f.read()

        # Write file
        with open(temp_path, 'w') as f:
            f.write("New content\n")

        # Get stats
        stats = tracker.get_stats()
        assert stats["total_operations"] >= 4  # open, read, open, write
        assert stats["total_bytes_read"] > 0
        assert stats["total_bytes_written"] > 0
        assert stats["avg_throughput_mbps"] >= 0

    finally:
        os.unlink(temp_path)
        tracker.stop()


def test_io_slowest_operations():
    """Test finding slowest IO operations"""
    tracker = IOTracker()
    tracker.start()

    # Create temp file
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        temp_path = f.name
        f.write("x" * 10000)

    try:
        # Multiple reads
        for _ in range(3):
            with open(temp_path, 'r') as f:
                _ = f.read()

        slowest = tracker.get_slowest_operations(n=3)
        assert len(slowest) <= 3
        assert all(isinstance(op.duration_ms, float) for op in slowest)

    finally:
        os.unlink(temp_path)
        tracker.stop()


def test_concurrency_tracker():
    """Test concurrency tracking"""
    tracker = ConcurrencyTracker()
    tracker.start()

    def worker():
        time.sleep(0.01)

    # Create threads
    threads = []
    for i in range(4):
        t = threading.Thread(target=worker, name=f"Worker-{i}")
        threads.append(t)
        t.start()

    # Wait for threads
    for t in threads:
        t.join()

    time.sleep(0.02)  # Give sampler time to collect

    # Get stats
    stats = tracker.get_stats()
    assert stats["total_threads"] >= 4
    assert stats["max_concurrent_threads"] >= 1

    tracker.stop()


def test_concurrency_gil_timeline():
    """Test GIL timeline tracking"""
    tracker = ConcurrencyTracker()
    tracker.start()

    def cpu_bound():
        # CPU-bound work to show GIL contention
        total = 0
        for i in range(100000):
            total += i

    threads = []
    for _ in range(2):
        t = threading.Thread(target=cpu_bound)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    time.sleep(0.02)  # Give sampler time

    # Get timeline
    timeline = tracker.get_gil_timeline()
    assert len(timeline) > 0

    # Check that we saw multiple threads
    max_threads = max(sample["active_threads"] for sample in timeline)
    assert max_threads >= 2

    tracker.stop()


def test_profiler_with_all_trackers():
    """Test profiler with lock, IO, and concurrency tracking"""
    with Profiler("test_all_trackers", capture_memory=True) as p:
        # Lock usage
        lock = p.lock_tracker.tracked_lock("test_lock")
        with lock:
            time.sleep(0.001)

        # IO usage
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            temp_path = f.name
            f.write("Test data\n")

        try:
            with open(temp_path, 'r') as f:
                _ = f.read()
        finally:
            os.unlink(temp_path)

        # Concurrency
        def worker():
            time.sleep(0.01)

        threads = []
        for _ in range(2):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

    # Check that all stats are captured
    # (They should be printed to console)
    # This test mainly checks that everything runs without errors


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
