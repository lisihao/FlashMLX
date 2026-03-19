"""
Concurrency and threading tracking
"""

import time
import threading
import sys
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class ThreadLifecycle:
    """Thread creation and destruction event"""
    thread_id: int
    thread_name: str
    created_at: float
    destroyed_at: Optional[float] = None
    lifetime_ms: Optional[float] = None


@dataclass
class GILSample:
    """GIL (Global Interpreter Lock) contention sample"""
    timestamp: float
    active_threads: int
    gil_switches: int  # Approximation


class ConcurrencyTracker:
    """
    Track threading, GIL contention, and parallel execution

    Example:
        tracker = ConcurrencyTracker()
        tracker.start()

        # Your multi-threaded code
        threads = []
        for i in range(4):
            t = threading.Thread(target=worker)
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        stats = tracker.get_stats()
    """

    def __init__(self):
        self._tracking = False
        self._threads: Dict[int, ThreadLifecycle] = {}
        self._gil_samples: List[GILSample] = []
        self._sample_interval = 0.01  # 10ms
        self._sampler_thread = None
        self._stop_sampling = threading.Event()

        # Original methods
        self._original_thread_run = None

    def start(self):
        """Start tracking concurrency"""
        if self._tracking:
            return

        self._tracking = True
        self._threads.clear()
        self._gil_samples.clear()
        self._stop_sampling.clear()

        # Patch threading.Thread
        self._patch_thread_methods()

        # Start GIL sampler
        self._start_gil_sampling()

    def stop(self):
        """Stop tracking concurrency"""
        if not self._tracking:
            return

        self._tracking = False
        self._stop_sampling.set()

        if self._sampler_thread:
            self._sampler_thread.join(timeout=1.0)

        self._restore_thread_methods()

    def _patch_thread_methods(self):
        """Monkey patch Thread.run to track creation"""
        self._original_thread_run = threading.Thread.run

        def tracked_thread_run(thread_self):
            # Record thread start (now ident is available)
            thread_id = threading.get_ident()
            thread_name = thread_self.name

            self._threads[thread_id] = ThreadLifecycle(
                thread_id=thread_id,
                thread_name=thread_name,
                created_at=time.perf_counter()
            )

            try:
                # Run original
                return self._original_thread_run(thread_self)
            finally:
                # Record thread end
                self.record_thread_destruction(thread_id)

        threading.Thread.run = tracked_thread_run

    def _restore_thread_methods(self):
        """Restore original Thread methods"""
        if self._original_thread_run:
            threading.Thread.run = self._original_thread_run

    def _start_gil_sampling(self):
        """Start background thread to sample GIL state"""
        def sampler():
            last_switch_count = sys.getswitchinterval()

            while not self._stop_sampling.is_set():
                # Sample current state
                active_threads = threading.active_count()

                # Approximate GIL switches (not perfect but useful)
                # In reality, we'd need C extension for accurate GIL tracking
                sample = GILSample(
                    timestamp=time.perf_counter(),
                    active_threads=active_threads,
                    gil_switches=0  # Placeholder
                )
                self._gil_samples.append(sample)

                time.sleep(self._sample_interval)

        self._sampler_thread = threading.Thread(target=sampler, name="GIL-Sampler", daemon=True)
        self._sampler_thread.start()

    def record_thread_destruction(self, thread_id: int):
        """Record thread destruction (call manually if needed)"""
        if thread_id in self._threads:
            lifecycle = self._threads[thread_id]
            lifecycle.destroyed_at = time.perf_counter()
            lifecycle.lifetime_ms = (lifecycle.destroyed_at - lifecycle.created_at) * 1000

    def get_stats(self) -> Dict[str, Any]:
        """
        Get concurrency statistics

        Returns:
            {
                "total_threads": 10,
                "active_threads": 4,
                "avg_thread_lifetime_ms": 500.0,
                "max_concurrent_threads": 8,
                "gil_contention_estimate": 0.75,  # 0-1 scale
                "threads": [
                    {"id": 123, "name": "Worker-1", "lifetime_ms": 450.0}
                ]
            }
        """
        # Thread statistics
        active_threads = sum(1 for t in self._threads.values() if t.destroyed_at is None)
        completed_threads = [t for t in self._threads.values() if t.destroyed_at is not None]

        avg_lifetime = 0.0
        if completed_threads:
            avg_lifetime = sum(t.lifetime_ms for t in completed_threads) / len(completed_threads)

        # GIL contention estimate
        max_concurrent = max((s.active_threads for s in self._gil_samples), default=1)

        # Estimate contention: if we see >1 active thread consistently, there's contention
        # This is a rough approximation
        gil_contention = 0.0
        if self._gil_samples:
            samples_with_contention = sum(1 for s in self._gil_samples if s.active_threads > 1)
            gil_contention = samples_with_contention / len(self._gil_samples)

        return {
            "total_threads": len(self._threads),
            "active_threads": active_threads,
            "completed_threads": len(completed_threads),
            "avg_thread_lifetime_ms": avg_lifetime,
            "max_concurrent_threads": max_concurrent,
            "gil_contention_estimate": gil_contention,
            "threads": [
                {
                    "id": t.thread_id,
                    "name": t.thread_name,
                    "lifetime_ms": t.lifetime_ms,
                    "active": t.destroyed_at is None
                }
                for t in self._threads.values()
            ]
        }

    def get_thread_timeline(self) -> List[Dict[str, Any]]:
        """
        Get timeline of thread activity

        Returns list of events:
        [
            {"time": 0.0, "event": "thread_start", "thread": "Worker-1"},
            {"time": 0.5, "event": "thread_end", "thread": "Worker-1"}
        ]
        """
        events = []

        base_time = min((t.created_at for t in self._threads.values()), default=0)

        for thread in self._threads.values():
            events.append({
                "time": (thread.created_at - base_time) * 1000,
                "event": "thread_start",
                "thread": thread.thread_name,
                "thread_id": thread.thread_id
            })

            if thread.destroyed_at:
                events.append({
                    "time": (thread.destroyed_at - base_time) * 1000,
                    "event": "thread_end",
                    "thread": thread.thread_name,
                    "thread_id": thread.thread_id
                })

        # Sort by time
        events.sort(key=lambda x: x["time"])
        return events

    def get_gil_timeline(self) -> List[Dict[str, Any]]:
        """
        Get timeline of GIL activity

        Returns list of samples:
        [
            {"time": 0.0, "active_threads": 4},
            {"time": 0.01, "active_threads": 3}
        ]
        """
        if not self._gil_samples:
            return []

        base_time = self._gil_samples[0].timestamp

        return [
            {
                "time": (s.timestamp - base_time) * 1000,
                "active_threads": s.active_threads
            }
            for s in self._gil_samples
        ]

    def detect_concurrency_issues(self) -> List[Dict[str, Any]]:
        """
        Detect potential concurrency issues

        Returns list of warnings:
        - High GIL contention (> 70%)
        - Thread pool exhaustion
        - Long-lived threads
        """
        warnings = []
        stats = self.get_stats()

        # High GIL contention
        if stats["gil_contention_estimate"] > 0.7:
            warnings.append({
                "type": "high_gil_contention",
                "severity": "warning",
                "contention": stats["gil_contention_estimate"],
                "message": f"High GIL contention: {stats['gil_contention_estimate']*100:.1f}% of samples show multiple active threads"
            })

        # Too many threads
        if stats["max_concurrent_threads"] > 20:
            warnings.append({
                "type": "thread_pool_exhaustion",
                "severity": "warning",
                "max_threads": stats["max_concurrent_threads"],
                "message": f"High thread count: {stats['max_concurrent_threads']} concurrent threads detected"
            })

        # Long-lived threads (> 10 seconds)
        for thread in stats["threads"]:
            if thread["lifetime_ms"] and thread["lifetime_ms"] > 10000:
                warnings.append({
                    "type": "long_lived_thread",
                    "severity": "info",
                    "thread": thread["name"],
                    "lifetime_ms": thread["lifetime_ms"],
                    "message": f"Thread {thread['name']} lived for {thread['lifetime_ms']:.0f}ms"
                })

        return warnings
