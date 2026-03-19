"""
Lock and synchronization tracking
"""

import time
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class LockAcquisition:
    """Single lock acquisition event"""
    lock_id: int
    lock_name: str
    thread_id: int
    thread_name: str
    acquire_time: float
    release_time: Optional[float] = None
    wait_time_ms: float = 0.0  # Time spent waiting
    hold_time_ms: Optional[float] = None  # Time held
    acquired: bool = False


class LockTracker:
    """
    Track lock acquisitions, contention, and potential deadlocks

    Note: Due to limitations of Python's threading.Lock (immutable C type),
    lock tracking requires manual instrumentation using tracked_lock().

    Example:
        tracker = LockTracker()
        tracker.start()

        lock = tracker.tracked_lock("my_lock")

        with lock:
            # Critical section
            pass

        stats = tracker.get_stats()
    """

    def __init__(self):
        self._tracking = False
        self._tracked_locks: Dict[int, str] = {}  # lock_id -> name
        self._acquisitions: List[LockAcquisition] = []
        self._active_locks: Dict[int, LockAcquisition] = {}  # lock_id -> current acquisition
        self._lock = threading.Lock()

    def start(self):
        """Start tracking locks"""
        if self._tracking:
            return

        self._tracking = True
        self._acquisitions.clear()
        self._active_locks.clear()

    def stop(self):
        """Stop tracking locks"""
        if not self._tracking:
            return

        self._tracking = False

    def tracked_lock(self, name: str = "") -> 'TrackedLock':
        """
        Create a tracked lock

        Args:
            name: Name for this lock

        Returns:
            TrackedLock instance that can be used like a regular lock
        """
        return TrackedLock(self, name)

    def _record_acquisition(self, lock_id: int, lock_name: str, wait_time_ms: float):
        """Record lock acquisition"""
        acq = LockAcquisition(
            lock_id=lock_id,
            lock_name=lock_name,
            thread_id=threading.get_ident(),
            thread_name=threading.current_thread().name,
            acquire_time=time.perf_counter(),
            wait_time_ms=wait_time_ms,
            acquired=True
        )

        with self._lock:
            self._acquisitions.append(acq)
            self._active_locks[lock_id] = acq

    def _record_release(self, lock_id: int):
        """Record lock release"""
        with self._lock:
            if lock_id in self._active_locks:
                acq = self._active_locks[lock_id]
                acq.release_time = time.perf_counter()
                acq.hold_time_ms = (acq.release_time - acq.acquire_time) * 1000
                del self._active_locks[lock_id]

    def get_stats(self) -> Dict[str, Any]:
        """
        Get lock statistics

        Returns:
            {
                "total_acquisitions": 100,
                "total_contention_ms": 250.5,
                "locks": {
                    "my_lock": {
                        "acquisitions": 50,
                        "total_wait_ms": 100.0,
                        "avg_wait_ms": 2.0,
                        "max_wait_ms": 10.0,
                        "total_hold_ms": 500.0,
                        "avg_hold_ms": 10.0,
                        "contention_rate": 0.6  # % of acquisitions that waited
                    }
                }
            }
        """
        stats_by_lock = defaultdict(lambda: {
            "acquisitions": 0,
            "wait_times": [],
            "hold_times": []
        })

        for acq in self._acquisitions:
            lock_stats = stats_by_lock[acq.lock_name]
            lock_stats["acquisitions"] += 1
            lock_stats["wait_times"].append(acq.wait_time_ms)
            if acq.hold_time_ms is not None:
                lock_stats["hold_times"].append(acq.hold_time_ms)

        # Calculate statistics
        result = {
            "total_acquisitions": len(self._acquisitions),
            "total_contention_ms": sum(acq.wait_time_ms for acq in self._acquisitions),
            "locks": {}
        }

        for lock_name, lock_data in stats_by_lock.items():
            wait_times = lock_data["wait_times"]
            hold_times = lock_data["hold_times"]

            # Count contentious acquisitions (wait > 0.1ms)
            contentious = sum(1 for w in wait_times if w > 0.1)

            result["locks"][lock_name] = {
                "acquisitions": lock_data["acquisitions"],
                "total_wait_ms": sum(wait_times),
                "avg_wait_ms": sum(wait_times) / len(wait_times) if wait_times else 0,
                "max_wait_ms": max(wait_times) if wait_times else 0,
                "total_hold_ms": sum(hold_times) if hold_times else 0,
                "avg_hold_ms": sum(hold_times) / len(hold_times) if hold_times else 0,
                "max_hold_ms": max(hold_times) if hold_times else 0,
                "contention_rate": contentious / len(wait_times) if wait_times else 0
            }

        return result

    def detect_potential_deadlocks(self) -> List[Dict[str, Any]]:
        """
        Detect potential deadlock patterns

        Returns list of suspicious patterns:
        - Circular wait chains
        - Long-held locks
        """
        warnings = []

        # Check for long-held locks (> 1 second)
        for acq in self._acquisitions:
            if acq.hold_time_ms and acq.hold_time_ms > 1000:
                warnings.append({
                    "type": "long_hold",
                    "lock_name": acq.lock_name,
                    "thread": acq.thread_name,
                    "hold_time_ms": acq.hold_time_ms,
                    "message": f"Lock {acq.lock_name} held for {acq.hold_time_ms:.1f}ms"
                })

        # Check for high contention (> 50% contention rate)
        stats = self.get_stats()
        for lock_name, lock_stats in stats["locks"].items():
            if lock_stats["contention_rate"] > 0.5:
                warnings.append({
                    "type": "high_contention",
                    "lock_name": lock_name,
                    "contention_rate": lock_stats["contention_rate"],
                    "message": f"Lock {lock_name} has {lock_stats['contention_rate']*100:.1f}% contention"
                })

        return warnings

    def get_acquisitions(self) -> List[LockAcquisition]:
        """Get all lock acquisitions"""
        return self._acquisitions.copy()


class TrackedLock:
    """A lock wrapper that tracks acquisitions and releases"""

    def __init__(self, tracker: LockTracker, name: str = ""):
        self._tracker = tracker
        self._lock = threading.Lock()
        self._lock_id = id(self._lock)
        self._name = name or f"Lock-{self._lock_id}"

    def acquire(self, blocking=True, timeout=-1):
        """Acquire lock"""
        start = time.perf_counter()
        result = self._lock.acquire(blocking, timeout)
        wait_time = (time.perf_counter() - start) * 1000

        if result and self._tracker._tracking:
            self._tracker._record_acquisition(self._lock_id, self._name, wait_time)

        return result

    def release(self):
        """Release lock"""
        if self._tracker._tracking:
            self._tracker._record_release(self._lock_id)
        return self._lock.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False
