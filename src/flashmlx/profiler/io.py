"""
IO operation tracking
"""

import time
import io
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import os


@dataclass
class IOOperation:
    """Single IO operation event"""
    op_type: str  # "read", "write", "open", "close"
    path: str
    size_bytes: int
    duration_ms: float
    thread_id: int
    timestamp: float


class IOTracker:
    """
    Track file I/O, network I/O, and Metal buffer operations

    Example:
        tracker = IOTracker()
        tracker.start()

        with open("data.txt", "r") as f:
            content = f.read()

        stats = tracker.get_stats()
    """

    def __init__(self):
        self._tracking = False
        self._operations: List[IOOperation] = []

        # Original methods
        self._original_open = None
        self._original_read = None
        self._original_write = None
        self._original_readinto = None

    def start(self):
        """Start tracking IO operations"""
        if self._tracking:
            return

        self._tracking = True
        self._operations.clear()

        # Monkey patch builtin open
        self._patch_io_methods()

    def stop(self):
        """Stop tracking IO operations"""
        if not self._tracking:
            return

        self._tracking = False
        self._restore_io_methods()

    def _patch_io_methods(self):
        """Monkey patch IO methods"""
        import builtins
        import threading

        self._original_open = builtins.open

        def tracked_open(file, mode='r', *args, **kwargs):
            start = time.perf_counter()
            f = self._original_open(file, mode, *args, **kwargs)
            duration = (time.perf_counter() - start) * 1000

            # Record open operation
            self._operations.append(IOOperation(
                op_type="open",
                path=str(file),
                size_bytes=0,
                duration_ms=duration,
                thread_id=threading.get_ident(),
                timestamp=time.perf_counter()
            ))

            # Wrap file object to track read/write
            return TrackedFile(f, str(file), self)

        builtins.open = tracked_open

    def _restore_io_methods(self):
        """Restore original IO methods"""
        if self._original_open:
            import builtins
            builtins.open = self._original_open

    def record_operation(self, op_type: str, path: str, size_bytes: int, duration_ms: float):
        """Record an IO operation"""
        import threading
        self._operations.append(IOOperation(
            op_type=op_type,
            path=path,
            size_bytes=size_bytes,
            duration_ms=duration_ms,
            thread_id=threading.get_ident(),
            timestamp=time.perf_counter()
        ))

    def get_stats(self) -> Dict[str, Any]:
        """
        Get IO statistics

        Returns:
            {
                "total_operations": 100,
                "total_bytes_read": 1024000,
                "total_bytes_written": 512000,
                "total_time_ms": 500.0,
                "avg_throughput_mbps": 25.0,
                "by_type": {
                    "read": {"count": 50, "bytes": 1024000, "time_ms": 200},
                    "write": {"count": 30, "bytes": 512000, "time_ms": 150}
                },
                "by_file": {
                    "data.txt": {"reads": 10, "writes": 5, "total_bytes": 10240}
                }
            }
        """
        stats_by_type = defaultdict(lambda: {
            "count": 0,
            "bytes": 0,
            "time_ms": 0.0
        })

        stats_by_file = defaultdict(lambda: {
            "reads": 0,
            "writes": 0,
            "total_bytes": 0
        })

        for op in self._operations:
            # By type
            type_stats = stats_by_type[op.op_type]
            type_stats["count"] += 1
            type_stats["bytes"] += op.size_bytes
            type_stats["time_ms"] += op.duration_ms

            # By file
            if op.op_type in ["read", "write"]:
                file_stats = stats_by_file[op.path]
                if op.op_type == "read":
                    file_stats["reads"] += 1
                else:
                    file_stats["writes"] += 1
                file_stats["total_bytes"] += op.size_bytes

        total_bytes_read = stats_by_type["read"]["bytes"]
        total_bytes_written = stats_by_type["write"]["bytes"]
        total_time_s = sum(s["time_ms"] for s in stats_by_type.values()) / 1000
        total_bytes = total_bytes_read + total_bytes_written

        # Calculate throughput (MB/s)
        avg_throughput_mbps = 0.0
        if total_time_s > 0:
            avg_throughput_mbps = (total_bytes / (1024 * 1024)) / total_time_s

        return {
            "total_operations": len(self._operations),
            "total_bytes_read": total_bytes_read,
            "total_bytes_written": total_bytes_written,
            "total_time_ms": total_time_s * 1000,
            "avg_throughput_mbps": avg_throughput_mbps,
            "by_type": dict(stats_by_type),
            "by_file": dict(stats_by_file)
        }

    def get_slowest_operations(self, n: int = 10) -> List[IOOperation]:
        """Get N slowest IO operations"""
        return sorted(self._operations, key=lambda x: x.duration_ms, reverse=True)[:n]

    def get_largest_operations(self, n: int = 10) -> List[IOOperation]:
        """Get N largest IO operations by size"""
        return sorted(self._operations, key=lambda x: x.size_bytes, reverse=True)[:n]


class TrackedFile:
    """Wrapper around file object to track read/write operations"""

    def __init__(self, file_obj, path: str, tracker: IOTracker):
        self._file = file_obj
        self._path = path
        self._tracker = tracker

    def read(self, size=-1):
        start = time.perf_counter()
        data = self._file.read(size)
        duration = (time.perf_counter() - start) * 1000

        self._tracker.record_operation(
            op_type="read",
            path=self._path,
            size_bytes=len(data) if isinstance(data, (bytes, str)) else 0,
            duration_ms=duration
        )

        return data

    def write(self, data):
        start = time.perf_counter()
        result = self._file.write(data)
        duration = (time.perf_counter() - start) * 1000

        self._tracker.record_operation(
            op_type="write",
            path=self._path,
            size_bytes=len(data) if isinstance(data, (bytes, str)) else 0,
            duration_ms=duration
        )

        return result

    def readline(self, size=-1):
        start = time.perf_counter()
        line = self._file.readline(size)
        duration = (time.perf_counter() - start) * 1000

        self._tracker.record_operation(
            op_type="read",
            path=self._path,
            size_bytes=len(line) if isinstance(line, (bytes, str)) else 0,
            duration_ms=duration
        )

        return line

    def readlines(self, hint=-1):
        start = time.perf_counter()
        lines = self._file.readlines(hint)
        duration = (time.perf_counter() - start) * 1000

        total_bytes = sum(len(line) for line in lines)
        self._tracker.record_operation(
            op_type="read",
            path=self._path,
            size_bytes=total_bytes,
            duration_ms=duration
        )

        return lines

    def close(self):
        start = time.perf_counter()
        result = self._file.close()
        duration = (time.perf_counter() - start) * 1000

        self._tracker.record_operation(
            op_type="close",
            path=self._path,
            size_bytes=0,
            duration_ms=duration
        )

        return result

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __getattr__(self, name):
        """Forward other attributes to wrapped file"""
        return getattr(self._file, name)

    def __iter__(self):
        return self

    def __next__(self):
        start = time.perf_counter()
        line = next(self._file)
        duration = (time.perf_counter() - start) * 1000

        self._tracker.record_operation(
            op_type="read",
            path=self._path,
            size_bytes=len(line) if isinstance(line, (bytes, str)) else 0,
            duration_ms=duration
        )

        return line
