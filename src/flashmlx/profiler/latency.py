"""
Latency tracking and analysis
"""

import time
import statistics
from typing import List, Dict, Any, Optional
from collections import defaultdict


class LatencyTracker:
    """Track latency metrics for generation and inference"""

    def __init__(self):
        self.latencies: Dict[str, List[float]] = defaultdict(list)
        self._timers: Dict[str, float] = {}

    def start_timer(self, name: str):
        """Start a latency timer"""
        self._timers[name] = time.perf_counter()

    def stop_timer(self, name: str) -> float:
        """Stop a latency timer and record the latency"""
        if name not in self._timers:
            return 0.0

        start_time = self._timers.pop(name)
        latency_ms = (time.perf_counter() - start_time) * 1000
        self.latencies[name].append(latency_ms)
        return latency_ms

    def record_latency(self, name: str, latency_ms: float):
        """Manually record a latency measurement"""
        self.latencies[name].append(latency_ms)

    def get_stats(self, name: str) -> Optional[Dict[str, float]]:
        """Get latency statistics for a metric"""
        if name not in self.latencies or not self.latencies[name]:
            return None

        latencies = self.latencies[name]

        return {
            "count": len(latencies),
            "min_ms": min(latencies),
            "max_ms": max(latencies),
            "mean_ms": statistics.mean(latencies),
            "median_ms": statistics.median(latencies),
            "p95_ms": self._percentile(latencies, 0.95),
            "p99_ms": self._percentile(latencies, 0.99),
            "stdev_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        }

    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all tracked metrics"""
        return {
            name: self.get_stats(name)
            for name in self.latencies
            if self.get_stats(name) is not None
        }

    @staticmethod
    def _percentile(data: List[float], percentile: float) -> float:
        """Calculate percentile"""
        if not data:
            return 0.0

        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile)
        index = min(index, len(sorted_data) - 1)
        return sorted_data[index]

    def format_stats(self, stats: Dict[str, float]) -> str:
        """Format latency statistics for display"""
        return (
            f"Mean: {stats['mean_ms']:.2f}ms, "
            f"P50: {stats['median_ms']:.2f}ms, "
            f"P95: {stats['p95_ms']:.2f}ms, "
            f"P99: {stats['p99_ms']:.2f}ms"
        )

    def clear(self):
        """Clear all latency data"""
        self.latencies.clear()
        self._timers.clear()


class GenerationLatencyTracker(LatencyTracker):
    """
    Specialized latency tracker for text generation

    Tracks:
    - Time to First Token (TTFT)
    - Inter-token Latency
    - Total generation time
    """

    def __init__(self):
        super().__init__()
        self._generation_start = None
        self._first_token_recorded = False
        self._token_times: List[float] = []

    def start_generation(self):
        """Mark the start of generation"""
        self._generation_start = time.perf_counter()
        self._first_token_recorded = False
        self._token_times = []

    def record_token(self):
        """Record a token generation"""
        if self._generation_start is None:
            return

        current_time = time.perf_counter()

        if not self._first_token_recorded:
            # First token - record TTFT
            ttft_ms = (current_time - self._generation_start) * 1000
            self.record_latency("ttft", ttft_ms)
            self._first_token_recorded = True
        else:
            # Subsequent tokens - record inter-token latency
            if self._token_times:
                inter_token_ms = (current_time - self._token_times[-1]) * 1000
                self.record_latency("inter_token", inter_token_ms)

        self._token_times.append(current_time)

    def end_generation(self) -> Dict[str, float]:
        """End generation and return summary"""
        if self._generation_start is None:
            return {}

        total_time_s = time.perf_counter() - self._generation_start
        num_tokens = len(self._token_times)

        summary = {
            "total_time_s": total_time_s,
            "num_tokens": num_tokens,
            "tokens_per_second": num_tokens / total_time_s if total_time_s > 0 else 0,
        }

        # Add TTFT stats
        ttft_stats = self.get_stats("ttft")
        if ttft_stats:
            summary["ttft_ms"] = ttft_stats["mean_ms"]

        # Add inter-token stats
        inter_token_stats = self.get_stats("inter_token")
        if inter_token_stats:
            summary["inter_token_mean_ms"] = inter_token_stats["mean_ms"]
            summary["inter_token_p95_ms"] = inter_token_stats["p95_ms"]

        self._generation_start = None
        return summary
