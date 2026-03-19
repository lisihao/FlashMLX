"""
Profile data analyzer
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict


class ProfileAnalyzer:
    """Analyze profiling data"""

    def __init__(self, profile_file: str):
        """
        Initialize analyzer

        Args:
            profile_file: Path to profile JSON file
        """
        self.profile_file = Path(profile_file)
        self.data = self._load_profile()
        self.events = self.data.get("events", [])
        self.metadata = self.data.get("metadata", {})

    def _load_profile(self) -> dict:
        """Load profile data from file"""
        with open(self.profile_file, 'r') as f:
            return json.load(f)

    def get_total_time(self) -> float:
        """Get total profiling time in seconds"""
        return self.metadata.get("total_time_s", 0.0)

    def get_function_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for each function

        Returns:
            Dictionary mapping function name to stats:
            {
                "function_name": {
                    "count": 10,
                    "total_ms": 100.0,
                    "avg_ms": 10.0,
                    "min_ms": 5.0,
                    "max_ms": 15.0,
                    "percent": 45.0
                }
            }
        """
        stats = defaultdict(lambda: {
            "count": 0,
            "total_ms": 0.0,
            "times": []
        })

        # Collect data
        for event in self.events:
            if event.get("event_type") == "function_call":
                name = event["name"]
                duration = event.get("duration_ms", 0.0)

                stats[name]["count"] += 1
                stats[name]["total_ms"] += duration
                stats[name]["times"].append(duration)

        # Calculate statistics
        total_time_ms = sum(s["total_ms"] for s in stats.values())

        result = {}
        for name, data in stats.items():
            times = data["times"]
            result[name] = {
                "count": data["count"],
                "total_ms": data["total_ms"],
                "avg_ms": data["total_ms"] / data["count"],
                "min_ms": min(times),
                "max_ms": max(times),
                "percent": (data["total_ms"] / total_time_ms * 100) if total_time_ms > 0 else 0
            }

        return result

    def get_top_hotspots(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Get top N hotspots by total time

        Args:
            n: Number of hotspots to return

        Returns:
            List of hotspots sorted by total time
        """
        stats = self.get_function_stats()

        hotspots = [
            {
                "name": name,
                **data
            }
            for name, data in stats.items()
        ]

        # Sort by total time
        hotspots.sort(key=lambda x: x["total_ms"], reverse=True)

        return hotspots[:n]

    def print_summary(self):
        """Print summary of profiling results"""
        print("=" * 80)
        print(f"Profile: {self.metadata.get('experiment_name', 'unknown')}")
        print("=" * 80)
        print(f"\nTotal Time: {self.get_total_time():.2f}s")
        print(f"Events: {len(self.events)}")

        # Print memory stats if available
        if "memory" in self.metadata:
            memory = self.metadata["memory"]
            peak = memory.get("peak", {})
            delta = memory.get("delta", {})
            print(f"\nMemory:")
            print(f"  Peak: Python {peak.get('python_mb', 0):.1f} MB, Metal {peak.get('metal_mb', 0):.1f} MB")
            print(f"  Delta: Python {delta.get('python_mb', 0):+.1f} MB, Metal {delta.get('metal_mb', 0):+.1f} MB")

        # Print latency stats if available
        if "latency" in self.metadata:
            print(f"\nLatency:")
            for name, stats in self.metadata["latency"].items():
                print(f"  {name}:")
                print(f"    Mean: {stats['mean_ms']:.2f}ms, P95: {stats['p95_ms']:.2f}ms, P99: {stats['p99_ms']:.2f}ms")

        print(f"\n{'Function':<30} {'Time (ms)':<12} {'Calls':<8} {'%':<8}")
        print("-" * 80)

        hotspots = self.get_top_hotspots(10)
        for hotspot in hotspots:
            print(
                f"{hotspot['name']:<30} "
                f"{hotspot['total_ms']:<12.2f} "
                f"{hotspot['count']:<8} "
                f"{hotspot['percent']:<8.1f}"
            )

        print("=" * 80)

    def generate_report(self, output_file: str):
        """Generate markdown report"""
        output_path = Path(output_file)

        with open(output_path, 'w') as f:
            f.write(f"# Profile Report: {self.metadata.get('experiment_name', 'unknown')}\n\n")
            f.write(f"**Date**: {self.metadata.get('timestamp', 'unknown')}\n\n")
            f.write(f"## Summary\n\n")
            f.write(f"- Total Time: {self.get_total_time():.2f}s\n")
            f.write(f"- Events: {len(self.events)}\n\n")

            f.write(f"## Top 10 Hotspots\n\n")
            f.write(f"| Function | Time (ms) | Calls | % |\n")
            f.write(f"|----------|-----------|-------|---|\n")

            hotspots = self.get_top_hotspots(10)
            for hotspot in hotspots:
                f.write(
                    f"| {hotspot['name']} | "
                    f"{hotspot['total_ms']:.2f} | "
                    f"{hotspot['count']} | "
                    f"{hotspot['percent']:.1f}% |\n"
                )

        print(f"✅ Report saved to: {output_path}")
