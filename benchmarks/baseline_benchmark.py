#!/usr/bin/env python3
"""
Compatibility wrapper for the updated generation benchmark.

The old file used stale profiler APIs. Keep this entry point so any existing
scripts still work, but delegate to the new benchmark implementation.
"""

from baseline_benchmark_simple import main


if __name__ == "__main__":
    raise SystemExit(main())
