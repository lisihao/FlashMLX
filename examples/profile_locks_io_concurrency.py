"""
Lock, IO, and Concurrency profiling example
"""

import time
import threading
import tempfile
import os
from flashmlx.profiler import Profiler, ProfileAnalyzer


def example_locks():
    """Example with lock contention"""
    print("\n[Example 1] Lock Contention Tracking")

    with Profiler("lock_example") as p:
        # Create tracked lock
        lock = p.lock_tracker.tracked_lock("shared_resource")

        results = []

        def worker(worker_id):
            # Each worker competes for the lock
            with lock:
                # Simulate work while holding lock
                time.sleep(0.005)  # 5ms
                results.append(worker_id)

        # Create threads that will contend
        threads = []
        for i in range(5):
            t = threading.Thread(target=worker, args=(i,), name=f"Worker-{i}")
            threads.append(t)
            t.start()

        # Wait for all
        for t in threads:
            t.join()

    analyzer = ProfileAnalyzer(str(p.output_file))
    analyzer.print_summary()

    # Show lock warnings
    warnings = p.lock_tracker.detect_potential_deadlocks()
    if warnings:
        print("\n⚠️  Lock Warnings:")
        for w in warnings:
            print(f"  - {w['message']}")


def example_io():
    """Example with IO operations"""
    print("\n[Example 2] IO Tracking")

    with Profiler("io_example") as p:
        # Create temp files
        temp_files = []
        for i in range(3):
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
                temp_files.append(f.name)
                # Write varying amounts of data
                f.write(f"File {i}\n" * (1000 * (i + 1)))

        # Read files
        total_content = ""
        for path in temp_files:
            with open(path, 'r') as f:
                total_content += f.read()

        # Write aggregated file
        output_path = tempfile.mktemp()
        with open(output_path, 'w') as f:
            f.write(total_content)

        # Clean up
        for path in temp_files + [output_path]:
            os.unlink(path)

    analyzer = ProfileAnalyzer(str(p.output_file))
    analyzer.print_summary()

    # Show slowest operations
    slowest = p.io_tracker.get_slowest_operations(n=5)
    print("\n📊 Slowest IO Operations:")
    for op in slowest:
        print(f"  {op.op_type}: {op.path} - {op.duration_ms:.2f}ms ({op.size_bytes} bytes)")


def example_concurrency():
    """Example with multi-threading"""
    print("\n[Example 3] Concurrency Tracking")

    with Profiler("concurrency_example") as p:
        def cpu_worker(worker_id):
            """CPU-bound work (shows GIL contention)"""
            total = 0
            for i in range(500000):
                total += i
            time.sleep(0.01)
            return total

        def io_worker(worker_id):
            """IO-bound work (releases GIL)"""
            time.sleep(0.02)

        # Mix of CPU and IO workers
        threads = []

        # CPU workers (will contend for GIL)
        for i in range(3):
            t = threading.Thread(target=cpu_worker, args=(i,), name=f"CPU-Worker-{i}")
            threads.append(t)
            t.start()

        # IO workers
        for i in range(2):
            t = threading.Thread(target=io_worker, args=(i,), name=f"IO-Worker-{i}")
            threads.append(t)
            t.start()

        # Wait for all
        for t in threads:
            t.join()

    analyzer = ProfileAnalyzer(str(p.output_file))
    analyzer.print_summary()

    # Show concurrency issues
    issues = p.concurrency_tracker.detect_concurrency_issues()
    if issues:
        print("\n⚠️  Concurrency Issues:")
        for issue in issues:
            print(f"  - [{issue['severity'].upper()}] {issue['message']}")


def example_combined():
    """Example combining all tracking features"""
    print("\n[Example 4] Combined Tracking - All Features")

    with Profiler(
        "combined_example",
        capture_memory=True  # Also track memory
    ) as p:
        # Setup
        lock = p.lock_tracker.tracked_lock("data_lock")
        shared_data = []

        # Create temp file for IO
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            temp_path = f.name
            f.write("Shared data\n" * 1000)

        def worker(worker_id):
            # Read from file (IO)
            with open(temp_path, 'r') as f:
                content = f.read()

            # Process data (CPU + lock contention)
            with lock:
                shared_data.append(len(content))
                time.sleep(0.002)

            # More CPU work
            total = sum(range(10000))

        # Create workers
        threads = []
        for i in range(4):
            t = threading.Thread(target=worker, args=(i,), name=f"Worker-{i}")
            threads.append(t)
            t.start()

        # Wait
        for t in threads:
            t.join()

        # Clean up
        os.unlink(temp_path)

    # Analyze results
    analyzer = ProfileAnalyzer(str(p.output_file))
    analyzer.print_summary()

    print("\n📈 Detailed Breakdown:")
    print(f"  Lock acquisitions: {p.lock_tracker.get_stats()['total_acquisitions']}")
    print(f"  IO operations: {p.io_tracker.get_stats()['total_operations']}")
    print(f"  Threads created: {p.concurrency_tracker.get_stats()['total_threads']}")
    print(f"  Max concurrent: {p.concurrency_tracker.get_stats()['max_concurrent_threads']}")


def main():
    print("=" * 80)
    print("FlashMLX Profiler - Lock, IO, and Concurrency Tracking")
    print("=" * 80)

    example_locks()
    example_io()
    example_concurrency()
    example_combined()

    print("\n✅ All examples completed!")


if __name__ == "__main__":
    main()
