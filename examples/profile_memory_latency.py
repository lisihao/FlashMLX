"""
Memory and Latency profiling example
"""

import mlx.core as mx
import time
from flashmlx.profiler import (
    Profiler,
    ProfileAnalyzer,
    GenerationLatencyTracker,
)


def memory_intensive_operation():
    """Operation that allocates a lot of memory"""
    # Allocate large arrays
    arrays = []
    for i in range(5):
        size = 1024 * (2 ** i)  # 1024, 2048, 4096, 8192, 16384
        a = mx.random.normal((size, size))
        b = mx.matmul(a, a)
        mx.eval(b)
        arrays.append(b)
    return arrays


def simulate_generation(num_tokens=10):
    """Simulate token generation with latency tracking"""
    # Simulate first token (slower - includes prefill)
    time.sleep(0.02)  # 20ms TTFT

    # Simulate subsequent tokens
    for _ in range(num_tokens - 1):
        time.sleep(0.003)  # 3ms per token


def main():
    print("=" * 80)
    print("FlashMLX Profiler - Memory & Latency Example")
    print("=" * 80)

    # Example 1: Memory Profiling
    print("\n[Example 1] Memory Profiling")
    with Profiler(
        "memory_test",
        capture_memory=True  # Enable memory tracking
    ) as p:
        arrays = memory_intensive_operation()

    analyzer = ProfileAnalyzer(str(p.output_file))
    analyzer.print_summary()

    # Example 2: Latency Tracking
    print("\n[Example 2] Latency Tracking")

    # Create latency tracker
    latency_tracker = GenerationLatencyTracker()

    with Profiler("latency_test") as p:
        # Track generation latencies
        for run in range(3):
            p.latency_tracker.start_timer(f"run_{run}")

            # Simulate generation
            latency_tracker.start_generation()
            for _ in range(10):
                latency_tracker.record_token()
                time.sleep(0.003)  # 3ms per token
            summary = latency_tracker.end_generation()

            p.latency_tracker.stop_timer(f"run_{run}")

            print(f"\nRun {run}:")
            print(f"  TTFT: {summary.get('ttft_ms', 0):.2f}ms")
            print(f"  Tokens/s: {summary.get('tokens_per_second', 0):.1f}")
            print(f"  Inter-token (mean): {summary.get('inter_token_mean_ms', 0):.2f}ms")

    analyzer = ProfileAnalyzer(str(p.output_file))
    analyzer.print_summary()

    # Example 3: Combined - Memory + Latency + Performance
    print("\n[Example 3] Combined Profiling")
    with Profiler(
        "combined_test",
        capture_memory=True
    ) as p:
        # Phase 1: Prefill (memory intensive)
        with p.region("prefill"):
            p.memory_tracker.snapshot("before_prefill")

            q = mx.random.normal((1, 2048, 32, 128))  # Large sequence
            k = mx.random.normal((1, 2048, 32, 128))
            v = mx.random.normal((1, 2048, 32, 128))

            scores = mx.matmul(q, mx.transpose(k, axes=[0, 1, 3, 2]))
            attn = mx.softmax(scores, axis=-1)
            output = mx.matmul(attn, v)
            mx.eval(output)

            p.memory_tracker.snapshot("after_prefill")

        # Phase 2: Decode (latency sensitive)
        with p.region("decode"):
            for i in range(5):
                p.latency_tracker.start_timer(f"decode_step_{i}")

                q_decode = mx.random.normal((1, 1, 32, 128))  # Single token
                scores = mx.matmul(q_decode, mx.transpose(k, axes=[0, 1, 3, 2]))
                attn = mx.softmax(scores, axis=-1)
                output = mx.matmul(attn, v)
                mx.eval(output)

                p.latency_tracker.stop_timer(f"decode_step_{i}")

    analyzer = ProfileAnalyzer(str(p.output_file))
    analyzer.print_summary()

    # Show memory snapshots
    print("\nMemory Snapshots:")
    for snapshot in p.memory_tracker.snapshots:
        print(f"  {snapshot['label']}: Metal {snapshot['metal_mb']:.1f} MB")


if __name__ == "__main__":
    main()
