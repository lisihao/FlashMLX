"""
Memory Benchmark: Verify > 70% Memory Savings

This script measures memory usage of CompactedKVCache and verifies
that it achieves > 70% memory savings compared to uncompressed cache.
"""

import mlx.core as mx
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'mlx-lm-source')))

from mlx_lm.models.compacted_cache import CompactedKVCache


def benchmark_memory(
    num_tokens,
    n_heads,
    head_dim,
    max_size,
    compression_ratio,
    use_quality_path=False
):
    """
    Benchmark memory usage for a given configuration.

    Returns:
        dict with memory stats
    """
    cache = CompactedKVCache(
        max_size=max_size,
        compression_ratio=compression_ratio,
        use_quality_path=use_quality_path,
        enable_compression=True
    )

    B = 1
    start_time = time.time()

    # Generate tokens
    for i in range(num_tokens // 10):
        new_keys = mx.random.normal((B, n_heads, 10, head_dim))
        new_values = mx.random.normal((B, n_heads, 10, head_dim))
        cache.update_and_fetch(new_keys, new_values)

    elapsed = time.time() - start_time

    # Get stats
    stats = cache.get_stats()
    memory_bytes = cache.nbytes
    memory_mb = memory_bytes / (1024 * 1024)

    # Calculate theoretical uncompressed memory
    uncompressed_tokens = num_tokens
    uncompressed_memory_bytes = (
        2 *  # keys + values
        n_heads *
        uncompressed_tokens *
        head_dim *
        4  # float32 = 4 bytes
    )
    uncompressed_memory_mb = uncompressed_memory_bytes / (1024 * 1024)

    # Calculate savings
    savings_pct = (1 - memory_mb / uncompressed_memory_mb) * 100

    return {
        'num_tokens': num_tokens,
        'cache_size': cache.offset,
        'memory_mb': memory_mb,
        'uncompressed_memory_mb': uncompressed_memory_mb,
        'savings_pct': savings_pct,
        'compressions': stats['num_compressions'],
        'avg_ratio': stats['avg_compression_ratio'],
        'time_sec': elapsed,
    }


def run_benchmarks():
    """
    Run comprehensive memory benchmarks.
    """
    print("=" * 70)
    print("Memory Benchmark: CompactedKVCache")
    print("Verify > 70% Memory Savings")
    print("=" * 70)

    n_heads = 8
    head_dim = 64

    # Configuration: (num_tokens, max_size, compression_ratio)
    configs = [
        ("Short (1K tokens, 5x)", 1000, 200, 5.0),
        ("Medium (4K tokens, 5x)", 4000, 1000, 5.0),
        ("Long (10K tokens, 5x)", 10000, 2000, 5.0),
        ("Very long (20K tokens, 5x)", 20000, 4000, 5.0),
        ("Aggressive (10K tokens, 10x)", 10000, 1000, 10.0),
        ("Conservative (10K tokens, 3x)", 10000, 3000, 3.0),
    ]

    results = []

    for config_name, num_tokens, max_size, compression_ratio in configs:
        print(f"\n[{config_name}]")

        # Fast Path
        print("  Fast Path:")
        result_fast = benchmark_memory(
            num_tokens, n_heads, head_dim,
            max_size, compression_ratio,
            use_quality_path=False
        )
        results.append(('Fast Path', config_name, result_fast))

        print(f"    Memory: {result_fast['memory_mb']:.2f} MB "
              f"(vs {result_fast['uncompressed_memory_mb']:.2f} MB uncompressed)")
        print(f"    Savings: {result_fast['savings_pct']:.1f}%")
        print(f"    Cache size: {result_fast['cache_size']} tokens (from {num_tokens})")
        print(f"    Compressions: {result_fast['compressions']}")
        print(f"    Time: {result_fast['time_sec']:.2f}s")

        # Quality Path
        print("  Quality Path:")
        result_quality = benchmark_memory(
            num_tokens, n_heads, head_dim,
            max_size, compression_ratio,
            use_quality_path=True
        )
        results.append(('Quality Path', config_name, result_quality))

        print(f"    Memory: {result_quality['memory_mb']:.2f} MB "
              f"(vs {result_quality['uncompressed_memory_mb']:.2f} MB uncompressed)")
        print(f"    Savings: {result_quality['savings_pct']:.1f}%")
        print(f"    Cache size: {result_quality['cache_size']} tokens (from {num_tokens})")
        print(f"    Compressions: {result_quality['compressions']}")
        print(f"    Time: {result_quality['time_sec']:.2f}s")
        print(f"    Overhead vs Fast: {(result_quality['time_sec'] / result_fast['time_sec'] - 1) * 100:.1f}%")

    # Summary
    print("\n" + "=" * 70)
    print("Summary: Memory Savings Verification")
    print("=" * 70)

    print("\n{:<20} {:<25} {:>12} {:>15}".format(
        "Path", "Configuration", "Memory (MB)", "Savings (%)"
    ))
    print("-" * 70)

    for path, config_name, result in results:
        print("{:<20} {:<25} {:>12.2f} {:>15.1f}".format(
            path, config_name,
            result['memory_mb'],
            result['savings_pct']
        ))

    # Verification
    print("\n" + "=" * 70)
    print("Verification: > 70% Memory Savings")
    print("=" * 70)

    all_passed = True
    for path, config_name, result in results:
        passed = result['savings_pct'] > 70.0
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} | {path:15s} | {config_name:30s} | {result['savings_pct']:.1f}%")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n🎉 All benchmarks achieve > 70% memory savings!")
        print("✅ Memory benchmark requirement: PASSED")
    else:
        print("\n⚠️ Some benchmarks did not achieve 70% savings")
        print("❌ Memory benchmark requirement: FAILED")

    return all_passed


def run_detailed_analysis():
    """
    Run detailed memory analysis showing the breakdown.
    """
    print("\n" + "=" * 70)
    print("Detailed Analysis: Memory Breakdown")
    print("=" * 70)

    num_tokens = 10000
    n_heads = 8
    head_dim = 64
    max_size = 2000
    compression_ratio = 5.0

    print(f"\nConfiguration:")
    print(f"  Tokens: {num_tokens}")
    print(f"  Heads: {n_heads}")
    print(f"  Head dim: {head_dim}")
    print(f"  Max size: {max_size}")
    print(f"  Compression ratio: {compression_ratio}x")

    # Uncompressed
    print(f"\n[Uncompressed Cache]")
    uncompressed_memory = 2 * n_heads * num_tokens * head_dim * 4 / (1024 * 1024)
    print(f"  Keys: {n_heads} × {num_tokens} × {head_dim} × 4 bytes = "
          f"{n_heads * num_tokens * head_dim * 4 / (1024 * 1024):.2f} MB")
    print(f"  Values: {n_heads} × {num_tokens} × {head_dim} × 4 bytes = "
          f"{n_heads * num_tokens * head_dim * 4 / (1024 * 1024):.2f} MB")
    print(f"  Total: {uncompressed_memory:.2f} MB")

    # Compressed (Quality Path)
    result = benchmark_memory(
        num_tokens, n_heads, head_dim,
        max_size, compression_ratio,
        use_quality_path=True
    )

    print(f"\n[Compressed Cache (Quality Path)]")
    compressed_tokens = result['cache_size']
    compressed_memory = result['memory_mb']
    print(f"  Keys: {n_heads} × {compressed_tokens} × {head_dim} × 4 bytes = "
          f"{n_heads * compressed_tokens * head_dim * 4 / (1024 * 1024):.2f} MB")
    print(f"  Values: {n_heads} × {compressed_tokens} × {head_dim} × 4 bytes = "
          f"{n_heads * compressed_tokens * head_dim * 4 / (1024 * 1024):.2f} MB")
    print(f"  Total: {compressed_memory:.2f} MB")

    print(f"\n[Memory Savings]")
    print(f"  Uncompressed: {uncompressed_memory:.2f} MB")
    print(f"  Compressed: {compressed_memory:.2f} MB")
    print(f"  Saved: {uncompressed_memory - compressed_memory:.2f} MB")
    print(f"  Savings: {result['savings_pct']:.1f}%")
    print(f"  Token reduction: {num_tokens} → {compressed_tokens} "
          f"({(1 - compressed_tokens/num_tokens) * 100:.1f}% fewer tokens)")


def main():
    """
    Run all benchmarks.
    """
    # Run benchmarks
    all_passed = run_benchmarks()

    # Run detailed analysis
    run_detailed_analysis()

    # Final message
    print("\n" + "=" * 70)
    print("Benchmark Complete")
    print("=" * 70)

    if all_passed:
        print("\n✅ Memory Benchmark: PASSED")
        print("   All configurations achieve > 70% memory savings")
        print("\n   Phase C.2 requirement: VERIFIED ✅")
    else:
        print("\n❌ Memory Benchmark: FAILED")
        print("   Some configurations did not achieve 70% savings")

    return 0 if all_passed else 1


if __name__ == '__main__':
    exit(main())
