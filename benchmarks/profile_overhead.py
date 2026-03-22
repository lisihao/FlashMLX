"""
性能分析：定位 CompactedKVCache 的 overhead

测试 3 个关键环节：
1. Cache update (append) overhead
2. Cache retrieval overhead
3. Compression overhead
"""

import time
import mlx.core as mx
from pathlib import Path
import json

# Add mlx-lm to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "mlx-lm-source"))

from mlx_lm import load
from mlx_lm.models.compacted_cache import CompactedKVCache
from mlx_lm.models.cache import KVCache


def profile_cache_operations():
    """
    对比 KVCache vs CompactedKVCache 的操作开销
    """
    print("=" * 80)
    print("Cache Operations Profiling")
    print("=" * 80)

    # 测试配置
    B, n_heads, head_dim = 1, 8, 128
    num_iterations = 100

    # Test 1: Append overhead
    print("\n[Test 1] Append overhead (100 iterations)")
    print("-" * 40)

    # Baseline KVCache
    kv_cache = KVCache()
    keys = mx.random.normal((B, n_heads, 10, head_dim))
    values = mx.random.normal((B, n_heads, 10, head_dim))

    start = time.time()
    for _ in range(num_iterations):
        kv_cache.update_and_fetch(keys, values)
        mx.eval(kv_cache.keys)
    kv_append_time = time.time() - start

    # CompactedKVCache
    compacted_cache = CompactedKVCache(
        max_size=4096,
        compression_ratio=5.0,
        enable_compression=True
    )

    start = time.time()
    for _ in range(num_iterations):
        compacted_cache.update_and_fetch(keys, values)
        mx.eval(compacted_cache.keys)
    compacted_append_time = time.time() - start

    print(f"  KVCache append:       {kv_append_time*1000:.2f} ms")
    print(f"  CompactedKVCache:     {compacted_append_time*1000:.2f} ms")
    print(f"  Overhead:             {(compacted_append_time/kv_append_time - 1)*100:.1f}%")

    # Test 2: Retrieval overhead at different sizes
    print("\n[Test 2] Retrieval overhead at different cache sizes")
    print("-" * 40)

    for cache_size in [512, 1024, 2048, 4096, 8192]:
        # Fill cache
        kv_cache = KVCache()
        compacted_cache = CompactedKVCache(
            max_size=4096,
            compression_ratio=5.0,
            enable_compression=False  # Disable compression for fair comparison
        )

        # Append to reach target size
        chunk_size = 64
        for _ in range(cache_size // chunk_size):
            keys = mx.random.normal((B, n_heads, chunk_size, head_dim))
            values = mx.random.normal((B, n_heads, chunk_size, head_dim))
            kv_cache.update_and_fetch(keys, values)
            compacted_cache.update_and_fetch(keys, values)

        # Measure retrieval time
        start = time.time()
        for _ in range(50):
            _ = kv_cache.keys[..., :kv_cache.offset, :]
            mx.eval(_)
        kv_retrieval_time = time.time() - start

        start = time.time()
        for _ in range(50):
            _ = compacted_cache.keys[..., :compacted_cache.offset, :]
            mx.eval(_)
        compacted_retrieval_time = time.time() - start

        overhead_pct = (compacted_retrieval_time / kv_retrieval_time - 1) * 100
        print(f"  Size {cache_size:5d}: KV={kv_retrieval_time*1000:5.2f}ms, Compacted={compacted_retrieval_time*1000:5.2f}ms, Overhead={overhead_pct:+5.1f}%")

    # Test 3: Compression overhead
    print("\n[Test 3] Compression overhead")
    print("-" * 40)

    # Fill cache to trigger compression
    compacted_cache = CompactedKVCache(
        max_size=4096,
        compression_ratio=5.0,
        enable_compression=True,
        use_quality_path=False
    )

    # Fill to 4096
    chunk_size = 64
    for _ in range(4096 // chunk_size):
        keys = mx.random.normal((B, n_heads, chunk_size, head_dim))
        values = mx.random.normal((B, n_heads, chunk_size, head_dim))
        compacted_cache.update_and_fetch(keys, values)

    # Trigger compression (next append will compress)
    print(f"  Cache size before compression: {compacted_cache.offset}")

    keys = mx.random.normal((B, n_heads, 100, head_dim))
    values = mx.random.normal((B, n_heads, 100, head_dim))

    start = time.time()
    compacted_cache.update_and_fetch(keys, values)
    mx.eval(compacted_cache.keys)
    compression_time = time.time() - start

    print(f"  Cache size after compression:  {compacted_cache.offset}")
    print(f"  Compression time:              {compression_time*1000:.2f} ms")
    print(f"  Compressions triggered:        {compacted_cache.get_stats()['num_compressions']}")


if __name__ == "__main__":
    profile_cache_operations()
