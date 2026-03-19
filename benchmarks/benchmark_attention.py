"""
Benchmark Flash Attention performance
"""

import mlx.core as mx
import time
from flashmlx.core import flash_attention
from flashmlx.utils import benchmark


def benchmark_flash_attention():
    """Benchmark Flash Attention at various sequence lengths"""

    print("=" * 60)
    print("Flash Attention Benchmark")
    print("=" * 60)

    batch_size = 1
    num_heads = 32
    head_dim = 128

    seq_lengths = [128, 256, 512, 1024, 2048, 4096]

    print(f"\nConfiguration:")
    print(f"  Batch Size: {batch_size}")
    print(f"  Num Heads: {num_heads}")
    print(f"  Head Dim: {head_dim}")
    print(f"\n{'Seq Length':<12} {'Time (ms)':<12} {'Tokens/s':<12}")
    print("-" * 60)

    for seq_len in seq_lengths:
        q = mx.random.normal((batch_size, seq_len, num_heads, head_dim))
        k = mx.random.normal((batch_size, seq_len, num_heads, head_dim))
        v = mx.random.normal((batch_size, seq_len, num_heads, head_dim))

        # Benchmark with forced evaluation
        def run_attention():
            output, _ = flash_attention(q, k, v)
            mx.eval(output)  # Force evaluation
            return output

        avg_time = benchmark(
            run_attention,
            warmup=3,
            iterations=10
        )

        tokens_per_sec = (seq_len * 1000) / avg_time

        print(f"{seq_len:<12} {avg_time:<12.2f} {tokens_per_sec:<12.1f}")

    print("=" * 60)


if __name__ == "__main__":
    benchmark_flash_attention()
