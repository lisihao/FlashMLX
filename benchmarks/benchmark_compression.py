"""
Performance benchmark for KV cache compression.

Measures:
1. Compression time vs sequence length
2. Inference speed (compressed vs uncompressed)
3. Memory savings
4. Quality vs compression ratio trade-off
"""
import time
import mlx.core as mx
import numpy as np
from typing import Dict, List, Tuple

from flashmlx.cache.compaction_algorithm import create_compaction_algorithm


class CompressionBenchmark:
    """Benchmark suite for compression algorithm"""

    def __init__(self):
        self.results = {
            'compression_time': [],
            'memory_savings': [],
            'quality': [],
            'inference_speed': [],
        }

    def benchmark_compression_time(
        self,
        sequence_lengths: List[int] = [256, 512, 1024, 2048],
        compression_ratios: List[int] = [2, 4, 8],
        n_queries: int = 50,
        head_dim: int = 128,
        n_runs: int = 5
    ):
        """
        Benchmark compression time vs sequence length.

        Args:
            sequence_lengths: List of original sequence lengths to test
            compression_ratios: List of compression ratios
            n_queries: Number of query samples
            head_dim: Head dimension
            n_runs: Number of runs for averaging
        """
        print(f"\n{'='*70}")
        print("Benchmark 1: Compression Time")
        print(f"{'='*70}")

        algo = create_compaction_algorithm(
            score_method='mean',
            beta_method='nnls',
            c2_method='lsq',
            c2_ridge_lambda=0.01
        )

        for T in sequence_lengths:
            for ratio in compression_ratios:
                t = T // ratio

                # Create synthetic data
                K = mx.random.normal(shape=(T, head_dim)) * 0.1
                V = mx.random.normal(shape=(T, head_dim)) * 0.1
                queries = mx.random.normal(shape=(n_queries, head_dim)) * 0.1

                # Warm-up
                algo.compute_compacted_cache(K, V, queries, t)

                # Benchmark
                times = []
                for _ in range(n_runs):
                    start = time.perf_counter()
                    C1, beta, C2, indices = algo.compute_compacted_cache(
                        K, V, queries, t
                    )
                    # Ensure computation is done (MLX is lazy)
                    mx.eval(C1, beta, C2)
                    end = time.perf_counter()
                    times.append((end - start) * 1000)  # Convert to ms

                avg_time = np.mean(times)
                std_time = np.std(times)

                result = {
                    'T': T,
                    't': t,
                    'ratio': ratio,
                    'time_ms': avg_time,
                    'std_ms': std_time,
                }
                self.results['compression_time'].append(result)

                print(f"  T={T:4d} → t={t:4d} ({ratio}x): "
                      f"{avg_time:6.2f} ± {std_time:5.2f} ms")

    def benchmark_memory_savings(
        self,
        T: int = 1024,
        compression_ratios: List[int] = [2, 4, 8],
        n_kv_heads: int = 8,
        head_dim: int = 128,
    ):
        """
        Calculate memory savings for different compression ratios.

        Args:
            T: Original sequence length
            compression_ratios: List of compression ratios
            n_kv_heads: Number of KV heads
            head_dim: Head dimension
        """
        print(f"\n{'='*70}")
        print("Benchmark 2: Memory Savings")
        print(f"{'='*70}")

        # Original memory: K + V (both T × head_dim)
        # Each float32 = 4 bytes
        bytes_per_token = n_kv_heads * head_dim * 2 * 4  # K+V

        for ratio in compression_ratios:
            t = T // ratio

            # Original size
            original_bytes = T * n_kv_heads * head_dim * 2 * 4

            # Compressed size: C1 + beta + C2
            # C1: (n_kv_heads, t, head_dim) float32
            # beta: (n_kv_heads, t) float32
            # C2: (n_kv_heads, t, head_dim) float32
            compressed_bytes = (
                t * n_kv_heads * head_dim * 4 +  # C1
                t * n_kv_heads * 4 +              # beta
                t * n_kv_heads * head_dim * 4     # C2
            )

            saved_bytes = original_bytes - compressed_bytes
            saved_pct = (saved_bytes / original_bytes) * 100

            result = {
                'ratio': ratio,
                'original_kb': original_bytes / 1024,
                'compressed_kb': compressed_bytes / 1024,
                'saved_kb': saved_bytes / 1024,
                'saved_pct': saved_pct,
            }
            self.results['memory_savings'].append(result)

            print(f"  {ratio}x compression:")
            print(f"    Original:   {original_bytes/1024:8.1f} KB")
            print(f"    Compressed: {compressed_bytes/1024:8.1f} KB")
            print(f"    Saved:      {saved_bytes/1024:8.1f} KB ({saved_pct:.1f}%)")

    def benchmark_quality_vs_ratio(
        self,
        T: int = 1024,
        compression_ratios: List[int] = [2, 4, 8, 16],
        n_queries: int = 50,
        head_dim: int = 128,
    ):
        """
        Measure quality degradation vs compression ratio.

        Args:
            T: Original sequence length
            compression_ratios: List of compression ratios
            n_queries: Number of query samples
            head_dim: Head dimension
        """
        print(f"\n{'='*70}")
        print("Benchmark 3: Quality vs Compression Ratio")
        print(f"{'='*70}")

        # Create data
        K = mx.random.normal(shape=(T, head_dim)) * 0.1
        V = mx.random.normal(shape=(T, head_dim)) * 0.1
        queries = mx.random.normal(shape=(n_queries, head_dim)) * 0.1

        # Compute original attention
        scale = 1.0 / mx.sqrt(mx.array(head_dim, dtype=K.dtype))
        attn_scores_orig = queries @ K.T * scale
        attn_weights_orig = mx.softmax(attn_scores_orig, axis=-1)
        attn_output_orig = attn_weights_orig @ V

        for ratio in compression_ratios:
            t = T // ratio

            # Compress
            algo = create_compaction_algorithm(
                score_method='mean',
                beta_method='nnls',
                c2_method='lsq',
                c2_ridge_lambda=0.01
            )
            C1, beta, C2, indices = algo.compute_compacted_cache(
                K, V, queries, t
            )

            # Compute compressed attention
            attn_scores_comp = queries @ C1.T * scale
            attn_weights_comp = mx.softmax(attn_scores_comp, axis=-1)
            attn_output_comp = attn_weights_comp @ C2

            # Compute similarity
            orig_norm = mx.linalg.norm(attn_output_orig, axis=1, keepdims=True)
            comp_norm = mx.linalg.norm(attn_output_comp, axis=1, keepdims=True)
            cosine_sim = mx.sum(attn_output_orig * attn_output_comp, axis=1) / \
                        (orig_norm.squeeze() * comp_norm.squeeze() + 1e-8)
            avg_cosine_sim = float(mx.mean(cosine_sim))

            # MSE
            mse = float(mx.mean((attn_output_orig - attn_output_comp) ** 2))

            result = {
                'ratio': ratio,
                't': t,
                'cosine_similarity': avg_cosine_sim,
                'mse': mse,
            }
            self.results['quality'].append(result)

            print(f"  {ratio:2d}x (T={T} → t={t:4d}): "
                  f"Similarity={avg_cosine_sim:.4f}, MSE={mse:.6f}")

    def benchmark_inference_overhead(
        self,
        T: int = 1024,
        t: int = 256,
        n_heads: int = 32,
        n_kv_heads: int = 8,
        head_dim: int = 128,
        new_tokens: int = 10,
        n_runs: int = 10
    ):
        """
        Benchmark inference overhead of compressed cache.

        Simulates attention computation with compressed vs uncompressed cache.

        Args:
            T: Original sequence length
            t: Compressed length
            n_heads: Number of query heads
            n_kv_heads: Number of KV heads
            head_dim: Head dimension
            new_tokens: Number of new tokens to generate
            n_runs: Number of runs for averaging
        """
        print(f"\n{'='*70}")
        print("Benchmark 4: Inference Overhead")
        print(f"{'='*70}")

        # Create compressed cache
        algo = create_compaction_algorithm()
        K_full = mx.random.normal(shape=(T, head_dim)) * 0.1
        V_full = mx.random.normal(shape=(T, head_dim)) * 0.1
        queries_compress = mx.random.normal(shape=(50, head_dim)) * 0.1

        C1, beta, C2, indices = algo.compute_compacted_cache(
            K_full, V_full, queries_compress, t
        )

        # Simulate attention computation
        def simulate_attention(K, V, q, scale):
            """Simulate attention computation"""
            scores = q @ K.T * scale
            weights = mx.softmax(scores, axis=-1)
            output = weights @ V
            mx.eval(output)
            return output

        scale = 1.0 / mx.sqrt(mx.array(head_dim, dtype=K_full.dtype))

        # Benchmark uncompressed (TTFT)
        print("\n  [Uncompressed Cache]")
        q_new = mx.random.normal(shape=(head_dim,)) * 0.1

        # Warm-up
        simulate_attention(K_full, V_full, q_new, scale)

        times_uncompressed = []
        for _ in range(n_runs):
            start = time.perf_counter()
            simulate_attention(K_full, V_full, q_new, scale)
            times_uncompressed.append((time.perf_counter() - start) * 1000)

        avg_uncompressed = np.mean(times_uncompressed)
        std_uncompressed = np.std(times_uncompressed)

        print(f"    TTFT: {avg_uncompressed:.3f} ± {std_uncompressed:.3f} ms")

        # Benchmark compressed (TTFT)
        print("\n  [Compressed Cache]")

        # Warm-up
        simulate_attention(C1, C2, q_new, scale)

        times_compressed = []
        for _ in range(n_runs):
            start = time.perf_counter()
            simulate_attention(C1, C2, q_new, scale)
            times_compressed.append((time.perf_counter() - start) * 1000)

        avg_compressed = np.mean(times_compressed)
        std_compressed = np.std(times_compressed)

        print(f"    TTFT: {avg_compressed:.3f} ± {std_compressed:.3f} ms")

        # Calculate overhead
        overhead_pct = ((avg_compressed - avg_uncompressed) / avg_uncompressed) * 100

        print(f"\n  Overhead: {overhead_pct:+.1f}%")
        print(f"  Speedup:  {avg_uncompressed/avg_compressed:.2f}x")

        result = {
            'T': T,
            't': t,
            'ratio': T / t,
            'uncompressed_ms': avg_uncompressed,
            'compressed_ms': avg_compressed,
            'overhead_pct': overhead_pct,
            'speedup': avg_uncompressed / avg_compressed,
        }
        self.results['inference_speed'].append(result)

    def print_summary(self):
        """Print benchmark summary"""
        print(f"\n{'='*70}")
        print("Benchmark Summary")
        print(f"{'='*70}")

        # Compression time summary
        if self.results['compression_time']:
            print("\n[Compression Time]")
            print("  Ratio | T=256  | T=512  | T=1024 | T=2048")
            print("  ------|--------|--------|--------|--------")

            for ratio in [2, 4, 8]:
                times_by_T = {}
                for r in self.results['compression_time']:
                    if r['ratio'] == ratio:
                        times_by_T[r['T']] = r['time_ms']

                row = f"  {ratio}x    |"
                for T in [256, 512, 1024, 2048]:
                    if T in times_by_T:
                        row += f" {times_by_T[T]:5.1f}ms |"
                    else:
                        row += "   --   |"
                print(row)

        # Memory savings summary
        if self.results['memory_savings']:
            print("\n[Memory Savings]")
            for r in self.results['memory_savings']:
                print(f"  {r['ratio']}x: {r['saved_pct']:.1f}% "
                      f"({r['saved_kb']:.0f} KB saved)")

        # Quality summary
        if self.results['quality']:
            print("\n[Quality Metrics]")
            print("  Ratio | Similarity | MSE")
            print("  ------|------------|----------")
            for r in self.results['quality']:
                print(f"  {r['ratio']:2d}x   | {r['cosine_similarity']:9.4f}  | {r['mse']:.6f}")

        # Inference speed summary
        if self.results['inference_speed']:
            print("\n[Inference Speed]")
            for r in self.results['inference_speed']:
                print(f"  {r['ratio']:.0f}x compression:")
                print(f"    Uncompressed: {r['uncompressed_ms']:.3f} ms")
                print(f"    Compressed:   {r['compressed_ms']:.3f} ms")
                print(f"    Overhead:     {r['overhead_pct']:+.1f}%")
                print(f"    Speedup:      {r['speedup']:.2f}x")

    def save_results(self, filepath: str = "benchmarks/benchmark_results.txt"):
        """Save benchmark results to file"""
        with open(filepath, 'w') as f:
            f.write("="*70 + "\n")
            f.write("KV Cache Compression Benchmark Results\n")
            f.write("="*70 + "\n\n")

            # Write each section
            for key, data in self.results.items():
                f.write(f"\n{key.upper().replace('_', ' ')}:\n")
                f.write("-" * 70 + "\n")
                for item in data:
                    f.write(str(item) + "\n")

        print(f"\nResults saved to: {filepath}")


def main():
    """Run all benchmarks"""
    print("="*70)
    print("KV Cache Compression Performance Benchmark")
    print("="*70)

    benchmark = CompressionBenchmark()

    # Run benchmarks
    benchmark.benchmark_compression_time(
        sequence_lengths=[256, 512, 1024, 2048],
        compression_ratios=[2, 4, 8],
        n_runs=3
    )

    benchmark.benchmark_memory_savings(
        T=1024,
        compression_ratios=[2, 4, 8]
    )

    benchmark.benchmark_quality_vs_ratio(
        T=1024,
        compression_ratios=[2, 4, 8, 16]
    )

    benchmark.benchmark_inference_overhead(
        T=1024,
        t=256,
        n_runs=10
    )

    # Print summary
    benchmark.print_summary()

    # Save results
    benchmark.save_results()

    print(f"\n{'='*70}")
    print("✅ Benchmark Complete")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
