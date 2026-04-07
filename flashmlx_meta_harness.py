#!/usr/bin/env python3
"""
FlashMLX Meta-Harness: End-to-End Optimization Framework

Inspired by Meta-Harness paper (arXiv:2603.28052) methodology:
- Automated hyperparameter search
- Multi-objective optimization (quality + speed + memory)
- Pareto frontier analysis
- Model-adaptive configuration generation

Usage:
    # Auto-tune for a specific model
    harness = FlashMLXMetaHarness("/path/to/model")
    best_config = harness.optimize(target='balanced', n_trials=20)

    # Visualize Pareto frontier
    harness.plot_pareto_frontier()
"""

import sys
sys.path.insert(0, "mlx-lm-source")

import json
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import numpy as np
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""
    # KV cache strategy
    kv_cache: str = "triple_pq"  # 'standard', 'triple_pq', etc.
    kv_warm_bits: Optional[int] = None
    strategy: Optional[str] = None  # 'polarquant', 'turboangle'

    # TurboAngle specific
    n_k: Optional[int] = None
    n_v: Optional[int] = None

    # Route 0 (Density Router)
    density_mode: Optional[str] = None  # 'balanced', 'ultra_long', 'recall_first'
    density_scale: Optional[float] = None

    # Context settings
    context_length: int = 4096

    def to_cache_kwargs(self) -> Dict[str, Any]:
        """Convert to cache factory kwargs."""
        kwargs = {}

        if self.kv_cache != "standard":
            kwargs["kv_cache"] = self.kv_cache

        if self.kv_warm_bits is not None:
            kwargs["kv_warm_bits"] = self.kv_warm_bits

        if self.strategy is not None:
            kwargs["kv_warm_quantizer"] = self.strategy

        if self.n_k is not None and self.n_v is not None:
            from mlx_lm.models.turboangle import TurboAngleQuantizer
            # Create per-layer quantizers
            num_layers = 32  # TODO: get from model
            kwargs["kv_layer_quantizers"] = {
                i: TurboAngleQuantizer(n_k=self.n_k, n_v=self.n_v, head_dim=128)
                for i in range(num_layers)
            }

        if self.density_mode is not None:
            kwargs["density_mode"] = self.density_mode

        if self.density_scale is not None:
            kwargs["density_scale"] = self.density_scale

        return kwargs

    def __hash__(self):
        """Make config hashable for deduplication."""
        return hash(json.dumps(asdict(self), sort_keys=True))


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    config: BenchmarkConfig

    # Performance metrics
    perplexity: float
    tokens_per_sec: float
    peak_memory_mb: float

    # Derived scores
    quality_score: float  # 1 / (1 + PPL)
    speed_score: float
    memory_score: float
    pareto_score: float  # Weighted combination

    # Metadata
    duration_sec: float
    timestamp: str


class FlashMLXMetaHarness:
    """
    End-to-end optimization framework for FlashMLX configurations.

    Implements multi-objective optimization to find optimal KV cache
    configurations balancing quality, speed, and memory usage.
    """

    def __init__(
        self,
        model_path: str,
        test_prompt: str = "The quick brown fox jumps over the lazy dog. " * 500,
        baseline_ppl: Optional[float] = None,
        baseline_speed: Optional[float] = None,
    ):
        """
        Initialize meta-harness.

        Parameters
        ----------
        model_path : str
            Path to MLX model
        test_prompt : str
            Text for benchmarking (default: ~500 tokens)
        baseline_ppl : float, optional
            Baseline perplexity for normalization
        baseline_speed : float, optional
            Baseline speed (tok/s) for normalization
        """
        self.model_path = model_path
        self.test_prompt = test_prompt
        self.baseline_ppl = baseline_ppl
        self.baseline_speed = baseline_speed

        # Load model once
        print(f"Loading model from {model_path}...")
        self.model, self.tokenizer = load(model_path)

        # Get layers (handle different model structures)
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            self.num_layers = len(self.model.model.layers)
        elif hasattr(self.model, 'language_model') and hasattr(self.model.language_model, 'model'):
            self.num_layers = len(self.model.language_model.model.layers)
        else:
            raise ValueError(f"Cannot find layers in model structure")

        # Tokenize test prompt
        self.test_tokens = mx.array([self.tokenizer.encode(test_prompt)])
        self.num_tokens = self.test_tokens.shape[1]
        print(f"Test prompt: {self.num_tokens} tokens")

        # Results storage
        self.results: List[BenchmarkResult] = []

    def get_search_space(self, target: str = 'balanced') -> List[BenchmarkConfig]:
        """
        Generate search space based on target optimization goal.

        Parameters
        ----------
        target : str
            'speed' | 'memory' | 'quality' | 'balanced'

        Returns
        -------
        configs : List[BenchmarkConfig]
            List of configurations to try
        """
        configs = []

        # Always include standard baseline
        configs.append(BenchmarkConfig(kv_cache="standard"))

        if target in ['memory', 'balanced']:
            # PolarQuant configurations (only supports 2, 3, 4 bits)
            for bits in [2, 3, 4]:
                configs.append(BenchmarkConfig(
                    kv_cache="triple_pq",
                    kv_warm_bits=bits,
                    strategy="polarquant",
                ))

            # Density router modes
            for mode in ['balanced', 'ultra_long', 'recall_first']:
                configs.append(BenchmarkConfig(
                    kv_cache="triple_pq",
                    kv_warm_bits=4,
                    strategy="polarquant",
                    density_mode=mode,
                ))

        if target in ['quality', 'balanced']:
            # TurboAngle configurations
            for n_k, n_v in [(128, 64), (256, 128), (64, 32)]:
                configs.append(BenchmarkConfig(
                    kv_cache="triple_pq",
                    strategy="turboangle",
                    n_k=n_k,
                    n_v=n_v,
                ))

        if target == 'speed':
            # Fast configurations (lower precision)
            configs.append(BenchmarkConfig(
                kv_cache="triple_pq",
                kv_warm_bits=2,
                strategy="polarquant",
            ))

        # Deduplicate
        return list(set(configs))

    def benchmark_config(self, config: BenchmarkConfig) -> BenchmarkResult:
        """
        Run benchmark for a single configuration.

        Parameters
        ----------
        config : BenchmarkConfig
            Configuration to benchmark

        Returns
        -------
        result : BenchmarkResult
            Benchmark results
        """
        print(f"\nBenchmarking: {config}")

        # Clean state
        mx.clear_cache()
        mx.reset_peak_memory()

        start_time = time.perf_counter()

        # Create cache
        cache_kwargs = config.to_cache_kwargs()
        cache = make_prompt_cache(self.model, **cache_kwargs)

        # Warmup
        _ = self.model(self.test_tokens[:, :10], cache=cache)
        mx.eval(_)

        # Reset for actual measurement
        mx.clear_cache()
        mx.reset_peak_memory()

        # Forward pass
        forward_start = time.perf_counter()
        logits = self.model(self.test_tokens, cache=cache)
        mx.eval(logits)
        forward_time = time.perf_counter() - forward_start

        # Measure memory
        peak_memory_mb = mx.get_peak_memory() / (1024**2)

        # Compute perplexity
        # Shift logits and tokens for next-token prediction
        shift_logits = logits[:, :-1, :]
        shift_tokens = self.test_tokens[:, 1:]

        # Cross-entropy loss (simplified calculation)
        # Take softmax, then compute negative log likelihood
        probs = mx.softmax(shift_logits, axis=-1)

        # For each position, gather the probability of the correct token
        # Use a simpler approach: compute full cross-entropy
        batch_size, seq_len, vocab_size = shift_logits.shape

        # Vectorized perplexity calculation (much faster!)
        # Use MLX's gather/take to extract correct token probabilities
        probs_flat = probs.reshape(-1, vocab_size)
        tokens_flat = shift_tokens.reshape(-1).astype(mx.int32)

        # Extract probabilities of correct tokens using advanced indexing
        # Create row indices
        row_indices = mx.arange(tokens_flat.shape[0])
        # Gather probabilities: probs_flat[row_indices, tokens_flat]
        correct_probs = probs_flat[row_indices, tokens_flat]

        # Compute log probabilities (vectorized)
        log_probs = mx.log(correct_probs + 1e-10)
        loss = -mx.mean(log_probs)
        perplexity = mx.exp(loss).item()

        # Compute speed (tokens/sec)
        tokens_per_sec = self.num_tokens / forward_time

        duration = time.perf_counter() - start_time

        # Normalize scores
        quality_score = 1.0 / (1.0 + perplexity)

        if self.baseline_speed is not None:
            speed_score = tokens_per_sec / self.baseline_speed
        else:
            speed_score = tokens_per_sec / 100.0  # Assume 100 tok/s baseline

        # Memory score: lower is better
        memory_score = 1.0 - min(peak_memory_mb / 10000.0, 1.0)

        # Pareto score: weighted combination
        pareto_score = (
            0.5 * quality_score +
            0.3 * speed_score +
            0.2 * memory_score
        )

        result = BenchmarkResult(
            config=config,
            perplexity=perplexity,
            tokens_per_sec=tokens_per_sec,
            peak_memory_mb=peak_memory_mb,
            quality_score=quality_score,
            speed_score=speed_score,
            memory_score=memory_score,
            pareto_score=pareto_score,
            duration_sec=duration,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

        print(f"  PPL={perplexity:.4f}, Speed={tokens_per_sec:.1f} tok/s, Memory={peak_memory_mb:.1f} MB")
        print(f"  Pareto Score={pareto_score:.4f}")

        # Cleanup
        del cache
        mx.clear_cache()

        return result

    def optimize(
        self,
        target: str = 'balanced',
        n_trials: Optional[int] = None,
    ) -> BenchmarkConfig:
        """
        Run optimization to find best configuration.

        Parameters
        ----------
        target : str
            'speed' | 'memory' | 'quality' | 'balanced'
        n_trials : int, optional
            Number of trials (default: all configs in search space)

        Returns
        -------
        best_config : BenchmarkConfig
            Best configuration found
        """
        print(f"\n{'='*80}")
        print(f"FlashMLX Meta-Harness Optimization")
        print(f"Target: {target}")
        print(f"{'='*80}\n")

        # Generate search space
        configs = self.get_search_space(target)

        if n_trials is not None:
            configs = configs[:n_trials]

        print(f"Search space: {len(configs)} configurations\n")

        # Benchmark all configs with progress tracking
        start_time = time.perf_counter()
        for i, config in enumerate(configs, 1):
            print(f"\n[{i}/{len(configs)}]")

            # Time estimate
            if i > 1:
                elapsed = time.perf_counter() - start_time
                avg_time = elapsed / (i - 1)
                remaining = avg_time * (len(configs) - i + 1)
                print(f"Progress: {i-1}/{len(configs)} complete, Est. remaining: {remaining/60:.1f} min")

            result = self.benchmark_config(config)
            self.results.append(result)

        # Total time
        total_time = time.perf_counter() - start_time
        print(f"\n{'='*80}")
        print(f"OPTIMIZATION COMPLETE")
        print(f"  Total time: {total_time/60:.1f} min ({total_time:.1f} sec)")
        print(f"  Avg per config: {total_time/len(configs):.1f} sec")
        print(f"{'='*80}\n")

        # Find best config by Pareto score
        best_result = max(self.results, key=lambda r: r.pareto_score)

        print(f"\n{'='*80}")
        print(f"BEST CONFIGURATION")
        print(f"{'='*80}")
        print(f"Config: {best_result.config}")
        print(f"  PPL: {best_result.perplexity:.4f}")
        print(f"  Speed: {best_result.tokens_per_sec:.1f} tok/s")
        print(f"  Memory: {best_result.peak_memory_mb:.1f} MB")
        print(f"  Pareto Score: {best_result.pareto_score:.4f}")

        return best_result.config

    def get_pareto_frontier(self) -> List[BenchmarkResult]:
        """
        Compute Pareto frontier from results.

        Returns
        -------
        frontier : List[BenchmarkResult]
            Non-dominated solutions
        """
        if not self.results:
            return []

        frontier = []

        for candidate in self.results:
            dominated = False

            for other in self.results:
                if candidate is other:
                    continue

                # Check if other dominates candidate
                # (better in at least one objective, not worse in any)
                better_quality = other.quality_score > candidate.quality_score
                better_speed = other.speed_score > candidate.speed_score
                better_memory = other.memory_score > candidate.memory_score

                worse_quality = other.quality_score < candidate.quality_score
                worse_speed = other.speed_score < candidate.speed_score
                worse_memory = other.memory_score < candidate.memory_score

                if (better_quality or better_speed or better_memory) and \
                   not (worse_quality or worse_speed or worse_memory):
                    dominated = True
                    break

            if not dominated:
                frontier.append(candidate)

        return frontier

    def save_results(self, output_path: str):
        """Save results to JSON file."""
        data = {
            'model_path': self.model_path,
            'num_tokens': self.num_tokens,
            'results': [
                {
                    'config': asdict(r.config),
                    'perplexity': r.perplexity,
                    'tokens_per_sec': r.tokens_per_sec,
                    'peak_memory_mb': r.peak_memory_mb,
                    'quality_score': r.quality_score,
                    'speed_score': r.speed_score,
                    'memory_score': r.memory_score,
                    'pareto_score': r.pareto_score,
                    'timestamp': r.timestamp,
                }
                for r in self.results
            ],
            'pareto_frontier': [
                asdict(r.config) for r in self.get_pareto_frontier()
            ],
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\nResults saved to {output_path}")

    def print_summary(self):
        """Print summary table of all results."""
        if not self.results:
            print("No results yet.")
            return

        print(f"\n{'='*100}")
        print(f"BENCHMARK SUMMARY ({len(self.results)} configurations)")
        print(f"{'='*100}")
        print(f"{'Config':<40} {'PPL':>8} {'Speed':>12} {'Memory':>12} {'Pareto':>10}")
        print(f"{'-'*100}")

        # Sort by Pareto score
        sorted_results = sorted(self.results, key=lambda r: r.pareto_score, reverse=True)

        for r in sorted_results:
            config_str = f"{r.config.strategy or 'standard'}"
            if r.config.kv_warm_bits:
                config_str += f" {r.config.kv_warm_bits}bit"
            if r.config.n_k and r.config.n_v:
                config_str += f" K{r.config.n_k}V{r.config.n_v}"
            if r.config.density_mode:
                config_str += f" {r.config.density_mode}"

            print(f"{config_str:<40} {r.perplexity:>8.4f} {r.tokens_per_sec:>10.1f}/s "
                  f"{r.peak_memory_mb:>10.1f}MB {r.pareto_score:>10.4f}")

        print(f"{'-'*100}")
        print(f"Pareto frontier: {len(self.get_pareto_frontier())} configurations")


def main():
    """Example usage."""
    import argparse

    parser = argparse.ArgumentParser(description="FlashMLX Meta-Harness Optimization")
    parser.add_argument("model_path", help="Path to MLX model")
    parser.add_argument("--target", choices=['speed', 'memory', 'quality', 'balanced'],
                        default='balanced', help="Optimization target")
    parser.add_argument("--n-trials", type=int, help="Number of trials (default: all)")
    parser.add_argument("--output", default="meta_harness_results.json",
                        help="Output JSON file")
    parser.add_argument("--context-len", type=int, default=4096,
                        help="Context length for testing")

    args = parser.parse_args()

    # Run optimization
    harness = FlashMLXMetaHarness(args.model_path)
    best_config = harness.optimize(target=args.target, n_trials=args.n_trials)

    # Print summary
    harness.print_summary()

    # Save results
    harness.save_results(args.output)


if __name__ == "__main__":
    main()
