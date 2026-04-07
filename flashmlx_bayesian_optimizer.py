#!/usr/bin/env python3
"""
FlashMLX Bayesian Optimizer for KV Cache Hyperparameter Search

Uses Tree-structured Parzen Estimator (TPE) for efficient hyperparameter optimization.
Reduces search trials from ~10 (grid search) to ~5-7 (Bayesian optimization).

Based on Meta-Harness methodology (arXiv:2603.28052) with Bayesian sampling.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from scipy.stats import norm
import time

from flashmlx_meta_harness import FlashMLXMetaHarness, BenchmarkConfig, BenchmarkResult


@dataclass
class HyperparameterSpace:
    """Define hyperparameter search space."""

    name: str
    type: str  # 'categorical', 'integer', 'continuous'
    choices: Optional[List[Any]] = None
    low: Optional[float] = None
    high: Optional[float] = None

    def sample_random(self) -> Any:
        """Sample uniformly from this space."""
        if self.type == 'categorical':
            value = np.random.choice(self.choices)
            # Convert numpy types to Python native types
            return value.item() if hasattr(value, 'item') else value
        elif self.type == 'integer':
            return int(np.random.randint(self.low, self.high + 1))
        elif self.type == 'continuous':
            return float(np.random.uniform(self.low, self.high))
        else:
            raise ValueError(f"Unknown parameter type: {self.type}")


class TreeStructuredParzenEstimator:
    """
    Tree-structured Parzen Estimator (TPE) for Bayesian Optimization.

    TPE maintains two distributions:
    - l(x): distribution of parameters that led to GOOD results
    - g(x): distribution of parameters that led to BAD results

    Acquisition function: EI(x) = (g(x) - l(x)) / g(x)
    """

    def __init__(self, space: List[HyperparameterSpace], gamma: float = 0.25):
        """
        Parameters
        ----------
        space : List[HyperparameterSpace]
            Search space definition
        gamma : float
            Quantile threshold for splitting good/bad results (default: 25%)
        """
        self.space = space
        self.gamma = gamma
        self.observations: List[Tuple[Dict[str, Any], float]] = []

    def observe(self, config: Dict[str, Any], score: float):
        """Record an observation (config -> score)."""
        self.observations.append((config, score))

    def suggest(self) -> Dict[str, Any]:
        """
        Suggest next configuration using Expected Improvement.

        Returns
        -------
        config : Dict[str, Any]
            Suggested configuration
        """
        if len(self.observations) < 3:
            # Not enough data: random sampling
            return self._random_sample()

        # Split into good/bad based on quantile
        scores = [score for _, score in self.observations]
        threshold = np.quantile(scores, 1.0 - self.gamma)  # Top 25%

        good_configs = [cfg for cfg, score in self.observations if score >= threshold]
        bad_configs = [cfg for cfg, score in self.observations if score < threshold]

        if len(good_configs) < 2:
            # Not enough good samples
            return self._random_sample()

        # Sample from good distribution
        # For categorical: use empirical distribution
        # For continuous: use KDE or simple Gaussian
        suggested = {}

        for param in self.space:
            if param.type == 'categorical':
                # Sample from good configs' empirical distribution
                good_values = [cfg[param.name] for cfg in good_configs if param.name in cfg]
                if good_values:
                    # Weight by frequency
                    from collections import Counter
                    counts = Counter(good_values)
                    total = sum(counts.values())
                    probs = [counts[val] / total for val in param.choices]
                    value = np.random.choice(param.choices, p=probs)
                    # Convert numpy types to Python native types
                    suggested[param.name] = value.item() if hasattr(value, 'item') else value
                else:
                    suggested[param.name] = param.sample_random()

            elif param.type in ['integer', 'continuous']:
                # Sample from Gaussian fit to good configs
                good_values = [cfg[param.name] for cfg in good_configs if param.name in cfg and cfg[param.name] is not None]
                if len(good_values) >= 2:
                    mean = np.mean(good_values)
                    std = max(np.std(good_values), 1e-6)

                    # Sample from Gaussian (clipped to bounds)
                    value = np.random.normal(mean, std)
                    if param.type == 'integer':
                        suggested[param.name] = int(np.clip(value, param.low, param.high))
                    else:
                        suggested[param.name] = float(np.clip(value, param.low, param.high))
                else:
                    suggested[param.name] = param.sample_random()

            else:
                suggested[param.name] = None

        return suggested

    def _random_sample(self) -> Dict[str, Any]:
        """Random sampling from the space."""
        return {param.name: param.sample_random() for param in self.space}


class BayesianMetaHarness:
    """
    Bayesian Optimization wrapper for Meta-Harness.

    Uses TPE to intelligently search the hyperparameter space,
    reducing trials from ~10 (grid search) to ~5-7.
    """

    def __init__(self, model_path: str, test_prompt: Optional[str] = None):
        """
        Parameters
        ----------
        model_path : str
            Path to MLX model
        test_prompt : str, optional
            Test prompt for perplexity measurement
        """
        if test_prompt is not None:
            self.harness = FlashMLXMetaHarness(model_path, test_prompt)
        else:
            self.harness = FlashMLXMetaHarness(model_path)
        self.optimizer: Optional[TreeStructuredParzenEstimator] = None

    def define_search_space(self, target: str = 'balanced') -> List[HyperparameterSpace]:
        """
        Define search space based on optimization target.

        Parameters
        ----------
        target : str
            'speed' | 'memory' | 'quality' | 'balanced'

        Returns
        -------
        space : List[HyperparameterSpace]
            Hyperparameter search space
        """
        space = []

        # KV cache strategy (categorical)
        if target in ['memory', 'balanced']:
            space.append(HyperparameterSpace(
                name='kv_cache',
                type='categorical',
                choices=['standard', 'triple_pq']
            ))
        else:
            space.append(HyperparameterSpace(
                name='kv_cache',
                type='categorical',
                choices=['triple_pq']  # Only compressed for speed
            ))

        # Quantization strategy (categorical)
        space.append(HyperparameterSpace(
            name='strategy',
            type='categorical',
            choices=['polarquant', 'turboangle', None]  # None = no quantization
        ))

        # Bit-width for PolarQuant (integer)
        if target in ['memory', 'balanced', 'speed']:
            space.append(HyperparameterSpace(
                name='kv_warm_bits',
                type='integer',
                low=2,
                high=4
            ))

        # TurboAngle codebook sizes (categorical)
        space.append(HyperparameterSpace(
            name='n_k',
            type='categorical',
            choices=[None, 64, 128, 256]
        ))
        space.append(HyperparameterSpace(
            name='n_v',
            type='categorical',
            choices=[None, 32, 64, 128]
        ))

        # Density mode (categorical)
        if target in ['memory', 'balanced']:
            space.append(HyperparameterSpace(
                name='density_mode',
                type='categorical',
                choices=[None, 'balanced', 'ultra_long', 'recall_first']
            ))

        return space

    def config_from_params(self, params: Dict[str, Any]) -> BenchmarkConfig:
        """
        Convert hyperparameter dict to BenchmarkConfig.

        Parameters
        ----------
        params : Dict[str, Any]
            Hyperparameter values

        Returns
        -------
        config : BenchmarkConfig
            Benchmark configuration
        """
        # Handle strategy-specific parameters
        strategy = params.get('strategy')

        # Validate combinations
        if strategy == 'polarquant':
            # Use kv_warm_bits, ignore n_k/n_v
            return BenchmarkConfig(
                kv_cache=params.get('kv_cache', 'standard'),
                kv_warm_bits=params.get('kv_warm_bits'),
                strategy='polarquant',
                density_mode=params.get('density_mode'),
            )
        elif strategy == 'turboangle':
            # Use n_k/n_v, ignore kv_warm_bits
            n_k = params.get('n_k')
            n_v = params.get('n_v')
            if n_k is None or n_v is None:
                # Invalid: skip
                return None
            return BenchmarkConfig(
                kv_cache=params.get('kv_cache', 'standard'),
                strategy='turboangle',
                n_k=n_k,
                n_v=n_v,
                density_mode=params.get('density_mode'),
            )
        else:
            # No quantization (standard)
            return BenchmarkConfig(
                kv_cache=params.get('kv_cache', 'standard'),
            )

    def optimize(self, target: str = 'balanced', n_trials: int = 7) -> BenchmarkConfig:
        """
        Run Bayesian optimization to find best configuration.

        Parameters
        ----------
        target : str
            Optimization goal: 'speed' | 'memory' | 'quality' | 'balanced'
        n_trials : int
            Number of trials (default: 7, vs 10+ for grid search)

        Returns
        -------
        best_config : BenchmarkConfig
            Best configuration found
        """
        print(f"\n{'='*80}")
        print(f"FlashMLX Bayesian Meta-Harness Optimization")
        print(f"Target: {target}")
        print(f"{'='*80}\n")

        # Define search space
        space = self.define_search_space(target)
        print(f"Search space: {len(space)} hyperparameters")

        # Initialize TPE optimizer
        self.optimizer = TreeStructuredParzenEstimator(space, gamma=0.25)

        # Always benchmark standard baseline first
        print(f"\n[Baseline]\n")
        baseline_config = BenchmarkConfig(kv_cache="standard")
        baseline_result = self.harness.benchmark_config(baseline_config)
        self.harness.results.append(baseline_result)  # Add to results
        self.optimizer.observe(
            {'kv_cache': 'standard', 'strategy': None},
            baseline_result.pareto_score
        )

        # Bayesian optimization loop
        for trial in range(1, n_trials):
            print(f"\n[{trial}/{n_trials - 1}]\n")

            # Suggest next configuration
            params = self.optimizer.suggest()
            config = self.config_from_params(params)

            if config is None:
                # Invalid combination: skip
                continue

            # Check if already evaluated (avoid duplicates)
            if any(r.config == config for r in self.harness.results):
                print(f"Skipping duplicate config: {config}")
                continue

            # Benchmark
            result = self.harness.benchmark_config(config)
            self.harness.results.append(result)  # Add to results

            # Record observation
            self.optimizer.observe(params, result.pareto_score)

            print(f"  PPL={result.perplexity:.4f}, Speed={result.tokens_per_sec:.1f} tok/s, Memory={result.peak_memory_mb:.1f} MB")
            print(f"  Pareto Score={result.pareto_score:.4f}")

        # Find best result
        print(f"\n{'='*80}")
        print("BEST CONFIGURATION")
        print(f"{'='*80}")

        best_result = max(self.harness.results, key=lambda r: r.pareto_score)
        print(f"Config: {best_result.config}")
        print(f"  PPL: {best_result.perplexity:.4f}")
        print(f"  Speed: {best_result.tokens_per_sec:.1f} tok/s")
        print(f"  Memory: {best_result.peak_memory_mb:.1f} MB")
        print(f"  Pareto Score: {best_result.pareto_score:.4f}")

        return best_result.config

    def save_results(self, output_path: str):
        """Save optimization results."""
        self.harness.save_results(output_path)

    def print_summary(self):
        """Print summary of all trials."""
        self.harness.print_summary()


def main():
    """CLI for Bayesian Meta-Harness."""
    import argparse

    parser = argparse.ArgumentParser(description="FlashMLX Bayesian Optimizer")
    parser.add_argument('model_path', type=str, help='Path to MLX model')
    parser.add_argument('--test-prompt', type=str, default=None, help='Test prompt file')
    parser.add_argument('--target', type=str, default='balanced',
                        choices=['speed', 'memory', 'quality', 'balanced'],
                        help='Optimization target')
    parser.add_argument('--n-trials', type=int, default=7,
                        help='Number of Bayesian optimization trials (default: 7)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for results')

    args = parser.parse_args()

    # Run Bayesian optimization
    optimizer = BayesianMetaHarness(args.model_path, args.test_prompt)
    best_config = optimizer.optimize(target=args.target, n_trials=args.n_trials)

    # Print summary
    optimizer.print_summary()

    # Save results
    if args.output:
        optimizer.save_results(args.output)
    else:
        # Auto-generate filename
        model_name = args.model_path.rstrip('/').split('/')[-1].lower()
        output_path = f"{model_name}_bayesian_results.json"
        optimizer.save_results(output_path)


if __name__ == '__main__':
    main()
