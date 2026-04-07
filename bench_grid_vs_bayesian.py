#!/usr/bin/env python3
"""
Compare Grid Search vs Bayesian Optimization for Meta-Harness.

Measures:
- Number of trials needed
- Time to find optimal configuration
- Quality of final result
"""

import time
import json
from flashmlx_meta_harness import FlashMLXMetaHarness
from flashmlx_bayesian_optimizer import BayesianMetaHarness


def run_grid_search(model_path: str, target: str = 'balanced'):
    """Run traditional grid search."""
    print(f"\n{'='*80}")
    print("GRID SEARCH (Traditional)")
    print(f"{'='*80}\n")

    harness = FlashMLXMetaHarness(model_path)

    start_time = time.perf_counter()
    best_config = harness.optimize(target=target)
    total_time = time.perf_counter() - start_time

    n_trials = len(harness.results)
    best_result = max(harness.results, key=lambda r: r.pareto_score)

    return {
        'method': 'grid_search',
        'n_trials': n_trials,
        'total_time': total_time,
        'best_pareto_score': best_result.pareto_score,
        'best_speed': best_result.tokens_per_sec,
        'best_memory': best_result.peak_memory_mb,
        'best_ppl': best_result.perplexity,
    }


def run_bayesian_optimization(model_path: str, target: str = 'balanced', n_trials: int = 7):
    """Run Bayesian optimization."""
    print(f"\n{'='*80}")
    print("BAYESIAN OPTIMIZATION (TPE)")
    print(f"{'='*80}\n")

    optimizer = BayesianMetaHarness(model_path)

    start_time = time.perf_counter()
    best_config = optimizer.optimize(target=target, n_trials=n_trials)
    total_time = time.perf_counter() - start_time

    best_result = max(optimizer.harness.results, key=lambda r: r.pareto_score)

    return {
        'method': 'bayesian_optimization',
        'n_trials': len(optimizer.harness.results),
        'total_time': total_time,
        'best_pareto_score': best_result.pareto_score,
        'best_speed': best_result.tokens_per_sec,
        'best_memory': best_result.peak_memory_mb,
        'best_ppl': best_result.perplexity,
    }


def compare_methods(model_path: str, target: str = 'balanced', bayesian_trials: int = 7):
    """
    Compare grid search vs Bayesian optimization.

    Parameters
    ----------
    model_path : str
        Path to MLX model
    target : str
        Optimization target ('balanced', 'speed', 'memory', 'quality')
    bayesian_trials : int
        Number of trials for Bayesian optimization (default: 7)
    """
    print(f"\n{'='*80}")
    print("META-HARNESS METHOD COMPARISON")
    print(f"Model: {model_path}")
    print(f"Target: {target}")
    print(f"{'='*80}\n")

    # Run grid search
    grid_results = run_grid_search(model_path, target)

    # Run Bayesian optimization
    bayesian_results = run_bayesian_optimization(model_path, target, bayesian_trials)

    # Compare results
    print(f"\n{'='*80}")
    print("COMPARISON RESULTS")
    print(f"{'='*80}\n")

    print(f"{'Metric':<30} {'Grid Search':>15} {'Bayesian':>15} {'Improvement':>15}")
    print(f"{'-'*80}")

    # Number of trials
    grid_trials = grid_results['n_trials']
    bayes_trials = bayesian_results['n_trials']
    trials_saved = grid_trials - bayes_trials
    trials_pct = (trials_saved / grid_trials) * 100
    print(f"{'Trials':<30} {grid_trials:>15} {bayes_trials:>15} {trials_saved:>14} ({trials_pct:+.0f}%)")

    # Total time
    grid_time = grid_results['total_time']
    bayes_time = bayesian_results['total_time']
    time_saved = grid_time - bayes_time
    time_pct = (time_saved / grid_time) * 100
    print(f"{'Time (seconds)':<30} {grid_time:>15.1f} {bayes_time:>15.1f} {time_saved:>14.1f} ({time_pct:+.0f}%)")

    # Quality comparison
    grid_score = grid_results['best_pareto_score']
    bayes_score = bayesian_results['best_pareto_score']
    score_diff = bayes_score - grid_score
    score_pct = (score_diff / grid_score) * 100
    print(f"{'Pareto Score':<30} {grid_score:>15.4f} {bayes_score:>15.4f} {score_diff:>14.4f} ({score_pct:+.1f}%)")

    # Speed comparison
    grid_speed = grid_results['best_speed']
    bayes_speed = bayesian_results['best_speed']
    speed_diff = bayes_speed - grid_speed
    speed_pct = (speed_diff / grid_speed) * 100
    print(f"{'Speed (tok/s)':<30} {grid_speed:>15.1f} {bayes_speed:>15.1f} {speed_diff:>14.1f} ({speed_pct:+.1f}%)")

    # Memory comparison
    grid_mem = grid_results['best_memory']
    bayes_mem = bayesian_results['best_memory']
    mem_diff = bayes_mem - grid_mem
    mem_pct = (mem_diff / grid_mem) * 100
    print(f"{'Memory (MB)':<30} {grid_mem:>15.1f} {bayes_mem:>15.1f} {mem_diff:>14.1f} ({mem_pct:+.1f}%)")

    print(f"\n{'='*80}")
    print("CONCLUSION")
    print(f"{'='*80}\n")

    if trials_saved > 0:
        print(f"✅ Bayesian optimization reduced trials by {trials_saved} ({trials_pct:.0f}%)")
    if time_saved > 0:
        print(f"✅ Bayesian optimization saved {time_saved:.1f} seconds ({time_pct:.0f}%)")
    if abs(score_pct) < 2:
        print(f"✅ Quality equivalent (Pareto score within 2%)")
    elif score_pct > 0:
        print(f"✅ Bayesian optimization found BETTER configuration (+{score_pct:.1f}%)")
    else:
        print(f"⚠️  Grid search found slightly better configuration ({score_pct:.1f}%)")

    # Save comparison results
    comparison = {
        'grid_search': grid_results,
        'bayesian_optimization': bayesian_results,
        'savings': {
            'trials': trials_saved,
            'trials_pct': trials_pct,
            'time': time_saved,
            'time_pct': time_pct,
        },
        'quality_comparison': {
            'pareto_score_diff': score_diff,
            'pareto_score_pct': score_pct,
        }
    }

    output_path = f"comparison_grid_vs_bayesian_{target}.json"
    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"\nResults saved to {output_path}")


def main():
    """CLI for method comparison."""
    import argparse

    parser = argparse.ArgumentParser(description="Compare Grid Search vs Bayesian Optimization")
    parser.add_argument('model_path', type=str, help='Path to MLX model')
    parser.add_argument('--target', type=str, default='balanced',
                        choices=['speed', 'memory', 'quality', 'balanced'],
                        help='Optimization target')
    parser.add_argument('--bayesian-trials', type=int, default=7,
                        help='Number of trials for Bayesian optimization (default: 7)')

    args = parser.parse_args()

    compare_methods(args.model_path, args.target, args.bayesian_trials)


if __name__ == '__main__':
    main()
