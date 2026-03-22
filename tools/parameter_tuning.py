"""
Parameter Tuning for Hybrid Cache (Task #81)

Finds optimal compression_ratio and budget configurations.
Generates Pareto frontier for memory vs performance trade-offs.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np


class ParameterTuner:
    """Parameter tuning for hybrid cache"""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Parameter search space
        self.compression_ratios = [2.0, 3.0, 4.0, 5.0]
        self.budget_sizes_mb = [64, 128, 256, 512]

        # Budget tier ratios (default configuration)
        self.default_tier_ratios = {
            "hot": 0.15,
            "warm": 0.25,
            "cold": 0.55,
            "pinned": 0.05
        }

        # Scenario definitions
        self.scenarios = {
            "short_context": {
                "context_tokens": 512,
                "generation_tokens": 100,
                "description": "Short context, quick response"
            },
            "medium_context": {
                "context_tokens": 2048,
                "generation_tokens": 200,
                "description": "Medium context, balanced usage"
            },
            "long_context": {
                "context_tokens": 4096,
                "generation_tokens": 500,
                "description": "Long context, extensive generation"
            }
        }

    def estimate_memory_savings(
        self,
        compression_ratio: float,
        budget_mb: int,
        context_tokens: int
    ) -> float:
        """
        Estimate memory savings percentage.

        Args:
            compression_ratio: Compression ratio for Attention layers
            budget_mb: Total cache budget in MB
            context_tokens: Context length in tokens

        Returns:
            Estimated memory savings percentage
        """
        # Qwen3.5: 40 layers (30 SSM + 10 Attention)
        num_attention_layers = 10
        num_ssm_layers = 30
        total_layers = 40

        # Estimate baseline KV cache size (simplified)
        # Assumptions:
        # - Each layer stores 2 arrays (K, V)
        # - Each array: batch_size × num_heads × seq_len × head_dim
        # - Qwen3.5: 8 heads × 64 head_dim × 4 bytes (float32)
        bytes_per_token_per_layer = 8 * 64 * 4 * 2  # K + V
        baseline_kv_mb = (total_layers * context_tokens * bytes_per_token_per_layer) / (1024 * 1024)

        # With compression:
        # - Attention layers: compressed by compression_ratio
        # - SSM layers: no compression
        attention_portion = num_attention_layers / total_layers
        ssm_portion = num_ssm_layers / total_layers

        # Compressed size
        attention_compressed_mb = (attention_portion * baseline_kv_mb) / compression_ratio
        ssm_mb = ssm_portion * baseline_kv_mb
        hybrid_kv_mb = attention_compressed_mb + ssm_mb

        # Savings
        savings_percent = ((baseline_kv_mb - hybrid_kv_mb) / baseline_kv_mb) * 100

        return savings_percent

    def estimate_performance_overhead(
        self,
        compression_ratio: float,
        budget_mb: int,
        context_tokens: int,
        generation_tokens: int
    ) -> Dict[str, float]:
        """
        Estimate performance overhead.

        Args:
            compression_ratio: Compression ratio
            budget_mb: Cache budget in MB
            context_tokens: Context length
            generation_tokens: Generation length

        Returns:
            Dictionary with TTFT and TBT overhead percentages
        """
        # Overhead sources:
        # 1. β calibration: ~0.5ms per layer
        # 2. Attention matching: ~10-50ms per layer (depends on compression ratio)
        # 3. Compressed KV retrieval: ~0.5-1.0ms per layer

        num_attention_layers = 10

        # TTFT overhead (prefill phase)
        beta_calibration_ms = 0.5 * num_attention_layers

        # Attention matching time increases with compression ratio
        # Higher compression = more computation
        attention_matching_base_ms = 30.0  # Base time per layer
        attention_matching_factor = 1.0 + (compression_ratio - 2.0) * 0.2  # +20% per unit
        attention_matching_ms = attention_matching_base_ms * attention_matching_factor * num_attention_layers

        total_prefill_overhead_ms = beta_calibration_ms + attention_matching_ms

        # Baseline TTFT scales with context length
        baseline_ttft_ms = context_tokens * 0.6  # ~0.6ms per token

        ttft_overhead_percent = (total_prefill_overhead_ms / baseline_ttft_ms) * 100

        # TBT overhead (decode phase)
        kv_retrieval_ms = 0.7 * num_attention_layers  # Per token
        baseline_tbt_ms = 17.0  # 17ms per token baseline

        tbt_overhead_percent = (kv_retrieval_ms / baseline_tbt_ms) * 100

        return {
            "ttft_overhead_percent": ttft_overhead_percent,
            "tbt_overhead_percent": tbt_overhead_percent
        }

    def evaluate_configuration(
        self,
        compression_ratio: float,
        budget_mb: int,
        scenario: str
    ) -> Dict:
        """
        Evaluate a single configuration.

        Args:
            compression_ratio: Compression ratio
            budget_mb: Cache budget in MB
            scenario: Scenario name

        Returns:
            Evaluation results
        """
        scenario_config = self.scenarios[scenario]

        # Memory savings
        memory_savings = self.estimate_memory_savings(
            compression_ratio,
            budget_mb,
            scenario_config["context_tokens"]
        )

        # Performance overhead
        overhead = self.estimate_performance_overhead(
            compression_ratio,
            budget_mb,
            scenario_config["context_tokens"],
            scenario_config["generation_tokens"]
        )

        # Quality score (simplified heuristic)
        # - Higher compression may reduce quality slightly
        # - Assume 2x = 100%, 5x = 95%
        quality_score = 100 - (compression_ratio - 2.0) * 1.67

        # Overall score (weighted)
        # Memory: 40%, Performance: 40%, Quality: 20%
        overall_score = (
            memory_savings * 0.4 -
            overhead["ttft_overhead_percent"] * 0.2 -
            overhead["tbt_overhead_percent"] * 0.2 +
            quality_score * 0.2
        )

        return {
            "compression_ratio": compression_ratio,
            "budget_mb": budget_mb,
            "memory_savings_percent": memory_savings,
            "ttft_overhead_percent": overhead["ttft_overhead_percent"],
            "tbt_overhead_percent": overhead["tbt_overhead_percent"],
            "quality_score": quality_score,
            "overall_score": overall_score
        }

    def run_parameter_sweep(self) -> Dict[str, List[Dict]]:
        """
        Run parameter sweep across all configurations.

        Returns:
            Results for each scenario
        """
        results = {}

        for scenario in self.scenarios.keys():
            print(f"\n{'='*60}")
            print(f"Scenario: {scenario}")
            print(f"{'='*60}")

            scenario_results = []

            for compression_ratio in self.compression_ratios:
                for budget_mb in self.budget_sizes_mb:
                    result = self.evaluate_configuration(
                        compression_ratio,
                        budget_mb,
                        scenario
                    )

                    scenario_results.append(result)

                    print(f"\nCompression: {compression_ratio}x, Budget: {budget_mb}MB")
                    print(f"  Memory savings: {result['memory_savings_percent']:.1f}%")
                    print(f"  TTFT overhead: {result['ttft_overhead_percent']:.1f}%")
                    print(f"  TBT overhead: {result['tbt_overhead_percent']:.1f}%")
                    print(f"  Quality score: {result['quality_score']:.1f}")
                    print(f"  Overall score: {result['overall_score']:.1f}")

            results[scenario] = scenario_results

        return results

    def find_pareto_frontier(
        self,
        results: List[Dict]
    ) -> List[Dict]:
        """
        Find Pareto frontier (non-dominated configurations).

        A configuration is Pareto optimal if no other configuration is better
        in all objectives (memory savings, low overhead).

        Args:
            results: List of configuration results

        Returns:
            Pareto optimal configurations
        """
        pareto_frontier = []

        for candidate in results:
            is_dominated = False

            for other in results:
                if other == candidate:
                    continue

                # Check if 'other' dominates 'candidate'
                # Domination: better in all objectives
                better_memory = other["memory_savings_percent"] > candidate["memory_savings_percent"]
                better_ttft = other["ttft_overhead_percent"] < candidate["ttft_overhead_percent"]
                better_tbt = other["tbt_overhead_percent"] < candidate["tbt_overhead_percent"]

                # If better in all, candidate is dominated
                if better_memory and better_ttft and better_tbt:
                    is_dominated = True
                    break

            if not is_dominated:
                pareto_frontier.append(candidate)

        return pareto_frontier

    def generate_recommendations(
        self,
        results: Dict[str, List[Dict]]
    ) -> Dict[str, Dict]:
        """
        Generate recommended configurations for each scenario.

        Args:
            results: Results from parameter sweep

        Returns:
            Recommended configurations
        """
        recommendations = {}

        for scenario, scenario_results in results.items():
            # Sort by overall score
            sorted_results = sorted(
                scenario_results,
                key=lambda x: x["overall_score"],
                reverse=True
            )

            # Top 3 configurations
            top_3 = sorted_results[:3]

            # Find Pareto frontier
            pareto = self.find_pareto_frontier(scenario_results)

            # Best balanced configuration (highest overall score)
            best_balanced = top_3[0]

            # Best memory savings
            best_memory = max(
                scenario_results,
                key=lambda x: x["memory_savings_percent"]
            )

            # Best performance (lowest overhead)
            best_performance = min(
                scenario_results,
                key=lambda x: x["ttft_overhead_percent"] + x["tbt_overhead_percent"]
            )

            recommendations[scenario] = {
                "best_balanced": best_balanced,
                "best_memory": best_memory,
                "best_performance": best_performance,
                "top_3": top_3,
                "pareto_frontier": pareto
            }

        return recommendations

    def plot_pareto_frontier(
        self,
        results: Dict[str, List[Dict]],
        recommendations: Dict[str, Dict]
    ):
        """
        Plot Pareto frontier for each scenario.

        Args:
            results: Parameter sweep results
            recommendations: Recommended configurations
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for idx, (scenario, scenario_results) in enumerate(results.items()):
            ax = axes[idx]

            # Extract data
            memory_savings = [r["memory_savings_percent"] for r in scenario_results]
            total_overhead = [
                r["ttft_overhead_percent"] + r["tbt_overhead_percent"]
                for r in scenario_results
            ]

            # Plot all configurations
            ax.scatter(memory_savings, total_overhead, alpha=0.5, s=50, label="All configs")

            # Plot Pareto frontier
            pareto = recommendations[scenario]["pareto_frontier"]
            pareto_memory = [r["memory_savings_percent"] for r in pareto]
            pareto_overhead = [
                r["ttft_overhead_percent"] + r["tbt_overhead_percent"]
                for r in pareto
            ]
            ax.scatter(pareto_memory, pareto_overhead, color='red', s=100,
                      marker='*', label="Pareto frontier", zorder=5)

            # Highlight best balanced
            best = recommendations[scenario]["best_balanced"]
            ax.scatter([best["memory_savings_percent"]],
                      [best["ttft_overhead_percent"] + best["tbt_overhead_percent"]],
                      color='green', s=150, marker='D', label="Best balanced", zorder=6)

            ax.set_xlabel("Memory Savings (%)", fontsize=12)
            ax.set_ylabel("Total Overhead (TTFT + TBT) (%)", fontsize=12)
            ax.set_title(f"{scenario.replace('_', ' ').title()}\n{self.scenarios[scenario]['description']}",
                        fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = self.output_dir / "pareto_frontier.png"
        plt.savefig(plot_path, dpi=300)
        print(f"\n✅ Pareto frontier plot saved: {plot_path}")

    def save_results(
        self,
        results: Dict[str, List[Dict]],
        recommendations: Dict[str, Dict]
    ):
        """
        Save results to JSON files.

        Args:
            results: Full parameter sweep results
            recommendations: Recommended configurations
        """
        # Save full results
        results_path = self.output_dir / "parameter_sweep_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✅ Full results saved: {results_path}")

        # Save recommendations
        recommendations_path = self.output_dir / "recommended_configurations.json"
        with open(recommendations_path, 'w') as f:
            json.dump(recommendations, f, indent=2)
        print(f"✅ Recommendations saved: {recommendations_path}")

    def generate_config_templates(
        self,
        recommendations: Dict[str, Dict]
    ):
        """
        Generate configuration file templates.

        Args:
            recommendations: Recommended configurations
        """
        templates_dir = self.output_dir / "config_templates"
        templates_dir.mkdir(exist_ok=True)

        for scenario, recs in recommendations.items():
            best = recs["best_balanced"]

            config = {
                "scenario": scenario,
                "description": self.scenarios[scenario]["description"],
                "hybrid_cache_config": {
                    "total_budget_bytes": best["budget_mb"] * 1024 * 1024,
                    "compression_ratio": best["compression_ratio"],
                    "beta_calibration": True,
                    "hot_budget_ratio": self.default_tier_ratios["hot"],
                    "warm_budget_ratio": self.default_tier_ratios["warm"],
                    "cold_budget_ratio": self.default_tier_ratios["cold"],
                    "pinned_budget_ratio": self.default_tier_ratios["pinned"]
                },
                "expected_performance": {
                    "memory_savings_percent": best["memory_savings_percent"],
                    "ttft_overhead_percent": best["ttft_overhead_percent"],
                    "tbt_overhead_percent": best["tbt_overhead_percent"],
                    "quality_score": best["quality_score"]
                }
            }

            config_path = templates_dir / f"{scenario}_config.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)

        print(f"✅ Configuration templates saved: {templates_dir}")

    def run(self):
        """Run complete parameter tuning pipeline"""
        print("\n" + "=" * 60)
        print("Hybrid Cache Parameter Tuning")
        print("=" * 60)

        # 1. Parameter sweep
        print("\n1. Running parameter sweep...")
        results = self.run_parameter_sweep()

        # 2. Generate recommendations
        print("\n2. Generating recommendations...")
        recommendations = self.generate_recommendations(results)

        # 3. Print recommendations
        print("\n" + "=" * 60)
        print("Recommended Configurations")
        print("=" * 60)
        for scenario, recs in recommendations.items():
            print(f"\n{scenario.upper()}:")
            best = recs["best_balanced"]
            print(f"  Compression ratio: {best['compression_ratio']}x")
            print(f"  Budget: {best['budget_mb']}MB")
            print(f"  Memory savings: {best['memory_savings_percent']:.1f}%")
            print(f"  TTFT overhead: {best['ttft_overhead_percent']:.1f}%")
            print(f"  TBT overhead: {best['tbt_overhead_percent']:.1f}%")

        # 4. Plot Pareto frontier
        print("\n3. Plotting Pareto frontier...")
        self.plot_pareto_frontier(results, recommendations)

        # 5. Save results
        print("\n4. Saving results...")
        self.save_results(results, recommendations)

        # 6. Generate config templates
        print("\n5. Generating configuration templates...")
        self.generate_config_templates(recommendations)

        print("\n" + "=" * 60)
        print("✅ Parameter tuning completed successfully!")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Hybrid Cache Parameter Tuning")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tuning_results"),
        help="Output directory for results"
    )

    args = parser.parse_args()

    tuner = ParameterTuner(args.output_dir)
    tuner.run()


if __name__ == "__main__":
    main()
