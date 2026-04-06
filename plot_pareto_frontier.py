#!/usr/bin/env python3
"""
Visualize Pareto frontier from meta-harness results.

Usage:
    python plot_pareto_frontier.py meta_harness_results.json
"""

import json
import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_results(json_path: str):
    """Load results from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def plot_2d_pareto(results, pareto_indices, x_key, y_key, x_label, y_label, title):
    """Plot 2D Pareto frontier."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # All points
    x_all = [r[x_key] for r in results]
    y_all = [r[y_key] for r in results]

    # Pareto points
    x_pareto = [results[i][x_key] for i in pareto_indices]
    y_pareto = [results[i][y_key] for i in pareto_indices]

    # Plot
    ax.scatter(x_all, y_all, alpha=0.5, s=100, label='All configs')
    ax.scatter(x_pareto, y_pareto, color='red', s=150, marker='*',
               label=f'Pareto frontier ({len(pareto_indices)} configs)', zorder=5)

    # Connect Pareto points
    if len(x_pareto) > 1:
        # Sort by x coordinate
        sorted_pairs = sorted(zip(x_pareto, y_pareto))
        x_sorted, y_sorted = zip(*sorted_pairs)
        ax.plot(x_sorted, y_sorted, 'r--', alpha=0.5, linewidth=2)

    # Labels
    for i, r in enumerate(results):
        config_name = r['config'].get('strategy', 'standard')
        if r['config'].get('kv_warm_bits'):
            config_name += f" {r['config']['kv_warm_bits']}bit"
        if r['config'].get('density_mode'):
            config_name += f"\n{r['config']['density_mode']}"

        ax.annotate(config_name, (r[x_key], r[y_key]),
                   fontsize=8, alpha=0.7, ha='center')

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def plot_3d_pareto(results, pareto_indices):
    """Plot 3D Pareto frontier (quality vs speed vs memory)."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # All points
    quality_all = [r['quality_score'] for r in results]
    speed_all = [r['speed_score'] for r in results]
    memory_all = [r['memory_score'] for r in results]

    # Pareto points
    quality_pareto = [results[i]['quality_score'] for i in pareto_indices]
    speed_pareto = [results[i]['speed_score'] for i in pareto_indices]
    memory_pareto = [results[i]['memory_score'] for i in pareto_indices]

    # Plot
    ax.scatter(quality_all, speed_all, memory_all, alpha=0.5, s=100, label='All configs')
    ax.scatter(quality_pareto, speed_pareto, memory_pareto,
               color='red', s=200, marker='*', label='Pareto frontier', zorder=5)

    # Labels
    ax.set_xlabel('Quality Score', fontsize=10)
    ax.set_ylabel('Speed Score', fontsize=10)
    ax.set_zlabel('Memory Score', fontsize=10)
    ax.set_title('3D Pareto Frontier: Quality vs Speed vs Memory', fontsize=12, fontweight='bold')
    ax.legend()

    return fig


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_pareto_frontier.py <results.json>")
        sys.exit(1)

    json_path = sys.argv[1]

    if not Path(json_path).exists():
        print(f"Error: File not found: {json_path}")
        sys.exit(1)

    # Load data
    data = load_results(json_path)
    results = data['results']

    # Find Pareto frontier indices
    pareto_configs = data.get('pareto_frontier', [])
    pareto_indices = []
    for i, r in enumerate(results):
        if r['config'] in pareto_configs:
            pareto_indices.append(i)

    print(f"Loaded {len(results)} results")
    print(f"Pareto frontier: {len(pareto_indices)} configurations\n")

    # Create output directory
    output_dir = Path("meta_harness_plots")
    output_dir.mkdir(exist_ok=True)

    # Plot 1: PPL vs Speed
    fig1 = plot_2d_pareto(
        results, pareto_indices,
        'perplexity', 'tokens_per_sec',
        'Perplexity (lower is better)', 'Speed (tokens/sec)',
        'Quality vs Speed Trade-off'
    )
    fig1.savefig(output_dir / "pareto_ppl_vs_speed.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'pareto_ppl_vs_speed.png'}")

    # Plot 2: Memory vs Speed
    fig2 = plot_2d_pareto(
        results, pareto_indices,
        'peak_memory_mb', 'tokens_per_sec',
        'Peak Memory (MB)', 'Speed (tokens/sec)',
        'Memory vs Speed Trade-off'
    )
    fig2.savefig(output_dir / "pareto_memory_vs_speed.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'pareto_memory_vs_speed.png'}")

    # Plot 3: Quality vs Memory
    fig3 = plot_2d_pareto(
        results, pareto_indices,
        'peak_memory_mb', 'perplexity',
        'Peak Memory (MB)', 'Perplexity (lower is better)',
        'Memory vs Quality Trade-off'
    )
    fig3.savefig(output_dir / "pareto_memory_vs_quality.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'pareto_memory_vs_quality.png'}")

    # Plot 4: 3D visualization
    try:
        fig4 = plot_3d_pareto(results, pareto_indices)
        fig4.savefig(output_dir / "pareto_3d.png", dpi=150, bbox_inches='tight')
        print(f"Saved: {output_dir / 'pareto_3d.png'}")
    except Exception as e:
        print(f"Warning: Could not create 3D plot: {e}")

    # Summary table
    print(f"\n{'='*80}")
    print("PARETO FRONTIER CONFIGURATIONS")
    print(f"{'='*80}")
    print(f"{'Config':<30} {'PPL':>8} {'Speed':>12} {'Memory':>12} {'Pareto':>10}")
    print(f"{'-'*80}")

    for i in pareto_indices:
        r = results[i]
        config_str = r['config'].get('strategy', 'standard')
        if r['config'].get('kv_warm_bits'):
            config_str += f" {r['config']['kv_warm_bits']}bit"
        if r['config'].get('density_mode'):
            config_str += f" {r['config']['density_mode']}"

        print(f"{config_str:<30} {r['perplexity']:>8.4f} {r['tokens_per_sec']:>10.1f}/s "
              f"{r['peak_memory_mb']:>10.1f}MB {r['pareto_score']:>10.4f}")

    print(f"{'-'*80}")
    print(f"\nPlots saved to: {output_dir}/")


if __name__ == "__main__":
    main()
