"""
FlashMLX Hybrid Cache - Monitoring Example

This example demonstrates how to monitor cache performance,
track statistics, and detect potential issues.
"""

from mlx_lm import load, generate
from flashmlx.cache import (
    inject_hybrid_cache_manager,
    create_layer_types_from_model,
    HybridCacheConfig
)
import json
import time
from typing import Dict, Any


class CacheMonitor:
    """Monitor hybrid cache performance and health."""

    def __init__(self, cache_wrapper):
        self.cache_wrapper = cache_wrapper
        self.history = []

    def snapshot(self) -> Dict[str, Any]:
        """Take a snapshot of current cache statistics."""
        stats = self.cache_wrapper.get_statistics()

        snapshot = {
            "timestamp": time.time(),
            "ssm": {
                "size": stats["ssm"]["local_cache"]["size"],
                "updates": stats["ssm"]["local_cache"]["total_updates"],
                "retrievals": stats["ssm"]["local_cache"]["total_retrievals"],
                "hit_rate": stats["ssm"]["local_cache"]["hit_rate"],
                "tiered": {
                    "hot": stats["ssm"]["tiered_cache"]["hot_size"],
                    "warm": stats["ssm"]["tiered_cache"]["warm_size"],
                    "cold": stats["ssm"]["tiered_cache"]["cold_size"],
                    "pinned": stats["ssm"]["tiered_cache"]["pinned_size"]
                }
            },
            "attention": {
                "size": stats["attention"]["local_cache"]["size"],
                "avg_compression": stats["attention"]["local_cache"]["avg_compression_ratio"],
                "compressions": stats["attention"]["local_cache"]["total_compressions"]
            }
        }

        self.history.append(snapshot)
        return snapshot

    def print_snapshot(self, snapshot: Dict[str, Any]):
        """Pretty print a cache snapshot."""
        print("\n" + "=" * 60)
        print(f"Cache Snapshot @ {time.strftime('%H:%M:%S', time.localtime(snapshot['timestamp']))}")
        print("=" * 60)

        print("\nSSM Cache:")
        print(f"  Total Size:    {snapshot['ssm']['size']} layers")
        print(f"  Updates:       {snapshot['ssm']['updates']}")
        print(f"  Retrievals:    {snapshot['ssm']['retrievals']}")
        print(f"  Hit Rate:      {snapshot['ssm']['hit_rate']:.2%}")

        print("\n  Tiered Distribution:")
        print(f"    Hot:    {snapshot['ssm']['tiered']['hot']:3d} layers")
        print(f"    Warm:   {snapshot['ssm']['tiered']['warm']:3d} layers")
        print(f"    Cold:   {snapshot['ssm']['tiered']['cold']:3d} layers")
        print(f"    Pinned: {snapshot['ssm']['tiered']['pinned']:3d} layers")

        print("\nAttention Cache:")
        print(f"  Total Size:         {snapshot['attention']['size']} layers")
        print(f"  Avg Compression:    {snapshot['attention']['avg_compression']:.2f}x")
        print(f"  Total Compressions: {snapshot['attention']['compressions']}")

    def check_health(self, snapshot: Dict[str, Any]) -> Dict[str, str]:
        """Check cache health and return warnings."""
        warnings = {}

        # Check SSM hit rate
        hit_rate = snapshot["ssm"]["hit_rate"]
        if hit_rate < 0.5:
            warnings["low_hit_rate"] = f"SSM hit rate very low: {hit_rate:.2%} (expected >50%)"
        elif hit_rate < 0.7:
            warnings["suboptimal_hit_rate"] = f"SSM hit rate suboptimal: {hit_rate:.2%} (expected >70%)"

        # Check compression ratio
        avg_compression = snapshot["attention"]["avg_compression"]
        if avg_compression > 5.0:
            warnings["high_compression"] = f"Very high compression: {avg_compression:.2f}x (may affect quality)"
        elif avg_compression < 2.0:
            warnings["low_compression"] = f"Low compression: {avg_compression:.2f}x (minimal memory savings)"

        # Check tier distribution
        hot_size = snapshot["ssm"]["tiered"]["hot"]
        total_size = snapshot["ssm"]["size"]
        if total_size > 0:
            hot_ratio = hot_size / total_size
            if hot_ratio > 0.5:
                warnings["hot_bloat"] = f"Hot tier too large: {hot_ratio:.0%} of cache (expected ~15-20%)"

        return warnings

    def print_health_report(self, snapshot: Dict[str, Any]):
        """Print health report with warnings."""
        warnings = self.check_health(snapshot)

        print("\n" + "=" * 60)
        print("Health Report")
        print("=" * 60)

        if not warnings:
            print("\n✓ All metrics healthy")
        else:
            print(f"\n⚠️  {len(warnings)} warning(s) detected:\n")
            for key, message in warnings.items():
                print(f"  • {message}")

        print("\nRecommendations:")
        if "low_hit_rate" in warnings or "suboptimal_hit_rate" in warnings:
            print("  → Increase total_budget_bytes or adjust tier ratios")
        if "high_compression" in warnings:
            print("  → Reduce compression_ratio to improve quality")
        if "low_compression" in warnings:
            print("  → Increase compression_ratio for more memory savings")
        if "hot_bloat" in warnings:
            print("  → Reduce hot_budget_ratio to allow more eviction")

    def compare_snapshots(self, snapshot1: Dict[str, Any], snapshot2: Dict[str, Any]):
        """Compare two snapshots and show deltas."""
        print("\n" + "=" * 60)
        print("Snapshot Comparison")
        print("=" * 60)

        # SSM deltas
        print("\nSSM Cache Changes:")
        print(f"  Updates:    {snapshot2['ssm']['updates'] - snapshot1['ssm']['updates']:+d}")
        print(f"  Retrievals: {snapshot2['ssm']['retrievals'] - snapshot1['ssm']['retrievals']:+d}")
        print(f"  Hit Rate:   {snapshot2['ssm']['hit_rate'] - snapshot1['ssm']['hit_rate']:+.2%}")

        # Attention deltas
        print("\nAttention Cache Changes:")
        print(f"  Compressions: {snapshot2['attention']['compressions'] - snapshot1['attention']['compressions']:+d}")
        print(f"  Avg Ratio:    {snapshot2['attention']['avg_compression'] - snapshot1['attention']['avg_compression']:+.2f}x")

        # Tier migration
        print("\nTier Migrations:")
        for tier in ["hot", "warm", "cold", "pinned"]:
            delta = snapshot2['ssm']['tiered'][tier] - snapshot1['ssm']['tiered'][tier]
            if delta != 0:
                print(f"  {tier.capitalize():8s}: {delta:+3d} layers")

    def save_history(self, filename: str = "cache_history.json"):
        """Save monitoring history to JSON file."""
        with open(filename, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"\n✓ Saved {len(self.history)} snapshots to {filename}")


def monitoring_example():
    """Demonstrate cache monitoring during text generation."""
    print("=" * 60)
    print("FlashMLX Hybrid Cache - Monitoring Example")
    print("=" * 60)

    # Setup
    print("\nSetting up hybrid cache...")
    model, tokenizer = load("mlx-community/Qwen3.5-35B-Instruct-4bit")

    layer_types = create_layer_types_from_model(
        model,
        attention_layer_pattern="every 4th"
    )

    config = HybridCacheConfig(
        total_budget_bytes=64 * 1024 * 1024,
        compression_ratio=4.0,
        beta_calibration=True
    )

    cache_wrapper = inject_hybrid_cache_manager(
        model=model,
        config=config,
        layer_types=layer_types,
        auto_inject=True
    )

    # Initialize monitor
    monitor = CacheMonitor(cache_wrapper)

    # Baseline snapshot
    print("\nTaking baseline snapshot...")
    snapshot_0 = monitor.snapshot()
    monitor.print_snapshot(snapshot_0)

    # Generate short text
    print("\n" + "=" * 60)
    print("Generating short text (512 tokens)...")
    print("=" * 60)

    prompt1 = "Explain quantum computing in 3 sentences."
    response1 = generate(model, tokenizer, prompt=prompt1, max_tokens=100, verbose=False)
    print(f"\nResponse: {response1[:200]}...")

    snapshot_1 = monitor.snapshot()
    monitor.print_snapshot(snapshot_1)
    monitor.print_health_report(snapshot_1)
    monitor.compare_snapshots(snapshot_0, snapshot_1)

    # Generate longer text
    print("\n" + "=" * 60)
    print("Generating long text (4096 tokens)...")
    print("=" * 60)

    prompt2 = """Write a detailed explanation of how neural networks learn through backpropagation.
    Include mathematical details and practical examples."""

    response2 = generate(model, tokenizer, prompt=prompt2, max_tokens=500, verbose=False)
    print(f"\nResponse: {response2[:200]}...")

    snapshot_2 = monitor.snapshot()
    monitor.print_snapshot(snapshot_2)
    monitor.print_health_report(snapshot_2)
    monitor.compare_snapshots(snapshot_1, snapshot_2)

    # Save history
    monitor.save_history("examples/cache_history.json")

    print("\n" + "=" * 60)
    print("✓ Monitoring example complete!")
    print("=" * 60)


def continuous_monitoring_example():
    """Example of continuous monitoring during a long-running session."""
    print("\n" + "=" * 60)
    print("Continuous Monitoring Example")
    print("=" * 60)

    model, tokenizer = load("mlx-community/Qwen3.5-35B-Instruct-4bit")
    layer_types = create_layer_types_from_model(model, attention_layer_pattern="every 4th")
    config = HybridCacheConfig()

    cache_wrapper = inject_hybrid_cache_manager(model, config, layer_types, auto_inject=True)
    monitor = CacheMonitor(cache_wrapper)

    # Simulate multi-turn conversation
    prompts = [
        "What is machine learning?",
        "How does deep learning differ?",
        "Explain neural network architectures.",
        "What are transformers?",
        "How does attention work?"
    ]

    for i, prompt in enumerate(prompts):
        print(f"\n{'=' * 60}")
        print(f"Turn {i+1}/{len(prompts)}")
        print(f"{'=' * 60}")

        response = generate(model, tokenizer, prompt=prompt, max_tokens=100, verbose=False)
        print(f"\nPrompt: {prompt}")
        print(f"Response: {response[:150]}...")

        snapshot = monitor.snapshot()
        monitor.print_snapshot(snapshot)

        if i > 0:
            monitor.compare_snapshots(monitor.history[-2], snapshot)

        monitor.print_health_report(snapshot)

        # Check for issues
        warnings = monitor.check_health(snapshot)
        if warnings:
            print("\n⚠️  Potential issues detected - consider adjusting configuration")

    # Final summary
    print("\n" + "=" * 60)
    print("Session Summary")
    print("=" * 60)

    final_snapshot = monitor.history[-1]
    initial_snapshot = monitor.history[0]

    print(f"\nTotal Generations: {len(prompts)}")
    print(f"Total SSM Updates: {final_snapshot['ssm']['updates'] - initial_snapshot['ssm']['updates']}")
    print(f"Total Compressions: {final_snapshot['attention']['compressions'] - initial_snapshot['attention']['compressions']}")
    print(f"Final Hit Rate: {final_snapshot['ssm']['hit_rate']:.2%}")

    monitor.save_history("examples/conversation_history.json")


def main():
    print("=" * 60)
    print("FlashMLX Hybrid Cache - Monitoring")
    print("=" * 60)
    print("\nSelect monitoring example:")
    print("1. Single-session monitoring (2 generations)")
    print("2. Continuous monitoring (5-turn conversation)")

    choice = input("\nEnter choice (1-2): ")

    if choice == "1":
        monitoring_example()
    elif choice == "2":
        continuous_monitoring_example()
    else:
        print("\nInvalid choice. Running single-session example...")
        monitoring_example()


if __name__ == "__main__":
    main()
