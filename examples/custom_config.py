"""
FlashMLX Hybrid Cache - Custom Configuration Example

This example shows how to customize cache configuration for different
use cases and performance requirements.
"""

from mlx_lm import load, generate
from flashmlx.cache import (
    inject_hybrid_cache_manager,
    create_layer_types_from_model,
    HybridCacheConfig,
    restore_original_cache
)
import json


def load_from_template(template_name: str) -> HybridCacheConfig:
    """Load pre-tuned configuration from template file."""
    template_path = f"tuning_results/config_templates/{template_name}_config.json"

    with open(template_path) as f:
        config_dict = json.load(f)

    return HybridCacheConfig(**config_dict["hybrid_cache_config"])


def custom_config_example_1():
    """Example 1: Maximum memory savings (aggressive compression)"""
    print("\n" + "=" * 60)
    print("Example 1: Maximum Memory Savings")
    print("=" * 60)

    config = HybridCacheConfig(
        total_budget_bytes=64 * 1024 * 1024,
        compression_ratio=5.0,  # Aggressive 5x compression
        beta_calibration=True,
        hot_budget_ratio=0.10,   # Reduce hot tier (more eviction)
        warm_budget_ratio=0.20,
        cold_budget_ratio=0.65,  # Increase cold tier (more archive)
        pinned_budget_ratio=0.05
    )

    print("\nConfiguration:")
    print(f"  - Compression: {config.compression_ratio}x (max savings)")
    print(f"  - Budget: {config.total_budget_bytes / (1024**2):.0f}MB")
    print(f"  - Tier allocation: Hot {config.hot_budget_ratio:.0%}, "
          f"Warm {config.warm_budget_ratio:.0%}, Cold {config.cold_budget_ratio:.0%}")

    print("\nExpected Trade-offs:")
    print("  ✓ Memory savings: ~20% (maximum possible)")
    print("  ⚠️ Quality score: ~95 (slight degradation)")
    print("  ⚠️ TTFT overhead: ~20%")

    return config


def custom_config_example_2():
    """Example 2: Balanced quality and performance"""
    print("\n" + "=" * 60)
    print("Example 2: Balanced Quality and Performance")
    print("=" * 60)

    config = HybridCacheConfig(
        total_budget_bytes=64 * 1024 * 1024,
        compression_ratio=3.0,  # Moderate compression
        beta_calibration=True,
        hot_budget_ratio=0.20,   # Larger hot tier (less eviction)
        warm_budget_ratio=0.30,
        cold_budget_ratio=0.45,
        pinned_budget_ratio=0.05
    )

    print("\nConfiguration:")
    print(f"  - Compression: {config.compression_ratio}x (balanced)")
    print(f"  - Budget: {config.total_budget_bytes / (1024**2):.0f}MB")
    print(f"  - Tier allocation: Hot {config.hot_budget_ratio:.0%}, "
          f"Warm {config.warm_budget_ratio:.0%}, Cold {config.cold_budget_ratio:.0%}")

    print("\nExpected Trade-offs:")
    print("  ✓ Memory savings: ~16.7%")
    print("  ✓ Quality score: ~98.3 (minimal degradation)")
    print("  ✓ TTFT overhead: ~15%")

    return config


def custom_config_example_3():
    """Example 3: Quality-first (minimal compression)"""
    print("\n" + "=" * 60)
    print("Example 3: Quality-First (Minimal Compression)")
    print("=" * 60)

    config = HybridCacheConfig(
        total_budget_bytes=128 * 1024 * 1024,  # Larger budget
        compression_ratio=2.0,  # Conservative compression
        beta_calibration=True,
        hot_budget_ratio=0.25,   # Maximum hot tier
        warm_budget_ratio=0.30,
        cold_budget_ratio=0.40,
        pinned_budget_ratio=0.05
    )

    print("\nConfiguration:")
    print(f"  - Compression: {config.compression_ratio}x (conservative)")
    print(f"  - Budget: {config.total_budget_bytes / (1024**2):.0f}MB (larger)")
    print(f"  - Tier allocation: Hot {config.hot_budget_ratio:.0%}, "
          f"Warm {config.warm_budget_ratio:.0%}, Cold {config.cold_budget_ratio:.0%}")

    print("\nExpected Trade-offs:")
    print("  ✓ Memory savings: ~12.5%")
    print("  ✓ Quality score: 100 (no degradation)")
    print("  ✓ TTFT overhead: ~12%")

    return config


def custom_config_example_4():
    """Example 4: Load from pre-tuned template"""
    print("\n" + "=" * 60)
    print("Example 4: Load from Pre-tuned Template")
    print("=" * 60)

    # Use pre-tuned configuration for long contexts
    config = load_from_template("long_context")

    print("\nTemplate: long_context_config.json")
    print(f"  - Compression: {config.compression_ratio}x")
    print(f"  - Budget: {config.total_budget_bytes / (1024**2):.0f}MB")
    print(f"  - β calibration: {config.beta_calibration}")

    print("\nPre-tuned for:")
    print("  - Context length: 4096+ tokens")
    print("  - Use case: Long documents, extensive RAG")
    print("  - Pareto optimal: Best memory/quality/performance balance")

    return config


def compare_configurations():
    """Compare all custom configurations side-by-side."""
    print("\n" + "=" * 60)
    print("Configuration Comparison")
    print("=" * 60)

    print("\n{:<20} {:<15} {:<15} {:<15} {:<15}".format(
        "Configuration", "Compression", "Memory Savings", "Quality", "TTFT Overhead"
    ))
    print("-" * 85)

    configs = [
        ("Max Savings", "5.0x", "20%", "95", "20%"),
        ("Balanced", "3.0x", "16.7%", "98.3", "15%"),
        ("Quality-First", "2.0x", "12.5%", "100", "12%"),
        ("Pre-tuned (4x)", "4.0x", "18.8%", "96.7", "17.3%"),
    ]

    for name, comp, mem, qual, ttft in configs:
        print("{:<20} {:<15} {:<15} {:<15} {:<15}".format(
            name, comp, mem, qual, ttft
        ))

    print("\n" + "=" * 60)


def test_custom_config(config: HybridCacheConfig, config_name: str):
    """Test a custom configuration with text generation."""
    print(f"\n{'=' * 60}")
    print(f"Testing: {config_name}")
    print(f"{'=' * 60}")

    # Load model
    print("\nLoading model...")
    model, tokenizer = load("mlx-community/Qwen3.5-35B-Instruct-4bit")

    # Detect layer types
    layer_types = create_layer_types_from_model(
        model,
        attention_layer_pattern="every 4th"
    )

    # Inject hybrid cache
    print("Injecting hybrid cache...")
    cache_wrapper = inject_hybrid_cache_manager(
        model=model,
        config=config,
        layer_types=layer_types,
        auto_inject=True
    )

    # Generate text
    print("Generating text...")
    prompt = "Write a short poem about artificial intelligence."

    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=100,
        verbose=False
    )

    print(f"\nResponse:\n{response}")

    # Check statistics
    stats = cache_wrapper.get_statistics()
    print(f"\nCache Statistics:")
    print(f"  - SSM cache size: {stats['ssm']['local_cache']['size']}")
    print(f"  - Attention cache size: {stats['attention']['local_cache']['size']}")
    print(f"  - Avg compression: {stats['attention']['local_cache']['avg_compression_ratio']:.2f}x")

    # Restore original cache
    print("\nRestoring original cache...")
    restore_original_cache(model, cache_wrapper)
    print("✓ Original cache restored")


def main():
    print("=" * 60)
    print("FlashMLX Hybrid Cache - Custom Configuration Examples")
    print("=" * 60)

    # Show all configuration options
    config1 = custom_config_example_1()
    config2 = custom_config_example_2()
    config3 = custom_config_example_3()
    config4 = custom_config_example_4()

    # Compare configurations
    compare_configurations()

    # Interactive selection
    print("\nWhich configuration would you like to test?")
    print("1. Maximum Memory Savings (5x compression)")
    print("2. Balanced Quality and Performance (3x compression)")
    print("3. Quality-First (2x compression)")
    print("4. Pre-tuned Template (4x compression)")
    print("5. Skip testing")

    choice = input("\nEnter choice (1-5): ")

    configs = {
        "1": (config1, "Maximum Memory Savings"),
        "2": (config2, "Balanced Quality and Performance"),
        "3": (config3, "Quality-First"),
        "4": (config4, "Pre-tuned Template")
    }

    if choice in configs:
        config, name = configs[choice]
        test_custom_config(config, name)
    else:
        print("\nSkipping testing.")

    print("\n" + "=" * 60)
    print("✓ Custom configuration examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
