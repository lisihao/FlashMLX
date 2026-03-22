"""
FlashMLX Hybrid Cache - Basic Usage Example

This example demonstrates the simplest way to enable hybrid cache
for a Qwen3.5 model with MLX-LM.
"""

from mlx_lm import load, generate
from flashmlx.cache import (
    inject_hybrid_cache_manager,
    create_layer_types_from_model,
    HybridCacheConfig
)


def main():
    print("=" * 60)
    print("FlashMLX Hybrid Cache - Basic Usage")
    print("=" * 60)

    # Step 1: Load model
    print("\n[1/5] Loading Qwen3.5-35B model...")
    model, tokenizer = load("mlx-community/Qwen3.5-35B-Instruct-4bit")
    print("✓ Model loaded")

    # Step 2: Detect layer types
    print("\n[2/5] Detecting layer types...")
    layer_types = create_layer_types_from_model(
        model,
        attention_layer_pattern="every 4th"  # Qwen3.5 default pattern
    )

    attention_count = sum(1 for lt in layer_types.values() if lt.value == "attention")
    ssm_count = len(layer_types) - attention_count
    print(f"✓ Detected {attention_count} Attention layers, {ssm_count} SSM layers")

    # Step 3: Configure hybrid cache
    print("\n[3/5] Configuring hybrid cache...")
    config = HybridCacheConfig(
        total_budget_bytes=64 * 1024 * 1024,  # 64MB (recommended)
        compression_ratio=4.0,                # 4x compression (optimal)
        beta_calibration=True                 # Enable β calibration for quality
    )
    print(f"✓ Configuration:")
    print(f"  - Budget: {config.total_budget_bytes / (1024**2):.0f}MB")
    print(f"  - Compression: {config.compression_ratio}x")
    print(f"  - β calibration: {config.beta_calibration}")

    # Step 4: Inject hybrid cache
    print("\n[4/5] Injecting hybrid cache...")
    cache_wrapper = inject_hybrid_cache_manager(
        model=model,
        config=config,
        layer_types=layer_types,
        auto_inject=True  # Automatically replace model.cache
    )
    print("✓ Hybrid cache injected (original cache saved)")

    # Step 5: Generate text
    print("\n[5/5] Generating text...")
    prompt = """Explain how a hybrid cache system works in a large language model,
    and why it's beneficial for memory-constrained environments."""

    print(f"\nPrompt: {prompt[:100]}...")

    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=200,
        verbose=False
    )

    print(f"\nResponse:\n{response}")

    # Check cache statistics
    print("\n" + "=" * 60)
    print("Cache Statistics")
    print("=" * 60)

    stats = cache_wrapper.get_statistics()

    print(f"\nSSM Cache:")
    print(f"  - Cached layers: {stats['ssm']['local_cache']['size']}")
    print(f"  - Total updates: {stats['ssm']['local_cache']['total_updates']}")
    print(f"  - Hit rate: {stats['ssm']['local_cache']['hit_rate']:.2%}")

    print(f"\nAttention Cache:")
    print(f"  - Cached layers: {stats['attention']['local_cache']['size']}")
    print(f"  - Avg compression: {stats['attention']['local_cache']['avg_compression_ratio']:.2f}x")
    print(f"  - Total compressions: {stats['attention']['local_cache']['total_compressions']}")

    print(f"\nExpected Performance:")
    print(f"  - Memory savings: ~18.8%")
    print(f"  - TTFT overhead: ~17.3% (acceptable for long contexts)")
    print(f"  - TBT overhead: ~5.0%")

    print("\n" + "=" * 60)
    print("✓ Basic usage complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
