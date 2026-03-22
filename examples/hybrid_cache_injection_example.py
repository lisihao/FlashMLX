"""
Hybrid Cache Injection Example

Demonstrates how to inject hybrid cache manager into MLX-LM models
for automatic SSM and Attention layer cache management.

This example shows three methods:
1. Manual layer type definition
2. Pattern-based layer type detection
3. Automatic detection from model structure
"""

import mlx.core as mx

from flashmlx.cache import (
    inject_hybrid_cache_manager,
    create_layer_types_from_model,
    HybridCacheConfig,
    LayerType,
    restore_original_cache
)


def example_1_manual_layer_types():
    """
    Example 1: Manual layer type definition

    Use this when you know exactly which layers are SSM vs Attention.
    Best for Qwen3.5 and other mixed-architecture models.
    """
    print("=" * 60)
    print("Example 1: Manual Layer Type Definition")
    print("=" * 60)

    # Mock model (in real usage, this would be your MLX-LM model)
    class MockModel:
        def __init__(self):
            self.layers = [None] * 40  # 40 layers
            self.cache = None

    model = MockModel()

    # Qwen3.5-35B: 40 layers (30 SSM + 10 Attention)
    # Every 4th layer is Attention
    layer_types = {}
    for i in range(40):
        is_attention = (i + 1) % 4 == 0
        layer_types[i] = LayerType.ATTENTION if is_attention else LayerType.SSM

    # Configure hybrid cache
    config = HybridCacheConfig(
        total_budget_bytes=256 * 1024 * 1024,  # 256MB
        hot_budget_ratio=0.3,                   # 30% for Hot tier
        warm_budget_ratio=0.5,                  # 50% for Warm tier
        cold_budget_ratio=0.15,                 # 15% for Cold tier
        pinned_budget_ratio=0.05,               # 5% for Pinned tier
        compression_ratio=3.0,                  # 3x compression for Attention
        beta_calibration=True                   # Enable β calibration
    )

    # Inject hybrid cache
    cache_wrapper = inject_hybrid_cache_manager(
        model=model,
        config=config,
        layer_types=layer_types,
        auto_inject=True  # Automatically replace model.cache
    )

    print(f"✅ Injected hybrid cache into model")
    print(f"   SSM layers: {sum(1 for t in layer_types.values() if t == LayerType.SSM)}")
    print(f"   Attention layers: {sum(1 for t in layer_types.values() if t == LayerType.ATTENTION)}")
    print(f"   Cache wrapper: {cache_wrapper}")

    # Use model normally
    # model.generate(prompt, max_tokens=100)

    # Get statistics
    stats = cache_wrapper.get_statistics()
    print(f"\n📊 Cache Statistics:")
    print(f"   SSM cache size: {stats['ssm']['local_cache']['size']}")
    print(f"   Attention cache size: {stats['attention']['local_cache']['size']}")

    # Restore original cache (optional)
    restore_original_cache(model, cache_wrapper)
    print(f"\n✅ Restored original cache")


def example_2_pattern_based_detection():
    """
    Example 2: Pattern-based layer type detection

    Use this when layers follow a predictable pattern.
    Supported patterns: "every Nth", "last N"
    """
    print("\n" + "=" * 60)
    print("Example 2: Pattern-Based Layer Type Detection")
    print("=" * 60)

    class MockModel:
        def __init__(self):
            self.layers = [None] * 40
            self.cache = None

    model = MockModel()

    # Method 1: "every Nth" pattern
    # Every 4th layer is Attention
    layer_types = create_layer_types_from_model(
        model,
        attention_layer_pattern="every 4th"
    )

    print(f"✅ Pattern 'every 4th' detected:")
    print(f"   Attention layers: {[i for i, t in layer_types.items() if t == LayerType.ATTENTION]}")

    # Method 2: "last N" pattern
    # Last 10 layers are Attention
    layer_types_alt = create_layer_types_from_model(
        model,
        attention_layer_pattern="last 10"
    )

    print(f"\n✅ Pattern 'last 10' detected:")
    print(f"   Attention layers: {[i for i, t in layer_types_alt.items() if t == LayerType.ATTENTION]}")

    # Inject with pattern-detected layer types
    config = HybridCacheConfig(total_budget_bytes=256 * 1024 * 1024)
    cache_wrapper = inject_hybrid_cache_manager(model, config, layer_types)

    print(f"\n✅ Injected hybrid cache with pattern-based detection")


def example_3_automatic_detection():
    """
    Example 3: Automatic detection from model structure

    Use this when model has 'self_attn' attribute on Attention layers.
    Works for standard Transformer models.
    """
    print("\n" + "=" * 60)
    print("Example 3: Automatic Detection from Model Structure")
    print("=" * 60)

    # Mock model with self_attn attributes
    class MockLayer:
        def __init__(self, has_attention: bool):
            if has_attention:
                self.self_attn = "mock_attention"

    class MockModel:
        def __init__(self):
            # Create 40 layers, every 4th has self_attn
            self.layers = []
            for i in range(40):
                has_attn = (i + 1) % 4 == 0
                self.layers.append(MockLayer(has_attention=has_attn))
            self.cache = None

    model = MockModel()

    # Automatic detection (checks for self_attn attribute)
    layer_types = create_layer_types_from_model(model)

    print(f"✅ Automatically detected layer types:")
    print(f"   Attention layers: {[i for i, t in layer_types.items() if t == LayerType.ATTENTION]}")

    # Inject with auto-detected layer types
    config = HybridCacheConfig(total_budget_bytes=256 * 1024 * 1024)
    cache_wrapper = inject_hybrid_cache_manager(model, config, layer_types)

    print(f"\n✅ Injected hybrid cache with automatic detection")


def example_4_explicit_indices():
    """
    Example 4: Explicit attention layer indices

    Use this when you have specific layer indices.
    Most explicit and error-proof method.
    """
    print("\n" + "=" * 60)
    print("Example 4: Explicit Attention Layer Indices")
    print("=" * 60)

    class MockModel:
        def __init__(self):
            self.layers = [None] * 40
            self.cache = None

    model = MockModel()

    # Explicit indices (e.g., from model documentation)
    attention_layer_indices = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]

    layer_types = create_layer_types_from_model(
        model,
        attention_layer_indices=attention_layer_indices
    )

    print(f"✅ Explicit indices defined:")
    print(f"   Attention layers: {attention_layer_indices}")

    # Inject with explicit indices
    config = HybridCacheConfig(total_budget_bytes=256 * 1024 * 1024)
    cache_wrapper = inject_hybrid_cache_manager(model, config, layer_types)

    print(f"\n✅ Injected hybrid cache with explicit indices")


def example_5_advanced_configuration():
    """
    Example 5: Advanced configuration

    Demonstrates fine-tuning cache budget and compression settings.
    """
    print("\n" + "=" * 60)
    print("Example 5: Advanced Configuration")
    print("=" * 60)

    class MockModel:
        def __init__(self):
            self.layers = [None] * 40
            self.cache = None

    model = MockModel()

    layer_types = {i: LayerType.SSM if i % 4 != 3 else LayerType.ATTENTION for i in range(40)}

    # Advanced configuration
    config = HybridCacheConfig(
        # Total budget: 512MB
        total_budget_bytes=512 * 1024 * 1024,

        # Budget ratios (must sum to 1.0)
        hot_budget_ratio=0.4,    # 40% Hot tier (frequently accessed)
        warm_budget_ratio=0.4,   # 40% Warm tier (staging area)
        cold_budget_ratio=0.15,  # 15% Cold tier (archive)
        pinned_budget_ratio=0.05, # 5% Pinned (control channels)

        # Attention Matching compression
        compression_ratio=4.0,    # Aggressive 4x compression
        beta_calibration=True,    # Enable β calibration

        # Migration thresholds (waterlines)
        hot_high_waterline=0.85,   # Hot tier demotion at 85% full
        warm_high_waterline=0.85,  # Warm tier demotion at 85% full
        warm_low_waterline=0.25    # Warm tier promotion at 25% full
    )

    cache_wrapper = inject_hybrid_cache_manager(model, config, layer_types)

    print(f"✅ Advanced configuration:")
    print(f"   Total budget: {config.total_budget_bytes / 1024 / 1024:.0f} MB")
    print(f"   Hot tier: {config.hot_budget_ratio * 100:.0f}%")
    print(f"   Warm tier: {config.warm_budget_ratio * 100:.0f}%")
    print(f"   Cold tier: {config.cold_budget_ratio * 100:.0f}%")
    print(f"   Pinned tier: {config.pinned_budget_ratio * 100:.0f}%")
    print(f"   Compression ratio: {config.compression_ratio:.1f}x")
    print(f"   β calibration: {config.beta_calibration}")

    # Monitor statistics during usage
    stats = cache_wrapper.get_statistics()
    print(f"\n📊 Runtime Statistics:")
    print(f"   SSM updates: {stats['ssm']['local_cache']['total_updates']}")
    print(f"   Attention compression: {stats['attention']['local_cache']['avg_compression_ratio']:.2f}x")


def example_6_usage_in_generation():
    """
    Example 6: Usage in generation

    Shows how injected cache works transparently during generation.
    """
    print("\n" + "=" * 60)
    print("Example 6: Usage in Generation")
    print("=" * 60)

    class MockModel:
        def __init__(self):
            self.layers = [None] * 40
            self.cache = None

        def generate(self, prompt: str, max_tokens: int = 100):
            """Mock generation (in real usage, this is MLX-LM's generate)"""
            print(f"   Generating with hybrid cache...")
            # Cache is automatically used by layers during forward pass
            return f"Generated {max_tokens} tokens"

    model = MockModel()

    # Setup hybrid cache
    layer_types = {i: LayerType.SSM if i % 4 != 3 else LayerType.ATTENTION for i in range(40)}
    config = HybridCacheConfig(total_budget_bytes=256 * 1024 * 1024)
    cache_wrapper = inject_hybrid_cache_manager(model, config, layer_types)

    print(f"✅ Model ready with hybrid cache")

    # Use model normally - cache is transparent
    result = model.generate("Hello, world!", max_tokens=100)
    print(f"   {result}")

    # Get post-generation statistics
    stats = cache_wrapper.get_statistics()
    print(f"\n📊 Post-Generation Statistics:")
    print(f"   SSM cache hits: {stats['ssm']['local_cache']['local_cache_hits']}")
    print(f"   Attention cache hits: {stats['attention']['local_cache']['local_cache_hits']}")


if __name__ == "__main__":
    # Run all examples
    example_1_manual_layer_types()
    example_2_pattern_based_detection()
    example_3_automatic_detection()
    example_4_explicit_indices()
    example_5_advanced_configuration()
    example_6_usage_in_generation()

    print("\n" + "=" * 60)
    print("✅ All examples completed successfully!")
    print("=" * 60)
