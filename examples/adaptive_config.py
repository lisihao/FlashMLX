"""
FlashMLX Hybrid Cache - Adaptive Configuration Example

This example demonstrates automatic configuration adjustment
based on context length and use case.
"""

from mlx_lm import load, generate
from flashmlx.cache import (
    inject_hybrid_cache_manager,
    restore_original_cache,
    create_layer_types_from_model,
    HybridCacheConfig
)
from typing import Optional


def get_adaptive_config(
    context_length: int,
    use_case: str = "general"
) -> Optional[HybridCacheConfig]:
    """
    Get adaptive configuration based on context length and use case.

    Args:
        context_length: Number of tokens in context
        use_case: "general", "quality", "memory", or "performance"

    Returns:
        HybridCacheConfig or None if hybrid cache should be disabled
    """
    print(f"\n{'=' * 60}")
    print(f"Adaptive Configuration")
    print(f"{'=' * 60}")
    print(f"\nContext length: {context_length} tokens")
    print(f"Use case: {use_case}")

    # Rule 1: Disable for very short contexts
    if context_length < 1000:
        print("\n⚠️  Context too short (<1000 tokens)")
        print("Recommendation: DISABLE hybrid cache")
        print("Reason: TTFT overhead (>50%) exceeds benefit")
        return None

    # Rule 2: Short-medium context (1000-2000 tokens)
    elif context_length < 2000:
        print("\n📊 Short-medium context (1000-2000 tokens)")

        if use_case == "quality":
            print("Strategy: Conservative compression for quality")
            config = HybridCacheConfig(
                total_budget_bytes=64 * 1024 * 1024,
                compression_ratio=2.0,  # Conservative
                beta_calibration=True
            )
            print(f"  - Compression: 2x")
            print(f"  - Expected memory savings: ~12.5%")
            print(f"  - Expected TTFT overhead: ~12%")

        elif use_case == "memory":
            print("Strategy: Moderate compression for memory")
            config = HybridCacheConfig(
                total_budget_bytes=64 * 1024 * 1024,
                compression_ratio=3.0,
                beta_calibration=True
            )
            print(f"  - Compression: 3x")
            print(f"  - Expected memory savings: ~16.7%")
            print(f"  - Expected TTFT overhead: ~15%")

        else:  # general or performance
            print("Strategy: Balanced configuration")
            config = HybridCacheConfig(
                total_budget_bytes=64 * 1024 * 1024,
                compression_ratio=2.5,
                beta_calibration=True
            )
            print(f"  - Compression: 2.5x")
            print(f"  - Expected memory savings: ~14.6%")
            print(f"  - Expected TTFT overhead: ~13%")

        return config

    # Rule 3: Medium context (2000-4000 tokens)
    elif context_length < 4000:
        print("\n📊 Medium context (2000-4000 tokens)")

        if use_case == "quality":
            print("Strategy: Moderate compression")
            config = HybridCacheConfig(
                total_budget_bytes=64 * 1024 * 1024,
                compression_ratio=3.0,
                beta_calibration=True
            )
            print(f"  - Compression: 3x")
            print(f"  - Expected memory savings: ~16.7%")
            print(f"  - Expected TTFT overhead: ~15%")

        elif use_case == "memory":
            print("Strategy: Aggressive compression")
            config = HybridCacheConfig(
                total_budget_bytes=64 * 1024 * 1024,
                compression_ratio=4.5,
                beta_calibration=True
            )
            print(f"  - Compression: 4.5x")
            print(f"  - Expected memory savings: ~19.6%")
            print(f"  - Expected TTFT overhead: ~18%")

        else:  # general or performance
            print("Strategy: Recommended 4x compression")
            config = HybridCacheConfig(
                total_budget_bytes=64 * 1024 * 1024,
                compression_ratio=4.0,
                beta_calibration=True
            )
            print(f"  - Compression: 4x")
            print(f"  - Expected memory savings: ~18.8%")
            print(f"  - Expected TTFT overhead: ~17%")

        return config

    # Rule 4: Long context (4000+ tokens) - optimal scenario
    else:
        print("\n📊 Long context (4000+ tokens) - OPTIMAL")

        if use_case == "quality":
            print("Strategy: Balanced quality-memory")
            config = HybridCacheConfig(
                total_budget_bytes=64 * 1024 * 1024,
                compression_ratio=3.5,
                beta_calibration=True
            )
            print(f"  - Compression: 3.5x")
            print(f"  - Expected memory savings: ~17.9%")
            print(f"  - Expected TTFT overhead: ~16%")

        elif use_case == "memory":
            print("Strategy: Maximum memory savings")
            config = HybridCacheConfig(
                total_budget_bytes=64 * 1024 * 1024,
                compression_ratio=5.0,  # Maximum
                beta_calibration=True
            )
            print(f"  - Compression: 5x")
            print(f"  - Expected memory savings: ~20%")
            print(f"  - Expected TTFT overhead: ~20%")
            print(f"  ⚠️  May have slight quality degradation")

        else:  # general or performance
            print("Strategy: Recommended 4x compression (Pareto optimal)")
            config = HybridCacheConfig(
                total_budget_bytes=64 * 1024 * 1024,
                compression_ratio=4.0,
                beta_calibration=True
            )
            print(f"  - Compression: 4x")
            print(f"  - Expected memory savings: ~18.8%")
            print(f"  - Expected TTFT overhead: ~17%")
            print(f"  ✓ Best balance for long contexts")

        return config


class AdaptiveHybridCache:
    """Automatically manage hybrid cache based on context."""

    def __init__(self, model, tokenizer, use_case: str = "general"):
        self.model = model
        self.tokenizer = tokenizer
        self.use_case = use_case
        self.cache_wrapper = None
        self.layer_types = create_layer_types_from_model(
            model,
            attention_layer_pattern="every 4th"
        )

    def generate(self, prompt: str, max_tokens: int = 200) -> str:
        """Generate text with adaptive hybrid cache configuration."""

        # Calculate context length
        context_length = len(self.tokenizer.encode(prompt))

        # Get adaptive configuration
        config = get_adaptive_config(context_length, self.use_case)

        # If hybrid cache recommended, inject it
        if config is not None:
            print(f"\n✓ Injecting hybrid cache...")
            self.cache_wrapper = inject_hybrid_cache_manager(
                self.model,
                config,
                self.layer_types,
                auto_inject=True
            )
        else:
            print(f"\n✓ Using baseline cache (hybrid cache disabled)")

        # Generate text
        print(f"\nGenerating text...")
        response = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False
        )

        # Restore if hybrid cache was used
        if self.cache_wrapper is not None:
            restore_original_cache(self.model, self.cache_wrapper)
            self.cache_wrapper = None

        return response


def adaptive_example():
    """Demonstrate adaptive configuration in action."""
    print("=" * 60)
    print("FlashMLX Hybrid Cache - Adaptive Configuration Example")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    model, tokenizer = load("mlx-community/Qwen3.5-35B-Instruct-4bit")

    # Create adaptive cache manager
    adaptive_cache = AdaptiveHybridCache(model, tokenizer, use_case="general")

    # Test 1: Very short context (should disable)
    print("\n" + "=" * 60)
    print("Test 1: Very Short Context")
    print("=" * 60)

    prompt1 = "What is AI?"  # ~5 tokens
    response1 = adaptive_cache.generate(prompt1, max_tokens=50)
    print(f"\nPrompt: {prompt1}")
    print(f"Response: {response1[:200]}...")

    # Test 2: Short-medium context
    print("\n" + "=" * 60)
    print("Test 2: Short-Medium Context")
    print("=" * 60)

    prompt2 = """Artificial intelligence (AI) is intelligence demonstrated by machines,
    in contrast to the natural intelligence displayed by humans and animals.
    Leading AI textbooks define the field as the study of intelligent agents.
    Explain the key differences between narrow AI and general AI."""  # ~500 tokens

    response2 = adaptive_cache.generate(prompt2, max_tokens=100)
    print(f"\nPrompt: {prompt2[:100]}...")
    print(f"Response: {response2[:200]}...")

    # Test 3: Long context (optimal)
    print("\n" + "=" * 60)
    print("Test 3: Long Context (Optimal)")
    print("=" * 60)

    # Create a long prompt by repeating context
    long_context = """Machine learning is a branch of artificial intelligence (AI) and computer science
    which focuses on the use of data and algorithms to imitate the way that humans learn,
    gradually improving its accuracy.""" * 50  # ~2000 tokens

    prompt3 = f"{long_context}\n\nSummarize the key concepts of machine learning mentioned above."

    response3 = adaptive_cache.generate(prompt3, max_tokens=200)
    print(f"\nPrompt length: ~{len(tokenizer.encode(prompt3))} tokens")
    print(f"Response: {response3[:200]}...")

    print("\n" + "=" * 60)
    print("✓ Adaptive configuration example complete!")
    print("=" * 60)


def use_case_comparison():
    """Compare different use case strategies."""
    print("\n" + "=" * 60)
    print("Use Case Strategy Comparison")
    print("=" * 60)

    test_context_length = 4096  # Long context
    use_cases = ["general", "quality", "memory", "performance"]

    print(f"\nContext length: {test_context_length} tokens")
    print("\nStrategies:\n")

    for use_case in use_cases:
        config = get_adaptive_config(test_context_length, use_case)
        if config:
            print("")  # Spacing between outputs


def interactive_mode():
    """Interactive mode for testing adaptive configuration."""
    print("\n" + "=" * 60)
    print("Interactive Adaptive Configuration")
    print("=" * 60)

    model, tokenizer = load("mlx-community/Qwen3.5-35B-Instruct-4bit")

    print("\nSelect use case:")
    print("1. General (balanced)")
    print("2. Quality-first (conservative compression)")
    print("3. Memory-first (aggressive compression)")
    print("4. Performance-first (minimal overhead)")

    choice = input("\nEnter choice (1-4): ")

    use_case_map = {
        "1": "general",
        "2": "quality",
        "3": "memory",
        "4": "performance"
    }

    use_case = use_case_map.get(choice, "general")
    print(f"\nSelected use case: {use_case}")

    adaptive_cache = AdaptiveHybridCache(model, tokenizer, use_case=use_case)

    while True:
        prompt = input("\nEnter prompt (or 'quit' to exit): ")
        if prompt.lower() == "quit":
            break

        response = adaptive_cache.generate(prompt, max_tokens=200)
        print(f"\nResponse:\n{response}\n")


def main():
    print("=" * 60)
    print("FlashMLX Hybrid Cache - Adaptive Configuration")
    print("=" * 60)
    print("\nSelect example:")
    print("1. Adaptive configuration demo (3 test cases)")
    print("2. Use case strategy comparison")
    print("3. Interactive mode")

    choice = input("\nEnter choice (1-3): ")

    if choice == "1":
        adaptive_example()
    elif choice == "2":
        use_case_comparison()
    elif choice == "3":
        interactive_mode()
    else:
        print("\nInvalid choice. Running adaptive example...")
        adaptive_example()


if __name__ == "__main__":
    main()
