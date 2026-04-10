"""
Advanced VLM Demo - Multi-turn Conversation & Batch Processing

Demonstrates advanced FlashMLX VLM usage patterns:
- Multi-turn conversation with shared cache
- Batch generation for multiple images
- Cache strategy comparison
- Performance monitoring
"""

import sys
from pathlib import Path
import time

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from flashmlx.vlm import load_vlm_components
from flashmlx.generation import VLMGenerator, create_vlm_cache
import mlx.core as mx


def demo_multi_turn():
    """Demonstrate multi-turn conversation with shared cache."""
    print("="*60)
    print("Demo 1: Multi-turn Conversation")
    print("="*60)

    # Load components
    print("\nLoading VLM...")
    model, tokenizer, processor, config = load_vlm_components(
        "mlx-community/Qwen2-VL-2B-Instruct-bf16"
    )

    # Create cache (reused across turns)
    cache = create_vlm_cache(model, kv_cache="standard")

    # Create generator
    generator = VLMGenerator(
        model=model,
        tokenizer=tokenizer,
        image_token_id=config.image_token_id,
        max_tokens=50,
    )

    # Multi-turn questions
    questions = [
        "What is machine learning?",
        "Can you explain neural networks?",
        "How does backpropagation work?",
    ]

    print("\nMulti-turn conversation (cache shared across turns):")
    total_time = 0

    for i, question in enumerate(questions, 1):
        print(f"\n--- Turn {i} ---")
        print(f"Q: {question}")

        start = time.time()
        response = generator.generate(
            prompt=question,
            cache=cache,
            use_chat_template=True,
        )
        elapsed = time.time() - start
        total_time += elapsed

        print(f"A: {response}")
        print(f"Time: {elapsed:.2f}s")

    print(f"\nTotal time: {total_time:.2f}s")
    print(f"Average time per turn: {total_time/len(questions):.2f}s")


def demo_batch_vision():
    """Demonstrate batch processing for multiple images."""
    print("\n" + "="*60)
    print("Demo 2: Batch Vision Processing")
    print("="*60)

    # Load components
    print("\nLoading VLM...")
    model, tokenizer, processor, config = load_vlm_components(
        "mlx-community/Qwen2-VL-2B-Instruct-bf16"
    )

    cache = create_vlm_cache(model, kv_cache="standard")
    generator = VLMGenerator(
        model=model,
        tokenizer=tokenizer,
        image_token_id=config.image_token_id,
        max_tokens=30,
    )

    # Prepare test image
    from test_real_weights import prepare_test_image
    pixel_values, grid_thw = prepare_test_image(processor)

    # Batch questions about the same image
    questions = [
        "What is in this image?",
        "What colors do you see?",
        "Is this indoors or outdoors?",
    ]

    print("\nBatch processing (same image, different questions):")
    results = []

    for i, question in enumerate(questions, 1):
        # Format prompt with image tokens
        image_tokens = "<|image_pad|>" * 256
        prompt = f"{image_tokens}\n{question}"

        print(f"\n--- Question {i} ---")
        print(f"Q: {question}")

        start = time.time()
        response = generator.generate(
            prompt=prompt,
            pixel_values=pixel_values,
            grid_thw=grid_thw,
            cache=cache,
            use_chat_template=True,
        )
        elapsed = time.time() - start

        print(f"A: {response}")
        print(f"Time: {elapsed:.2f}s")

        results.append({
            "question": question,
            "response": response,
            "time": elapsed,
        })

    # Summary
    total_time = sum(r["time"] for r in results)
    print(f"\nBatch summary:")
    print(f"  Total questions: {len(results)}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average time: {total_time/len(results):.2f}s")


def demo_cache_comparison():
    """Compare different cache strategies."""
    print("\n" + "="*60)
    print("Demo 3: Cache Strategy Comparison")
    print("="*60)

    # Load components
    print("\nLoading VLM...")
    model, tokenizer, processor, config = load_vlm_components(
        "mlx-community/Qwen2-VL-2B-Instruct-bf16"
    )

    # Test prompt
    prompt = "Explain deep learning in one sentence."

    # Test different cache strategies
    strategies = ["standard", "triple_pq"]
    results = {}

    for strategy in strategies:
        print(f"\n--- Testing: {strategy} ---")

        # Create cache
        cache = create_vlm_cache(model, kv_cache=strategy)

        # Create generator
        generator = VLMGenerator(
            model=model,
            tokenizer=tokenizer,
            image_token_id=config.image_token_id,
            max_tokens=30,
        )

        # Generate
        start = time.time()
        response = generator.generate(
            prompt=prompt,
            cache=cache,
            use_chat_template=True,
        )
        elapsed = time.time() - start

        print(f"Response: {response}")
        print(f"Time: {elapsed:.2f}s")

        results[strategy] = {
            "response": response,
            "time": elapsed,
        }

    # Comparison
    print("\n" + "="*60)
    print("Cache Comparison Summary")
    print("="*60)

    standard_time = results["standard"]["time"]
    compressed_time = results["triple_pq"]["time"]
    speedup = ((standard_time - compressed_time) / standard_time * 100)

    print(f"\nStandard cache: {standard_time:.2f}s")
    print(f"Compressed cache: {compressed_time:.2f}s")
    print(f"Speedup: {speedup:+.1f}%")

    # Check quality
    standard_resp = results["standard"]["response"]
    compressed_resp = results["triple_pq"]["response"]

    if standard_resp == compressed_resp:
        print("\n✅ Responses identical (perfect quality preservation)")
    else:
        print(f"\n⚠️  Responses differ:")
        print(f"  Standard length: {len(standard_resp)} chars")
        print(f"  Compressed length: {len(compressed_resp)} chars")


def demo_performance_monitoring():
    """Monitor generation performance metrics."""
    print("\n" + "="*60)
    print("Demo 4: Performance Monitoring")
    print("="*60)

    # Load components
    print("\nLoading VLM...")
    model, tokenizer, processor, config = load_vlm_components(
        "mlx-community/Qwen2-VL-2B-Instruct-bf16"
    )

    cache = create_vlm_cache(model, kv_cache="standard")
    generator = VLMGenerator(
        model=model,
        tokenizer=tokenizer,
        image_token_id=config.image_token_id,
        max_tokens=50,
    )

    # Test prompt
    prompt = "What is the difference between AI and machine learning?"

    print(f"\nPrompt: {prompt}")
    print("\nGenerating with performance monitoring...")

    # Memory before
    mx.eval(model.parameters())
    mem_before = mx.metal.get_active_memory() / 1024**2  # MB

    # Generate
    start = time.time()
    response = generator.generate(
        prompt=prompt,
        cache=cache,
        use_chat_template=True,
    )
    elapsed = time.time() - start

    # Memory after
    mem_after = mx.metal.get_active_memory() / 1024**2  # MB
    mem_peak = mx.metal.get_peak_memory() / 1024**2  # MB

    # Token metrics
    response_tokens = len(tokenizer.encode(response))
    tokens_per_sec = response_tokens / elapsed if elapsed > 0 else 0

    # Display results
    print(f"\nResponse: {response}")

    print(f"\n📊 Performance Metrics:")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Tokens generated: {response_tokens}")
    print(f"  Speed: {tokens_per_sec:.1f} tokens/sec")
    print(f"  Memory (before): {mem_before:.1f} MB")
    print(f"  Memory (after): {mem_after:.1f} MB")
    print(f"  Memory (peak): {mem_peak:.1f} MB")
    print(f"  Memory delta: {mem_after - mem_before:+.1f} MB")


def main():
    """Run all advanced demos."""
    print("="*60)
    print("FlashMLX VLM - Advanced Demos")
    print("="*60)

    demos = [
        ("Multi-turn Conversation", demo_multi_turn),
        ("Batch Vision Processing", demo_batch_vision),
        ("Cache Comparison", demo_cache_comparison),
        ("Performance Monitoring", demo_performance_monitoring),
    ]

    for i, (name, demo_func) in enumerate(demos, 1):
        try:
            print(f"\n\n{'='*60}")
            print(f"Running Demo {i}/{len(demos)}: {name}")
            print(f"{'='*60}")

            demo_func()

            print(f"\n✅ Demo {i} complete!")

        except Exception as e:
            print(f"\n❌ Demo {i} failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n\n" + "="*60)
    print("All Demos Complete!")
    print("="*60)

    print("\nKey takeaways:")
    print("  ✓ Multi-turn: Cache reuse speeds up conversations")
    print("  ✓ Batch: Process multiple queries efficiently")
    print("  ✓ Cache strategies: Standard for quality, compressed for speed")
    print("  ✓ Monitoring: Track performance for optimization")


if __name__ == "__main__":
    main()
