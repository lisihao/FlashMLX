"""
Gemma 4 + FlashMLX Demo

Demonstrates running Gemma 4 with FlashMLX cache optimization.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# vlm_gemma4 will auto-patch mlx-vlm on import
from flashmlx.vlm_gemma4 import load_gemma4_with_flashmlx
import time


def test_text_generation():
    """Test text-only generation with FlashMLX cache."""
    print("="*60)
    print("Test 1: Text Generation with FlashMLX")
    print("="*60)

    # Load Gemma 4 with FlashMLX
    model, processor, generator, cache = load_gemma4_with_flashmlx(
        model_path="/Volumes/toshiba/models/gemma-4-E4B",
        cache_strategy="standard",  # Try: triple_pq for compression
        max_tokens=100,
    )

    # Test prompt
    prompt = "What is machine learning? Explain briefly."
    print(f"\nPrompt: {prompt}")

    # Generate
    start = time.time()
    response = generator.generate(
        prompt=prompt,
        cache=cache,
        temperature=0.0,
    )
    elapsed = time.time() - start

    print(f"\nResponse:\n{response}")
    print(f"\nTime: {elapsed:.2f}s")


def test_vision_generation():
    """Test vision+text generation with FlashMLX cache."""
    print("\n\n" + "="*60)
    print("Test 2: Vision+Text Generation with FlashMLX")
    print("="*60)

    # Load Gemma 4
    model, processor, generator, cache = load_gemma4_with_flashmlx(
        model_path="/Volumes/toshiba/models/gemma-4-E4B",
        cache_strategy="standard",
        max_tokens=100,
    )

    # Check if test image exists
    image_path = project_root / "tests/datasets/fixtures/cat.jpg"
    if not image_path.exists():
        print(f"\n⚠️  Test image not found: {image_path}")
        print("Skipping vision test")
        return

    # Test prompt
    prompt = "What is in this image? Describe it in detail."
    print(f"\nPrompt: {prompt}")
    print(f"Image: {image_path}")

    # Generate
    start = time.time()
    response = generator.generate(
        prompt=prompt,
        image=str(image_path),
        cache=cache,
        temperature=0.0,
    )
    elapsed = time.time() - start

    print(f"\nResponse:\n{response}")
    print(f"\nTime: {elapsed:.2f}s")


def test_cache_comparison():
    """Compare standard vs compressed cache."""
    print("\n\n" + "="*60)
    print("Test 3: Cache Strategy Comparison")
    print("="*60)

    prompt = "Explain deep learning in one sentence."

    strategies = ["standard", "triple_pq"]
    results = {}

    for strategy in strategies:
        print(f"\n--- Testing: {strategy} ---")

        # Load with different cache
        model, processor, generator, cache = load_gemma4_with_flashmlx(
            model_path="/Volumes/toshiba/models/gemma-4-E4B",
            cache_strategy=strategy,
            max_tokens=50,
        )

        # Generate
        start = time.time()
        response = generator.generate(
            prompt=prompt,
            cache=cache,
            temperature=0.0,
        )
        elapsed = time.time() - start

        print(f"Response: {response[:100]}...")
        print(f"Time: {elapsed:.2f}s")

        results[strategy] = {
            "response": response,
            "time": elapsed,
        }

    # Compare
    print("\n" + "="*60)
    print("Comparison")
    print("="*60)

    standard_time = results["standard"]["time"]
    compressed_time = results["triple_pq"]["time"]
    speedup = ((standard_time - compressed_time) / standard_time * 100)

    print(f"\nStandard cache: {standard_time:.2f}s")
    print(f"Compressed cache: {compressed_time:.2f}s")
    print(f"Speedup: {speedup:+.1f}%")

    if results["standard"]["response"] == results["triple_pq"]["response"]:
        print("\n✅ Outputs identical (perfect quality)")
    else:
        print(f"\n⚠️  Outputs differ")


def main():
    """Run all tests."""
    print("="*60)
    print("Gemma 4 + FlashMLX Demo")
    print("="*60)

    tests = [
        ("Text Generation", test_text_generation),
        ("Vision+Text Generation", test_vision_generation),
        ("Cache Comparison", test_cache_comparison),
    ]

    for name, test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"\n❌ {name} failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n\n" + "="*60)
    print("✅ Demo Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
