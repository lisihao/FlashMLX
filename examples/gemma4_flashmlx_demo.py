"""
Gemma 4 + FlashMLX Deep Integration Demo

Demonstrates using Gemma 4 with FlashMLX optimized cache through VLM Bridge.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from flashmlx.vlm_bridge import load_vlm_model, create_vlm_cache, generate_vlm


def demo_text_generation():
    """Demo 1: Text-only generation with FlashMLX cache."""
    print("=" * 60)
    print("Demo 1: Text Generation with FlashMLX Cache")
    print("=" * 60)

    # Load Gemma 4 from our fork
    model, processor = load_vlm_model("/Volumes/toshiba/models/gemma-4-E4B")

    # Create standard cache (no compression)
    cache = create_vlm_cache(model, strategy="standard")

    # Generate
    response = generate_vlm(
        model, processor,
        prompt="Explain machine learning in one sentence.",
        cache=cache,
        max_tokens=50,
        verbose=False
    )

    print(f"\nResponse: {response.text}")
    print(f"Tokens: {response.generation_tokens}")
    print(f"Speed: {response.generation_tps:.1f} tok/s")


def demo_flashmlx_compression():
    """Demo 2: FlashMLX cache compression."""
    print("\n\n" + "=" * 60)
    print("Demo 2: FlashMLX Cache Compression")
    print("=" * 60)

    model, processor = load_vlm_model("/Volumes/toshiba/models/gemma-4-E4B")

    # Create compressed cache (scored_kv_direct)
    cache = create_vlm_cache(
        model,
        strategy="scored_kv_direct",
        density_mode="balanced"  # Route 0: Density Router
    )

    response = generate_vlm(
        model, processor,
        prompt="What is deep learning?",
        cache=cache,
        max_tokens=50,
        verbose=False
    )

    print(f"\nResponse: {response.text}")
    print(f"Cache type: {type(cache[0]).__name__}")
    print(f"Memory: {response.peak_memory:.2f} MB")


def demo_vision_text():
    """Demo 3: Vision+Text generation."""
    print("\n\n" + "=" * 60)
    print("Demo 3: Vision+Text Generation")
    print("=" * 60)

    # Check if test image exists
    image_path = project_root / "tests/datasets/fixtures/cat.jpg"
    if not image_path.exists():
        print(f"⚠️  Test image not found: {image_path}")
        print("Skipping vision demo")
        return

    model, processor = load_vlm_model("/Volumes/toshiba/models/gemma-4-E4B")

    # Use ultra_long mode for vision (280 vision tokens!)
    cache = create_vlm_cache(
        model,
        strategy="scored_kv_direct",
        density_mode="ultra_long"  # 10x compression
    )

    response = generate_vlm(
        model, processor,
        prompt="Describe this image in detail.",
        image=str(image_path),
        cache=cache,
        max_tokens=100,
        verbose=False
    )

    print(f"\nResponse: {response.text}")
    print(f"Vision tokens: ~280")
    print(f"Total tokens: {response.total_tokens}")
    print(f"Memory: {response.peak_memory:.2f} MB")


def demo_cache_comparison():
    """Demo 4: Compare cache strategies."""
    print("\n\n" + "=" * 60)
    print("Demo 4: Cache Strategy Comparison")
    print("=" * 60)

    model, processor = load_vlm_model("/Volumes/toshiba/models/gemma-4-E4B")

    strategies = [
        ("standard", {}),
        ("scored_kv_direct", {"density_mode": "balanced"}),
        ("scored_kv_direct", {"density_mode": "ultra_long"}),
    ]

    prompt = "Explain neural networks briefly."

    for strategy, kwargs in strategies:
        cache = create_vlm_cache(model, strategy=strategy, **kwargs)

        response = generate_vlm(
            model, processor,
            prompt=prompt,
            cache=cache,
            max_tokens=30,
            verbose=False
        )

        mode = kwargs.get("density_mode", "none")
        print(f"\n{strategy} ({mode}):")
        print(f"  Speed: {response.generation_tps:.1f} tok/s")
        print(f"  Memory: {response.peak_memory:.2f} MB")


def main():
    """Run all demos."""
    print("=" * 60)
    print("Gemma 4 + FlashMLX Deep Integration Demos")
    print("=" * 60)

    demos = [
        demo_text_generation,
        demo_flashmlx_compression,
        demo_vision_text,
        demo_cache_comparison,
    ]

    for demo in demos:
        try:
            demo()
        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n\n" + "=" * 60)
    print("✅ All Demos Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
