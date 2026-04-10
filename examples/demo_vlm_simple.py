"""
Simple VLM Demo - Component-Based API

Demonstrates the FlashMLX VLM component-based API.
Uses working test_real_weights.py loading + VLMGenerator + cache optimization.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from flashmlx.vlm import load_vlm_components
from flashmlx.generation import VLMGenerator, create_vlm_cache


def main():
    """Simple VLM demo."""
    print("="*60)
    print("FlashMLX VLM - Simple Demo")
    print("="*60)

    # Step 1: Load VLM components
    print("\n[1/4] Loading VLM components...")
    model, tokenizer, processor, config = load_vlm_components(
        "mlx-community/Qwen2-VL-2B-Instruct-bf16"
    )
    print("✅ Model, tokenizer, processor loaded")

    # Step 2: Create optimized cache
    print("\n[2/4] Creating cache...")
    cache = create_vlm_cache(model, kv_cache="standard")  # Production-ready
    print("✅ Standard cache created")

    # Step 3: Create generator
    print("\n[3/4] Creating generator...")
    generator = VLMGenerator(
        model=model,
        tokenizer=tokenizer,
        image_token_id=config.image_token_id,
        max_tokens=100,
    )
    print("✅ Generator ready")

    # Step 4: Generate text
    print("\n[4/4] Testing generation")
    print("="*60)

    # Text-only generation
    print("\n📝 Text-only generation:")
    prompt1 = "What is MLX? Explain in one sentence."
    print(f"  Prompt: {prompt1}")

    response1 = generator.generate(
        prompt=prompt1,
        cache=cache,
        use_chat_template=True,
    )
    print(f"  Response: {response1}")

    # Vision+text generation (if image exists)
    image_path = project_root / "tests/datasets/fixtures/cat.jpg"
    if image_path.exists():
        print("\n🖼️  Vision+text generation:")
        from test_real_weights import prepare_test_image

        pixel_values, grid_thw = prepare_test_image(processor)

        # Format prompt with image tokens
        image_tokens = "<|image_pad|>" * 256
        prompt2 = f"{image_tokens}\nWhat is in this image? Describe it briefly."

        print(f"  Prompt: What is in this image? Describe it briefly.")
        print(f"  Image: {image_path}")

        response2 = generator.generate(
            prompt=prompt2,
            pixel_values=pixel_values,
            grid_thw=grid_thw,
            cache=cache,
            use_chat_template=True,
        )
        print(f"  Response: {response2}")
    else:
        print(f"\n⚠️  Image not found: {image_path}, skipping vision test")

    print("\n" + "="*60)
    print("✅ Demo complete!")
    print("="*60)

    print("\nUsage summary:")
    print("  1. Load: model, tokenizer, processor, config = load_vlm_components(path)")
    print("  2. Cache: cache = create_vlm_cache(model, kv_cache='standard')")
    print("  3. Generator: generator = VLMGenerator(model, tokenizer, image_token_id)")
    print("  4. Generate: response = generator.generate(prompt, cache=cache)")


if __name__ == "__main__":
    main()
