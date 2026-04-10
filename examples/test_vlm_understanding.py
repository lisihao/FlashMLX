"""
Test VLM Understanding with Synthetic Images

Uses simple test images to verify model can actually understand visual content.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import mlx.core as mx
from PIL import Image

# Add paths
models_path = project_root / "src" / "flashmlx" / "models"
processors_path = project_root / "src" / "flashmlx" / "processors"
generation_path = project_root / "src" / "flashmlx" / "generation"
sys.path.insert(0, str(models_path))
sys.path.insert(0, str(processors_path))
sys.path.insert(0, str(generation_path))

from image_processing import ImageProcessor


def prepare_image(image_path, processor):
    """Prepare image for VLM input."""
    image = Image.open(image_path)

    # Preprocess
    pixel_values = processor.preprocess(image)
    pixel_values = mx.expand_dims(pixel_values, axis=0)
    pixel_values = mx.expand_dims(pixel_values, axis=2)

    # Ensure temporal dimension divisibility
    temporal_patch_size = 2
    if pixel_values.shape[2] < temporal_patch_size:
        pixel_values = mx.tile(pixel_values, (1, 1, temporal_patch_size, 1, 1))

    # Calculate grid
    patch_size = 14
    t_patches = pixel_values.shape[2] // temporal_patch_size
    h_patches = pixel_values.shape[3] // patch_size
    w_patches = pixel_values.shape[4] // patch_size
    grid_thw = mx.array([[t_patches, h_patches, w_patches]])

    return pixel_values, grid_thw


def test_image_understanding(model, tokenizer, processor, config):
    """Test model understanding with synthetic images."""
    from vlm_generator import VLMGenerator

    print(f"\\n{'='*60}")
    print(f"Testing VLM Understanding")
    print(f"{'='*60}")

    generator = VLMGenerator(
        model=model,
        tokenizer=tokenizer,
        image_token_id=config.image_token_id,
        max_tokens=100,
    )

    fixtures_dir = project_root / "tests" / "datasets" / "fixtures"

    # Test cases
    test_cases = [
        {
            "image": "shapes.jpg",
            "question": "<image>What shapes do you see in this image?",
            "expected_keywords": ["circle", "square", "triangle", "star"],
        },
        {
            "image": "text.jpg",
            "question": "<image>What text is written in this image?",
            "expected_keywords": ["hello", "world"],
        },
        {
            "image": "colors.jpg",
            "question": "<image>Describe the colors you see.",
            "expected_keywords": ["red", "green", "blue", "yellow"],
        },
        {
            "image": "checkerboard.jpg",
            "question": "<image>What pattern is shown in this image?",
            "expected_keywords": ["checkerboard", "chess", "pattern", "squares"],
        },
    ]

    results = []

    for i, test in enumerate(test_cases, 1):
        print(f"\\n{'='*60}")
        print(f"Test {i}/{len(test_cases)}: {test['image']}")
        print(f"{'='*60}")

        image_path = fixtures_dir / test["image"]
        if not image_path.exists():
            print(f"⚠️  Image not found: {image_path}")
            continue

        # Prepare image
        pixel_values, grid_thw = prepare_image(image_path, processor)

        # Generate response
        print(f"Question: {test['question']}")
        print(f"\\nGenerating...")

        try:
            response = generator.generate(
                prompt=test["question"],
                pixel_values=pixel_values,
                grid_thw=grid_thw,
                max_tokens=100,
                temperature=0.0,
            )

            print(f"\\nResponse:")
            print(f"  {response}")

            # Check if response contains expected keywords
            response_lower = response.lower()
            found_keywords = [kw for kw in test["expected_keywords"] if kw in response_lower]

            if found_keywords:
                print(f"\\n✅ Found keywords: {found_keywords}")
                results.append((test["image"], True, found_keywords))
            else:
                print(f"\\n⚠️  No expected keywords found")
                print(f"   Expected: {test['expected_keywords']}")
                results.append((test["image"], False, []))

        except Exception as e:
            print(f"\\n❌ Generation failed: {e}")
            results.append((test["image"], False, []))

    # Summary
    print(f"\\n{'='*60}")
    print(f"Test Summary")
    print(f"{'='*60}")

    passed = sum(1 for _, success, _ in results if success)
    total = len(results)

    for image, success, keywords in results:
        status = "✅" if success else "❌"
        kw_str = f" ({', '.join(keywords)})" if keywords else ""
        print(f"{status} {image}{kw_str}")

    print(f"\\nPassed: {passed}/{total}")

    if passed >= total * 0.5:
        print(f"\\n✅ Model shows understanding of visual content!")
    else:
        print(f"\\n⚠️  Model may need better weights or more testing")

    return passed, total


def main(model=None, tokenizer=None, processor=None, config=None):
    """Main test function.

    Can be called with pre-loaded model or will load from scratch.
    """
    if model is None:
        # Load model
        print("Loading model...")
        from test_real_weights import download_and_load_model
        model, tokenizer, processor, config = download_and_load_model(use_4bit=True)

    # Run tests
    passed, total = test_image_understanding(model, tokenizer, processor, config)

    return passed, total


if __name__ == "__main__":
    passed, total = main()
    sys.exit(0 if passed >= total * 0.5 else 1)
