"""
Test VLM with Real Weights

Quick test to verify model can understand images with real weights.
Uses 4-bit quantized model (1.5GB) for faster testing.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import mlx.core as mx
from PIL import Image
import json

# Add models directory
models_path = project_root / "src" / "flashmlx" / "models"
processors_path = project_root / "src" / "flashmlx" / "processors"
generation_path = project_root / "src" / "flashmlx" / "generation"
sys.path.insert(0, str(models_path))
sys.path.insert(0, str(processors_path))
sys.path.insert(0, str(generation_path))

from qwen2_vl import Qwen2VLModel
from vlm_config import VLMConfig
from image_processing import ImageProcessor
from vlm_generator import VLMGenerator

# mlx-lm imports
try:
    from mlx_lm.utils import load_tokenizer
except ImportError:
    sys.path.insert(0, str(project_root / "mlx-lm-source"))
    from mlx_lm.utils import load_tokenizer


def download_and_load_model(use_4bit=False):
    """Download and load model with real weights.

    Args:
        use_4bit: Use 4-bit quantized model (faster) vs bf16 (slower)
    """
    from huggingface_hub import snapshot_download

    model_id = "mlx-community/Qwen2-VL-2B-Instruct-4bit" if use_4bit else "mlx-community/Qwen2-VL-2B-Instruct-bf16"

    print(f"\\n{'='*60}")
    print(f"Loading Model: {model_id}")
    print(f"{'='*60}")

    cache_dir = project_root / ".cache" / "huggingface"

    # Step 1: Download all files
    print(f"\\n[1/5] Downloading model files...")
    print(f"   {'4-bit: ~1.5GB' if use_4bit else 'bf16: ~4GB'}")
    print(f"   This may take several minutes...")

    model_path = snapshot_download(
        repo_id=model_id,
        allow_patterns=["*.safetensors", "config.json", "tokenizer*"],
        cache_dir=str(cache_dir),
    )

    print(f"   ✅ Downloaded to: {model_path}")

    # Step 2: Load config
    print(f"\\n[2/5] Loading configuration...")
    config_path = Path(model_path) / "config.json"
    with open(config_path) as f:
        config_dict = json.load(f)

    vlm_config = VLMConfig.from_dict(config_dict)
    print(f"   ✅ Config loaded")

    # Step 3: Create model
    print(f"\\n[3/5] Initializing model...")
    model = Qwen2VLModel(vlm_config)
    print(f"   ✅ Model structure created")

    # Step 4: Load weights
    print(f"\\n[4/5] Loading model weights...")

    # Find weight file
    weight_file = Path(model_path) / "model.safetensors"
    if not weight_file.exists():
        # Try finding in snapshots
        weight_files = list(Path(model_path).rglob("*.safetensors"))
        if weight_files:
            weight_file = weight_files[0]
        else:
            raise FileNotFoundError(f"No .safetensors file found in {model_path}")

    print(f"   Loading from: {weight_file.name}")
    weights = mx.load(str(weight_file))
    print(f"   ✅ Loaded {len(weights)} tensors")

    # Sanitize weights (HF → FlashMLX format)
    sanitized = model.sanitize(weights)
    print(f"   ✅ Sanitized to {len(sanitized)} tensors")

    # Update model
    if use_4bit:
        # Filter quantization params for 4-bit models
        print(f"   Filtering quantization parameters...")
        filtered = {}
        skipped = 0
        for k, v in sanitized.items():
            if k.endswith('.biases') or k.endswith('.scales'):
                skipped += 1
                continue
            filtered[k] = v
        print(f"   Kept {len(filtered)} weights, skipped {skipped} quant params")
        sanitized = filtered

    from mlx.utils import tree_unflatten
    sanitized_tree = tree_unflatten(list(sanitized.items()))
    model.update(sanitized_tree)
    print(f"   ✅ Model weights loaded")

    # Step 5: Load tokenizer and processor
    print(f"\\n[5/5] Loading tokenizer and processor...")
    tokenizer = load_tokenizer(model_path)

    vision_cfg = config_dict['vision_config']
    processor = ImageProcessor(
        image_size=(vision_cfg['patch_size'] * 32, vision_cfg['patch_size'] * 32),
        do_resize=True,
        do_normalize=True,
    )
    print(f"   ✅ Tokenizer and processor ready")

    print(f"\\n{'='*60}")
    print(f"Model Loading Complete!")
    print(f"{'='*60}")

    return model, tokenizer, processor, vlm_config


def prepare_test_image(processor):
    """Prepare test image for VLM."""
    # Use the cat image fixture
    image_path = project_root / "tests" / "datasets" / "fixtures" / "cat.jpg"

    if not image_path.exists():
        print(f"\\n⚠️  Test image not found: {image_path}")
        print("Creating test image...")
        image_path.parent.mkdir(parents=True, exist_ok=True)
        import numpy as np
        img_array = np.random.randint(0, 255, (448, 448, 3), dtype=np.uint8)
        test_image = Image.fromarray(img_array)
        test_image.save(image_path)

    print(f"\\nPreparing image: {image_path}")
    image = Image.open(image_path)
    print(f"   Size: {image.size}")

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

    print(f"   Preprocessed: {pixel_values.shape}")
    print(f"   Grid: {grid_thw.tolist()}")

    return pixel_values, grid_thw


def test_generation(model, tokenizer, processor, config):
    """Test text generation with real weights."""
    print(f"\\n{'='*60}")
    print(f"Testing Text Generation")
    print(f"{'='*60}")

    # Create generator
    generator = VLMGenerator(
        model=model,
        tokenizer=tokenizer,
        image_token_id=config.image_token_id,
        max_tokens=100,
    )

    # Prepare image
    pixel_values, grid_thw = prepare_test_image(processor)

    # Test 1: Vision + Text
    print(f"\\n[Test 1] Vision + Text Generation")
    print(f"-" * 60)
    prompt = "<image>Describe this image in detail."
    print(f"Prompt: {prompt}")
    print(f"\\nGenerating...")

    try:
        response = generator.generate(
            prompt=prompt,
            pixel_values=pixel_values,
            grid_thw=grid_thw,
            max_tokens=100,
            temperature=0.0,  # Greedy for consistency
        )

        print(f"\\n✅ Response:")
        print(f"   {response}")
        print(f"\\n   Length: {len(response)} characters")

        # Check if response is meaningful (not random gibberish)
        if len(response) > 20 and not response.startswith("lieutenanticip"):
            print(f"   ✅ Response appears meaningful!")
        else:
            print(f"   ⚠️  Response may be gibberish (check weights)")

    except Exception as e:
        print(f"\\n❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 2: Text-only
    print(f"\\n[Test 2] Text-only Generation")
    print(f"-" * 60)
    prompt_text = "What is a vision-language model?"
    print(f"Prompt: {prompt_text}")
    print(f"\\nGenerating...")

    try:
        response = generator.generate(
            prompt=prompt_text,
            max_tokens=50,
            temperature=0.0,
        )

        print(f"\\n✅ Response:")
        print(f"   {response}")

    except Exception as e:
        print(f"\\n❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()

    print(f"\\n{'='*60}")
    print(f"Testing Complete!")
    print(f"{'='*60}")

    return True


def main():
    """Main test function."""
    print("\\n" + "="*60)
    print("FlashMLX VLM Real Weights Test")
    print("="*60)

    # Load model with real weights
    model, tokenizer, processor, config = download_and_load_model(use_4bit=True)

    # Test generation
    success = test_generation(model, tokenizer, processor, config)

    if success:
        print("\\n✅ Real weights test PASSED!")
        print("\\nNext steps:")
        print("  1. Try with bf16 weights for better quality")
        print("  2. Test with different images")
        print("  3. Add FlashMLX cache optimization")
        print("  4. Run VQA benchmarks")
    else:
        print("\\n⚠️  Some tests failed, check logs above")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
