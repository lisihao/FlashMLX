"""
FlashMLX VLM Inference Demo

使用真实图片进行 Vision-Language 推理
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import mlx.core as mx
from PIL import Image
import json

# Add models directory for direct import
models_path = project_root / "src" / "flashmlx" / "models"
processors_path = project_root / "src" / "flashmlx" / "processors"
sys.path.insert(0, str(models_path))
sys.path.insert(0, str(processors_path))

from qwen2_vl import Qwen2VLModel
from vlm_config import VLMConfig
from image_processing import ImageProcessor


def load_model_and_processor(model_id="mlx-community/Qwen2-VL-2B-Instruct-4bit"):
    """Load VLM model and image processor from Hugging Face"""
    print(f"Loading model: {model_id}")
    print("="*60)

    # Load config
    from huggingface_hub import snapshot_download

    cache_dir = project_root / ".cache" / "huggingface"
    model_path = snapshot_download(
        repo_id=model_id,
        allow_patterns=["config.json"],
        cache_dir=str(cache_dir),
    )

    config_path = Path(model_path) / "config.json"
    with open(config_path) as f:
        config_dict = json.load(f)

    print(f"✅ Config loaded")
    print(f"   Model type: {config_dict.get('model_type')}")
    print(f"   Vision blocks: {config_dict['vision_config']['depth']}")
    print(f"   Language layers: {config_dict['num_hidden_layers']}")

    # Create model
    vlm_config = VLMConfig.from_dict(config_dict)
    model = Qwen2VLModel(vlm_config)

    print(f"\n✅ Model initialized")

    # Create image processor
    vision_cfg = config_dict['vision_config']
    processor = ImageProcessor(
        image_size=(vision_cfg['patch_size'] * 32, vision_cfg['patch_size'] * 32),  # 448x448
        do_resize=True,
        do_normalize=True,
    )

    print(f"✅ Image processor created")
    print("="*60)

    return model, processor, vlm_config


def prepare_image(image_path, processor):
    """Load and preprocess image"""
    print(f"\nPreparing image: {image_path}")

    # Load image
    image = Image.open(image_path)
    original_size = image.size
    print(f"   Original size: {original_size}")

    # Preprocess
    # For VLM, we need [B, C, T, H, W] format
    pixel_values = processor.preprocess(image)  # [C, H, W]

    # Add batch and temporal dimensions
    pixel_values = mx.expand_dims(pixel_values, axis=0)  # [1, C, H, W]
    # Add temporal dimension (for video support, use 1 frame for images)
    pixel_values = mx.expand_dims(pixel_values, axis=2)  # [1, C, 1, H, W]
    # Reorder to [B, C, T, H, W]
    pixel_values = pixel_values.transpose(0, 1, 2, 3, 4)

    print(f"   Preprocessed: {pixel_values.shape}")

    # Calculate grid_thw based on actual processing
    # grid_thw represents the grid AFTER patch embedding but BEFORE merge
    # It's the grid that rot_pos_emb will process

    # Get actual values from vision config
    patch_size = 14
    temporal_patch_size = 2
    spatial_merge_size = 2

    # After patch embedding: divide by patch_size
    h_patches = pixel_values.shape[3] // patch_size  # 448 / 14 = 32
    w_patches = pixel_values.shape[4] // patch_size  # 448 / 14 = 32

    # For temporal: need at least temporal_patch_size frames
    # If we have 1 frame, duplicate to 2
    if pixel_values.shape[2] < temporal_patch_size:
        pixel_values = mx.tile(pixel_values, (1, 1, temporal_patch_size, 1, 1))

    t_frames = pixel_values.shape[2]
    t_patches = t_frames // temporal_patch_size  # 2 / 2 = 1

    # grid_thw is BEFORE spatial merge
    # After temporal merge: t_patches
    # After spatial processing: h_patches, w_patches (not divided by merge size yet)
    grid_thw = mx.array([[t_patches, h_patches, w_patches]])

    print(f"   Pixel values: {pixel_values.shape}")
    print(f"   Grid (T, H, W): {grid_thw.tolist()}")

    # Calculate expected vision tokens AFTER merger
    vision_tokens = t_patches * \
                   (h_patches // spatial_merge_size) * \
                   (w_patches // spatial_merge_size)
    print(f"   Expected vision tokens: {vision_tokens}")

    return pixel_values, grid_thw


def create_prompt_with_image(text_prompt, image_token_id=151655):
    """Create prompt with <image> token"""
    # Simple prompt format: <image> + question
    # In actual use, should follow Qwen2-VL chat format

    prompt_text = f"<image>{text_prompt}"

    # Tokenize (simplified - in real use, need proper tokenizer)
    # For demo, create dummy token IDs
    # Format: [<image>, token1, token2, ...]

    # Dummy tokenization (replace with actual tokenizer)
    tokens = [image_token_id]  # <image> token
    # Add some dummy text tokens
    for _ in range(10):
        tokens.append(100)  # Placeholder

    input_ids = mx.array([tokens])  # [1, seq_len]

    print(f"\n✅ Prompt created:")
    print(f"   Text: {text_prompt}")
    print(f"   Tokens: {len(tokens)} ({tokens[:5]}...)")

    return input_ids


def inference_demo():
    """Run VLM inference demo"""
    print("\n" + "="*60)
    print("FlashMLX VLM Inference Demo")
    print("="*60)

    # Load model
    model, processor, config = load_model_and_processor()

    # Check if test fixture exists
    fixture_path = project_root / "tests" / "datasets" / "fixtures" / "cat.jpg"

    if not fixture_path.exists():
        print(f"\n⚠️  Test image not found: {fixture_path}")
        print("Creating a test image...")

        # Create test image
        fixture_path.parent.mkdir(parents=True, exist_ok=True)
        from PIL import Image
        import numpy as np

        # Create a simple gradient image
        img_array = np.random.randint(0, 255, (448, 448, 3), dtype=np.uint8)
        test_image = Image.fromarray(img_array)
        test_image.save(fixture_path)
        print(f"✅ Test image created: {fixture_path}")

    # Prepare image
    pixel_values, grid_thw = prepare_image(fixture_path, processor)

    # Create prompt
    input_ids = create_prompt_with_image(
        "What is in this image?",
        image_token_id=config.image_token_id
    )

    # Inference
    print(f"\n" + "="*60)
    print("Running inference...")
    print("="*60)

    try:
        # Note: Weights not loaded yet, this will test the forward pass structure
        print("⚠️  Running with random weights (model not loaded)")
        print("   This tests the architecture, not actual VLM capability")

        logits = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            grid_thw=grid_thw,
        )

        print(f"\n✅ Inference successful!")
        print(f"   Input: {input_ids.shape}")
        print(f"   Image: {pixel_values.shape}")
        print(f"   Output logits: {logits.shape}")
        print(f"   Logit range: [{logits.min().item():.2f}, {logits.max().item():.2f}]")

        # Get top predictions (simplified)
        # In real use, decode with tokenizer
        last_logits = logits[0, -1, :]  # Last position
        top_k = 5
        top_indices = mx.argpartition(-last_logits, kth=top_k)[:top_k]

        print(f"\n   Top-{top_k} predicted token IDs: {top_indices.tolist()}")

    except Exception as e:
        print(f"\n❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "="*60)
    print("Demo completed!")
    print("="*60)
    print("\nNext steps:")
    print("1. Load actual model weights (see test_weight_loading.py)")
    print("2. Use proper tokenizer for text encoding/decoding")
    print("3. Implement generation loop for full responses")
    print("4. Add FlashMLX cache for optimized inference")

    return True


if __name__ == "__main__":
    success = inference_demo()
    sys.exit(0 if success else 1)
