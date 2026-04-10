"""
Tests for FlashMLX Qwen2-VL Model

验证 Vision-Language 融合逻辑的正确性
"""

import sys
from pathlib import Path

# Add models directory to path for direct import
project_root = Path(__file__).parent.parent.parent
models_path = project_root / "src" / "flashmlx" / "models"
sys.path.insert(0, str(models_path))

import numpy as np
import mlx.core as mx
import pytest

# Now import directly
from qwen2_vl import Qwen2VLModel
from vlm_config import VLMConfig
from vision import VisionConfig


class TestMergeInputIdsWithImageFeatures:
    """Test the core vision-text fusion logic"""

    def test_single_image_token(self):
        """Test merging with single <image> token"""
        batch_size = 1
        seq_len = 5
        hidden_dim = 128
        image_token_id = 151655

        # Create input_ids: ["A", "<image>", "B", "C", "D"]
        # Position 1 is <image> token
        input_ids = mx.array([[1, image_token_id, 3, 4, 5]])

        # Create dummy text embeddings
        text_embeds = mx.random.normal((batch_size, seq_len, hidden_dim))

        # Create vision features (1 image → 256 tokens, but simplified to 1 for test)
        vision_features = mx.random.normal((1, hidden_dim))

        # Merge
        merged = Qwen2VLModel.merge_input_ids_with_image_features(
            image_token_id=image_token_id,
            video_token_id=151656,
            image_features=vision_features,
            inputs_embeds=text_embeds,
            input_ids=input_ids,
        )

        # Check shape unchanged
        assert merged.shape == text_embeds.shape

        # Check position 1 replaced with vision features
        assert not mx.array_equal(merged[0, 1], text_embeds[0, 1])
        assert mx.array_equal(merged[0, 1], vision_features[0])

        # Check other positions unchanged
        assert mx.array_equal(merged[0, 0], text_embeds[0, 0])
        assert mx.array_equal(merged[0, 2], text_embeds[0, 2])

        print(f"✅ Single image token: Position 1 replaced")

    def test_multiple_image_tokens(self):
        """Test merging with multiple <image> tokens"""
        batch_size = 1
        seq_len = 6
        hidden_dim = 128
        image_token_id = 151655

        # Create input_ids: ["A", "<image>", "<image>", "B", "C", "D"]
        # Positions 1, 2 are <image> tokens
        input_ids = mx.array([[1, image_token_id, image_token_id, 4, 5, 6]])

        text_embeds = mx.random.normal((batch_size, seq_len, hidden_dim))

        # 2 vision features
        vision_features = mx.random.normal((2, hidden_dim))

        merged = Qwen2VLModel.merge_input_ids_with_image_features(
            image_token_id=image_token_id,
            video_token_id=151656,
            image_features=vision_features,
            inputs_embeds=text_embeds,
            input_ids=input_ids,
        )

        # Check positions 1, 2 replaced
        assert mx.array_equal(merged[0, 1], vision_features[0])
        assert mx.array_equal(merged[0, 2], vision_features[1])

        # Check other positions unchanged
        assert mx.array_equal(merged[0, 0], text_embeds[0, 0])
        assert mx.array_equal(merged[0, 3], text_embeds[0, 3])

        print(f"✅ Multiple image tokens: Positions 1, 2 replaced")

    def test_batch_processing(self):
        """Test merging with batch size > 1"""
        batch_size = 2
        seq_len = 5
        hidden_dim = 128
        image_token_id = 151655

        # Batch 0: 1 <image> at position 1
        # Batch 1: 2 <image> at positions 2, 3
        input_ids = mx.array([
            [1, image_token_id, 3, 4, 5],
            [10, 11, image_token_id, image_token_id, 15],
        ])

        text_embeds = mx.random.normal((batch_size, seq_len, hidden_dim))

        # Total 3 vision features (1 + 2)
        vision_features = mx.random.normal((3, hidden_dim))

        merged = Qwen2VLModel.merge_input_ids_with_image_features(
            image_token_id=image_token_id,
            video_token_id=151656,
            image_features=vision_features,
            inputs_embeds=text_embeds,
            input_ids=input_ids,
        )

        # Batch 0: position 1 replaced with feature 0
        assert mx.array_equal(merged[0, 1], vision_features[0])
        assert mx.array_equal(merged[0, 0], text_embeds[0, 0])

        # Batch 1: positions 2, 3 replaced with features 1, 2
        assert mx.array_equal(merged[1, 2], vision_features[1])
        assert mx.array_equal(merged[1, 3], vision_features[2])
        assert mx.array_equal(merged[1, 0], text_embeds[1, 0])

        print(f"✅ Batch processing: 2 batches with 1+2 images")

    def test_no_image_tokens(self):
        """Test text-only case (no <image> tokens)"""
        batch_size = 1
        seq_len = 5
        hidden_dim = 128
        image_token_id = 151655

        # No <image> tokens
        input_ids = mx.array([[1, 2, 3, 4, 5]])
        text_embeds = mx.random.normal((batch_size, seq_len, hidden_dim))

        # Empty vision features
        vision_features = mx.zeros((0, hidden_dim))

        merged = Qwen2VLModel.merge_input_ids_with_image_features(
            image_token_id=image_token_id,
            video_token_id=151656,
            image_features=vision_features,
            inputs_embeds=text_embeds,
            input_ids=input_ids,
        )

        # Should be unchanged
        assert mx.array_equal(merged, text_embeds)

        print(f"✅ No image tokens: Embeddings unchanged")

    def test_video_token_fallback(self):
        """Test fallback to <video> token if no <image> tokens"""
        batch_size = 1
        seq_len = 4
        hidden_dim = 128
        image_token_id = 151655
        video_token_id = 151656

        # Use <video> token instead of <image>
        input_ids = mx.array([[1, video_token_id, 3, 4]])
        text_embeds = mx.random.normal((batch_size, seq_len, hidden_dim))
        vision_features = mx.random.normal((1, hidden_dim))

        merged = Qwen2VLModel.merge_input_ids_with_image_features(
            image_token_id=image_token_id,
            video_token_id=video_token_id,
            image_features=vision_features,
            inputs_embeds=text_embeds,
            input_ids=input_ids,
        )

        # Position 1 (<video>) should be replaced
        assert mx.array_equal(merged[0, 1], vision_features[0])

        print(f"✅ Video token fallback: Position 1 replaced")


class TestQwen2VLModel:
    """Test Qwen2VLModel initialization and structure"""

    def test_model_initialization(self):
        """Test basic model initialization"""
        # Create minimal config
        config = VLMConfig(
            model_type="qwen2_vl",
            vision_config={
                "embed_dim": 1152,
                "depth": 2,  # Small for testing
                "num_heads": 16,
                "hidden_size": 3584,
            },
            text_config={
                "hidden_size": 3584,
                "vocab_size": 151936,
            },
            image_token_id=151655,
            video_token_id=151656,
        )

        model = Qwen2VLModel(config)

        # Check components exist
        assert model.vision_tower is not None
        assert model.config.image_token_id == 151655

        print(f"✅ Model initialization: Vision tower created")

    def test_sanitize_weights(self):
        """Test weight key transformation"""
        config = VLMConfig(
            vision_config={"embed_dim": 1152, "depth": 1},
        )
        model = Qwen2VLModel(config)

        # Test Hugging Face → FlashMLX key transformation
        hf_weights = {
            "visual.patch_embed.weight": mx.zeros((10,)),
            "model.embed_tokens.weight": mx.zeros((100,)),
            "lm_head.weight": mx.zeros((50,)),
        }

        sanitized = model.sanitize(hf_weights)

        # Check transformations
        assert "vision_tower.patch_embed.weight" in sanitized
        assert "language_model.model.embed_tokens.weight" in sanitized
        assert "language_model.lm_head.weight" in sanitized

        print(f"✅ Weight sanitization: HF → FlashMLX keys")


def test_qwen2vl_integration():
    """Integration test: Full fusion pipeline (without language model)"""
    print("\n" + "="*60)
    print("Qwen2-VL Fusion Integration Test")
    print("="*60)

    batch_size = 2
    seq_len = 8
    hidden_dim = 256
    image_token_id = 151655

    print(f"\n1. Setup:")
    print(f"   Batch size: {batch_size}")
    print(f"   Sequence length: {seq_len}")
    print(f"   Hidden dimension: {hidden_dim}")

    # Create input_ids with mixed text and image tokens
    # Batch 0: [A, <img>, <img>, B, C, D, E, F]
    # Batch 1: [X, Y, <img>, Z, W, V, U, T]
    input_ids = mx.array([
        [1, image_token_id, image_token_id, 4, 5, 6, 7, 8],
        [10, 11, image_token_id, 13, 14, 15, 16, 17],
    ])

    print(f"\n2. Input IDs:")
    print(f"   Batch 0: 2 <image> tokens at positions 1, 2")
    print(f"   Batch 1: 1 <image> token at position 2")

    # Create embeddings
    text_embeds = mx.random.normal((batch_size, seq_len, hidden_dim))
    vision_features = mx.random.normal((3, hidden_dim))  # 2 + 1 = 3 total

    print(f"\n3. Embeddings:")
    print(f"   Text embeddings: {text_embeds.shape}")
    print(f"   Vision features: {vision_features.shape} (3 total)")

    # Merge
    merged = Qwen2VLModel.merge_input_ids_with_image_features(
        image_token_id=image_token_id,
        video_token_id=151656,
        image_features=vision_features,
        inputs_embeds=text_embeds,
        input_ids=input_ids,
    )

    print(f"\n4. Merged embeddings: {merged.shape}")

    # Verify
    assert merged.shape == text_embeds.shape
    assert mx.array_equal(merged[0, 1], vision_features[0])
    assert mx.array_equal(merged[0, 2], vision_features[1])
    assert mx.array_equal(merged[1, 2], vision_features[2])

    print(f"\n✅ Integration test PASSED")
    print(f"   Vision features correctly merged at <image> positions")
    print("="*60)


if __name__ == "__main__":
    # Run all tests
    print("Testing FlashMLX Qwen2-VL Model...\n")

    # Test merge logic
    test_merge = TestMergeInputIdsWithImageFeatures()
    test_merge.test_single_image_token()
    test_merge.test_multiple_image_tokens()
    test_merge.test_batch_processing()
    test_merge.test_no_image_tokens()
    test_merge.test_video_token_fallback()

    # Test model
    test_model = TestQwen2VLModel()
    test_model.test_model_initialization()
    test_model.test_sanitize_weights()

    # Integration test
    test_qwen2vl_integration()

    print("\n" + "="*60)
    print("All Qwen2-VL Model tests PASSED ✅")
    print("="*60)
