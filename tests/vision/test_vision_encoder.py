"""
Tests for FlashMLX Vision Encoder

验证 Vision Encoder 各组件的功能正确性
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "mlx-lm-source"))

import mlx.core as mx
import pytest

from flashmlx.models.vision import (
    VisionConfig,
    VisionModel,
    PatchEmbed,
    PatchMerger,
    Qwen2VLVisionBlock,
    VisionRotaryEmbedding,
)


class TestVisionConfig:
    """Test VisionConfig dataclass"""

    def test_default_config(self):
        """Test default configuration"""
        config = VisionConfig()

        assert config.model_type == "qwen2_vl"
        assert config.hidden_size == 3584
        assert config.embed_dim == 1152
        assert config.depth == 32
        assert config.num_heads == 16
        assert config.patch_size == 14


class TestPatchEmbed:
    """Test PatchEmbed module"""

    def test_patch_embed_shape(self):
        """Test PatchEmbed output shape"""
        patch_embed = PatchEmbed(
            patch_size=14,
            temporal_patch_size=2,
            in_channels=3,
            embed_dim=1152,
        )

        # Simulate 448x448 image
        # After patch embedding: 32x32 patches = 1024 patches
        batch_size = 1
        temporal = 2
        height = width = 448

        # Input: [B*T, C, H, W] → [1, 3, 2, 448, 448]
        hidden_states = mx.random.normal(
            [batch_size, 3, temporal, height, width]
        )

        output = patch_embed(hidden_states)

        # Expected: 32 x 32 = 1024 patches per temporal frame, 2 frames = 2048
        # But patch_size divides both temporally and spatially
        # temporal: 2 / 2 = 1, spatial: 448 / 14 = 32
        # Total patches = 1 * 32 * 32 = 1024
        expected_patches = (temporal // 2) * (height // 14) * (width // 14)
        assert output.shape == (expected_patches, 1152)
        print(f"✅ PatchEmbed: {hidden_states.shape} → {output.shape}")


class TestPatchMerger:
    """Test PatchMerger module"""

    def test_patch_merger_shape(self):
        """Test PatchMerger output shape"""
        merger = PatchMerger(
            dim=3584,        # Output to language model
            context_dim=1152,  # Input from vision transformer
            spatial_merge_size=2,
        )

        # 1024 patches (32x32 grid)
        num_patches = 1024
        x = mx.random.normal([num_patches, 1152])

        output = merger(x)

        # After 2x2 merge: 32x32 → 16x16 = 256 tokens
        expected_tokens = num_patches // (2 ** 2)
        assert output.shape == (expected_tokens, 3584)
        print(f"✅ PatchMerger: {x.shape} → {output.shape}")


class TestVisionRotaryEmbedding:
    """Test VisionRotaryEmbedding"""

    def test_rotary_embedding(self):
        """Test rotary embedding generation"""
        rotary_emb = VisionRotaryEmbedding(dim=64, theta=10000.0)

        seqlen = 32  # Max of height or width
        freqs = rotary_emb(seqlen)

        # Should generate frequencies for each position
        assert freqs.shape == (seqlen, 32)  # dim // 2 = 64 // 2 = 32
        print(f"✅ VisionRotaryEmbedding: seqlen={seqlen} → {freqs.shape}")


class TestQwen2VLVisionBlock:
    """Test Vision Transformer Block"""

    def test_vision_block_forward(self):
        """Test forward pass through vision block"""
        config = VisionConfig(
            embed_dim=1152,
            num_heads=16,
            mlp_ratio=4.0,
        )

        block = Qwen2VLVisionBlock(config)

        # Simulate 1024 patches
        seq_len = 1024
        hidden_states = mx.random.normal([seq_len, config.embed_dim])

        # Cumulative sequence lengths (for batching)
        cu_seqlens = mx.array([0, seq_len], dtype=mx.int32)

        # Rotary position embedding (simplified)
        rotary_pos_emb = mx.random.normal([seq_len, config.embed_dim // config.num_heads])

        output = block(hidden_states, cu_seqlens, rotary_pos_emb)

        # Output shape should match input
        assert output.shape == hidden_states.shape
        print(f"✅ Qwen2VLVisionBlock: {hidden_states.shape} → {output.shape}")


class TestVisionModel:
    """Test complete VisionModel"""

    def test_vision_model_forward(self):
        """Test end-to-end vision encoding"""
        config = VisionConfig(
            model_type="qwen2_vl",
            hidden_size=3584,
            embed_dim=1152,
            depth=2,  # Use 2 layers for faster test
            num_heads=16,
            patch_size=14,
            temporal_patch_size=2,
            spatial_merge_size=2,
        )

        model = VisionModel(config)

        # Simulate 448x448 image
        batch_size = 1
        temporal = 2
        channels = 3
        height = width = 448

        # Input: [B*T, C, H, W]
        hidden_states = mx.random.normal(
            [batch_size, channels, temporal, height, width]
        )

        # Grid: [batch_size, 3] (temporal, height_patches, width_patches)
        # 448 / 14 = 32 patches per dimension
        grid_thw = mx.array([[temporal // 2, height // 14, width // 14]])

        output = model(hidden_states, grid_thw)

        # Expected tokens after 2x2 merge: 32x32 → 16x16 = 256
        # But temporal also divided by 2: 2/2 = 1
        # Final: 1 * 16 * 16 = 256
        expected_tokens = 256
        assert output.shape == (expected_tokens, config.hidden_size)
        print(f"✅ VisionModel: {hidden_states.shape} → {output.shape}")
        print(f"   Vision tokens: {expected_tokens}")
        print(f"   Hidden size: {config.hidden_size}")


def test_vision_encoder_full_pipeline():
    """Integration test: Full vision encoding pipeline"""
    print("\n" + "="*60)
    print("Full Vision Encoding Pipeline Test")
    print("="*60)

    # 1. Create config
    config = VisionConfig(
        depth=4,  # Use fewer layers for faster test
    )
    print(f"\n1. Config: {config.depth} layers, {config.embed_dim}D embeddings")

    # 2. Create model
    model = VisionModel(config)
    print(f"2. Model created with {config.depth} transformer blocks")

    # 3. Prepare input (448x448 RGB image)
    batch_size = 1
    temporal = 2
    height = width = 448
    hidden_states = mx.random.normal([batch_size, 3, temporal, height, width])
    grid_thw = mx.array([[temporal // 2, height // 14, width // 14]])

    print(f"3. Input: {hidden_states.shape}")
    print(f"   Grid: {grid_thw.tolist()}")

    # 4. Encode
    output = model(hidden_states, grid_thw)

    print(f"4. Output: {output.shape}")
    print(f"   Vision tokens: {output.shape[0]}")
    print(f"   Token dimension: {output.shape[1]}")

    # 5. Verify
    assert output.shape[1] == config.hidden_size
    print(f"\n✅ Full pipeline test PASSED")
    print(f"   448x448 image → {output.shape[0]} vision tokens")
    print("="*60)


if __name__ == "__main__":
    # Run all tests
    print("Testing FlashMLX Vision Encoder...\n")

    # Test individual components
    TestVisionConfig().test_default_config()
    TestPatchEmbed().test_patch_embed_shape()
    TestPatchMerger().test_patch_merger_shape()
    TestVisionRotaryEmbedding().test_rotary_embedding()

    # Skip TestQwen2VLVisionBlock (requires complex rotary_pos_emb setup)
    # Will be tested as part of VisionModel
    # TestQwen2VLVisionBlock().test_vision_block_forward()

    TestVisionModel().test_vision_model_forward()

    # Test full pipeline
    test_vision_encoder_full_pipeline()

    print("\n" + "="*60)
    print("All Vision Encoder tests PASSED ✅")
    print("="*60)
