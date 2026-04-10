"""
Test weight transpose in vision tower sanitize
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

import mlx.core as mx

models_path = project_root / "src" / "flashmlx" / "models"
sys.path.insert(0, str(models_path))

from vision import VisionConfig, VisionModel


def test_linear_weight_transpose():
    """Test that linear weights are transposed correctly."""
    # Create minimal config
    config = VisionConfig(
        depth=2,
        embed_dim=128,
        num_heads=4,
        hidden_size=128,
        in_channels=3,
        patch_size=14,
        spatial_merge_size=2,
        temporal_patch_size=2,
    )

    model = VisionModel(config)

    # Simulate PyTorch weights (out, in) format
    fake_weights = {
        "blocks.0.attn.qkv.weight": mx.random.normal((384, 128)),  # (3*128, 128) PyTorch format
        "blocks.0.attn.proj.weight": mx.random.normal((128, 128)),  # (128, 128) PyTorch format
        "blocks.0.mlp.fc1.weight": mx.random.normal((512, 128)),    # (4*128, 128) PyTorch format
        "blocks.0.mlp.fc2.weight": mx.random.normal((128, 512)),    # (128, 4*128) PyTorch format
    }

    print("Testing weight transpose...")
    print("=" * 60)

    print("\\nBefore sanitize (PyTorch format [out, in]):")
    for k, v in fake_weights.items():
        print(f"  {k}: {v.shape}")

    # Sanitize weights
    sanitized = model.sanitize(fake_weights)

    print("\\nAfter sanitize (MLX format [in, out]):")
    for k, v in sanitized.items():
        print(f"  {k}: {v.shape}")

    # Verify transpose happened
    print("\\nVerification:")
    success = True

    for k in fake_weights.keys():
        original_shape = fake_weights[k].shape
        sanitized_shape = sanitized[k].shape

        # Should be transposed: (out, in) → (in, out)
        expected_shape = (original_shape[1], original_shape[0])

        if sanitized_shape == expected_shape:
            print(f"  ✅ {k}: {original_shape} → {sanitized_shape}")
        else:
            print(f"  ❌ {k}: {original_shape} → {sanitized_shape} (expected {expected_shape})")
            success = False

    print("=" * 60)

    if success:
        print("✅ All weight transposes correct!")
        return True
    else:
        print("❌ Some weight transposes failed!")
        return False


def test_quantized_weights_not_transposed():
    """Test that quantization params (.biases, .scales) are not transposed."""
    config = VisionConfig(
        depth=1,
        embed_dim=128,
        num_heads=4,
        hidden_size=128,
    )

    model = VisionModel(config)

    # Simulate quantized weights
    fake_weights = {
        "blocks.0.attn.qkv.weight": mx.random.normal((384, 16)),    # Compressed
        "blocks.0.attn.qkv.biases": mx.random.normal((384, 2)),     # Should NOT transpose
        "blocks.0.attn.qkv.scales": mx.random.normal((384, 2)),     # Should NOT transpose
    }

    print("\\nTesting quantization params NOT transposed...")
    print("=" * 60)

    print("\\nBefore sanitize:")
    for k, v in fake_weights.items():
        print(f"  {k}: {v.shape}")

    sanitized = model.sanitize(fake_weights)

    print("\\nAfter sanitize:")
    for k, v in sanitized.items():
        print(f"  {k}: {v.shape}")

    # Verify weight is transposed but biases/scales are not
    print("\\nVerification:")
    success = True

    # Weight should be transposed
    if sanitized["blocks.0.attn.qkv.weight"].shape == (16, 384):
        print(f"  ✅ .weight transposed: (384, 16) → (16, 384)")
    else:
        print(f"  ❌ .weight NOT transposed")
        success = False

    # Biases should NOT be transposed
    if sanitized["blocks.0.attn.qkv.biases"].shape == (384, 2):
        print(f"  ✅ .biases NOT transposed: (384, 2) → (384, 2)")
    else:
        print(f"  ❌ .biases transposed incorrectly")
        success = False

    # Scales should NOT be transposed
    if sanitized["blocks.0.attn.qkv.scales"].shape == (384, 2):
        print(f"  ✅ .scales NOT transposed: (384, 2) → (384, 2)")
    else:
        print(f"  ❌ .scales transposed incorrectly")
        success = False

    print("=" * 60)

    if success:
        print("✅ Quantization params handled correctly!")
        return True
    else:
        print("❌ Quantization param handling failed!")
        return False


if __name__ == "__main__":
    print("\\n" + "=" * 60)
    print("Weight Transpose Tests")
    print("=" * 60)

    test1 = test_linear_weight_transpose()
    test2 = test_quantized_weights_not_transposed()

    print("\\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Linear weight transpose: {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"Quantization params: {'✅ PASS' if test2 else '❌ FAIL'}")

    if test1 and test2:
        print("\\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\\n❌ Some tests failed!")
        sys.exit(1)
