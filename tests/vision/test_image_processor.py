"""
Tests for FlashMLX Image Processor

验证图像预处理流程的正确性
"""

import sys
from pathlib import Path

# Add processors module directly to path (avoid flashmlx.__init__)
project_root = Path(__file__).parent.parent.parent
processors_path = project_root / "src" / "flashmlx" / "processors"
sys.path.insert(0, str(processors_path))

import numpy as np
import mlx.core as mx
from PIL import Image
import pytest

# Direct import bypassing package initialization
from image_processing import ImageProcessor


class TestImageProcessor:
    """Test ImageProcessor class"""

    def test_initialization(self):
        """Test ImageProcessor initialization with default parameters"""
        processor = ImageProcessor()

        assert processor.image_size == (448, 448)
        assert processor.do_resize is True
        assert processor.do_normalize is True
        assert processor.do_convert_rgb is True
        np.testing.assert_array_almost_equal(
            processor.image_mean.flatten(),
            [0.485, 0.456, 0.406],
            decimal=5
        )
        np.testing.assert_array_almost_equal(
            processor.image_std.flatten(),
            [0.229, 0.224, 0.225],
            decimal=5
        )
        print("✅ ImageProcessor initialization")

    def test_initialization_custom(self):
        """Test ImageProcessor with custom parameters"""
        processor = ImageProcessor(
            image_size=(224, 224),
            image_mean=[0.5, 0.5, 0.5],
            image_std=[0.5, 0.5, 0.5],
            do_resize=False,
        )

        assert processor.image_size == (224, 224)
        assert processor.do_resize is False
        np.testing.assert_array_almost_equal(
            processor.image_mean.flatten(),
            [0.5, 0.5, 0.5],
            decimal=5
        )
        print("✅ ImageProcessor custom initialization")

    def test_smart_resize_explicit(self):
        """Test smart_resize with explicit target size"""
        processor = ImageProcessor(image_size=448)

        # Create test image 1024x1024
        image = Image.new("RGB", (1024, 1024), color=(128, 128, 128))

        # Resize to 448x448
        resized = processor.smart_resize(image, target_size=(448, 448))

        assert resized.size == (448, 448)
        print(f"✅ Smart resize explicit: (1024, 1024) → {resized.size}")

    def test_smart_resize_aspect_ratio(self):
        """Test smart_resize preserves aspect ratio"""
        processor = ImageProcessor(
            min_pixels=224 * 224,
            max_pixels=1280 * 1280
        )

        # Create rectangular image 800x600
        image = Image.new("RGB", (800, 600), color=(128, 128, 128))

        # Smart resize (should maintain aspect ratio)
        resized = processor.smart_resize(image, target_size=None)

        # Should be rounded to multiples of 28
        width, height = resized.size
        assert width % 28 == 0
        assert height % 28 == 0

        # Aspect ratio approximately preserved
        original_ratio = 800 / 600
        new_ratio = width / height
        assert abs(original_ratio - new_ratio) < 0.1

        print(f"✅ Smart resize aspect ratio: (800, 600) → {resized.size}")

    def test_to_numpy(self):
        """Test PIL Image to numpy conversion"""
        processor = ImageProcessor()

        # RGB image
        image = Image.new("RGB", (100, 100), color=(255, 128, 64))
        array = processor.to_numpy(image)

        assert array.shape == (100, 100, 3)
        assert array.dtype == np.float32
        np.testing.assert_array_equal(array[0, 0], [255.0, 128.0, 64.0])

        print(f"✅ to_numpy: RGB image → {array.shape}")

    def test_to_numpy_grayscale(self):
        """Test grayscale image conversion to RGB"""
        processor = ImageProcessor()

        # Grayscale image
        image = Image.new("L", (100, 100), color=128)
        array = processor.to_numpy(image)

        # Should be converted to 3-channel
        assert array.shape == (100, 100, 3)
        np.testing.assert_array_equal(array[0, 0], [128.0, 128.0, 128.0])

        print(f"✅ to_numpy: Grayscale → {array.shape}")

    def test_normalize(self):
        """Test pixel normalization with ImageNet stats"""
        processor = ImageProcessor()

        # Create test array with known values
        pixel_values = np.ones((100, 100, 3), dtype=np.float32) * 128.0

        normalized = processor.normalize(pixel_values)

        # Check shape unchanged
        assert normalized.shape == (100, 100, 3)

        # Check approximate values
        # 128 / 255 = 0.502, (0.502 - 0.485) / 0.229 ≈ 0.074
        expected = (128.0 / 255.0 - processor.image_mean) / processor.image_std
        np.testing.assert_array_almost_equal(
            normalized[0, 0],
            expected.flatten(),
            decimal=3
        )

        print(f"✅ normalize: range [0, 255] → [{normalized.min():.2f}, {normalized.max():.2f}]")

    def test_to_mlx(self):
        """Test numpy to MLX conversion and HWC → CHW transpose"""
        processor = ImageProcessor()

        # Create test array HWC
        array = np.random.rand(100, 100, 3).astype(np.float32)

        mlx_array = processor.to_mlx(array)

        # Check shape transposed to CHW
        assert mlx_array.shape == (3, 100, 100)
        assert isinstance(mlx_array, mx.array)

        print(f"✅ to_mlx: (100, 100, 3) → {mlx_array.shape}")

    def test_preprocess_pil(self):
        """Test full preprocessing pipeline with PIL Image"""
        processor = ImageProcessor(image_size=448)

        # Create test image
        image = Image.new("RGB", (1024, 1024), color=(255, 128, 64))

        # Preprocess
        pixel_values = processor.preprocess(image)

        # Check output
        assert pixel_values.shape == (3, 448, 448)
        assert isinstance(pixel_values, mx.array)

        # Check normalization applied (values should be roughly in [-2, 2])
        assert -3 < pixel_values.min().item() < 3
        assert -3 < pixel_values.max().item() < 3

        print(f"✅ preprocess PIL: (1024, 1024) → {pixel_values.shape}")

    def test_preprocess_numpy(self):
        """Test preprocessing with numpy array input"""
        processor = ImageProcessor(image_size=224)

        # Create test numpy array (HWC format)
        array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

        # Preprocess
        pixel_values = processor.preprocess(array)

        assert pixel_values.shape == (3, 224, 224)
        assert isinstance(pixel_values, mx.array)

        print(f"✅ preprocess numpy: (512, 512, 3) → {pixel_values.shape}")

    def test_preprocess_return_numpy(self):
        """Test preprocess with return_numpy=True"""
        processor = ImageProcessor(image_size=336)

        image = Image.new("RGB", (800, 600), color=(128, 128, 128))

        # Return numpy instead of MLX
        pixel_values = processor.preprocess(image, return_numpy=True)

        assert pixel_values.shape == (3, 336, 336)
        assert isinstance(pixel_values, np.ndarray)

        print(f"✅ preprocess return_numpy: {pixel_values.shape}")

    def test_preprocess_batch(self):
        """Test batch preprocessing"""
        processor = ImageProcessor(image_size=448)

        # Create batch of 3 images with different sizes
        images = [
            Image.new("RGB", (1024, 768), color=(255, 0, 0)),
            Image.new("RGB", (800, 600), color=(0, 255, 0)),
            Image.new("RGB", (640, 480), color=(0, 0, 255)),
        ]

        # Preprocess batch
        batch = processor.preprocess_batch(images)

        # Check batch shape
        assert batch.shape == (3, 3, 448, 448)  # (B, C, H, W)
        assert isinstance(batch, mx.array)

        print(f"✅ preprocess_batch: 3 images → {batch.shape}")

    def test_preprocess_batch_numpy(self):
        """Test batch preprocessing with numpy return"""
        processor = ImageProcessor(image_size=224)

        images = [
            Image.new("RGB", (512, 512), color=(100, 100, 100)),
            Image.new("RGB", (512, 512), color=(200, 200, 200)),
        ]

        batch = processor.preprocess_batch(images, return_numpy=True)

        assert batch.shape == (2, 3, 224, 224)
        assert isinstance(batch, np.ndarray)

        print(f"✅ preprocess_batch numpy: 2 images → {batch.shape}")

    def test_repr(self):
        """Test string representation"""
        processor = ImageProcessor(image_size=448)
        repr_str = repr(processor)

        assert "ImageProcessor" in repr_str
        assert "image_size=(448, 448)" in repr_str
        assert "do_resize=True" in repr_str

        print(f"✅ __repr__: {len(repr_str)} characters")


class TestImageProcessorRealImages:
    """Test with real test fixture images"""

    def test_with_fixture_image(self):
        """Test preprocessing with actual fixture image"""
        processor = ImageProcessor(image_size=448)

        # Try to load fixture image
        fixture_path = Path(__file__).parent.parent / "datasets" / "fixtures" / "cat.jpg"

        if not fixture_path.exists():
            print(f"⚠️  Fixture image not found: {fixture_path}")
            print("   Skipping real image test")
            return

        # Load and preprocess
        image = Image.open(fixture_path)
        original_size = image.size
        pixel_values = processor.preprocess(image)

        assert pixel_values.shape == (3, 448, 448)

        print(f"✅ Real image preprocessing: {original_size} → {pixel_values.shape}")


def test_image_processor_integration():
    """Integration test: Full preprocessing pipeline"""
    print("\n" + "="*60)
    print("Image Processor Integration Test")
    print("="*60)

    # 1. Create processor
    processor = ImageProcessor(
        image_size=448,
        do_resize=True,
        do_normalize=True
    )
    print(f"\n1. Processor created:")
    print(f"   {processor}")

    # 2. Create test image
    image = Image.new("RGB", (1920, 1080), color=(128, 64, 192))
    print(f"\n2. Test image: {image.size} ({image.mode})")

    # 3. Preprocess
    pixel_values = processor.preprocess(image)
    print(f"\n3. Preprocessed: {pixel_values.shape}")
    print(f"   Value range: [{pixel_values.min().item():.2f}, {pixel_values.max().item():.2f}]")

    # 4. Verify
    assert pixel_values.shape == (3, 448, 448)
    assert -3 < pixel_values.min().item() < 3
    print(f"\n✅ Integration test PASSED")
    print("="*60)


if __name__ == "__main__":
    # Run all tests
    print("Testing FlashMLX Image Processor...\n")

    # Test ImageProcessor class
    test_class = TestImageProcessor()
    test_class.test_initialization()
    test_class.test_initialization_custom()
    test_class.test_smart_resize_explicit()
    test_class.test_smart_resize_aspect_ratio()
    test_class.test_to_numpy()
    test_class.test_to_numpy_grayscale()
    test_class.test_normalize()
    test_class.test_to_mlx()
    test_class.test_preprocess_pil()
    test_class.test_preprocess_numpy()
    test_class.test_preprocess_return_numpy()
    test_class.test_preprocess_batch()
    test_class.test_preprocess_batch_numpy()
    test_class.test_repr()

    # Test with real images
    test_real = TestImageProcessorRealImages()
    test_real.test_with_fixture_image()

    # Integration test
    test_image_processor_integration()

    print("\n" + "="*60)
    print("All Image Processor tests PASSED ✅")
    print("="*60)
