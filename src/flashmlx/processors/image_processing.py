"""
Image Processing for FlashMLX Vision Models

Handles image preprocessing for Vision-Language Models (Qwen2-VL, Qwen3-VL).
"""

from typing import Union, List, Tuple, Optional
from PIL import Image
import numpy as np
import mlx.core as mx


class ImageProcessor:
    """Image processor for Vision Transformer models.

    Converts PIL Images to preprocessed MLX arrays ready for Vision Encoder.
    Supports Qwen2-VL preprocessing with flexible resizing and normalization.

    Args:
        image_size: Target image size (height, width) or single int for square
        min_pixels: Minimum number of pixels (for smart resizing)
        max_pixels: Maximum number of pixels (for smart resizing)
        do_resize: Whether to resize images
        do_normalize: Whether to normalize images
        image_mean: Mean values for normalization (RGB channels)
        image_std: Standard deviation values for normalization (RGB channels)
        do_convert_rgb: Whether to convert images to RGB

    Examples:
        >>> from PIL import Image
        >>> processor = ImageProcessor(image_size=448)
        >>> image = Image.open("cat.jpg")
        >>> pixel_values = processor.preprocess(image)
        >>> print(pixel_values.shape)  # (3, 448, 448)
    """

    def __init__(
        self,
        image_size: Union[int, Tuple[int, int]] = 448,
        min_pixels: int = 256 * 28 * 28,  # Qwen2-VL default
        max_pixels: int = 1280 * 28 * 28,  # Qwen2-VL default
        do_resize: bool = True,
        do_normalize: bool = True,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        do_convert_rgb: bool = True,
    ):
        # Image size
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = image_size

        # Pixel constraints for smart resizing
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        # Processing flags
        self.do_resize = do_resize
        self.do_normalize = do_normalize
        self.do_convert_rgb = do_convert_rgb

        # Normalization parameters (ImageNet defaults)
        self.image_mean = np.array(
            image_mean if image_mean is not None else [0.485, 0.456, 0.406],
            dtype=np.float32
        ).reshape(1, 1, 3)

        self.image_std = np.array(
            image_std if image_std is not None else [0.229, 0.224, 0.225],
            dtype=np.float32
        ).reshape(1, 1, 3)

    def smart_resize(
        self,
        image: Image.Image,
        target_size: Optional[Tuple[int, int]] = None
    ) -> Image.Image:
        """Resize image with aspect ratio preservation.

        Implements Qwen2-VL smart resizing:
        - Maintains aspect ratio
        - Ensures pixel count within [min_pixels, max_pixels]
        - Rounds to multiples of patch size (28 for Qwen2-VL)

        Args:
            image: PIL Image to resize
            target_size: Optional explicit target size (height, width)

        Returns:
            Resized PIL Image
        """
        if target_size is not None:
            # Explicit resize
            return image.resize((target_size[1], target_size[0]), Image.BICUBIC)

        # Get original dimensions
        width, height = image.size
        original_pixels = width * height

        # Calculate scale factor to fit within [min_pixels, max_pixels]
        if original_pixels > self.max_pixels:
            scale = (self.max_pixels / original_pixels) ** 0.5
        elif original_pixels < self.min_pixels:
            scale = (self.min_pixels / original_pixels) ** 0.5
        else:
            scale = 1.0

        # Apply scale
        new_width = int(width * scale)
        new_height = int(height * scale)

        # Round to multiples of 28 (patch size * spatial_merge_size: 14 * 2)
        patch_grid_size = 28
        new_width = (new_width // patch_grid_size) * patch_grid_size
        new_height = (new_height // patch_grid_size) * patch_grid_size

        # Ensure minimum size
        new_width = max(new_width, patch_grid_size)
        new_height = max(new_height, patch_grid_size)

        return image.resize((new_width, new_height), Image.BICUBIC)

    def to_numpy(self, image: Image.Image) -> np.ndarray:
        """Convert PIL Image to numpy array.

        Args:
            image: PIL Image

        Returns:
            Numpy array in HWC format, shape (H, W, 3), dtype float32
        """
        # Convert to RGB if needed
        if self.do_convert_rgb and image.mode != "RGB":
            image = image.convert("RGB")

        # To numpy array
        pixel_values = np.array(image, dtype=np.float32)

        # Ensure 3 channels (grayscale → RGB)
        if pixel_values.ndim == 2:
            pixel_values = np.stack([pixel_values] * 3, axis=-1)

        return pixel_values

    def normalize(self, pixel_values: np.ndarray) -> np.ndarray:
        """Normalize pixel values with ImageNet statistics.

        Formula: (pixel / 255.0 - mean) / std

        Args:
            pixel_values: Numpy array in HWC format, range [0, 255]

        Returns:
            Normalized array, range approximately [-2, 2]
        """
        # Scale to [0, 1]
        pixel_values = pixel_values / 255.0

        # Normalize with ImageNet stats
        pixel_values = (pixel_values - self.image_mean) / self.image_std

        return pixel_values

    def to_mlx(self, pixel_values: np.ndarray) -> mx.array:
        """Convert numpy array to MLX array and transpose to CHW format.

        Args:
            pixel_values: Numpy array in HWC format

        Returns:
            MLX array in CHW format, shape (3, H, W)
        """
        # Convert to MLX
        pixel_values = mx.array(pixel_values)

        # Transpose HWC → CHW
        pixel_values = pixel_values.transpose(2, 0, 1)

        return pixel_values

    def preprocess(
        self,
        image: Union[Image.Image, np.ndarray],
        return_numpy: bool = False
    ) -> Union[mx.array, np.ndarray]:
        """Preprocess a single image for Vision Encoder.

        Full pipeline:
        1. Convert to RGB (if needed)
        2. Resize (smart or fixed)
        3. Convert to numpy array
        4. Normalize (ImageNet stats)
        5. Convert to MLX array (CHW format)

        Args:
            image: PIL Image or numpy array (HWC format)
            return_numpy: If True, return numpy array instead of MLX

        Returns:
            Preprocessed image as MLX array (C, H, W) or numpy array

        Examples:
            >>> processor = ImageProcessor(image_size=448)
            >>> image = Image.open("cat.jpg")
            >>> pixel_values = processor.preprocess(image)
            >>> print(pixel_values.shape)  # (3, 448, 448)
        """
        # Convert numpy to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))

        # Resize
        if self.do_resize:
            image = self.smart_resize(image, target_size=self.image_size)

        # To numpy
        pixel_values = self.to_numpy(image)

        # Normalize
        if self.do_normalize:
            pixel_values = self.normalize(pixel_values)

        # Return numpy or MLX
        if return_numpy:
            # Transpose to CHW for numpy too
            return pixel_values.transpose(2, 0, 1)
        else:
            return self.to_mlx(pixel_values)

    def preprocess_batch(
        self,
        images: List[Union[Image.Image, np.ndarray]],
        return_numpy: bool = False
    ) -> Union[mx.array, np.ndarray]:
        """Preprocess a batch of images.

        Args:
            images: List of PIL Images or numpy arrays
            return_numpy: If True, return numpy array instead of MLX

        Returns:
            Batched preprocessed images, shape (B, C, H, W)

        Note:
            All images are resized to the same size for batching.
        """
        processed = [self.preprocess(img, return_numpy=return_numpy) for img in images]

        if return_numpy:
            return np.stack(processed, axis=0)
        else:
            return mx.stack(processed, axis=0)

    def __repr__(self) -> str:
        return (
            f"ImageProcessor(\n"
            f"  image_size={self.image_size},\n"
            f"  min_pixels={self.min_pixels},\n"
            f"  max_pixels={self.max_pixels},\n"
            f"  do_resize={self.do_resize},\n"
            f"  do_normalize={self.do_normalize},\n"
            f"  image_mean={self.image_mean.flatten().tolist()},\n"
            f"  image_std={self.image_std.flatten().tolist()}\n"
            f")"
        )
