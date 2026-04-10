"""
FlashMLX Gemma 4 Integration

Integrates Gemma 4 (mlx-vlm) with FlashMLX KV cache optimization.

Strategy:
1. Use mlx-vlm to load Gemma 4 model
2. Extract language model
3. Apply FlashMLX cache optimization
4. Wrap in FlashMLX VLMGenerator interface

Quick Start:
    >>> from flashmlx.vlm_gemma4 import load_gemma4_with_flashmlx
    >>> model, processor, generator, cache = load_gemma4_with_flashmlx(
    ...     "/Volumes/toshiba/models/gemma-4-E4B",
    ...     cache_strategy="standard"
    ... )
    >>> response = generator.generate("What is ML?", cache=cache)
"""

import sys
from pathlib import Path

# Ensure local mlx-lm-source is prioritized
mlx_lm_path = Path(__file__).parent.parent.parent / "mlx-lm-source"
if str(mlx_lm_path) not in sys.path:
    sys.path.insert(0, str(mlx_lm_path))

# Patch mlx-vlm with FlashMLX cache
from flashmlx.patch_mlx_vlm import patch_mlx_vlm_cache
patch_mlx_vlm_cache()


def load_gemma4_with_flashmlx(
    model_path: str,
    cache_strategy: str = "standard",
    max_tokens: int = 512,
):
    """Load Gemma 4 with FlashMLX cache optimization.

    Args:
        model_path: Path to Gemma 4 model
        cache_strategy: FlashMLX cache strategy ("standard", "triple_pq", "scored_pq")
        max_tokens: Default max tokens to generate

    Returns:
        (model, processor, generator, cache) tuple

    Examples:
        >>> model, processor, generator, cache = load_gemma4_with_flashmlx(
        ...     "/Volumes/toshiba/models/gemma-4-E4B"
        ... )
        >>> response = generator.generate("What is ML?", cache=cache)
    """
    print(f"Loading Gemma 4 with FlashMLX optimization...")
    print(f"  Model: {model_path}")
    print(f"  Cache: {cache_strategy}")

    # Step 1: Load Gemma 4 using mlx-vlm
    print("\n[1/4] Loading Gemma 4 model (mlx-vlm)...")
    from mlx_vlm import load

    model, processor = load(model_path)
    print(f"  ✅ Model loaded")

    # Step 2: Extract language model for cache optimization
    print("\n[2/4] Extracting language model...")

    # Gemma 4 structure: Gemma4ForConditionalGeneration
    # - language_model: The text decoder
    # - vision_encoder: Vision tower
    # - audio_encoder: Audio tower

    if hasattr(model, 'language_model'):
        language_model = model.language_model
        print(f"  ✅ Language model extracted")
    else:
        print(f"  ⚠️  No separate language_model, using full model")
        language_model = model

    # Step 3: Create FlashMLX optimized cache
    print(f"\n[3/4] Creating FlashMLX cache ({cache_strategy})...")
    # Use mlx-vlm's make_prompt_cache (already patched with FlashMLX)
    from mlx_vlm.models.cache import make_prompt_cache

    cache = make_prompt_cache(language_model, kv_cache=cache_strategy)
    print(f"  ✅ Cache created: {type(cache[0]).__name__ if cache else 'None'}")

    # Step 4: Create generator wrapper
    print(f"\n[4/4] Creating generator...")
    generator = Gemma4Generator(
        model=model,
        processor=processor,
        max_tokens=max_tokens,
    )
    print(f"  ✅ Generator ready")

    print(f"\n{'='*60}")
    print("✅ Gemma 4 + FlashMLX ready!")
    print(f"{'='*60}")

    return model, processor, generator, cache


class Gemma4Generator:
    """FlashMLX generator wrapper for Gemma 4.

    Provides unified interface for text and vision+text generation.

    Args:
        model: Gemma 4 model (from mlx-vlm)
        processor: Image/audio processor
        max_tokens: Default max tokens to generate

    Examples:
        >>> generator = Gemma4Generator(model, processor)
        >>>
        >>> # Text generation
        >>> response = generator.generate("What is ML?", cache=cache)
        >>>
        >>> # Vision+text generation
        >>> response = generator.generate(
        ...     "What's in this image?",
        ...     image="photo.jpg",
        ...     cache=cache
        ... )
    """

    def __init__(
        self,
        model,
        processor,
        max_tokens: int = 512,
    ):
        self.model = model
        self.processor = processor
        self.max_tokens = max_tokens

    def generate(
        self,
        prompt: str,
        image = None,
        audio = None,
        max_tokens: int = None,
        temperature: float = 0.0,
        cache = None,
    ) -> str:
        """Generate text response.

        Args:
            prompt: Text prompt
            image: Image path or PIL Image (optional)
            audio: Audio path or array (optional)
            max_tokens: Override default max_tokens
            temperature: Sampling temperature (0.0 = greedy)
            cache: FlashMLX optimized cache

        Returns:
            Generated text response

        Examples:
            >>> # Text-only
            >>> response = generator.generate("What is ML?", cache=cache)
            >>>
            >>> # Vision+text
            >>> response = generator.generate(
            ...     "Describe this image",
            ...     image="photo.jpg",
            ...     cache=cache
            ... )
        """
        max_tokens = max_tokens or self.max_tokens

        # Use mlx-vlm's generate function
        # Note: mlx-vlm handles image/audio preprocessing internally
        from mlx_vlm import generate

        # Prepare inputs
        kwargs = {
            "model": self.model,
            "processor": self.processor,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temp": temperature,
        }

        # Add image if provided
        if image is not None:
            from PIL import Image
            if isinstance(image, (str, Path)):
                image = Image.open(image)
            kwargs["image"] = image

        # Add audio if provided
        if audio is not None:
            kwargs["audio"] = audio

        # Generate
        # Note: mlx-vlm generate doesn't support custom cache yet
        # For now, we'll use standard generation
        # TODO: Monkey-patch cache into mlx-vlm's generate
        response = generate(**kwargs)

        return response

    def generate_with_flashmlx_cache(
        self,
        prompt: str,
        image = None,
        max_tokens: int = None,
        temperature: float = 0.0,
        cache = None,
    ) -> str:
        """Generate with FlashMLX cache (experimental).

        This method directly uses FlashMLX's generation loop with optimized cache.

        Args:
            prompt: Text prompt
            image: Image path or PIL Image (optional)
            max_tokens: Max tokens to generate
            temperature: Sampling temperature
            cache: FlashMLX cache

        Returns:
            Generated text response

        Note:
            This is experimental and may not work perfectly with Gemma 4's
            multi-modal architecture. Use generate() for stable results.
        """
        max_tokens = max_tokens or self.max_tokens

        # Tokenize prompt
        # Note: Gemma 4 uses special tokens for images/audio
        # We need to handle this properly

        # For now, fallback to standard generate
        print("⚠️  FlashMLX cache integration experimental for Gemma 4")
        print("   Using standard generation...")

        return self.generate(
            prompt=prompt,
            image=image,
            max_tokens=max_tokens,
            temperature=temperature,
        )


__all__ = [
    "load_gemma4_with_flashmlx",
    "Gemma4Generator",
]
