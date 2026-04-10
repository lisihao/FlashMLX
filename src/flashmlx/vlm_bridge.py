"""
FlashMLX VLM Bridge - Deep Integration with mlx-vlm

Provides deep integration between FlashMLX and mlx-vlm (fork).
Unlike monkey-patching, this allows full control over:
- Model architecture modification
- Generation loop optimization
- Custom attention mechanisms
- Vision/Audio encoder enhancement

Architecture:
    FlashMLX → VLM Bridge → mlx-vlm-source (our fork)

Usage:
    >>> from flashmlx.vlm_bridge import load_vlm_model, generate_vlm
    >>> model, processor = load_vlm_model("gemma-4-E4B")
    >>> response = generate_vlm(model, processor, "What is ML?", cache="scored_pq")
"""

import sys
from pathlib import Path
from typing import Optional, Union, List, Dict, Any
from PIL import Image

# Ensure mlx-vlm-source is prioritized
_vlm_source_path = Path(__file__).parent.parent.parent / "mlx-vlm-source"
if str(_vlm_source_path) not in sys.path:
    sys.path.insert(0, str(_vlm_source_path))

# Ensure mlx-lm-source is also available (for cache)
_lm_source_path = Path(__file__).parent.parent.parent / "mlx-lm-source"
if str(_lm_source_path) not in sys.path:
    sys.path.insert(0, str(_lm_source_path))


def load_vlm_model(
    model_path: str,
    lazy: bool = False,
    **kwargs
):
    """Load VLM model from our mlx-vlm fork.

    Args:
        model_path: Path to model directory or HF model ID
        lazy: Whether to lazy load model
        **kwargs: Additional arguments for mlx_vlm.load()

    Returns:
        (model, processor) tuple

    Examples:
        >>> model, processor = load_vlm_model("gemma-4-E4B")
        >>> model, processor = load_vlm_model("/Volumes/toshiba/models/gemma-4-E4B")
    """
    print(f"[FlashMLX VLM Bridge] Loading model from fork: {model_path}")

    # Import from our fork
    from mlx_vlm import load

    model, processor = load(model_path, lazy=lazy, **kwargs)

    print(f"[FlashMLX VLM Bridge] Model loaded: {type(model).__name__}")
    print(f"[FlashMLX VLM Bridge] Using mlx-vlm from: {_vlm_source_path}")

    return model, processor


def create_vlm_cache(
    model,
    strategy: str = "standard",
    max_kv_size: Optional[int] = None,
    **cache_kwargs
):
    """Create FlashMLX optimized cache for VLM model.

    Args:
        model: VLM model (full model, not language_model)
        strategy: Cache strategy (standard, triple_pq, scored_pq, scored_kv_direct)
        max_kv_size: Maximum KV cache size
        **cache_kwargs: Additional cache arguments (density_mode, h0_quant, etc.)

    Returns:
        FlashMLX optimized cache

    Examples:
        >>> cache = create_vlm_cache(model, strategy="scored_pq")
        >>> cache = create_vlm_cache(model, strategy="scored_kv_direct",
        ...                          density_mode="ultra_long")
    """
    print(f"[FlashMLX VLM Bridge] Creating cache: {strategy}")

    # Extract language model for cache creation
    if hasattr(model, 'language_model'):
        language_model = model.language_model
        print(f"[FlashMLX VLM Bridge] Using language_model for cache")
    else:
        language_model = model
        print(f"[FlashMLX VLM Bridge] Using full model for cache")

    # Import FlashMLX cache factory
    from mlx_lm.models.cache_factory import make_optimized_cache

    # Create optimized cache
    cache = make_optimized_cache(
        language_model,
        strategy=strategy,
        max_kv_size=max_kv_size,
        **cache_kwargs
    )

    print(f"[FlashMLX VLM Bridge] Cache created: {type(cache[0]).__name__ if cache else 'None'}")

    return cache


def generate_vlm(
    model,
    processor,
    prompt: str,
    image: Optional[Union[str, Path, Image.Image]] = None,
    audio: Optional[Union[str, Path]] = None,
    cache = None,
    max_tokens: int = 512,
    temperature: float = 0.0,
    verbose: bool = False,
    **kwargs
) -> str:
    """Generate text using VLM model with FlashMLX optimization.

    Args:
        model: VLM model
        processor: Image/audio processor
        prompt: Text prompt
        image: Image path or PIL Image (optional)
        audio: Audio path (optional)
        cache: FlashMLX optimized cache (optional)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.0 = greedy)
        verbose: Whether to print generation progress
        **kwargs: Additional arguments for mlx_vlm.generate()

    Returns:
        Generated text response

    Examples:
        >>> # Text-only generation
        >>> response = generate_vlm(model, processor, "What is ML?", cache=cache)
        >>>
        >>> # Vision+text generation
        >>> response = generate_vlm(
        ...     model, processor,
        ...     "Describe this image",
        ...     image="photo.jpg",
        ...     cache=cache
        ... )
    """
    # Import from our fork
    from mlx_vlm import generate

    # Prepare kwargs
    gen_kwargs = {
        "model": model,
        "processor": processor,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temp": temperature,
        "verbose": verbose,
    }

    # Add image if provided
    if image is not None:
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        gen_kwargs["image"] = image

    # Add audio if provided
    if audio is not None:
        gen_kwargs["audio"] = audio

    # Add FlashMLX cache if provided
    # mlx-vlm's generate() accepts prompt_cache parameter (line 718-722)
    if cache is not None:
        print(f"[FlashMLX VLM Bridge] Using FlashMLX cache: {type(cache[0]).__name__}")
        gen_kwargs["prompt_cache"] = cache

    # Merge additional kwargs
    gen_kwargs.update(kwargs)

    # Generate
    response = generate(**gen_kwargs)

    return response


class VLMGenerationHooks:
    """Hooks for deep modification of VLM generation pipeline.

    Allows injecting custom logic at key points:
    - Pre/post vision encoding
    - Pre/post attention
    - Pre/post generation step

    Examples:
        >>> hooks = VLMGenerationHooks()
        >>> hooks.pre_vision_encode = lambda x: print(f"Encoding {x.shape}")
        >>> # Apply hooks to model
    """

    def __init__(self):
        self.pre_vision_encode: Optional[callable] = None
        self.post_vision_encode: Optional[callable] = None
        self.pre_attention: Optional[callable] = None
        self.post_attention: Optional[callable] = None
        self.pre_generation_step: Optional[callable] = None
        self.post_generation_step: Optional[callable] = None

    def apply_to_model(self, model):
        """Apply hooks to VLM model.

        This modifies the model in-place to call hooks at appropriate points.

        Args:
            model: VLM model to apply hooks to
        """
        print("[FlashMLX VLM Bridge] Applying generation hooks...")

        # TODO: Implement hook injection into mlx-vlm model
        # Will require modifying:
        # - mlx-vlm-source/mlx_vlm/models/*/vision.py
        # - mlx-vlm-source/mlx_vlm/models/*/language.py
        # - mlx-vlm-source/mlx_vlm/generate.py

        raise NotImplementedError("Hook system requires mlx-vlm-source modification")


def get_vlm_info(model) -> Dict[str, Any]:
    """Get architectural information about VLM model.

    Args:
        model: VLM model

    Returns:
        Dictionary with model architecture info

    Examples:
        >>> info = get_vlm_info(model)
        >>> print(f"Layers: {info['num_layers']}, KV heads: {info['num_kv_heads']}")
    """
    info = {}

    # Get language model
    language_model = getattr(model, 'language_model', model)

    # Extract architecture info
    if hasattr(language_model, 'model'):
        inner_model = language_model.model

        # Number of layers
        if hasattr(inner_model, 'layers'):
            info['num_layers'] = len(inner_model.layers)

        # Get config from first layer if available
        if hasattr(inner_model, 'layers') and len(inner_model.layers) > 0:
            first_layer = inner_model.layers[0]

            # Self-attention info
            if hasattr(first_layer, 'self_attn'):
                attn = first_layer.self_attn
                if hasattr(attn, 'n_heads'):
                    info['num_attention_heads'] = attn.n_heads
                if hasattr(attn, 'n_kv_heads'):
                    info['num_kv_heads'] = attn.n_kv_heads
                if hasattr(attn, 'head_dim'):
                    info['head_dim'] = attn.head_dim

    # Vision encoder info
    if hasattr(model, 'vision_encoder'):
        vision = model.vision_encoder
        if hasattr(vision, 'vision_model') and hasattr(vision.vision_model, 'encoder'):
            encoder = vision.vision_model.encoder
            if hasattr(encoder, 'layers'):
                info['vision_layers'] = len(encoder.layers)

        # Vision tokens
        if hasattr(model.config, 'vision_config'):
            vision_config = model.config.vision_config
            if hasattr(vision_config, 'vision_soft_tokens_per_image'):
                info['vision_tokens_per_image'] = vision_config.vision_soft_tokens_per_image

    # Audio encoder info
    if hasattr(model, 'audio_encoder'):
        if hasattr(model.config, 'audio_config'):
            audio_config = model.config.audio_config
            if hasattr(audio_config, 'num_hidden_layers'):
                info['audio_layers'] = audio_config.num_hidden_layers

    return info


__all__ = [
    'load_vlm_model',
    'create_vlm_cache',
    'generate_vlm',
    'VLMGenerationHooks',
    'get_vlm_info',
]
