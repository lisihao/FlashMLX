"""
Configuration classes for FlashMLX Vision-Language Models
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class VLMConfig:
    """Configuration for Vision-Language Models.

    Combines vision and text model configurations with VLM-specific settings.

    Args:
        model_type: Model architecture type (e.g., "qwen2_vl")
        vision_config: Configuration for vision encoder
        text_config: Configuration for language model
        image_token_id: Token ID for <image> placeholder
        video_token_id: Token ID for <video> placeholder (optional)
        vision_start_token_id: Token ID marking vision token start (optional)
        vocab_size: Total vocabulary size
        hidden_size: Hidden dimension size for text model
        tie_word_embeddings: Whether to tie input/output embeddings
    """

    model_type: str = "qwen2_vl"

    # Vision configuration
    vision_config: Optional[dict] = None

    # Text configuration
    text_config: Optional[dict] = None

    # Special token IDs
    image_token_id: int = 151655  # Qwen2-VL default
    video_token_id: int = 151656  # Qwen2-VL default
    vision_start_token_id: Optional[int] = 151652  # Qwen2-VL default

    # Model parameters
    vocab_size: int = 151936
    hidden_size: int = 3584
    tie_word_embeddings: bool = False

    @classmethod
    def from_dict(cls, config_dict: dict) -> "VLMConfig":
        """Create VLMConfig from dictionary.

        Args:
            config_dict: Configuration dictionary from model config.json

        Returns:
            VLMConfig instance
        """
        return cls(
            model_type=config_dict.get("model_type", "qwen2_vl"),
            vision_config=config_dict.get("vision_config"),
            text_config=config_dict.get("text_config"),
            image_token_id=config_dict.get("image_token_id", 151655),
            video_token_id=config_dict.get("video_token_id", 151656),
            vision_start_token_id=config_dict.get("vision_start_token_id", 151652),
            vocab_size=config_dict.get("vocab_size", 151936),
            hidden_size=config_dict.get("hidden_size", 3584),
            tie_word_embeddings=config_dict.get("tie_word_embeddings", False),
        )
