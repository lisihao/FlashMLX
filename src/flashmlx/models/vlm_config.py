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

        Note:
            Handles both formats:
            - Nested: {text_config: {...}, vision_config: {...}}
            - Flat: {hidden_size: ..., vision_config: {...}} (HF format)
        """
        # Check if text_config is explicitly provided
        text_config = config_dict.get("text_config")

        # If not, extract text params from root level (HF format)
        if text_config is None:
            # Extract text model parameters from root level
            text_params = {}
            text_param_keys = [
                "model_type", "hidden_size", "num_hidden_layers",
                "intermediate_size", "num_attention_heads", "num_key_value_heads",
                "vocab_size", "rms_norm_eps", "rope_theta", "rope_traditional",
                "rope_scaling", "tie_word_embeddings", "attention_dropout",
                "hidden_act", "max_position_embeddings", "max_window_layers",
                "sliding_window", "use_sliding_window",
            ]

            for key in text_param_keys:
                if key in config_dict:
                    text_params[key] = config_dict[key]

            # Only create text_config if we found parameters
            if text_params:
                text_config = text_params

        return cls(
            model_type=config_dict.get("model_type", "qwen2_vl"),
            vision_config=config_dict.get("vision_config"),
            text_config=text_config,
            image_token_id=config_dict.get("image_token_id", 151655),
            video_token_id=config_dict.get("video_token_id", 151656),
            vision_start_token_id=config_dict.get("vision_start_token_id", 151652),
            vocab_size=config_dict.get("vocab_size", 151936),
            hidden_size=config_dict.get("hidden_size", 3584),
            tie_word_embeddings=config_dict.get("tie_word_embeddings", False),
        )
