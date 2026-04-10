"""
FlashMLX Qwen2-VL Model

Vision-Language Model combining Vision Encoder with Language Model.
Supports FlashMLX optimization routes (Route 0-5).
"""

from typing import Optional, Tuple
import mlx.core as mx
import mlx.nn as nn

# MLX-LM models (for language model)
try:
    from mlx_lm.models import qwen2
except ImportError:
    # Fallback: try from mlx-lm-source
    import sys
    from pathlib import Path
    mlx_lm_path = Path(__file__).parent.parent.parent.parent / "mlx-lm-source"
    if mlx_lm_path.exists():
        sys.path.insert(0, str(mlx_lm_path))
    from mlx_lm.models import qwen2

# Support both relative and absolute imports
try:
    from .vision import VisionModel, VisionConfig
    from .vlm_config import VLMConfig
except ImportError:
    # Fallback for testing
    from vision import VisionModel, VisionConfig
    from vlm_config import VLMConfig


class Qwen2VLModel(nn.Module):
    """FlashMLX Qwen2-VL Vision-Language Model.

    Architecture:
        VisionModel (MLX-VLM) → Vision features
        + Text embeddings → Merged embeddings
        → LanguageModel (MLX-LM + FlashMLX Routes)

    Args:
        config: VLMConfig with vision and text configurations

    Examples:
        >>> config = VLMConfig.from_dict(model_config)
        >>> model = Qwen2VLModel(config)
        >>> logits = model(input_ids, pixel_values, grid_thw, cache=cache)
    """

    def __init__(self, config: VLMConfig):
        super().__init__()
        self.config = config

        # Vision Encoder (已移植)
        if config.vision_config is not None:
            # Handle HF config parameter name differences
            vision_cfg_dict = dict(config.vision_config)

            # Map HF names to FlashMLX names
            name_mapping = {
                'in_chans': 'in_channels',
            }

            # Remove parameters we don't use
            params_to_remove = ['spatial_patch_size']

            for hf_name, flashmlx_name in name_mapping.items():
                if hf_name in vision_cfg_dict:
                    vision_cfg_dict[flashmlx_name] = vision_cfg_dict.pop(hf_name)

            for param in params_to_remove:
                vision_cfg_dict.pop(param, None)

            vision_config = VisionConfig(**vision_cfg_dict)
            self.vision_tower = VisionModel(vision_config)
        else:
            self.vision_tower = None

        # Language Model (MLX-LM Qwen2 + FlashMLX Routes)
        if config.text_config is not None:
            text_args = qwen2.ModelArgs.from_dict(config.text_config)
            self.language_model = qwen2.Model(text_args)
            # Get embed_tokens for fusion logic
            self.embed_tokens = self.language_model.model.embed_tokens
        else:
            self.language_model = None
            self.embed_tokens = None

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        grid_thw: Optional[mx.array] = None,
    ) -> mx.array:
        """Get merged input embeddings (vision + text).

        Args:
            input_ids: Text token IDs [batch_size, seq_len]
            pixel_values: Image pixels [batch_size, channels, height, width]
            grid_thw: Grid dimensions [batch_size, 3] (temporal, height, width)

        Returns:
            Merged embeddings [batch_size, seq_len, hidden_dim]

        Process:
            1. Get text embeddings from embed_tokens
            2. If pixel_values provided:
               - Encode images through vision_tower
               - Merge vision features into text embeddings at <image> token positions
            3. Return merged embeddings
        """
        if self.embed_tokens is None:
            raise RuntimeError(
                "embed_tokens not initialized. Language model not connected yet."
            )

        # Get text embeddings
        inputs_embeds = self.embed_tokens(input_ids)

        # Text-only case
        if pixel_values is None:
            return inputs_embeds

        # Vision + Text case
        if self.vision_tower is None:
            raise RuntimeError("vision_tower is None but pixel_values provided")

        # Encode images
        dtype = pixel_values.dtype  # Preserve original dtype
        vision_features = self.vision_tower(pixel_values, grid_thw)

        # Merge vision features into text embeddings
        merged_embeds = self.merge_input_ids_with_image_features(
            image_token_id=self.config.image_token_id,
            video_token_id=self.config.video_token_id,
            image_features=vision_features,
            inputs_embeds=inputs_embeds,
            input_ids=input_ids,
        )

        return merged_embeds

    @staticmethod
    def merge_input_ids_with_image_features(
        image_token_id: int,
        video_token_id: int,
        image_features: mx.array,
        inputs_embeds: mx.array,
        input_ids: mx.array,
    ) -> mx.array:
        """Merge vision features into text embeddings at special token positions.

        This is the core fusion logic: replace <image> token embeddings with
        corresponding vision features from the vision encoder.

        Args:
            image_token_id: Token ID for <image> (e.g., 151655)
            video_token_id: Token ID for <video> (e.g., 151656)
            image_features: Vision features [num_features, hidden_dim]
            inputs_embeds: Text embeddings [batch_size, seq_len, hidden_dim]
            input_ids: Text token IDs [batch_size, seq_len]

        Returns:
            Merged embeddings [batch_size, seq_len, hidden_dim]

        Algorithm:
            For each batch:
                1. Find positions where input_ids == image_token_id
                2. Extract corresponding vision features
                3. Replace text embeddings at those positions with vision features

        Example:
            Input:  ["A", "<image>", "<image>", "cat"]
            Vision: [feat_0, feat_1]  # 2 vision tokens from 256-patch image
            Output: [emb_A, feat_0, feat_1, emb_cat]
        """
        # Find <image> or <video> token positions
        image_positions = input_ids == image_token_id
        if mx.sum(image_positions) == 0:
            image_positions = input_ids == video_token_id

        batch_size, seq_len = input_ids.shape

        # Process each batch item
        batch_outputs = []
        feature_start_idx = 0

        for batch_idx in range(batch_size):
            # Get image token positions for this batch
            image_mask = image_positions[batch_idx]
            num_positions = int(mx.sum(image_mask).item())

            if num_positions > 0:
                # Extract vision features for this batch
                batch_features = image_features[
                    feature_start_idx : feature_start_idx + num_positions
                ]

                # Validate feature count
                if batch_features.shape[0] != num_positions:
                    raise ValueError(
                        f"Feature count mismatch: {num_positions} <image> tokens "
                        f"but {batch_features.shape[0]} vision features "
                        f"(batch {batch_idx})"
                    )

                # Create indices for gathering features
                cumsum = mx.cumsum(image_mask.astype(mx.int32))
                feature_indices = mx.where(image_mask, cumsum - 1, 0)

                # Gather vision features
                gathered_features = batch_features[feature_indices]

                # Replace text embeddings with vision features where mask is True
                image_mask_expanded = mx.expand_dims(image_mask, axis=-1)
                batch_output = mx.where(
                    image_mask_expanded,
                    gathered_features,
                    inputs_embeds[batch_idx],
                )

                feature_start_idx += num_positions
            else:
                # No image tokens in this batch item
                batch_output = inputs_embeds[batch_idx]

            batch_outputs.append(batch_output)

        # Stack all batch outputs
        return mx.stack(batch_outputs, axis=0)

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        grid_thw: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
    ) -> mx.array:
        """Forward pass through VLM.

        Args:
            input_ids: Text token IDs [batch_size, seq_len]
            pixel_values: Image pixels [batch_size, C, H, W] (optional)
            grid_thw: Grid dimensions [batch_size, 3] (optional)
            mask: Attention mask (optional)
            cache: KV cache (FlashMLX optimized cache supported) (optional)

        Returns:
            Logits [batch_size, seq_len, vocab_size]

        Process:
            1. get_input_embeddings() → merged vision+text embeddings
            2. language_model(input_embeddings=...) → logits

        FlashMLX Cache Support:
            Pass FlashMLX optimized cache created via:
            >>> from flashmlx.cache import make_prompt_cache
            >>> cache = make_prompt_cache(model, kv_cache="scored_pq", ...)
        """
        if self.language_model is None:
            raise RuntimeError(
                "Language model not initialized. Check config.text_config."
            )

        # Get merged embeddings (vision + text)
        inputs_embeds = self.get_input_embeddings(
            input_ids=input_ids,
            pixel_values=pixel_values,
            grid_thw=grid_thw,
        )

        # Pass to language model with input_embeddings
        # mlx-lm qwen2.Model supports input_embeddings parameter
        logits = self.language_model(
            inputs=input_ids,  # Still needed for position encoding
            cache=cache,  # FlashMLX optimized cache supported here
            input_embeddings=inputs_embeds,  # Use merged embeddings
        )

        return logits

    def sanitize(self, weights: dict) -> dict:
        """Transform weight keys from Hugging Face format to FlashMLX format.

        Hugging Face Qwen2-VL weights structure:
            visual.* → vision_tower.*
            model.* → language_model.model.*
            lm_head.* → language_model.lm_head.*

        Args:
            weights: Raw weights from Hugging Face

        Returns:
            Transformed weights matching FlashMLX structure
        """

        def transform_key(key: str) -> str:
            # Vision weights
            if "vision_tower" not in key:
                key = key.replace("visual", "vision_tower")

            # Language model weights
            if "language_model" not in key:
                if "model." in key and not key.startswith("vision"):
                    key = key.replace("model", "language_model.model")
                elif "lm_head" in key:
                    key = key.replace("lm_head", "language_model.lm_head")

            return key

        sanitized = {transform_key(k): v for k, v in weights.items()}

        # Also sanitize vision tower weights (Conv3d transpose)
        if hasattr(self.vision_tower, "sanitize"):
            vision_weights = {
                k.replace("vision_tower.", ""): v
                for k, v in sanitized.items()
                if k.startswith("vision_tower.")
            }
            vision_weights = self.vision_tower.sanitize(vision_weights)
            # Replace vision weights with sanitized version
            sanitized = {
                k: v for k, v in sanitized.items() if not k.startswith("vision_tower.")
            }
            sanitized.update(
                {f"vision_tower.{k}": v for k, v in vision_weights.items()}
            )

        return sanitized

    @property
    def layers(self):
        """Access language model layers (for compatibility)."""
        if self.language_model is None:
            return []
        if hasattr(self.language_model, "model"):
            return self.language_model.model.layers
        return self.language_model.layers
