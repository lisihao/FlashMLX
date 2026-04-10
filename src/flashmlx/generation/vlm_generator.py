"""
FlashMLX VLM Text Generation

Simple generation utilities for Vision-Language Models.
Supports text-only and vision+text generation with proper tokenization.
"""

from typing import Optional, List
import mlx.core as mx


class VLMGenerator:
    """VLM text generator with tokenizer integration.

    Provides simple greedy generation for Vision-Language Models.
    Uses mlx-lm tokenizer for proper text encoding/decoding.

    Args:
        model: FlashMLX VLM model (Qwen2VLModel)
        tokenizer: mlx-lm TokenizerWrapper
        image_token_id: Token ID for <image> placeholder (default: 151655)
        max_tokens: Maximum tokens to generate (default: 100)

    Examples:
        >>> from mlx_lm.utils import load_tokenizer
        >>> tokenizer = load_tokenizer(model_path)
        >>> generator = VLMGenerator(model, tokenizer)
        >>>
        >>> # Text-only generation
        >>> response = generator.generate("What is MLX?")
        >>>
        >>> # Vision+text generation
        >>> response = generator.generate(
        ...     "What is in this image?",
        ...     pixel_values=pixel_values,
        ...     grid_thw=grid_thw
        ... )
    """

    def __init__(
        self,
        model,
        tokenizer,
        image_token_id: int = 151655,
        max_tokens: int = 100,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.image_token_id = image_token_id
        self.max_tokens = max_tokens

    def _format_prompt(self, prompt: str, use_chat_template: bool = True) -> str:
        """Format prompt using chat template if available.

        Args:
            prompt: User prompt text
            use_chat_template: Whether to use chat template (default: True)

        Returns:
            Formatted prompt string
        """
        if not use_chat_template or not hasattr(self.tokenizer, 'apply_chat_template'):
            return prompt

        # Format as chat message
        messages = [
            {"role": "user", "content": prompt}
        ]

        try:
            # Apply chat template
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,  # Add <|im_start|>assistant
            )
            return formatted
        except Exception as e:
            # Fallback to plain text if template fails
            print(f"  [WARN] Chat template failed: {e}, using plain text")
            return prompt

    def generate(
        self,
        prompt: str,
        pixel_values: Optional[mx.array] = None,
        grid_thw: Optional[mx.array] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
        cache=None,
        use_chat_template: bool = True,
    ) -> str:
        """Generate text response from prompt and optional image.

        Args:
            prompt: Text prompt (can include <image> placeholder)
            pixel_values: Image pixels [batch, C, T, H, W] (optional)
            grid_thw: Grid dimensions [batch, 3] (optional)
            max_tokens: Override default max_tokens (optional)
            temperature: Sampling temperature (0.0 = greedy, default)
            cache: FlashMLX optimized cache (optional)
            use_chat_template: Whether to format with chat template (default: True)

        Returns:
            Generated text response

        Process:
            1. Format prompt with chat template (if available)
            2. Tokenize prompt (handles <image> tokens automatically)
            3. Run generation loop:
               - Forward pass through model
               - Sample next token
               - Append to sequence
            4. Detokenize output tokens to text
        """
        max_tokens = max_tokens or self.max_tokens

        # Format prompt with chat template
        formatted_prompt = self._format_prompt(prompt, use_chat_template)

        # Tokenize prompt
        # For Qwen2-VL, <image> is automatically tokenized to image_token_id
        input_ids = mx.array(self.tokenizer.encode(formatted_prompt))
        input_ids = mx.expand_dims(input_ids, axis=0)  # [1, seq_len]

        # Initialize generation
        generated_tokens = []
        current_ids = input_ids

        # Greedy generation loop
        for i in range(max_tokens):
            # Forward pass
            logits = self.model(
                input_ids=current_ids,
                pixel_values=pixel_values if i == 0 else None,  # Only process image once
                grid_thw=grid_thw if i == 0 else None,
                cache=cache,
            )

            # Get next token (greedy or sampling)
            next_token_logits = logits[0, -1, :]  # [vocab_size]

            if temperature == 0.0:
                # Greedy sampling
                next_token = mx.argmax(next_token_logits, keepdims=True)
            else:
                # Temperature sampling
                next_token_logits = next_token_logits / temperature
                probs = mx.softmax(next_token_logits)
                next_token = mx.random.categorical(probs, num_samples=1)

            next_token_id = next_token.item()

            # Check for EOS token
            if self._is_eos_token(next_token_id):
                # Debug: print when hitting EOS
                # print(f"  [DEBUG] Hit EOS at token {i+1}: {next_token_id}")
                break

            # Append token
            generated_tokens.append(next_token_id)

            # Update input_ids for next iteration
            # If using cache, only pass new token; otherwise pass full sequence
            if cache is not None:
                next_token = mx.array([[next_token_id]])
                current_ids = next_token
            else:
                # No cache: need full sequence for autoregressive generation
                next_token = mx.array([[next_token_id]])
                current_ids = mx.concatenate([current_ids, next_token], axis=1)

        # Detokenize
        if not generated_tokens:
            return ""

        response_text = self.tokenizer.decode(generated_tokens)
        return response_text

    def _is_eos_token(self, token_id: int) -> bool:
        """Check if token is EOS token."""
        # Check multiple possible EOS token types
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)

        if eos_token_id is not None:
            if isinstance(eos_token_id, list):
                return token_id in eos_token_id
            else:
                return token_id == eos_token_id

        return False

    def generate_batch(
        self,
        prompts: List[str],
        pixel_values_batch: Optional[List[mx.array]] = None,
        grid_thw_batch: Optional[List[mx.array]] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
    ) -> List[str]:
        """Generate responses for a batch of prompts.

        Note: Currently implemented as sequential processing.
        Batched inference can be added as an optimization later.

        Args:
            prompts: List of text prompts
            pixel_values_batch: List of image tensors (optional)
            grid_thw_batch: List of grid dimensions (optional)
            max_tokens: Override default max_tokens (optional)
            temperature: Sampling temperature (default: 0.0)

        Returns:
            List of generated text responses
        """
        responses = []

        for i, prompt in enumerate(prompts):
            pixel_values = pixel_values_batch[i] if pixel_values_batch else None
            grid_thw = grid_thw_batch[i] if grid_thw_batch else None

            response = self.generate(
                prompt=prompt,
                pixel_values=pixel_values,
                grid_thw=grid_thw,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            responses.append(response)

        return responses
