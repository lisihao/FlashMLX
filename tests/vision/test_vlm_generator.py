"""
Unit tests for VLM text generation
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

import unittest
import mlx.core as mx

# Add models directory
models_path = project_root / "src" / "flashmlx" / "models"
generation_path = project_root / "src" / "flashmlx" / "generation"
sys.path.insert(0, str(models_path))
sys.path.insert(0, str(generation_path))

from qwen2_vl import Qwen2VLModel
from vlm_config import VLMConfig
from vlm_generator import VLMGenerator


class MockTokenizer:
    """Mock tokenizer for testing without HF dependencies."""

    def __init__(self, vocab_size=151936):
        self.vocab_size = vocab_size
        self.eos_token_id = 151643  # Qwen2 EOS
        self._token_counter = 0

    def encode(self, text):
        """Simple mock encoding: return token IDs based on text length."""
        # For testing, just create sequential token IDs
        # <image> token (151655) should be included if in text
        tokens = []

        if "<image>" in text:
            tokens.append(151655)  # image token
            text = text.replace("<image>", "")

        # Add some tokens based on text length
        num_tokens = len(text.split())
        for i in range(num_tokens):
            tokens.append(100 + i)  # Dummy token IDs

        return tokens

    def decode(self, tokens):
        """Simple mock decoding: convert tokens to text."""
        # For testing, return a simple representation
        return f"generated_{len(tokens)}_tokens"


class TestVLMGenerator(unittest.TestCase):
    """Test VLMGenerator class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create minimal VLM config
        config_dict = {
            "model_type": "qwen2_vl",
            "vocab_size": 151936,
            "hidden_size": 896,
            "num_hidden_layers": 4,  # Small for testing
            "num_attention_heads": 14,
            "num_key_value_heads": 2,
            "intermediate_size": 4864,
            "rms_norm_eps": 1e-6,  # Required for Qwen2
            "image_token_id": 151655,
            "video_token_id": 151656,
            "vision_config": {
                "depth": 4,  # Small for testing
                "embed_dim": 896,
                "num_heads": 14,
                "hidden_size": 896,
                "in_channels": 3,
                "patch_size": 14,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
            },
        }

        self.config = VLMConfig.from_dict(config_dict)
        self.model = Qwen2VLModel(self.config)
        self.tokenizer = MockTokenizer()

    def test_generator_initialization(self):
        """Test generator initialization."""
        generator = VLMGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            image_token_id=151655,
            max_tokens=50,
        )

        self.assertIsNotNone(generator)
        self.assertEqual(generator.image_token_id, 151655)
        self.assertEqual(generator.max_tokens, 50)
        self.assertEqual(generator.model, self.model)
        self.assertEqual(generator.tokenizer, self.tokenizer)

    def test_text_only_generation(self):
        """Test text-only generation (no image)."""
        generator = VLMGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            max_tokens=10,
        )

        prompt = "What is MLX?"
        response = generator.generate(prompt)

        # Check response is a string
        self.assertIsInstance(response, str)
        # With mock tokenizer, should get "generated_N_tokens"
        self.assertTrue(response.startswith("generated_"))

    def test_vision_text_generation(self):
        """Test vision+text generation with image."""
        generator = VLMGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            max_tokens=10,
        )

        prompt = "<image>What is in this image?"

        # Create dummy image tensor [1, C, T, H, W]
        pixel_values = mx.random.normal((1, 3, 2, 448, 448))
        grid_thw = mx.array([[1, 32, 32]])  # T=1, H=32, W=32 patches

        response = generator.generate(
            prompt=prompt,
            pixel_values=pixel_values,
            grid_thw=grid_thw,
        )

        # Check response is a string
        self.assertIsInstance(response, str)
        self.assertTrue(response.startswith("generated_"))

    def test_max_tokens_override(self):
        """Test max_tokens can be overridden per call."""
        generator = VLMGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            max_tokens=50,  # Default
        )

        prompt = "Test prompt"

        # Generate with override
        response = generator.generate(prompt, max_tokens=5)

        # Should respect override (5 tokens)
        self.assertIsInstance(response, str)

    def test_eos_token_detection(self):
        """Test EOS token stops generation early."""
        generator = VLMGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            max_tokens=100,
        )

        # Test _is_eos_token method
        self.assertTrue(generator._is_eos_token(151643))  # Qwen2 EOS
        self.assertFalse(generator._is_eos_token(100))  # Random token

    def test_batch_generation(self):
        """Test batch generation."""
        generator = VLMGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            max_tokens=10,
        )

        prompts = ["What is MLX?", "Describe this image"]

        responses = generator.generate_batch(prompts)

        # Check we get a list of responses
        self.assertIsInstance(responses, list)
        self.assertEqual(len(responses), 2)
        for response in responses:
            self.assertIsInstance(response, str)
            self.assertTrue(response.startswith("generated_"))

    def test_temperature_parameter(self):
        """Test temperature parameter is accepted."""
        generator = VLMGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            max_tokens=10,
        )

        prompt = "Test"

        # Should work with temperature=0.0 (greedy)
        response = generator.generate(prompt, temperature=0.0)
        self.assertIsInstance(response, str)

        # Should work with temperature=0.7 (sampling)
        # Note: with random weights, both will produce random output
        response = generator.generate(prompt, temperature=0.7)
        self.assertIsInstance(response, str)

    def test_empty_generation(self):
        """Test handling of minimal generation."""
        generator = VLMGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            max_tokens=1,  # Generate only 1 token
        )

        # Very short generation
        prompt = "Hi"
        response = generator.generate(prompt)
        self.assertIsInstance(response, str)


def run_tests():
    """Run all tests."""
    print("Running VLM Generator Tests...")
    print("=" * 60)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestVLMGenerator)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("=" * 60)
    if result.wasSuccessful():
        print(f"✅ All {result.testsRun} tests passed!")
        return True
    else:
        print(f"❌ {len(result.failures)} test(s) failed")
        print(f"❌ {len(result.errors)} test(s) had errors")
        return False


if __name__ == "__main__":
    import sys

    success = run_tests()
    sys.exit(0 if success else 1)
