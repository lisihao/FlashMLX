"""
Debug VLM generation - check raw token IDs
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import mlx.core as mx

models_path = project_root / "src" / "flashmlx" / "models"
processors_path = project_root / "src" / "flashmlx" / "processors"
generation_path = project_root / "src" / "flashmlx" / "generation"
sys.path.insert(0, str(models_path))
sys.path.insert(0, str(processors_path))
sys.path.insert(0, str(generation_path))

from test_real_weights import download_and_load_model, prepare_test_image


def debug_generation():
    """Debug generation by checking raw token IDs."""
    print("Loading model...")
    model, tokenizer, processor, config = download_and_load_model(use_4bit=False)

    print("\nPreparing image...")
    pixel_values, grid_thw = prepare_test_image(processor)

    print("\nTesting tokenizer...")
    test_text = "Hello world"
    test_tokens = tokenizer.encode(test_text)
    test_decoded = tokenizer.decode(test_tokens)
    print(f"  Input: '{test_text}'")
    print(f"  Tokens: {test_tokens}")
    print(f"  Decoded: '{test_decoded}'")

    print("\nTesting simple generation...")
    prompt = "What is MLX?"
    input_ids = mx.array(tokenizer.encode(prompt))
    input_ids = mx.expand_dims(input_ids, axis=0)

    print(f"  Prompt: {prompt}")
    print(f"  Input IDs: {input_ids.tolist()[0][:10]}...")

    # Forward pass
    logits = model(
        input_ids=input_ids,
        pixel_values=None,
        grid_thw=None,
    )

    print(f"  Logits shape: {logits.shape}")

    # Get next token
    next_token_logits = logits[0, -1, :]
    next_token = mx.argmax(next_token_logits)
    next_token_id = int(next_token.item())

    print(f"  Next token ID: {next_token_id}")
    print(f"  Next token decoded: '{tokenizer.decode([next_token_id])}'")

    # Check if it's EOS
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    print(f"  EOS token ID: {eos_token_id}")
    print(f"  Is EOS: {next_token_id == eos_token_id if eos_token_id else 'N/A'}")

    # Generate a few more tokens manually
    print("\nGenerating 10 tokens manually...")
    generated_ids = []
    current_ids = input_ids

    for i in range(10):
        logits = model(input_ids=current_ids, pixel_values=None, grid_thw=None)
        next_token_logits = logits[0, -1, :]
        next_token = mx.argmax(next_token_logits)
        next_token_id = int(next_token.item())

        generated_ids.append(next_token_id)

        # Check if EOS
        if eos_token_id and next_token_id == eos_token_id:
            print(f"  Token {i+1}: {next_token_id} (EOS) - STOPPING")
            break

        decoded = tokenizer.decode([next_token_id])
        print(f"  Token {i+1}: {next_token_id} → '{decoded}'")

        # Update for next iteration
        current_ids = mx.array([[next_token_id]])

    print(f"\nGenerated token IDs: {generated_ids}")
    print(f"Decoded: '{tokenizer.decode(generated_ids)}'")


if __name__ == "__main__":
    debug_generation()
