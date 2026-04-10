"""
Debug vision+text generation token by token
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import mlx.core as mx
sys.path.insert(0, str(project_root / "src/flashmlx/models"))
sys.path.insert(0, str(project_root / "src/flashmlx/processors"))
sys.path.insert(0, str(project_root / "examples"))

from test_real_weights import download_and_load_model, prepare_test_image

def debug_vision_generation():
    """Debug vision+text generation step by step."""
    print("Loading model...")
    model, tokenizer, processor, config = download_and_load_model(use_4bit=False)

    print("\nPreparing image...")
    pixel_values, grid_thw = prepare_test_image(processor)

    print("\nTokenizing prompt with <image>...")
    prompt = "<image>What is this?"
    input_ids = mx.array(tokenizer.encode(prompt))
    input_ids = mx.expand_dims(input_ids, axis=0)

    print(f"  Prompt: {prompt}")
    print(f"  Input IDs: {input_ids.tolist()[0]}")
    print(f"  Image token ID: {config.image_token_id}")

    # Check where <image> token is
    image_token_positions = [i for i, tid in enumerate(input_ids.tolist()[0]) if tid == config.image_token_id]
    print(f"  <image> token at positions: {image_token_positions}")

    print("\nRunning first forward pass with vision...")
    logits = model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        grid_thw=grid_thw,
    )

    print(f"  Logits shape: {logits.shape}")

    # Get next token
    next_token_logits = logits[0, -1, :]
    next_token = mx.argmax(next_token_logits)
    next_token_id = int(next_token.item())

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    print(f"  EOS token ID: {eos_token_id}")
    print(f"  Next token ID: {next_token_id}")
    print(f"  Is EOS: {next_token_id == eos_token_id}")

    if next_token_id == eos_token_id:
        print("\\n⚠️  Model generated EOS immediately after processing image!")
        print("  This explains the empty response.")
        return

    decoded = tokenizer.decode([next_token_id])
    print(f"  Next token decoded: '{decoded}'")

    # Generate a few more tokens
    print("\\nGenerating 10 more tokens...")
    generated_ids = [next_token_id]
    current_ids = input_ids

    for i in range(10):
        # Append latest token
        next_token_arr = mx.array([[next_token_id]])
        current_ids = mx.concatenate([current_ids, next_token_arr], axis=1)

        # Forward pass (no image after first)
        logits = model(input_ids=current_ids, pixel_values=None, grid_thw=None)
        next_token_logits = logits[0, -1, :]
        next_token = mx.argmax(next_token_logits)
        next_token_id = int(next_token.item())

        if next_token_id == eos_token_id:
            print(f"  Token {i+2}: {next_token_id} (EOS) - STOPPING")
            break

        generated_ids.append(next_token_id)
        decoded = tokenizer.decode([next_token_id])
        print(f"  Token {i+2}: {next_token_id} → '{decoded}'")

    print(f"\\nGenerated: '{tokenizer.decode(generated_ids)}'")


if __name__ == "__main__":
    debug_vision_generation()
