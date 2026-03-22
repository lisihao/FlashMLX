#!/usr/bin/env python3
"""
快速测试 SSM state 捕获和注入功能
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "mlx-lm-source"))

import mlx.core as mx
from mlx_lm import load
from mlx_lm.compaction.critical_channels_profiler import CriticalChannelsProfiler


def identify_ssm_layers(model):
    """识别 SSM 层"""
    ssm_layers = []
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'linear_attn') or hasattr(layer, 'mamba_block'):
            ssm_layers.append(i)
    return ssm_layers


def main():
    print("=" * 60)
    print("Testing SSM State Capture & Injection")
    print("=" * 60)
    print()

    # Load model
    print("Loading model...")
    model_path = "/Volumes/toshiba/models/qwen3.5-35b-mlx"
    model, tokenizer = load(model_path)
    print(f"✅ Model loaded: {len(model.layers)} layers")

    # Identify SSM layers
    ssm_layers = identify_ssm_layers(model)
    print(f"✅ Identified {len(ssm_layers)} SSM layers")
    print(f"   First 5: {ssm_layers[:5]}")
    print()

    # Create profiler
    profiler = CriticalChannelsProfiler(
        model=model,
        tokenizer=tokenizer
    )

    # Test: Capture state from first SSM layer
    test_layer = ssm_layers[0]
    print(f"Testing layer {test_layer}...")
    print()

    # Step 1: Capture state
    print("Step 1: Capturing SSM state...")
    try:
        state = profiler.capture_ssm_state_at_layer(test_layer)
        print(f"✅ State captured!")
        print(f"   Shape: {state.shape}")
        print(f"   Dtype: {state.dtype}")
        print(f"   Mean: {float(mx.mean(state).item()):.6f}")
        print(f"   Std: {float(mx.std(state).item()):.6f}")
    except Exception as e:
        print(f"❌ Failed to capture state: {e}")
        import traceback
        traceback.print_exc()
        return

    print()

    # Step 2: Generate with original state
    print("Step 2: Generating with original state...")
    try:
        logits_original, tokens_original = profiler.generate_with_perturbed_state(
            state, test_layer, max_tokens=5
        )
        print(f"✅ Generated with original state!")
        print(f"   Logits shape: {logits_original.shape}")
        print(f"   Tokens: {tokens_original}")
        decoded = tokenizer.decode(tokens_original)
        print(f"   Decoded: '{decoded}'")
    except Exception as e:
        print(f"❌ Failed to generate: {e}")
        import traceback
        traceback.print_exc()
        return

    print()

    # Step 3: Perturb state
    print("Step 3: Perturbing channel 0...")
    perturbed_state = profiler.perturb_channel(state, channel_idx=0, strength=0.1)
    print(f"✅ State perturbed!")
    diff = mx.abs(state - perturbed_state).mean()
    print(f"   Average difference: {float(diff.item()):.6f}")

    print()

    # Step 4: Generate with perturbed state
    print("Step 4: Generating with perturbed state...")
    try:
        logits_perturbed, tokens_perturbed = profiler.generate_with_perturbed_state(
            perturbed_state, test_layer, max_tokens=5
        )
        print(f"✅ Generated with perturbed state!")
        print(f"   Tokens: {tokens_perturbed}")
        decoded = tokenizer.decode(tokens_perturbed)
        print(f"   Decoded: '{decoded}'")
    except Exception as e:
        print(f"❌ Failed to generate: {e}")
        import traceback
        traceback.print_exc()
        return

    print()

    # Step 5: Compare outputs
    print("Step 5: Comparing outputs...")
    tokens_changed = sum(1 for a, b in zip(tokens_original, tokens_perturbed) if a != b)
    print(f"   Tokens changed: {tokens_changed}/{len(tokens_original)}")

    logits_diff = mx.abs(logits_original - logits_perturbed).mean()
    print(f"   Logits diff (mean): {float(logits_diff.item()):.6f}")

    print()
    print("=" * 60)
    print("✅ State capture & injection working!")
    print("=" * 60)
    print()
    print("Ready to run full profiling.")


if __name__ == "__main__":
    main()
