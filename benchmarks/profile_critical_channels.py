#!/usr/bin/env python3
"""
Run critical channels profiling for SSM layers

This script profiles Qwen3.5-35B SSM layers to identify which channels
are critical for maintaining output quality.

Usage:
    python3 benchmarks/profile_critical_channels.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "mlx-lm-source"))

from mlx_lm import load
from mlx_lm.compaction.critical_channels_profiler import CriticalChannelsProfiler


def identify_ssm_layers(model):
    """Identify which layers are SSM layers"""
    ssm_layers = []

    for i, layer in enumerate(model.layers):
        # Check if layer has SSM components
        if hasattr(layer, 'linear_attn') or hasattr(layer, 'mamba_block'):
            ssm_layers.append(i)

    return ssm_layers


def main():
    print("=" * 60)
    print("Critical Channels Profiling")
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
    print(f"   Layer indices: {ssm_layers}")
    print()

    # Create profiler
    profiler = CriticalChannelsProfiler(
        model=model,
        tokenizer=tokenizer,
        perturbation_strength=0.1
    )

    # Profile all SSM layers
    output_dir = ".solar/calibration"
    print(f"Output directory: {output_dir}")
    print()

    print(f"🚀 Full profiling mode: All {len(ssm_layers)} SSM layers")
    print(f"   Estimated time: ~{len(ssm_layers) * 3.5:.0f} minutes ({len(ssm_layers) * 3.5 / 60:.1f} hours)")
    print()

    profiler.profile_all_layers(
        ssm_layer_indices=ssm_layers,
        output_dir=output_dir
    )

    print()
    print("=" * 60)
    print("✅ Profiling complete!")
    print("=" * 60)
    print()
    print(f"Calibration files: {output_dir}")
    print()
    print("Next steps:")
    print("1. Review calibration files")
    print("2. Implement three-stage cache")
    print("3. Test performance and quality")


if __name__ == "__main__":
    main()
