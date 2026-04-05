#!/usr/bin/env python3
"""
TurboAngle KV Cache Compression Usage Examples

Shows how to use TurboAngle quantization with FlashMLX for:
1. Baseline compression (3.25 angle bits)
2. Per-layer early-boost (higher precision for critical layers)
3. Integration with make_prompt_cache
4. Model Card configuration

Performance expectations (from paper):
- Mistral-7B @ 6.56 total bits: ΔPPL = +0.0014
- 4/7 models achieve lossless compression (ΔPPL ≤ 0)
- 14.8× better than TurboQuant at 4.0 bits
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mlx-lm-source"))

import flashmlx
from mlx_lm import load


def example1_baseline():
    """Example 1: Baseline TurboAngle (uniform across all layers)."""
    print("=" * 80)
    print("Example 1: Baseline TurboAngle")
    print("=" * 80)
    print()

    # Create quantizer
    quantizer = flashmlx.get_quantizer(
        'turboangle',
        n_k=128,       # K cache: 128 angle bins (3.5 bits)
        n_v=64,        # V cache: 64 angle bins (3.0 bits)
        k_norm_bits=8, # K norms: 8-bit linear
        v_norm_bits=4, # V norms: 4-bit log-space
        head_dim=128,  # Must match model
    )

    print(f"Quantizer: {quantizer}")
    print(f"  Angle bits: 3.25 (average of K and V)")
    print(f"  Total bits: 6.75 (with norms)")
    print(f"  Compression: {quantizer.get_compression_ratio():.2f}×")
    print()

    # Use with make_prompt_cache
    # model, tokenizer = load("path/to/model")
    # cache = flashmlx.make_prompt_cache(
    #     model,
    #     strategy="scored_kv_direct",
    #     flat_quant="turboangle",
    #     flat_quant_kwargs={"n_k": 128, "n_v": 64, "head_dim": 128}
    # )

    print("✓ Created baseline TurboAngle quantizer")
    print()


def example2_early_boost():
    """Example 2: Per-layer early-boost (higher precision for first 4 layers)."""
    print("=" * 80)
    print("Example 2: Early-Boost for Critical Layers")
    print("=" * 80)
    print()

    print("论文发现：前 4 层对量化最敏感（TinyLlama, Mistral, OLMo）")
    print()

    # Early layers (0-3): Higher precision
    quantizer_early = flashmlx.get_quantizer(
        'turboangle',
        n_k=256,       # K: 8.0 bits (2× baseline)
        n_v=128,       # V: 7.0 bits (2× baseline)
        k_norm_bits=8,
        v_norm_bits=4,
        head_dim=128,
    )

    # Remaining layers (4+): Baseline
    quantizer_base = flashmlx.get_quantizer(
        'turboangle',
        n_k=128,
        n_v=64,
        k_norm_bits=8,
        v_norm_bits=4,
        head_dim=128,
    )

    print(f"Early layers (0-3): {quantizer_early}")
    print(f"  Angle bits: 3.75")
    print(f"  Total bits: 7.25")
    print()

    print(f"Base layers (4+): {quantizer_base}")
    print(f"  Angle bits: 3.25")
    print(f"  Total bits: 6.75")
    print()

    # Average across 32 layers (Mistral-7B)
    total_layers = 32
    early_layers = 4
    avg_bits = (
        early_layers * 7.25 + (total_layers - early_layers) * 6.75
    ) / total_layers

    print(f"Average total bits: {avg_bits:.2f}")
    print(f"Expected result: ΔPPL ≈ +0.0002 (Mistral-7B, 论文 Table 2)")
    print()

    # TODO: Integration with per-layer cache factory
    # Currently FlashMLX doesn't support per-layer quantizers yet
    # This would require extending make_prompt_cache to accept:
    # layer_quantizers = {
    #     range(0, 4): quantizer_early,
    #     range(4, 32): quantizer_base,
    # }

    print("✓ Configured per-layer quantizers")
    print("⚠️  Per-layer cache factory integration pending")
    print()


def example3_model_card():
    """Example 3: Model Card configuration for TurboAngle."""
    print("=" * 80)
    print("Example 3: Model Card Configuration")
    print("=" * 80)
    print()

    # Example Model Card (JSON format)
    model_card = {
        "model_name": "Qwen3-8B",
        "quantization": {
            "strategy": "turboangle",
            "default": {
                "n_k": 128,
                "n_v": 64,
                "k_norm_bits": 8,
                "v_norm_bits": 4,
            },
            "per_layer": [
                {
                    "layers": [0, 1, 2, 3],  # E4: First 4 layers
                    "n_k": 256,
                    "n_v": 128,
                    "reason": "Concentrated sensitivity (论文 Table 3)",
                },
                {
                    "layers": [4, 35],       # Remaining layers
                    "n_k": 128,
                    "n_v": 64,
                    "reason": "Baseline precision sufficient",
                }
            ],
            "bottleneck_type": "K-dominated",  # From paper Table 3
            "expected_ppl_delta": 0.0,         # Lossless for this config
        }
    }

    print("Example Model Card (qwen3-8b-turboangle.json):")
    print("-" * 80)
    import json
    print(json.dumps(model_card, indent=2))
    print()

    print("✓ Model Card shows recommended configuration per model")
    print()


def example4_sensitivity_patterns():
    """Example 4: Different sensitivity patterns from paper."""
    print("=" * 80)
    print("Example 4: Per-Model Sensitivity Patterns (论文发现)")
    print("=" * 80)
    print()

    patterns = [
        {
            "model": "TinyLlama-1.1B",
            "pattern": "Concentrated (E4)",
            "config": "E4: K128V256 (V-dominated)",
            "result": "ΔPPL = -0.0022 (lossless)",
        },
        {
            "model": "Mistral-7B",
            "pattern": "Concentrated (E4)",
            "config": "E4: K256V128 (K-dominated)",
            "result": "ΔPPL = +0.0002 (lossless)",
        },
        {
            "model": "SmolLM2-1.7B",
            "pattern": "Broad (E20)",
            "config": "E20: K256V128 (20/24 layers)",
            "result": "ΔPPL = -0.0003 (lossless)",
        },
        {
            "model": "phi-1.5",
            "pattern": "Selective (0-7, 16-23)",
            "config": "K256V128 但跳过 8-15 (negative transfer)",
            "result": "ΔPPL = 0.0000 (lossless)",
        },
        {
            "model": "StableLM-2-1.6B",
            "pattern": "Broad (E24)",
            "config": "E24: K256V128 (全部 32 层)",
            "result": "ΔPPL = +0.0012",
        },
        {
            "model": "StarCoder2-3B",
            "pattern": "Non-monotonic (E16)",
            "config": "E16 最优（E12 反而更差）",
            "result": "ΔPPL = -0.0007 (lossless)",
        },
        {
            "model": "OLMo-1B",
            "pattern": "K-only (E4)",
            "config": "E4: K256V64 (V 不需要提升)",
            "result": "ΔPPL = +0.0063",
        },
    ]

    for p in patterns:
        print(f"{p['model']}")
        print(f"  Pattern: {p['pattern']}")
        print(f"  Config: {p['config']}")
        print(f"  Result: {p['result']}")
        print()

    print("✓ 不同模型有不同的敏感性模式")
    print("✓ 可以从论文 Table 3 复用配置作为起点")
    print()


def example5_comparison():
    """Example 5: Comparison with other quantizers."""
    print("=" * 80)
    print("Example 5: TurboAngle vs Other Quantizers")
    print("=" * 80)
    print()

    quantizers = {
        "Q4_0 (FlashMLX default)": {
            "bits": 4.0,
            "compression": "2.0×",
            "quality": "Medium",
            "calibration": "No",
        },
        "PolarQuant 4-bit": {
            "bits": 4.0,
            "compression": "3.8×",
            "quality": "cosine sim > 0.95",
            "calibration": "No",
        },
        "TurboAngle baseline": {
            "bits": 6.75,
            "compression": "2.37×",
            "quality": "cosine sim > 0.999",
            "calibration": "No",
        },
        "TurboQuant sym4-g4": {
            "bits": 4.0,
            "compression": "4.0×",
            "quality": "ΔPPL = +0.0148 (Mistral)",
            "calibration": "No",
        },
        "KVQuant-4b": {
            "bits": 4.32,
            "compression": "3.7×",
            "quality": "ΔPPL = +0.01 (LLaMA-7B)",
            "calibration": "Yes",
        },
    }

    print(f"{'Method':<25} {'Bits':<8} {'Compression':<12} {'Quality':<25} {'Calibration'}")
    print("-" * 100)
    for name, spec in quantizers.items():
        print(f"{name:<25} {spec['bits']:<8} {spec['compression']:<12} {spec['quality']:<25} {spec['calibration']}")

    print()
    print("关键观察:")
    print("  1. TurboAngle 用更多 bits (6.75) 换极高质量 (cosine sim > 0.999)")
    print("  2. 零校准，适合 FlashMLX 多模型支持")
    print("  3. 在 Mistral-7B 上比 TurboQuant 质量好 14.8×")
    print()

    print("✓ TurboAngle 是"高质量"方案，不是"极限压缩"方案")
    print()


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 22 + "TurboAngle Usage Examples" + " " * 31 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    example1_baseline()
    example2_early_boost()
    example3_model_card()
    example4_sensitivity_patterns()
    example5_comparison()

    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 25 + "Examples Complete" + " " * 35 + "║")
    print("╚" + "=" * 78 + "╝")
    print()


if __name__ == "__main__":
    main()
