#!/usr/bin/env python3
"""
Test TurboAngle per-layer configuration.

Verifies:
1. Preset loading
2. Layer quantizer creation
3. Correct quantizer assignment per layer
4. Integration with existing tests
"""

import sys
sys.path.insert(0, "mlx-lm-source")

import mlx.core as mx
from mlx_lm.models.turboangle_config import (
    get_preset,
    create_layer_quantizers,
    TURBOANGLE_PRESETS,
)
from mlx_lm.models.turboangle import TurboAngleQuantizer


def test_preset_loading():
    """Test preset loading and lookup."""
    print("=" * 80)
    print("Test 1: Preset Loading")
    print("=" * 80)
    print()

    # Test 1: Exact match
    preset = get_preset('mistral-7b')
    assert preset is not None, "Should find mistral-7b"
    assert preset['model_name'] == "Mistral-7B-v0.1"
    print(f"✓ Found preset: {preset['model_name']}")

    # Test 2: Partial match
    preset = get_preset('mistral')
    assert preset is not None, "Should find with partial match"
    print(f"✓ Partial match works: 'mistral' → {preset['model_name']}")

    # Test 3: Case insensitive
    preset = get_preset('TINYLLAMA')
    assert preset is not None, "Should be case insensitive"
    print(f"✓ Case insensitive: 'TINYLLAMA' → {preset['model_name']}")

    # Test 4: Not found
    preset = get_preset('nonexistent-model')
    assert preset is None, "Should return None for unknown model"
    print(f"✓ Returns None for unknown model")

    print()


def test_layer_quantizer_creation():
    """Test creating per-layer quantizers."""
    print("=" * 80)
    print("Test 2: Layer Quantizer Creation")
    print("=" * 80)
    print()

    # Test Mistral-7B preset (E4: 0-3 boosted, 4-31 baseline)
    preset = get_preset('mistral-7b')
    quantizers = create_layer_quantizers(preset)

    print(f"Model: {preset['model_name']}")
    print(f"Total layers: {preset['num_layers']}")
    print(f"Quantizers created: {len(quantizers)}")
    print()

    # Verify all layers have quantizers
    assert len(quantizers) == 32, f"Should have 32 quantizers, got {len(quantizers)}"
    print(f"✓ All 32 layers have quantizers")

    # Verify early layers (0-3) use E4 boost
    for i in range(4):
        q = quantizers[i]
        assert q.n_k == 256, f"Layer {i} should have n_k=256, got {q.n_k}"
        assert q.n_v == 128, f"Layer {i} should have n_v=128, got {q.n_v}"
    print(f"✓ Layers 0-3 use E4 boost (K256V128)")

    # Verify remaining layers (4-31) use baseline
    for i in range(4, 32):
        q = quantizers[i]
        assert q.n_k == 128, f"Layer {i} should have n_k=128, got {q.n_k}"
        assert q.n_v == 64, f"Layer {i} should have n_v=64, got {q.n_v}"
    print(f"✓ Layers 4-31 use baseline (K128V64)")

    # Verify head_dim is correct
    assert quantizers[0].head_dim == 128, "Should have head_dim=128 for Mistral"
    print(f"✓ Head dimension: {quantizers[0].head_dim}")

    print()


def test_phi15_selective():
    """Test phi-1.5 selective pattern (skip layers 8-15)."""
    print("=" * 80)
    print("Test 3: phi-1.5 Selective Pattern")
    print("=" * 80)
    print()

    preset = get_preset('phi-1.5')
    quantizers = create_layer_quantizers(preset)

    print(f"Model: {preset['model_name']}")
    print(f"Pattern: {preset['pattern']}")
    print()

    # Layers 0-7: boosted
    for i in range(8):
        q = quantizers[i]
        assert q.n_k == 256 and q.n_v == 128, f"Layer {i} should be boosted"
    print(f"✓ Layers 0-7: K256V128 (boosted)")

    # Layers 8-15: baseline (not boosted)
    for i in range(8, 16):
        q = quantizers[i]
        assert q.n_k == 128 and q.n_v == 64, f"Layer {i} should be baseline"
    print(f"✓ Layers 8-15: K128V64 (baseline, skipped)")

    # Layers 16-23: boosted again
    for i in range(16, 24):
        q = quantizers[i]
        assert q.n_k == 256 and q.n_v == 128, f"Layer {i} should be boosted"
    print(f"✓ Layers 16-23: K256V128 (boosted)")

    print()


def test_quantizer_functionality():
    """Test that per-layer quantizers actually work."""
    print("=" * 80)
    print("Test 4: Per-Layer Quantizer Functionality")
    print("=" * 80)
    print()

    preset = get_preset('tinyllama')
    quantizers = create_layer_quantizers(preset)

    # Create test data
    B, H, S, D = 1, 4, 8, 64  # TinyLlama has d=64
    keys = mx.random.normal(shape=(B, H, S, D)).astype(mx.bfloat16)
    values = mx.random.normal(shape=(B, H, S, D)).astype(mx.bfloat16)

    print(f"Test data shape: {keys.shape}")
    print()

    # Test layer 0 (V-boosted: K128V256)
    q0 = quantizers[0]
    print(f"Layer 0 quantizer: {q0}")

    quant_k, quant_v, meta = q0.quantize(keys, values)
    rec_k, rec_v = q0.dequantize(quant_k, quant_v, meta)

    # Compute cosine similarity
    def cosine_sim(a, b):
        a_flat = a.reshape(-1, D)
        b_flat = b.reshape(-1, D)
        dot = (a_flat * b_flat).sum(axis=-1)
        norm_a = mx.sqrt((a_flat * a_flat).sum(axis=-1))
        norm_b = mx.sqrt((b_flat * b_flat).sum(axis=-1))
        return (dot / (norm_a * norm_b + 1e-8)).mean()

    sim_k = cosine_sim(keys, rec_k).item()
    sim_v = cosine_sim(values, rec_v).item()

    print(f"  Cosine similarity K: {sim_k:.6f}")
    print(f"  Cosine similarity V: {sim_v:.6f}")
    assert sim_k > 0.99, f"K similarity too low: {sim_k}"
    assert sim_v > 0.99, f"V similarity too low: {sim_v}"
    print(f"✓ Layer 0 quantization quality good")

    # Test layer 10 (baseline: K128V64)
    q10 = quantizers[10]
    print(f"\nLayer 10 quantizer: {q10}")

    quant_k, quant_v, meta = q10.quantize(keys, values)
    rec_k, rec_v = q10.dequantize(quant_k, quant_v, meta)

    sim_k = cosine_sim(keys, rec_k).item()
    sim_v = cosine_sim(values, rec_v).item()

    print(f"  Cosine similarity K: {sim_k:.6f}")
    print(f"  Cosine similarity V: {sim_v:.6f}")
    assert sim_k > 0.99, f"K similarity too low: {sim_k}"
    assert sim_v > 0.95, f"V similarity too low: {sim_v}"  # Baseline V has fewer bins
    print(f"✓ Layer 10 quantization quality good")

    print()


def test_all_presets():
    """Test all 7 presets load correctly."""
    print("=" * 80)
    print("Test 5: All Presets")
    print("=" * 80)
    print()

    for key in TURBOANGLE_PRESETS.keys():
        preset = get_preset(key)
        quantizers = create_layer_quantizers(preset)

        expected_layers = preset['num_layers']
        actual_layers = len(quantizers)

        assert actual_layers == expected_layers, \
            f"{key}: Expected {expected_layers} layers, got {actual_layers}"

        print(f"✓ {preset['model_name']:20s} - {expected_layers:2d} layers, "
              f"{preset['pattern']:25s}, ΔPPL={preset['expected_ppl_delta']:+.4f}")

    print()


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "TurboAngle Per-Layer Tests" + " " * 32 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    test_preset_loading()
    test_layer_quantizer_creation()
    test_phi15_selective()
    test_quantizer_functionality()
    test_all_presets()

    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 30 + "ALL TESTS PASSED" + " " * 32 + "║")
    print("╚" + "=" * 78 + "╝")
    print()


if __name__ == "__main__":
    main()
