#!/usr/bin/env python3
"""
Test TurboAngle quantizer implementation.

Verifies:
1. FWHT is self-inverse and norm-preserving
2. Quantize + dequantize round-trip
3. Compression ratio calculations
4. Integration with quantization_strategies registry
"""

import sys
sys.path.insert(0, "mlx-lm-source")

import mlx.core as mx
import numpy as np

from mlx_lm.models.turboangle import fwht, create_random_diagonal, TurboAngleQuantizer
from mlx_lm.models.quantization_strategies import get_quantizer


def test_fwht():
    """Test FWHT properties."""
    print("=" * 80)
    print("Test 1: FWHT Properties")
    print("=" * 80)

    d = 128
    x = mx.random.normal(shape=(2, 4, 10, d))  # [B=2, H=4, S=10, D=128]

    # Test 1: Self-inverse
    y = fwht(x)
    x_rec = fwht(y)

    error = mx.abs(x - x_rec).max()
    print(f"✓ Self-inverse: max error = {error.item():.2e} (should be ~1e-6)")
    assert error < 1e-5, "FWHT not self-inverse"

    # Test 2: Norm preservation
    norm_x = mx.sqrt((x * x).sum(axis=-1))
    norm_y = mx.sqrt((y * y).sum(axis=-1))

    norm_error = mx.abs(norm_x - norm_y).max()
    print(f"✓ Norm preservation: max error = {norm_error.item():.2e} (should be ~1e-6)")
    assert norm_error < 1e-5, "FWHT doesn't preserve norm"

    print()


def test_random_diagonal():
    """Test random diagonal generation."""
    print("=" * 80)
    print("Test 2: Random Diagonal")
    print("=" * 80)

    D = create_random_diagonal(128, seed=42)

    # Test 1: All ±1
    D_np = np.array(D.tolist())
    unique_values = np.unique(D_np)
    print(f"✓ Unique values: {unique_values.tolist()} (should be [-1.0, 1.0])")
    assert len(unique_values) == 2

    # Test 2: Approximately balanced
    n_pos = (D > 0).sum().item()
    n_neg = (D < 0).sum().item()
    balance = abs(n_pos - n_neg) / len(D)
    print(f"✓ Balance: {n_pos} positive, {n_neg} negative (diff={balance:.1%})")

    # Test 3: Reproducibility
    D2 = create_random_diagonal(128, seed=42)
    assert mx.array_equal(D, D2), "Random diagonal not reproducible"
    print(f"✓ Reproducible with same seed")

    print()


def test_quantizer_roundtrip():
    """Test quantize + dequantize round-trip."""
    print("=" * 80)
    print("Test 3: Quantizer Round-Trip")
    print("=" * 80)

    # Create test data
    B, H, S, D = 2, 8, 16, 128
    keys = mx.random.normal(shape=(B, H, S, D)).astype(mx.bfloat16)
    values = mx.random.normal(shape=(B, H, S, D)).astype(mx.bfloat16)

    # Test multiple configurations
    configs = [
        {"n_k": 128, "n_v": 64, "name": "Baseline (3.25 bits)"},
        {"n_k": 256, "n_v": 128, "name": "E4 Boost (3.75 bits)"},
        {"n_k": 64, "n_v": 32, "name": "Aggressive (2.75 bits)"},
    ]

    for config in configs:
        name = config.pop("name")
        print(f"\n{name}")
        print("-" * 40)

        quantizer = TurboAngleQuantizer(head_dim=D, **config)
        print(f"  Config: {quantizer}")

        # Quantize
        quant_k, quant_v, meta = quantizer.quantize(keys, values)

        # Dequantize
        rec_k, rec_v = quantizer.dequantize(quant_k, quant_v, meta)

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
        print(f"  Compression ratio: {quantizer.get_compression_ratio():.2f}×")

        # Should be high similarity (論文報告 >0.95 for similar configs)
        assert sim_k > 0.85, f"K similarity too low: {sim_k}"
        assert sim_v > 0.85, f"V similarity too low: {sim_v}"

    print("\n✓ All round-trip tests passed")
    print()


def test_registry_integration():
    """Test integration with quantization_strategies registry."""
    print("=" * 80)
    print("Test 4: Registry Integration")
    print("=" * 80)

    # Test 1: Can create via registry
    quantizer = get_quantizer('turboangle', n_k=128, n_v=64, head_dim=128)
    print(f"✓ Created via registry: {quantizer}")

    # Test 2: Implements QuantizationStrategy interface
    B, H, S, D = 1, 4, 8, 128
    keys = mx.random.normal(shape=(B, H, S, D)).astype(mx.bfloat16)
    values = mx.random.normal(shape=(B, H, S, D)).astype(mx.bfloat16)

    quant_k, quant_v, meta = quantizer.quantize(keys, values)
    rec_k, rec_v = quantizer.dequantize(quant_k, quant_v, meta)

    print(f"✓ Quantize/dequantize works via registry")
    print(f"  Input shape: {keys.shape}")
    print(f"  Reconstructed shape: {rec_k.shape}")

    # Test 3: Compression ratio
    ratio = quantizer.get_compression_ratio()
    print(f"✓ Compression ratio: {ratio:.2f}×")

    # Test 4: Memory estimation
    mem = quantizer.estimate_memory(num_tokens=1000, head_dim=128, num_heads=32)
    print(f"✓ Memory estimation: {mem / (1024**2):.2f} MB for 1K tokens")

    print()


def test_angle_uniformity():
    """Test that angles are approximately uniform after FWHT + rotation."""
    print("=" * 80)
    print("Test 5: Angle Uniformity (Core TurboAngle Property)")
    print("=" * 80)

    D = 128
    N = 10000  # Large sample

    # Generate random vectors
    x = mx.random.normal(shape=(N, D))

    # Apply FWHT + random diagonal
    D_diag = create_random_diagonal(D, seed=42)
    x_rotated = x * D_diag
    y = fwht(x_rotated)

    # Extract first pair and compute angle
    y_even = y[:, 0].tolist()
    y_odd = y[:, 1].tolist()

    angles = [np.arctan2(odd, even) for even, odd in zip(y_even, y_odd)]
    angles = [(a + 2 * np.pi) % (2 * np.pi) for a in angles]  # Shift to [0, 2π)

    # Check uniformity using histogram
    bins = 8
    counts, edges = np.histogram(angles, bins=bins, range=(0, 2 * np.pi))
    expected = N / bins

    print(f"  Sample size: {N}")
    print(f"  Angle distribution ({bins} bins):")
    for i, count in enumerate(counts):
        bar = "█" * int(count / expected * 40)
        print(f"    Bin {i}: {count:5d} {bar}")

    # Chi-square test for uniformity
    chi_square = sum((c - expected) ** 2 / expected for c in counts)
    print(f"\n  Chi-square statistic: {chi_square:.2f}")
    print(f"  Expected (uniform): ~{bins - 1:.1f} (degrees of freedom)")

    # Should be approximately uniform (chi-square < 2 × dof)
    assert chi_square < 2 * bins, "Angles not approximately uniform"
    print(f"✓ Angles are approximately uniform")

    print()


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "TurboAngle Implementation Tests" + " " * 27 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    test_fwht()
    test_random_diagonal()
    test_quantizer_roundtrip()
    test_registry_integration()
    test_angle_uniformity()

    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 30 + "ALL TESTS PASSED" + " " * 32 + "║")
    print("╚" + "=" * 78 + "╝")
    print()


if __name__ == "__main__":
    main()
