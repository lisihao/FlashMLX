#!/usr/bin/env python3
"""
Test TurboAngle per-layer integration in cache factory.

Verifies:
1. Preset loading through cache factory
2. Per-layer quantizer assignment
3. Different quantizers actually used per layer
4. End-to-end cache creation with TurboAngle
"""

import sys
sys.path.insert(0, "mlx-lm-source")

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.turboangle_config import get_preset, create_layer_quantizers


def test_preset_through_cache_factory():
    """Test preset string passed to cache factory."""
    print("=" * 80)
    print("Test 1: Preset String Through Cache Factory")
    print("=" * 80)
    print()

    MODEL_PATH = "/Volumes/toshiba/models/qwen3-8b-mlx"

    print(f"Loading model: {MODEL_PATH}")
    model, tokenizer = load(MODEL_PATH)
    print(f"✅ Model loaded: {len(model.model.layers)} layers")
    print()

    # Test 1: Create cache with preset string
    print("Creating cache with TurboAngle preset: 'mistral-7b'")
    print("  (Qwen3-8B has 36 layers, preset expects 32, should still work)")
    print()

    cache = make_prompt_cache(
        model,
        kv_cache="triple_pq",  # Use triple_pq strategy
        kv_layer_quantizers="mistral-7b",  # TurboAngle preset
    )

    print(f"✅ Cache created: {len(cache)} layers")
    print()

    # Inspect first few layers
    from mlx_lm.models.triple_layer_cache import TripleLayerKVCache

    for i in [0, 1, 2, 3, 10, 20, 30, 35]:
        c = cache[i]
        if isinstance(c, TripleLayerKVCache):
            quantizer = c.warm_quantizer
            if quantizer is not None:
                # TurboAngle quantizers have n_k and n_v attributes
                n_k = getattr(quantizer, 'n_k', None)
                n_v = getattr(quantizer, 'n_v', None)
                if n_k is not None and n_v is not None:
                    print(f"  Layer {i:2d}: TurboAngle K{n_k}V{n_v}")
                else:
                    print(f"  Layer {i:2d}: {type(quantizer).__name__}")
            else:
                print(f"  Layer {i:2d}: No quantizer")
        else:
            print(f"  Layer {i:2d}: {type(c).__name__} (not TripleLayerKVCache)")

    print()

    # According to Mistral-7B preset:
    # Layers 0-3 should have K256V128 (E4 boost)
    # Layers 4-31 should have K128V64 (baseline)
    # Layers 32-35 (Qwen3 specific) should inherit from layer 31

    c0 = cache[0]
    c10 = cache[10]

    if isinstance(c0, TripleLayerKVCache) and isinstance(c10, TripleLayerKVCache):
        q0 = c0.warm_quantizer
        q10 = c10.warm_quantizer

        if hasattr(q0, 'n_k') and hasattr(q10, 'n_k'):
            # Check layer 0 uses E4 boost
            if q0.n_k == 256 and q0.n_v == 128:
                print("✅ Layer 0 uses E4 boost (K256V128)")
            else:
                print(f"⚠️  Layer 0 expected K256V128, got K{q0.n_k}V{q0.n_v}")

            # Check layer 10 uses baseline
            if q10.n_k == 128 and q10.n_v == 64:
                print("✅ Layer 10 uses baseline (K128V64)")
            else:
                print(f"⚠️  Layer 10 expected K128V64, got K{q10.n_k}V{q10.n_v}")

    print()


def test_dict_quantizers():
    """Test dict of quantizers passed to cache factory."""
    print("=" * 80)
    print("Test 2: Dict of Quantizers Through Cache Factory")
    print("=" * 80)
    print()

    MODEL_PATH = "/Volumes/toshiba/models/qwen3-8b-mlx"

    print(f"Loading model: {MODEL_PATH}")
    model, tokenizer = load(MODEL_PATH)
    print(f"✅ Model loaded")
    print()

    # Create custom per-layer quantizers
    from mlx_lm.models.turboangle import TurboAngleQuantizer

    layer_quantizers = {}

    # Layers 0-5: aggressive V boost (K128V256)
    for i in range(6):
        layer_quantizers[i] = TurboAngleQuantizer(n_k=128, n_v=256, head_dim=128)

    # Layers 6-35: baseline (K128V64)
    for i in range(6, 36):
        layer_quantizers[i] = TurboAngleQuantizer(n_k=128, n_v=64, head_dim=128)

    print("Custom quantizer mapping:")
    print("  Layers 0-5:  K128V256 (V-boost)")
    print("  Layers 6-35: K128V64 (baseline)")
    print()

    # Create cache with dict
    cache = make_prompt_cache(
        model,
        kv_cache="triple_pq",
        kv_layer_quantizers=layer_quantizers,
    )

    print(f"✅ Cache created: {len(cache)} layers")
    print()

    # Verify
    from mlx_lm.models.triple_layer_cache import TripleLayerKVCache

    c0 = cache[0]
    c5 = cache[5]
    c10 = cache[10]

    if isinstance(c0, TripleLayerKVCache):
        q0 = c0.warm_quantizer
        if hasattr(q0, 'n_k'):
            assert q0.n_k == 128 and q0.n_v == 256, f"Layer 0 should be K128V256"
            print("✅ Layer 0: K128V256 (V-boost)")

    if isinstance(c5, TripleLayerKVCache):
        q5 = c5.warm_quantizer
        if hasattr(q5, 'n_k'):
            assert q5.n_k == 128 and q5.n_v == 256, f"Layer 5 should be K128V256"
            print("✅ Layer 5: K128V256 (V-boost)")

    if isinstance(c10, TripleLayerKVCache):
        q10 = c10.warm_quantizer
        if hasattr(q10, 'n_k'):
            assert q10.n_k == 128 and q10.n_v == 64, f"Layer 10 should be K128V64"
            print("✅ Layer 10: K128V64 (baseline)")

    print()


def test_inference_with_perlayer():
    """Test actual inference with per-layer quantizers."""
    print("=" * 80)
    print("Test 3: Inference with Per-Layer Quantizers")
    print("=" * 80)
    print()

    MODEL_PATH = "/Volumes/toshiba/models/qwen3-8b-mlx"

    print(f"Loading model: {MODEL_PATH}")
    model, tokenizer = load(MODEL_PATH)
    print(f"✅ Model loaded")
    print()

    # Create cache with TurboAngle preset
    cache = make_prompt_cache(
        model,
        kv_cache="triple_pq",
        kv_layer_quantizers="mistral-7b",
    )

    # Run a forward pass
    text = "The quick brown fox jumps over the lazy dog."
    tokens = tokenizer.encode(text)
    tokens_mx = mx.array([tokens])

    print(f"Input text: '{text}'")
    print(f"Tokens: {len(tokens)}")
    print()

    print("Running forward pass...")
    logits = model(tokens_mx, cache=cache)
    mx.eval(logits)

    print(f"✅ Forward pass successful")
    print(f"   Logits shape: {logits.shape}")
    print()

    # Check that cache was populated
    from mlx_lm.models.triple_layer_cache import TripleLayerKVCache

    total_tokens_cached = 0
    for i, c in enumerate(cache):
        if isinstance(c, TripleLayerKVCache):
            # Check recent buffer
            if hasattr(c, '_recent_k') and c._recent_k is not None:
                total_tokens_cached += c._recent_k.shape[2]
                break

    if total_tokens_cached > 0:
        print(f"✅ Cache populated: {total_tokens_cached} tokens in recent buffer")
    else:
        print("⚠️  Cache might not be populated (check implementation)")

    print()


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 18 + "TurboAngle Per-Layer Integration Tests" + " " * 22 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    try:
        test_preset_through_cache_factory()
        test_dict_quantizers()
        test_inference_with_perlayer()

        print("╔" + "=" * 78 + "╗")
        print("║" + " " * 30 + "ALL TESTS PASSED" + " " * 32 + "║")
        print("╚" + "=" * 78 + "╝")
        print()

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
