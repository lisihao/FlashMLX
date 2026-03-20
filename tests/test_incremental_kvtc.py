#!/usr/bin/env python3
"""Test incremental KVTC cache."""

import sys
sys.path.insert(0, '.')

import numpy as np
import mlx.core as mx

from mlx_lm.models.incremental_kvtc_cache import IncrementalKVTCCache


def test_basic_encode_decode():
    """Test basic encode-decode correctness."""
    print("=" * 70)
    print("Test 1: Basic Encode-Decode")
    print("=" * 70)

    np.random.seed(42)

    # Create initial cache
    batch, heads, tokens, dim = 1, 8, 10, 64
    keys = mx.random.normal((batch, heads, tokens, dim))
    values = mx.random.normal((batch, heads, tokens, dim))

    # Create incremental cache (calibration will be fitted on the actual data)
    cache = IncrementalKVTCCache.from_cache(
        keys, values,
        energy=0.99, bits=4, group_size=32, sample_limit=100
    )
    print(f"✅ Created cache: {cache}")

    # Decode
    decoded_keys, decoded_values = cache.decode()

    # Check correctness
    key_diff = mx.max(mx.abs(keys - decoded_keys))
    value_diff = mx.max(mx.abs(values - decoded_values))

    print(f"Key diff:   {key_diff:.6f}")
    print(f"Value diff: {value_diff:.6f}")

    if key_diff < 0.5 and value_diff < 0.5:
        print("✅ Basic encode-decode test PASSED!")
        return True
    else:
        print("❌ Basic encode-decode test FAILED!")
        return False


def test_incremental_append():
    """Test incremental append functionality."""
    print("\n" + "=" * 70)
    print("Test 2: Incremental Append")
    print("=" * 70)

    np.random.seed(123)

    # Create initial cache (5 tokens)
    batch, heads, dim = 1, 8, 64
    initial_tokens = 5

    keys_initial = mx.random.normal((batch, heads, initial_tokens, dim))
    values_initial = mx.random.normal((batch, heads, initial_tokens, dim))

    # Create cache (calibration will be fitted on the actual data)
    cache = IncrementalKVTCCache.from_cache(
        keys_initial, values_initial,
        energy=0.99, bits=4, group_size=32, sample_limit=100
    )
    print(f"Initial cache: {cache}")

    # Append new tokens (3 tokens)
    new_tokens = 3
    keys_new = mx.random.normal((batch, heads, new_tokens, dim))
    values_new = mx.random.normal((batch, heads, new_tokens, dim))

    print(f"Appending {new_tokens} new tokens...")
    cache.append(keys_new, values_new)
    print(f"After append: {cache}")

    # Decode full cache
    decoded_keys, decoded_values = cache.decode()

    # Expected: concatenation of initial + new
    expected_keys = mx.concatenate([keys_initial, keys_new], axis=2)
    expected_values = mx.concatenate([values_initial, values_new], axis=2)

    # Check shape
    if decoded_keys.shape != expected_keys.shape:
        print(f"❌ Shape mismatch: {decoded_keys.shape} vs {expected_keys.shape}")
        return False

    # Check correctness
    key_diff = mx.max(mx.abs(expected_keys - decoded_keys))
    value_diff = mx.max(mx.abs(expected_values - decoded_values))

    print(f"Key diff:   {key_diff:.6f}")
    print(f"Value diff: {value_diff:.6f}")

    if key_diff < 0.5 and value_diff < 0.5:
        print("✅ Incremental append test PASSED!")
        return True
    else:
        print("❌ Incremental append test FAILED!")
        return False


def test_multiple_appends():
    """Test multiple incremental appends."""
    print("\n" + "=" * 70)
    print("Test 3: Multiple Incremental Appends")
    print("=" * 70)

    np.random.seed(456)

    # Create initial cache (2 tokens)
    batch, heads, dim = 1, 4, 32
    tokens_per_append = 2

    # Initial cache (calibration will be fitted on the actual data)
    keys_list = [mx.random.normal((batch, heads, tokens_per_append, dim))]
    values_list = [mx.random.normal((batch, heads, tokens_per_append, dim))]

    cache = IncrementalKVTCCache.from_cache(
        keys_list[0], values_list[0],
        energy=0.99, bits=4, group_size=16, sample_limit=50
    )
    print(f"Initial: {cache}")

    # Append 3 times
    num_appends = 3
    for i in range(num_appends):
        new_keys = mx.random.normal((batch, heads, tokens_per_append, dim))
        new_values = mx.random.normal((batch, heads, tokens_per_append, dim))

        keys_list.append(new_keys)
        values_list.append(new_values)

        cache.append(new_keys, new_values)
        print(f"After append {i+1}: {cache}")

    # Decode
    decoded_keys, decoded_values = cache.decode()

    # Expected: concatenation of all chunks
    expected_keys = mx.concatenate(keys_list, axis=2)
    expected_values = mx.concatenate(values_list, axis=2)

    # Check
    key_diff = mx.max(mx.abs(expected_keys - decoded_keys))
    value_diff = mx.max(mx.abs(expected_values - decoded_values))

    print(f"\nFinal cache: {cache}")
    print(f"Key diff:   {key_diff:.6f}")
    print(f"Value diff: {value_diff:.6f}")

    if key_diff < 0.5 and value_diff < 0.5:
        print("✅ Multiple appends test PASSED!")
        return True
    else:
        print("❌ Multiple appends test FAILED!")
        return False


def main():
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "Incremental KVTC Cache Test Suite" + " " * 20 + "║")
    print("╚" + "=" * 68 + "╝")
    print()

    all_passed = True

    try:
        all_passed &= test_basic_encode_decode()
        all_passed &= test_incremental_append()
        all_passed &= test_multiple_appends()

        print("\n" + "=" * 70)
        if all_passed:
            print("🎉 All tests PASSED!")
        else:
            print("❌ Some tests FAILED!")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
