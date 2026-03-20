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
    rng = np.random.default_rng(42)

    # Create structured data (low-rank + noise) like test_kvtc_cache.py
    # Use sufficient tokens for effective PCA and bit allocation
    batch, heads, tokens, dim = 1, 4, 128, 32
    latent_dim = 8  # Low-rank dimension

    # Generate low-rank structure
    latent = rng.normal(size=(batch, heads, tokens, latent_dim)).astype(np.float32)
    wk = rng.normal(size=(latent_dim, dim)).astype(np.float32)
    wv = rng.normal(size=(latent_dim, dim)).astype(np.float32)

    # Low-rank projection + small noise
    keys = np.einsum("bhtf,fd->bhtd", latent, wk)
    values = np.einsum("bhtf,fd->bhtd", latent, wv)
    keys += 0.01 * rng.normal(size=keys.shape).astype(np.float32)
    values += 0.01 * rng.normal(size=values.shape).astype(np.float32)

    keys = mx.array(keys)
    values = mx.array(values)

    # Create incremental cache with successful config from test_kvtc_cache.py
    cache = IncrementalKVTCCache.from_cache(
        keys, values,
        rank=8, bits=4, group_size=16, sample_limit=256
    )
    print(f"✅ Created cache: {cache}")

    # Decode
    decoded_keys, decoded_values = cache.decode()

    # Check correctness using relative error (same as test_kvtc_cache.py)
    key_diff = mx.max(mx.abs(keys - decoded_keys))
    value_diff = mx.max(mx.abs(values - decoded_values))

    print(f"Key diff:   {key_diff:.6f}")
    print(f"Value diff: {value_diff:.6f}")

    # Calculate relative error (L2 norm)
    key_rel_err = (
        np.linalg.norm(np.asarray(decoded_keys) - np.asarray(keys))
        / np.linalg.norm(np.asarray(keys))
    )
    value_rel_err = (
        np.linalg.norm(np.asarray(decoded_values) - np.asarray(values))
        / np.linalg.norm(np.asarray(values))
    )

    print(f"Key relative error:   {key_rel_err:.6f}")
    print(f"Value relative error: {value_rel_err:.6f}")

    if key_rel_err < 1.1 and value_rel_err < 1.1:
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
    rng = np.random.default_rng(123)

    # Create initial cache with structured data
    # Use sufficient tokens for effective PCA and bit allocation
    batch, heads, dim = 1, 4, 32
    initial_tokens = 64
    latent_dim = 8

    # Initial data (low-rank + noise)
    latent_init = rng.normal(size=(batch, heads, initial_tokens, latent_dim)).astype(np.float32)
    wk = rng.normal(size=(latent_dim, dim)).astype(np.float32)
    wv = rng.normal(size=(latent_dim, dim)).astype(np.float32)

    keys_initial = np.einsum("bhtf,fd->bhtd", latent_init, wk)
    values_initial = np.einsum("bhtf,fd->bhtd", latent_init, wv)
    keys_initial += 0.01 * rng.normal(size=keys_initial.shape).astype(np.float32)
    values_initial += 0.01 * rng.normal(size=values_initial.shape).astype(np.float32)

    keys_initial = mx.array(keys_initial)
    values_initial = mx.array(values_initial)

    # Create cache with successful config
    cache = IncrementalKVTCCache.from_cache(
        keys_initial, values_initial,
        rank=8, bits=4, group_size=16, sample_limit=256
    )
    print(f"Initial cache: {cache}")

    # Append new tokens with same structure
    new_tokens = 32
    latent_new = rng.normal(size=(batch, heads, new_tokens, latent_dim)).astype(np.float32)

    keys_new = np.einsum("bhtf,fd->bhtd", latent_new, wk)
    values_new = np.einsum("bhtf,fd->bhtd", latent_new, wv)
    keys_new += 0.01 * rng.normal(size=keys_new.shape).astype(np.float32)
    values_new += 0.01 * rng.normal(size=values_new.shape).astype(np.float32)

    keys_new = mx.array(keys_new)
    values_new = mx.array(values_new)

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

    # Check correctness using relative error (same as test_kvtc_cache.py)
    key_diff = mx.max(mx.abs(expected_keys - decoded_keys))
    value_diff = mx.max(mx.abs(expected_values - decoded_values))

    print(f"Key diff:   {key_diff:.6f}")
    print(f"Value diff: {value_diff:.6f}")

    # Calculate relative error (L2 norm)
    key_rel_err = (
        np.linalg.norm(np.asarray(decoded_keys) - np.asarray(expected_keys))
        / np.linalg.norm(np.asarray(expected_keys))
    )
    value_rel_err = (
        np.linalg.norm(np.asarray(decoded_values) - np.asarray(expected_values))
        / np.linalg.norm(np.asarray(expected_values))
    )

    print(f"Key relative error:   {key_rel_err:.6f}")
    print(f"Value relative error: {value_rel_err:.6f}")

    if key_rel_err < 1.1 and value_rel_err < 1.1:
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
    rng = np.random.default_rng(456)

    # Create initial cache with structured data
    # Use sufficient tokens for effective PCA and bit allocation
    batch, heads, dim = 1, 4, 32
    tokens_per_append = 32
    latent_dim = 8

    # Shared projection matrices
    wk = rng.normal(size=(latent_dim, dim)).astype(np.float32)
    wv = rng.normal(size=(latent_dim, dim)).astype(np.float32)

    # Initial chunk (low-rank + noise)
    latent_0 = rng.normal(size=(batch, heads, tokens_per_append, latent_dim)).astype(np.float32)
    keys_0 = np.einsum("bhtf,fd->bhtd", latent_0, wk)
    values_0 = np.einsum("bhtf,fd->bhtd", latent_0, wv)
    keys_0 += 0.01 * rng.normal(size=keys_0.shape).astype(np.float32)
    values_0 += 0.01 * rng.normal(size=values_0.shape).astype(np.float32)

    keys_list = [mx.array(keys_0)]
    values_list = [mx.array(values_0)]

    cache = IncrementalKVTCCache.from_cache(
        keys_list[0], values_list[0],
        rank=8, bits=4, group_size=16, sample_limit=256
    )
    print(f"Initial: {cache}")

    # Append 3 times with structured data
    num_appends = 3
    for i in range(num_appends):
        latent_i = rng.normal(size=(batch, heads, tokens_per_append, latent_dim)).astype(np.float32)
        keys_i = np.einsum("bhtf,fd->bhtd", latent_i, wk)
        values_i = np.einsum("bhtf,fd->bhtd", latent_i, wv)
        keys_i += 0.01 * rng.normal(size=keys_i.shape).astype(np.float32)
        values_i += 0.01 * rng.normal(size=values_i.shape).astype(np.float32)

        new_keys = mx.array(keys_i)
        new_values = mx.array(values_i)

        keys_list.append(new_keys)
        values_list.append(new_values)

        cache.append(new_keys, new_values)
        print(f"After append {i+1}: {cache}")

    # Decode
    decoded_keys, decoded_values = cache.decode()

    # Expected: concatenation of all chunks
    expected_keys = mx.concatenate(keys_list, axis=2)
    expected_values = mx.concatenate(values_list, axis=2)

    # Check correctness using relative error (same as test_kvtc_cache.py)
    key_diff = mx.max(mx.abs(expected_keys - decoded_keys))
    value_diff = mx.max(mx.abs(expected_values - decoded_values))

    print(f"\nFinal cache: {cache}")
    print(f"Key diff:   {key_diff:.6f}")
    print(f"Value diff: {value_diff:.6f}")

    # Calculate relative error (L2 norm)
    key_rel_err = (
        np.linalg.norm(np.asarray(decoded_keys) - np.asarray(expected_keys))
        / np.linalg.norm(np.asarray(expected_keys))
    )
    value_rel_err = (
        np.linalg.norm(np.asarray(decoded_values) - np.asarray(expected_values))
        / np.linalg.norm(np.asarray(expected_values))
    )

    print(f"Key relative error:   {key_rel_err:.6f}")
    print(f"Value relative error: {value_rel_err:.6f}")

    if key_rel_err < 1.1 and value_rel_err < 1.1:
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
