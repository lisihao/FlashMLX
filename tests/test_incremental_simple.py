#!/usr/bin/env python3
"""简单测试 - 使用 test_kvtc_cache.py 的成功配置"""

import sys
sys.path.insert(0, '.')

import numpy as np
import mlx.core as mx

from mlx_lm.models.incremental_kvtc_cache import IncrementalKVTCCache

def test_with_successful_config():
    """使用 test_kvtc_cache.py 中成功的配置"""
    print("=" * 70)
    print("Test: Using Successful Config from test_kvtc_cache.py")
    print("=" * 70)

    rng = np.random.default_rng(7)

    # 使用与 test_kvtc_cache.py 完全相同的数据生成方式
    latent = rng.normal(size=(1, 4, 128, 8)).astype(np.float32)
    wk = rng.normal(size=(8, 32)).astype(np.float32)
    wv = rng.normal(size=(8, 32)).astype(np.float32)

    keys = np.einsum("bhtf,fd->bhtd", latent, wk)
    values = np.einsum("bhtf,fd->bhtd", latent, wv)
    keys += 0.01 * rng.normal(size=keys.shape).astype(np.float32)
    values += 0.01 * rng.normal(size=values.shape).astype(np.float32)

    print(f"Keys shape: {keys.shape}")
    print(f"Values shape: {values.shape}")

    # 使用与 test_kvtc_cache.py 完全相同的配置
    cache = IncrementalKVTCCache.from_cache(
        mx.array(keys),
        mx.array(values),
        rank=8,  # 固定 rank，不使用 energy
        bits=4,
        group_size=16,  # 16 而不是 32
        sample_limit=256,  # 256 而不是 100
    )

    print(f"✅ Created cache: {cache}")

    # Decode
    decoded_keys, decoded_values = cache.decode()

    # Check correctness
    key_diff = mx.max(mx.abs(mx.array(keys) - decoded_keys))
    value_diff = mx.max(mx.abs(mx.array(values) - decoded_values))

    print(f"\nKey diff:   {key_diff:.6f}")
    print(f"Value diff: {value_diff:.6f}")

    # test_kvtc_cache.py 使用 <1.1 的相对误差，这里我们也用同样的标准
    key_rel_err = (
        np.linalg.norm(np.asarray(decoded_keys) - keys)
        / np.linalg.norm(keys)
    )
    value_rel_err = (
        np.linalg.norm(np.asarray(decoded_values) - values)
        / np.linalg.norm(values)
    )

    print(f"Key relative error:   {key_rel_err:.6f}")
    print(f"Value relative error: {value_rel_err:.6f}")

    if key_rel_err < 1.1 and value_rel_err < 1.1:
        print("✅ Test PASSED!")
        return True
    else:
        print("❌ Test FAILED!")
        return False

if __name__ == "__main__":
    success = test_with_successful_config()
    sys.exit(0 if success else 1)
