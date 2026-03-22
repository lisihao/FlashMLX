"""
集成测试：CompactedKVCache

验证 CompactedKVCache 与 mlx-lm 的集成。
"""

import unittest
import mlx.core as mx

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'mlx-lm-source')))

from mlx_lm.models.compacted_cache import CompactedKVCache


class TestCompactedCacheIntegration(unittest.TestCase):
    """测试 CompactedKVCache 集成"""

    def test_basic_usage(self):
        """测试基本使用"""
        cache = CompactedKVCache(max_size=1000, compression_ratio=5.0)

        # Simulate adding keys/values
        B, n_heads, num_steps, head_dim = 1, 32, 100, 128

        for i in range(15):  # 15 * 100 = 1500 tokens total
            keys = mx.random.normal((B, n_heads, num_steps, head_dim))
            values = mx.random.normal((B, n_heads, num_steps, head_dim))

            cached_keys, cached_values = cache.update_and_fetch(keys, values)

            # Verify shapes
            expected_len = min((i + 1) * num_steps, int(1000 / 5.0))  # After compression
            if (i + 1) * num_steps <= 1000:
                # Before first compression
                self.assertEqual(cached_keys.shape[2], (i + 1) * num_steps)
            else:
                # After compression
                self.assertLess(cached_keys.shape[2], 1000)
                self.assertGreater(cached_keys.shape[2], 0)

        # Check statistics
        stats = cache.get_stats()
        print(f"\nCache statistics:")
        print(f"  Compressions: {stats['num_compressions']}")
        print(f"  Current size: {stats['current_size']}")
        print(f"  Avg compression ratio: {stats['avg_compression_ratio']:.2f}x")

        # Should have compressed at least once
        self.assertGreater(stats['num_compressions'], 0)

    def test_disable_compression(self):
        """测试禁用压缩"""
        cache = CompactedKVCache(max_size=1000, enable_compression=False)

        B, n_heads, num_steps, head_dim = 1, 32, 100, 128

        for i in range(15):
            keys = mx.random.normal((B, n_heads, num_steps, head_dim))
            values = mx.random.normal((B, n_heads, num_steps, head_dim))

            cached_keys, cached_values = cache.update_and_fetch(keys, values)

        # No compression should have happened
        stats = cache.get_stats()
        self.assertEqual(stats['num_compressions'], 0)
        self.assertEqual(stats['current_size'], 1500)  # 15 * 100

    def test_compression_ratio(self):
        """测试不同压缩比"""
        for ratio in [2.0, 5.0, 10.0]:
            cache = CompactedKVCache(max_size=1000, compression_ratio=ratio)

            B, n_heads, num_steps, head_dim = 1, 32, 100, 128

            # Add enough to trigger compression
            for i in range(15):
                keys = mx.random.normal((B, n_heads, num_steps, head_dim))
                values = mx.random.normal((B, n_heads, num_steps, head_dim))
                cache.update_and_fetch(keys, values)

            stats = cache.get_stats()
            print(f"\nCompression ratio {ratio}x:")
            print(f"  Compressions: {stats['num_compressions']}")
            print(f"  Current size: {stats['current_size']}")

            # After compression, size should be less than max_size
            self.assertLess(stats['current_size'], 1000)

    def test_batch_size(self):
        """测试不同 batch size"""
        cache = CompactedKVCache(max_size=1000, compression_ratio=5.0)

        for B in [1, 2, 4]:
            cache = CompactedKVCache(max_size=1000, compression_ratio=5.0)
            n_heads, num_steps, head_dim = 32, 100, 128

            keys = mx.random.normal((B, n_heads, num_steps, head_dim))
            values = mx.random.normal((B, n_heads, num_steps, head_dim))

            cached_keys, cached_values = cache.update_and_fetch(keys, values)

            # Verify batch dimension preserved
            self.assertEqual(cached_keys.shape[0], B)
            self.assertEqual(cached_values.shape[0], B)

    def test_state_save_load(self):
        """测试状态保存和恢复"""
        cache = CompactedKVCache(max_size=1000, compression_ratio=5.0)

        B, n_heads, num_steps, head_dim = 1, 32, 100, 128

        # Add some data
        for i in range(5):
            keys = mx.random.normal((B, n_heads, num_steps, head_dim))
            values = mx.random.normal((B, n_heads, num_steps, head_dim))
            cache.update_and_fetch(keys, values)

        # Save state
        state = cache.state
        meta_state = cache.meta_state

        # Create new cache and restore
        new_cache = CompactedKVCache.from_state(state, meta_state)

        # Verify restored
        self.assertEqual(new_cache.offset, cache.offset)
        self.assertEqual(new_cache.max_size, cache.max_size)
        self.assertEqual(new_cache.compression_ratio, cache.compression_ratio)


if __name__ == '__main__':
    unittest.main(verbosity=2)
