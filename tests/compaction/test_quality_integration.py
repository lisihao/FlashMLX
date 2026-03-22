"""
Test Quality Path Integration into CompactedKVCache

Verifies that Quality Path works correctly when integrated into the
CompactedKVCache class.
"""

import unittest
import mlx.core as mx

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'mlx-lm-source')))

from mlx_lm.models.compacted_cache import CompactedKVCache


class TestQualityPathIntegration(unittest.TestCase):
    """Test Quality Path integration in CompactedKVCache"""

    def test_quality_path_basic(self):
        """
        Test that CompactedKVCache can use Quality Path.
        """
        # Create cache with Quality Path enabled
        cache = CompactedKVCache(
            max_size=100,
            compression_ratio=2.0,
            use_quality_path=True,
            quality_fit_beta=True,
            quality_fit_c2=True
        )

        # Simulate adding keys/values
        B, n_heads, head_dim = 1, 4, 32
        num_steps = 120  # Exceeds max_size

        keys = mx.random.normal((B, n_heads, num_steps, head_dim))
        values = mx.random.normal((B, n_heads, num_steps, head_dim))

        # Update cache (should trigger compression)
        cached_keys, cached_values = cache.update_and_fetch(keys, values)

        # Verify compression happened
        self.assertGreater(cache.num_compressions, 0)
        self.assertLess(cache.offset, num_steps)  # Cache was compressed

        # Verify cache size
        target_budget = int(num_steps / cache.compression_ratio)
        self.assertLessEqual(cache.offset, target_budget + 10)  # Within tolerance

        print(f"\nQuality Path basic test:")
        print(f"  Original size: {num_steps}")
        print(f"  Compressed size: {cache.offset}")
        print(f"  Compressions: {cache.num_compressions}")

    def test_fast_vs_quality_path(self):
        """
        Compare Fast Path vs Quality Path compression.

        Both should compress, but Quality Path should handle random data better.
        """
        B, n_heads, head_dim = 1, 4, 32
        num_steps = 150
        max_size = 100

        # Generate random data
        keys = mx.random.normal((B, n_heads, num_steps, head_dim))
        values = mx.random.normal((B, n_heads, num_steps, head_dim))

        # Fast Path cache
        cache_fast = CompactedKVCache(
            max_size=max_size,
            compression_ratio=2.0,
            use_quality_path=False
        )
        cached_keys_fast, cached_values_fast = cache_fast.update_and_fetch(keys, values)

        # Quality Path cache
        cache_quality = CompactedKVCache(
            max_size=max_size,
            compression_ratio=2.0,
            use_quality_path=True
        )
        cached_keys_quality, cached_values_quality = cache_quality.update_and_fetch(keys, values)

        # Both should compress
        self.assertGreater(cache_fast.num_compressions, 0)
        self.assertGreater(cache_quality.num_compressions, 0)

        # Both should have similar final sizes
        self.assertLess(cache_fast.offset, num_steps)
        self.assertLess(cache_quality.offset, num_steps)

        print(f"\nFast vs Quality Path:")
        print(f"  Fast Path size: {cache_fast.offset}")
        print(f"  Quality Path size: {cache_quality.offset}")

        # Verify shapes match
        self.assertEqual(cached_keys_fast.shape[2], cache_fast.offset)
        self.assertEqual(cached_keys_quality.shape[2], cache_quality.offset)

    def test_quality_path_ablation(self):
        """
        Test Quality Path with different configurations.

        Test:
        1. No beta, no C2
        2. Beta only
        3. C2 only
        4. Both (full)
        """
        B, n_heads, head_dim = 1, 4, 32
        num_steps = 150

        keys = mx.random.normal((B, n_heads, num_steps, head_dim))
        values = mx.random.normal((B, n_heads, num_steps, head_dim))

        configs = [
            ("No beta, no C2", False, False),
            ("Beta only", True, False),
            ("C2 only", False, True),
            ("Full", True, True),
        ]

        print(f"\nQuality Path ablation:")
        for name, fit_beta, fit_c2 in configs:
            cache = CompactedKVCache(
                max_size=100,
                compression_ratio=2.0,
                use_quality_path=True,
                quality_fit_beta=fit_beta,
                quality_fit_c2=fit_c2
            )

            cached_keys, cached_values = cache.update_and_fetch(keys, values)

            print(f"  {name:15s}: size={cache.offset}, compressions={cache.num_compressions}")

            # All configs should work
            self.assertGreater(cache.num_compressions, 0)
            self.assertLess(cache.offset, num_steps)

    def test_quality_path_multiple_compressions(self):
        """
        Test Quality Path with multiple compression cycles.
        """
        cache = CompactedKVCache(
            max_size=50,
            compression_ratio=2.0,
            use_quality_path=True
        )

        B, n_heads, head_dim = 1, 4, 32

        # Add data multiple times to trigger multiple compressions
        for i in range(5):
            keys = mx.random.normal((B, n_heads, 30, head_dim))
            values = mx.random.normal((B, n_heads, 30, head_dim))
            cache.update_and_fetch(keys, values)

        # Should have triggered multiple compressions
        self.assertGreater(cache.num_compressions, 1)

        # Check statistics
        stats = cache.get_stats()
        print(f"\nMultiple compressions:")
        print(f"  Num compressions: {stats['num_compressions']}")
        print(f"  Avg compression ratio: {stats['avg_compression_ratio']:.2f}")
        print(f"  Current size: {stats['current_size']}")

        self.assertEqual(stats['num_compressions'], cache.num_compressions)
        self.assertGreater(stats['avg_compression_ratio'], 1.0)

    def test_quality_path_state_persistence(self):
        """
        Test that Quality Path configuration is saved/restored correctly.
        """
        # Create cache with Quality Path
        cache1 = CompactedKVCache(
            max_size=100,
            compression_ratio=3.0,
            use_quality_path=True,
            quality_fit_beta=False,
            quality_fit_c2=True
        )

        # Add data
        B, n_heads, head_dim = 1, 4, 32
        keys = mx.random.normal((B, n_heads, 120, head_dim))
        values = mx.random.normal((B, n_heads, 120, head_dim))
        cache1.update_and_fetch(keys, values)

        # Save state
        state = cache1.state
        meta_state = cache1.meta_state

        # Create new cache and restore
        cache2 = CompactedKVCache()
        cache2.meta_state = meta_state
        cache2.state = state

        # Verify configuration restored
        self.assertEqual(cache2.use_quality_path, True)
        self.assertEqual(cache2.quality_fit_beta, False)
        self.assertEqual(cache2.quality_fit_c2, True)
        self.assertEqual(cache2.max_size, 100)
        self.assertEqual(cache2.compression_ratio, 3.0)
        self.assertEqual(cache2.offset, cache1.offset)

        print(f"\nState persistence:")
        print(f"  use_quality_path: {cache2.use_quality_path}")
        print(f"  quality_fit_beta: {cache2.quality_fit_beta}")
        print(f"  quality_fit_c2: {cache2.quality_fit_c2}")
        print(f"  offset: {cache2.offset}")

    def test_quality_path_backward_compatibility(self):
        """
        Test that old caches (without Quality Path params) load correctly.
        """
        # Simulate old meta_state format (8 values)
        old_meta_state = ('50', '100', '2.0', '0.5', '1', '1', '100', '50')

        cache = CompactedKVCache()
        cache.meta_state = old_meta_state

        # Should load with default Quality Path settings
        self.assertEqual(cache.offset, 50)
        self.assertEqual(cache.max_size, 100)
        self.assertEqual(cache.use_quality_path, False)  # Default
        self.assertEqual(cache.quality_fit_beta, True)   # Default
        self.assertEqual(cache.quality_fit_c2, True)     # Default

        print(f"\nBackward compatibility:")
        print(f"  Loaded offset: {cache.offset}")
        print(f"  Default use_quality_path: {cache.use_quality_path}")

    def test_quality_path_large_scale(self):
        """
        Test Quality Path at realistic scale.
        """
        cache = CompactedKVCache(
            max_size=512,
            compression_ratio=4.0,
            use_quality_path=True
        )

        B, n_heads, head_dim = 1, 8, 64

        # Simulate long sequence
        keys = mx.random.normal((B, n_heads, 600, head_dim))
        values = mx.random.normal((B, n_heads, 600, head_dim))

        cached_keys, cached_values = cache.update_and_fetch(keys, values)

        # Should compress
        self.assertGreater(cache.num_compressions, 0)
        self.assertLess(cache.offset, 600)

        # Verify shapes
        self.assertEqual(cached_keys.shape, (B, n_heads, cache.offset, head_dim))
        self.assertEqual(cached_values.shape, (B, n_heads, cache.offset, head_dim))

        stats = cache.get_stats()
        print(f"\nLarge scale test:")
        print(f"  Original: 600 tokens")
        print(f"  Compressed: {cache.offset} tokens")
        print(f"  Compression ratio: {600 / cache.offset:.2f}x")
        print(f"  Memory saved: {(1 - cache.offset/600) * 100:.1f}%")


if __name__ == '__main__':
    unittest.main(verbosity=2)
