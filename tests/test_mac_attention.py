"""
Tests for MAC-Attention Metal/MLX port.

Test cases:
  1. Ring buffer basic operations (write, read, wrap)
  2. Match correctness: Metal kernel vs pure MLX reference
  3. Merge identity: merge(cached, fresh) = full attention
  4. Downdate correctness: full - window = rest
  5. E2E miss→hit: step 0 misses, step 1 with same query hits
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ring_cache():
    from flashmlx.mac import MACRingCache
    return MACRingCache(max_requests=4, capacity=8, num_heads=2, head_dim=16)


@pytest.fixture
def small_cache():
    """Tiny cache for focused tests."""
    from flashmlx.mac import MACRingCache
    return MACRingCache(max_requests=2, capacity=4, num_heads=1, head_dim=8)


# ---------------------------------------------------------------------------
# Test 1: Ring buffer operations
# ---------------------------------------------------------------------------

class TestRingCache:
    def test_init_shapes(self, ring_cache):
        assert ring_cache.query_cache.shape == (4, 8, 2, 16)
        assert ring_cache.attn_cache.shape == (4, 8, 2, 16)
        assert ring_cache.lse_cache.shape == (4, 8, 2)
        assert ring_cache.request_length.shape == (4,)

    def test_update_single(self, ring_cache):
        req_ids = mx.array([0], dtype=mx.int32)
        q = mx.ones((1, 2, 16), dtype=mx.bfloat16)
        a = mx.ones((1, 2, 16), dtype=mx.bfloat16) * 2
        lse = mx.zeros((1, 2), dtype=mx.float32)

        ring_cache.update(req_ids, q, a, lse)
        mx.eval(ring_cache.query_cache, ring_cache.request_length)

        # Should be written at slot 0 (request_length was 0)
        assert ring_cache.request_length[0].item() == 1
        assert mx.allclose(
            ring_cache.query_cache[0, 0].astype(mx.float32),
            mx.ones((2, 16), dtype=mx.float32),
        )

    def test_ring_wrap(self, small_cache):
        """Write more entries than capacity → oldest overwritten."""
        from flashmlx.mac import MACRingCache
        cache = small_cache  # capacity=4
        req_ids = mx.array([0], dtype=mx.int32)

        # Write 6 entries (capacity=4, so first 2 are overwritten)
        for i in range(6):
            q = mx.full((1, 1, 8), float(i), dtype=mx.bfloat16)
            a = mx.zeros((1, 1, 8), dtype=mx.bfloat16)
            lse = mx.zeros((1, 1), dtype=mx.float32)
            cache.update(req_ids, q, a, lse)
            mx.eval(cache.query_cache, cache.request_length)

        assert cache.request_length[0].item() == 6
        # Slot 0 = entry 4 (written at step 4, request_length=4, slot=4%4=0)
        # Slot 1 = entry 5 (written at step 5, request_length=5, slot=5%4=1)
        # Slot 2 = entry 2 (written at step 2, not overwritten)
        # Slot 3 = entry 3 (written at step 3, not overwritten)
        val_slot0 = cache.query_cache[0, 0, 0, 0].item()
        val_slot1 = cache.query_cache[0, 1, 0, 0].item()
        assert abs(val_slot0 - 4.0) < 0.1, f"Expected 4.0, got {val_slot0}"
        assert abs(val_slot1 - 5.0) < 0.1, f"Expected 5.0, got {val_slot1}"

    def test_fetch(self, ring_cache):
        req_ids = mx.array([0], dtype=mx.int32)
        q = mx.ones((1, 2, 16), dtype=mx.bfloat16) * 3
        a = mx.ones((1, 2, 16), dtype=mx.bfloat16) * 7
        lse = mx.full((1, 2), 1.5, dtype=mx.float32)

        ring_cache.update(req_ids, q, a, lse)
        mx.eval(ring_cache.attn_cache, ring_cache.lse_cache)

        # Fetch at slot 0 for both heads
        indices = mx.zeros((1, 2), dtype=mx.int32)
        attn_out, lse_out = ring_cache.fetch(req_ids, indices)
        mx.eval(attn_out, lse_out)

        assert attn_out.shape == (1, 2, 16)
        assert mx.allclose(
            attn_out[0, 0].astype(mx.float32),
            mx.full((16,), 7.0, dtype=mx.float32),
            atol=0.1,
        )
        assert abs(lse_out[0, 0].item() - 1.5) < 0.01

    def test_reset(self, ring_cache):
        req_ids = mx.array([0, 1], dtype=mx.int32)
        q = mx.ones((2, 2, 16), dtype=mx.bfloat16)
        a = mx.ones((2, 2, 16), dtype=mx.bfloat16)
        lse = mx.zeros((2, 2), dtype=mx.float32)
        ring_cache.update(req_ids, q, a, lse)
        mx.eval(ring_cache.request_length)

        ring_cache.reset(mx.array([0], dtype=mx.int32))
        mx.eval(ring_cache.request_length)
        assert ring_cache.request_length[0].item() == 0
        assert ring_cache.request_length[1].item() == 1


# ---------------------------------------------------------------------------
# Test 2: Merge identity
# ---------------------------------------------------------------------------

class TestMerge:
    def test_merge_equal_weights(self):
        """When lse_cached == lse_fresh, output is average."""
        from flashmlx.mac import merge_attention_states

        N, H, D = 2, 4, 8
        o1 = mx.ones((N, H, D))
        o2 = mx.ones((N, H, D)) * 3
        lse = mx.zeros((N, H))  # equal LSE

        merged_o, merged_lse = merge_attention_states(o1, lse, o2, lse)
        mx.eval(merged_o, merged_lse)

        # Equal LSE → equal weights → average
        expected = mx.full((N, H, D), 2.0)
        assert mx.allclose(merged_o, expected, atol=1e-5)

    def test_merge_dominant(self):
        """When one LSE >> other, output ≈ dominant."""
        from flashmlx.mac import merge_attention_states

        N, H, D = 1, 1, 4
        o_big = mx.ones((N, H, D)) * 10
        o_small = mx.ones((N, H, D)) * 0

        lse_big = mx.array([[100.0]])  # dominant
        lse_small = mx.array([[-100.0]])  # negligible

        merged_o, _ = merge_attention_states(o_big, lse_big, o_small, lse_small)
        mx.eval(merged_o)

        assert mx.allclose(merged_o, o_big, atol=1e-3)


# ---------------------------------------------------------------------------
# Test 3: Downdate correctness
# ---------------------------------------------------------------------------

class TestDowndate:
    def test_downdate_recovers_rest(self):
        """downdate(full, window) should recover the rest contribution."""
        from flashmlx.mac import downdate_attention, merge_attention_states

        N, H, D = 1, 2, 8
        # Construct: full = merge(rest, window)
        rest_o = mx.random.normal((N, H, D))
        rest_lse = mx.random.normal((N, H))
        window_o = mx.random.normal((N, H, D))
        window_lse = mx.random.normal((N, H))

        # Forward: merge rest + window → full
        full_o, full_lse = merge_attention_states(rest_o, rest_lse, window_o, window_lse)

        # Inverse: downdate full - window → recovered_rest
        recovered_o, recovered_lse = downdate_attention(full_o, full_lse, window_o, window_lse)
        mx.eval(recovered_o, recovered_lse)

        # Should recover rest_o and rest_lse
        assert mx.allclose(recovered_o, rest_o, atol=1e-3), (
            f"rest_o mismatch: max diff = {mx.max(mx.abs(recovered_o - rest_o)).item()}"
        )
        assert mx.allclose(recovered_lse, rest_lse, atol=1e-3), (
            f"rest_lse mismatch: max diff = {mx.max(mx.abs(recovered_lse - rest_lse)).item()}"
        )


# ---------------------------------------------------------------------------
# Test 4: Match — Metal kernel vs reference
# ---------------------------------------------------------------------------

class TestMatch:
    @pytest.mark.skipif(
        not mx.metal.is_available(),
        reason="Metal not available",
    )
    def test_match_kernel_vs_reference(self):
        """Metal match kernel should produce same results as pure MLX reference."""
        from flashmlx.mac import MACRingCache, mac_ring_match, mac_ring_match_reference

        R, M, H, D = 2, 16, 2, 16
        cache = MACRingCache(R, M, H, D)

        # Populate cache with random data for request 0
        mx.random.seed(42)
        for i in range(10):
            q = mx.random.normal((1, H, D)).astype(mx.bfloat16)
            a = mx.random.normal((1, H, D)).astype(mx.bfloat16)
            lse = mx.random.normal((1, H)).astype(mx.float32)
            cache.update(mx.array([0], dtype=mx.int32), q, a, lse)
            mx.eval(cache.query_cache, cache.request_length)

        # Query
        queries = mx.random.normal((1, H, D)).astype(mx.bfloat16)
        req_ids = mx.array([0], dtype=mx.int32)

        # Metal kernel
        hit_metal, left_metal, idx_metal = mac_ring_match(
            cache, queries, req_ids, threshold=0.6, band_r=8
        )
        mx.eval(hit_metal, left_metal, idx_metal)

        # Reference
        hit_ref, left_ref, idx_ref = mac_ring_match_reference(
            cache, queries, req_ids, threshold=0.6, band_r=8
        )
        mx.eval(hit_ref, left_ref, idx_ref)

        # Compare
        assert mx.array_equal(hit_metal, hit_ref), (
            f"hit mismatch: metal={hit_metal.tolist()}, ref={hit_ref.tolist()}"
        )
        assert mx.array_equal(idx_metal, idx_ref), (
            f"idx mismatch: metal={idx_metal.tolist()}, ref={idx_ref.tolist()}"
        )
        assert mx.array_equal(left_metal, left_ref), (
            f"left mismatch: metal={left_metal.tolist()}, ref={left_ref.tolist()}"
        )

    @pytest.mark.skipif(
        not mx.metal.is_available(),
        reason="Metal not available",
    )
    def test_match_exact_query_hits(self):
        """A query identical to a cached entry should always hit (threshold < 1)."""
        from flashmlx.mac import MACRingCache, mac_ring_match

        R, M, H, D = 1, 8, 1, 16
        cache = MACRingCache(R, M, H, D)

        # Write a known query
        known_q = mx.random.normal((1, H, D)).astype(mx.bfloat16)
        cache.update(
            mx.array([0], dtype=mx.int32),
            known_q,
            mx.zeros((1, H, D), dtype=mx.bfloat16),
            mx.zeros((1, H), dtype=mx.float32),
        )
        mx.eval(cache.query_cache, cache.request_length)

        # Match with the exact same query
        hit, left, idx = mac_ring_match(
            cache, known_q, mx.array([0], dtype=mx.int32),
            threshold=0.5, band_r=4,
        )
        mx.eval(hit)

        assert hit[0, 0].item() is True, "Exact query should hit"


# ---------------------------------------------------------------------------
# Test 5: E2E miss → hit
# ---------------------------------------------------------------------------

class TestE2E:
    @pytest.mark.skipif(
        not mx.metal.is_available(),
        reason="Metal not available",
    )
    def test_miss_then_hit(self):
        """Step 0 with empty cache → miss. Step 1 with same query → hit."""
        from flashmlx.mac import MACDecodeWrapper

        N, Hq, Hkv, D, S = 1, 2, 1, 16, 32
        mac = MACDecodeWrapper(
            max_requests=4, capacity=16,
            num_heads=Hq, num_kv_heads=Hkv, head_dim=D,
            threshold=0.3, band_r=8, window_left=8,
        )

        mx.random.seed(123)
        q = mx.random.normal((N, Hq, D)).astype(mx.bfloat16)
        k = mx.random.normal((N, S, Hkv, D)).astype(mx.bfloat16)
        v = mx.random.normal((N, S, Hkv, D)).astype(mx.bfloat16)
        req_ids = mx.array([0], dtype=mx.int32)

        # Step 0: empty cache → should miss
        out0 = mac(q, k, v, req_ids)
        mx.eval(out0)
        stats0 = mac.last_stats
        assert stats0 is not None
        assert stats0.hit_rate == 0.0, f"Step 0 should be all miss, got hit_rate={stats0.hit_rate}"

        # Step 1: same query → should hit (cache now populated)
        out1 = mac(q, k, v, req_ids)
        mx.eval(out1)
        stats1 = mac.last_stats
        assert stats1 is not None
        assert stats1.hit_rate > 0.0, f"Step 1 should have some hits, got hit_rate={stats1.hit_rate}"

    def test_output_shape(self):
        """MACDecodeWrapper output shape matches input query shape."""
        from flashmlx.mac import MACDecodeWrapper

        N, Hq, Hkv, D, S = 2, 4, 2, 16, 64
        mac = MACDecodeWrapper(
            max_requests=4, capacity=8,
            num_heads=Hq, num_kv_heads=Hkv, head_dim=D,
        )

        q = mx.random.normal((N, Hq, D)).astype(mx.bfloat16)
        k = mx.random.normal((N, S, Hkv, D)).astype(mx.bfloat16)
        v = mx.random.normal((N, S, Hkv, D)).astype(mx.bfloat16)
        req_ids = mx.arange(N).astype(mx.int32)

        out = mac(q, k, v, req_ids)
        mx.eval(out)
        assert out.shape == (N, Hq, D)
