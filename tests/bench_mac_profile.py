#!/usr/bin/env python3
"""Profile MAC-Attention: isolate each step's latency."""

from __future__ import annotations

import sys
import time

sys.path.insert(0, "src")
sys.path.insert(0, "mlx-lm-source")

import mlx.core as mx


def sync():
    mx.eval(mx.array(0))


def measure_ms(fn, warmup=5, iters=30):
    for _ in range(warmup):
        fn()
        sync()
    times = []
    for _ in range(iters):
        sync()
        t0 = time.perf_counter()
        result = fn()
        if isinstance(result, (tuple, list)):
            mx.eval(*result)
        elif result is not None:
            mx.eval(result)
        sync()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


def profile_steps():
    from flashmlx.mac import (
        MACDecodeWrapper,
        mac_fused_partial_attention,
        mac_partial_attention,
        merge_attention_states,
    )
    from flashmlx.mac.attention import _mac_partial_attention_reference
    from flashmlx.mac.match import mac_ring_match

    print("=" * 70)
    print("Step-by-step MAC profiling")
    print("=" * 70)

    for S, Hq, Hkv, D, label in [
        (1024, 8, 2, 128, "Small (1K, 8Hq)"),
        (4096, 32, 8, 128, "Medium (4K, 32Hq)"),
        (8192, 32, 8, 128, "Large (8K, 32Hq)"),
    ]:
        N, M = 1, 512
        print(f"\n--- {label}: S={S}, Hq={Hq}, Hkv={Hkv} ---")

        mac = MACDecodeWrapper(
            max_requests=4,
            capacity=M,
            num_heads=Hq,
            num_kv_heads=Hkv,
            head_dim=D,
            threshold=0.5,
            band_r=min(256, S // 4),
            window_left=min(256, S // 4),
        )

        mx.random.seed(42)
        k = mx.random.normal((N, S, Hkv, D)).astype(mx.bfloat16)
        v = mx.random.normal((N, S, Hkv, D)).astype(mx.bfloat16)
        req_ids = mx.array([0], dtype=mx.int32)
        mx.eval(k, v)

        # Warm up cache
        for _ in range(5):
            q_w = mx.random.normal((N, Hq, D)).astype(mx.bfloat16)
            mx.eval(q_w)
            mac(q_w, k, v, req_ids)
            _ = mac.last_stats  # force eval
            sync()

        q = mx.random.normal((N, Hq, D)).astype(mx.bfloat16)
        mx.eval(q)

        # 1. SDPA baseline (MLX built-in)
        q_sdpa = q[:, :, None, :]
        groups = Hq // Hkv
        k_sdpa = mx.repeat(k, groups, axis=2).transpose(0, 2, 1, 3)
        v_sdpa = mx.repeat(v, groups, axis=2).transpose(0, 2, 1, 3)
        mx.eval(k_sdpa, v_sdpa)
        scale = D**-0.5

        t_sdpa = measure_ms(
            lambda: mx.fast.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa, scale=scale
            )
        )

        # 2. Our fused kernel (full attention, start_pos=0)
        start_zero = mx.zeros((N, Hq), dtype=mx.int32)
        t_fused_full = measure_ms(
            lambda: mac_fused_partial_attention(q, k, v, start_zero, scale)
        )

        # 3. Reference unfused (for comparison)
        t_ref = measure_ms(
            lambda: _mac_partial_attention_reference(q, k, v, start_zero, scale)
        )

        # 4. Match kernel only
        t_match = measure_ms(
            lambda: mac_ring_match(
                mac.ring_cache, q, req_ids, threshold=0.5, band_r=256
            )
        )

        # 5. Fetch from ring cache
        hit, left_start, indices = mac_ring_match(
            mac.ring_cache, q, req_ids, threshold=0.5, band_r=256
        )
        mx.eval(hit, left_start, indices)

        t_fetch = measure_ms(
            lambda: mac.ring_cache.fetch(req_ids, indices)
        )

        # 6. Merge
        cached_o, cached_lse = mac.ring_cache.fetch(req_ids, indices)
        fresh_o, fresh_lse = mac_fused_partial_attention(
            q, k, v, left_start, scale
        )
        mx.eval(cached_o, cached_lse, fresh_o, fresh_lse)

        t_merge = measure_ms(
            lambda: merge_attention_states(
                cached_o.astype(mx.float32), cached_lse, fresh_o, fresh_lse
            )
        )

        # 7. Windowed attention (S-window to S)
        window_left = min(256, S // 4)
        win_start = mx.full((N, Hq), max(0, S - window_left), dtype=mx.int32)
        t_window = measure_ms(
            lambda: mac_fused_partial_attention(q, k, v, win_start, scale)
        )

        # 8. Full E2E MAC decode
        t_mac = measure_ms(lambda: mac(q, k, v, req_ids))

        print(f"  SDPA (MLX built-in):     {t_sdpa:>8.3f} ms")
        print(f"  Fused kernel (full):     {t_fused_full:>8.3f} ms  ({t_fused_full/t_sdpa:.1f}x vs SDPA)")
        print(f"  Reference unfused:       {t_ref:>8.3f} ms  ({t_ref/t_sdpa:.1f}x vs SDPA)")
        print(f"  Match kernel:            {t_match:>8.3f} ms")
        print(f"  Fetch (vectorized):      {t_fetch:>8.3f} ms")
        print(f"  Merge:                   {t_merge:>8.3f} ms")
        print(f"  Window attn ({window_left}tok):  {t_window:>8.3f} ms")
        print(f"  ---")
        print(f"  Sum of parts:            {t_match + t_fused_full + t_fetch + t_merge + t_window:>8.3f} ms")
        print(f"  Full MAC E2E:            {t_mac:>8.3f} ms")
        print(f"  MAC / SDPA ratio:        {t_mac/t_sdpa:>8.1f}x")


if __name__ == "__main__":
    print(f"Device: {mx.default_device()}")
    profile_steps()
