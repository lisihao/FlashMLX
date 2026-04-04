#!/usr/bin/env python3
"""
MAC hit 耗时分解 + 命中率调参

问题1: hit 0.3-0.4ms 太慢，瓶颈在哪？
问题2: 命中率能不能更高？
"""

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


def breakdown():
    """问题1: MAC hit 每一步到底花多久"""
    from flashmlx.mac import (
        MACDecodeWrapper,
        mac_fused_partial_attention,
        merge_attention_states,
        mac_ring_match,
    )
    from flashmlx.mac.attention import (
        mac_windowed_attention,
        downdate_attention,
    )

    print("=" * 72)
    print("问题1: MAC hit 每一步耗时分解")
    print("=" * 72)
    print()

    N, Hq, Hkv, D = 1, 32, 8, 128
    scale = D ** -0.5

    for S in [8192, 32768]:
        M = 512
        band_r = 256
        window_left = 256

        mac = MACDecodeWrapper(
            max_requests=4, capacity=M,
            num_heads=Hq, num_kv_heads=Hkv, head_dim=D,
            threshold=0.5, band_r=band_r, window_left=window_left,
        )

        mx.random.seed(42)
        k = mx.random.normal((N, S, Hkv, D)).astype(mx.bfloat16)
        v = mx.random.normal((N, S, Hkv, D)).astype(mx.bfloat16)
        req_ids = mx.array([0], dtype=mx.int32)
        mx.eval(k, v)

        # 填满 cache + 模拟 prefill
        for step in range(M + 5):
            q_f = mx.random.normal((N, Hq, D)).astype(mx.bfloat16)
            mx.eval(q_f)
            mac(q_f, k, v, req_ids)
            sync()
        mac.ring_cache.request_length = mx.array([S], dtype=mx.int32)
        mx.eval(mac.ring_cache.request_length)

        q = mx.random.normal((N, Hq, D)).astype(mx.bfloat16)
        mx.eval(q)
        mac(q, k, v, req_ids)
        sync()

        # 准备各步的输入
        hit, left_start, indices = mac_ring_match(
            mac.ring_cache, q, req_ids, threshold=0.5, band_r=band_r
        )
        mx.eval(hit, left_start, indices)

        cached_o, cached_lse = mac.ring_cache.fetch(req_ids, indices)
        fresh_o, fresh_lse = mac_fused_partial_attention(q, k, v, left_start, scale)
        mx.eval(cached_o, cached_lse, fresh_o, fresh_lse)

        skip_pct = float(left_start.astype(mx.float32).mean().item()) / S

        print(f"--- S={S}, 跳过 {skip_pct:.0%} ---")

        # 1. Match
        t1 = measure_ms(lambda: mac_ring_match(
            mac.ring_cache, q, req_ids, threshold=0.5, band_r=band_r
        ))

        # 2. Fetch
        t2 = measure_ms(lambda: mac.ring_cache.fetch(req_ids, indices))

        # 3. Partial attention (只算 left_start 以后)
        t3 = measure_ms(lambda: mac_fused_partial_attention(q, k, v, left_start, scale))

        # 4. Merge
        t4 = measure_ms(lambda: merge_attention_states(
            cached_o.astype(mx.float32), cached_lse, fresh_o, fresh_lse
        ))

        # 5. Window attention (rectify 用)
        t5 = measure_ms(lambda: mac_windowed_attention(q, k, v, window_left, scale))

        # 6. Downdate
        full_o = fresh_o  # 近似
        full_lse = fresh_lse
        window_o, window_lse = mac_windowed_attention(q, k, v, window_left, scale)
        mx.eval(window_o, window_lse)
        t6 = measure_ms(lambda: downdate_attention(full_o, full_lse, window_o, window_lse))

        # 7. Ring cache update
        rest_o, rest_lse = downdate_attention(full_o, full_lse, window_o, window_lse)
        mx.eval(rest_o, rest_lse)
        t7 = measure_ms(lambda: mac.ring_cache.update(
            req_ids, q, rest_o.astype(mx.bfloat16), rest_lse
        ))

        # 8. E2E (对照)
        t_e2e = measure_ms(lambda: mac(q, k, v, req_ids))

        total = t1 + t2 + t3 + t4 + t5 + t6 + t7

        print(f"  ① Match (Metal L2 搜索):      {t1:>6.3f} ms  ({t1/total*100:>4.0f}%)")
        print(f"  ② Fetch (从 ring cache 取):    {t2:>6.3f} ms  ({t2/total*100:>4.0f}%)")
        print(f"  ③ Partial attn (跳过前缀):     {t3:>6.3f} ms  ({t3/total*100:>4.0f}%)")
        print(f"  ④ Merge (cached + fresh):      {t4:>6.3f} ms  ({t4/total*100:>4.0f}%)")
        print(f"  ⑤ Window attn (rectify 用):    {t5:>6.3f} ms  ({t5/total*100:>4.0f}%)")
        print(f"  ⑥ Downdate (full - window):    {t6:>6.3f} ms  ({t6/total*100:>4.0f}%)")
        print(f"  ⑦ Cache update (写回 ring):    {t7:>6.3f} ms  ({t7/total*100:>4.0f}%)")
        print(f"  -----------------------------------------")
        print(f"  各步总和:                       {total:>6.3f} ms")
        print(f"  实测 E2E:                       {t_e2e:>6.3f} ms")
        print()


def hitrate_sweep():
    """问题2: 不同 threshold 下的命中率"""
    from flashmlx.mac import MACDecodeWrapper

    print("=" * 72)
    print("问题2: 命中率 vs threshold 调参")
    print("=" * 72)
    print()

    N, Hq, Hkv, D = 1, 32, 8, 128
    S = 8192
    M = 512

    mx.random.seed(42)
    k = mx.random.normal((N, S, Hkv, D)).astype(mx.bfloat16)
    v = mx.random.normal((N, S, Hkv, D)).astype(mx.bfloat16)
    req_ids = mx.array([0], dtype=mx.int32)
    mx.eval(k, v)

    print("  场景A: 随机 query (最差情况 — query 之间无关联)")
    print(f"  {'threshold':>10} {'hit率':>6} {'质量(cos)':>10}")
    print(f"  " + "-" * 30)

    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        mac = MACDecodeWrapper(
            max_requests=4, capacity=M,
            num_heads=Hq, num_kv_heads=Hkv, head_dim=D,
            threshold=thresh, band_r=256, window_left=256,
        )
        # 填 cache
        for _ in range(M + 5):
            q_f = mx.random.normal((N, Hq, D)).astype(mx.bfloat16)
            mx.eval(q_f)
            mac(q_f, k, v, req_ids)
            sync()

        # 测 10 个新 query 的 hit 率
        hits = 0
        total = 0
        cos_sims = []
        from flashmlx.mac import mac_partial_attention
        start_zero = mx.zeros((N, Hq), dtype=mx.int32)

        for _ in range(20):
            q_t = mx.random.normal((N, Hq, D)).astype(mx.bfloat16)
            mx.eval(q_t)

            full_o, _ = mac_partial_attention(q_t, k, v, start_zero)
            mac_o = mac(q_t, k, v, req_ids)
            mx.eval(full_o, mac_o)

            stats = mac.last_stats
            if stats:
                hits += stats.num_hits
                total += stats.num_total

            f = full_o.astype(mx.float32)
            m = mac_o.astype(mx.float32)
            fn = mx.sqrt(mx.sum(f * f)).item()
            mn = mx.sqrt(mx.sum(m * m)).item()
            cos = mx.sum(f * m).item() / max(fn * mn, 1e-10)
            cos_sims.append(cos)
            sync()

        hr = hits / max(total, 1)
        avg_cos = sum(cos_sims) / len(cos_sims)
        print(f"  {thresh:>10.1f} {hr:>5.0%} {avg_cos:>10.5f}")

    print()
    print("  场景B: 相似 query (真实推理 — 连续 token 的 query 有相关性)")
    print(f"  {'threshold':>10} {'hit率':>6} {'质量(cos)':>10}")
    print(f"  " + "-" * 30)

    # 模拟相似 query：在基础 query 上加小噪声
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        mac = MACDecodeWrapper(
            max_requests=4, capacity=M,
            num_heads=Hq, num_kv_heads=Hkv, head_dim=D,
            threshold=thresh, band_r=256, window_left=256,
        )

        # 用「漂移」query 填 cache：每步在前一个基础上加小噪声
        base_q = mx.random.normal((N, Hq, D)).astype(mx.bfloat16)
        mx.eval(base_q)
        for step in range(M + 5):
            noise = mx.random.normal((N, Hq, D)).astype(mx.bfloat16) * 0.1
            q_f = base_q + noise
            mx.eval(q_f)
            mac(q_f, k, v, req_ids)
            base_q = q_f  # 缓慢漂移
            sync()

        # 测相似 query
        hits = 0
        total = 0
        cos_sims = []

        for _ in range(20):
            noise = mx.random.normal((N, Hq, D)).astype(mx.bfloat16) * 0.1
            q_t = base_q + noise
            mx.eval(q_t)

            full_o, _ = mac_partial_attention(q_t, k, v, start_zero)
            mac_o = mac(q_t, k, v, req_ids)
            mx.eval(full_o, mac_o)

            stats = mac.last_stats
            if stats:
                hits += stats.num_hits
                total += stats.num_total

            f = full_o.astype(mx.float32)
            m = mac_o.astype(mx.float32)
            fn = mx.sqrt(mx.sum(f * f)).item()
            mn = mx.sqrt(mx.sum(m * m)).item()
            cos = mx.sum(f * m).item() / max(fn * mn, 1e-10)
            cos_sims.append(cos)
            base_q = q_t
            sync()

        hr = hits / max(total, 1)
        avg_cos = sum(cos_sims) / len(cos_sims)
        print(f"  {thresh:>10.1f} {hr:>5.0%} {avg_cos:>10.5f}")

    print()
    print("  结论：")
    print("  • 随机 query：hit 率很低（随机数据之间没相关性）")
    print("  • 相似 query：hit 率大幅提升（模拟真实推理场景）")
    print("  • threshold 越低越容易 hit，但质量下降")
    print("  • 推荐 threshold=0.5-0.7：平衡 hit 率和质量")


if __name__ == "__main__":
    print(f"设备: {mx.default_device()}")
    print()
    breakdown()
    hitrate_sweep()
