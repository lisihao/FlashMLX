#!/usr/bin/env python3
"""
MAC-Attention 性能分解 - 找出真正的瓶颈

分析各个操作的时间占比：
1. Query normalization
2. mac_ring_match (Match)
3. mac_partial_attention (Attention)
4. merge_attention_states (Merge)
5. mx.where (Select)
6. mac_rectify_and_update (Rectify + Update)
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
        else:
            mx.eval(result)
        sync()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


print("=" * 80)
print("MAC-Attention 性能分解")
print("=" * 80)
print()

# Setup
from flashmlx.mac import MACDecodeWrapper
from flashmlx.mac.attention import mac_partial_attention, merge_attention_states
from flashmlx.mac.match import mac_ring_match

N, Hq, Hkv, D = 1, 32, 8, 128
S = 8192

mac = MACDecodeWrapper(
    max_requests=4,
    capacity=512,
    num_heads=Hq,
    num_kv_heads=Hkv,
    head_dim=D,
    threshold=0.5,
    band_r=256,
    window_left=256,
    normalize_queries=True,
)

mx.random.seed(42)
k = mx.random.normal((N, S, Hkv, D)).astype(mx.bfloat16)
v = mx.random.normal((N, S, Hkv, D)).astype(mx.bfloat16)
req_ids = mx.array([0], dtype=mx.int32)
mx.eval(k, v)

# Warmup
print("Warming up...")
for i in range(100):
    q = mx.random.normal((N, Hq, D)).astype(mx.bfloat16)
    mx.eval(q)
    mac(q, k, v, req_ids)
    if i % 50 == 49:
        sync()

mac.ring_cache.request_length = mx.array([S], dtype=mx.int32)
mx.eval(mac.ring_cache.request_length)

# Test query
q_test = mx.random.normal((N, Hq, D)).astype(mx.bfloat16)
mx.eval(q_test)
scale = D**-0.5

print()
print("分解各操作时间（30 次中位数）：")
print("=" * 80)

# 1. Query normalization
def step1_normalize():
    q_f32 = q_test.astype(mx.float32)
    norms = mx.sqrt(mx.sum(q_f32 * q_f32, axis=-1, keepdims=True))
    norms = mx.maximum(norms, 1e-8)
    return (q_f32 / norms).astype(q_test.dtype)


t1 = measure_ms(step1_normalize)
print(f"1. Query normalization:     {t1:6.3f} ms")

queries_normalized = step1_normalize()
mx.eval(queries_normalized)

# 2. mac_ring_match
def step2_match():
    return mac_ring_match(
        mac.ring_cache,
        queries_normalized,
        req_ids,
        threshold=mac._match_threshold,
        band_r=mac.band_r,
        rows_per_tile=mac.rows_per_tile,
    )


t2 = measure_ms(step2_match)
print(f"2. Match (mac_ring_match):  {t2:6.3f} ms")

hit, left_start, indices = step2_match()
mx.eval(hit, left_start, indices)

# 3. mac_partial_attention
def step3_attention():
    return mac_partial_attention(q_test, k, v, left_start, scale)


t3 = measure_ms(step3_attention)
print(f"3. Partial Attention:       {t3:6.3f} ms")

fresh_o, fresh_lse = step3_attention()
mx.eval(fresh_o, fresh_lse)

# 4. Fetch from cache
def step4_fetch():
    return mac.ring_cache.fetch(req_ids, indices)


t4 = measure_ms(step4_fetch)
print(f"4. Fetch from cache:        {t4:6.3f} ms")

cached_o, cached_lse = step4_fetch()
mx.eval(cached_o, cached_lse)

# 5. merge_attention_states
def step5_merge():
    return merge_attention_states(
        cached_o.astype(fresh_o.dtype), cached_lse, fresh_o, fresh_lse
    )


t5 = measure_ms(step5_merge)
print(f"5. Merge attention states:  {t5:6.3f} ms")

merged_o, merged_lse = step5_merge()
mx.eval(merged_o, merged_lse)

# 6. mx.where (select)
def step6_select():
    o = mx.where(hit[..., None], merged_o, fresh_o)
    lse = mx.where(hit, merged_lse, fresh_lse)
    return o, lse


t6 = measure_ms(step6_select)
print(f"6. Select (mx.where):       {t6:6.3f} ms")

output, output_lse = step6_select()
mx.eval(output, output_lse)

# 7. mac_rectify_and_update
from flashmlx.mac.attention import mac_rectify_and_update


def step7_rectify():
    mac_rectify_and_update(
        q_test,
        k,
        v,
        output,
        output_lse,
        mac.ring_cache,
        req_ids,
        window_left=mac.window_left,
        scale=scale,
    )
    return mx.array(0)


t7 = measure_ms(step7_rectify)
print(f"7. Rectify + Update:        {t7:6.3f} ms")

print()
print("=" * 80)
total = t1 + t2 + t3 + t4 + t5 + t6 + t7
print(f"总计（理论）:               {total:6.3f} ms")
print()

# Actual end-to-end
def full_mac():
    return mac(q_test, k, v, req_ids)


t_full = measure_ms(full_mac)
print(f"实际端到端:                 {t_full:6.3f} ms")
print(f"差异（调度开销）:           {t_full - total:6.3f} ms")
print()

# Percentages
print("=" * 80)
print("时间占比分析：")
print("=" * 80)
print(f"1. Query normalization:     {t1:6.3f} ms ({t1/total*100:5.1f}%)")
print(f"2. Match (mac_ring_match):  {t2:6.3f} ms ({t2/total*100:5.1f}%)")
print(f"3. Partial Attention:       {t3:6.3f} ms ({t3/total*100:5.1f}%)")
print(f"4. Fetch from cache:        {t4:6.3f} ms ({t4/total*100:5.1f}%)")
print(f"5. Merge attention states:  {t5:6.3f} ms ({t5/total*100:5.1f}%)")
print(f"6. Select (mx.where):       {t6:6.3f} ms ({t6/total*100:5.1f}%)")
print(f"7. Rectify + Update:        {t7:6.3f} ms ({t7/total*100:5.1f}%)")
print()

print("=" * 80)
print("优化建议：")
print("=" * 80)
if t3 > total * 0.3:
    print(f"⚡ Partial Attention 占 {t3/total*100:.1f}% - 最大瓶颈！")
if t2 > total * 0.2:
    print(f"⚡ Match 占 {t2/total*100:.1f}% - 可优化！")
if t5 < total * 0.1:
    print(f"  Merge 仅占 {t5/total*100:.1f}% - 优化收益小")
print()
