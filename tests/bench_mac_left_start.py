#!/usr/bin/env python3
"""
检查 MAC 的实际 left_start 值

如果 left_start 接近 0 → 虽然命中，但还是要计算大部分
如果 left_start 接近 S → 真正的缓存复用
"""

from __future__ import annotations

import sys

sys.path.insert(0, "src")
sys.path.insert(0, "mlx-lm-source")

import mlx.core as mx


def sync():
    mx.eval(mx.array(0))


print("=" * 80)
print("MAC left_start 分析")
print("=" * 80)
print()

from flashmlx.mac import MACDecodeWrapper

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

print("Scenario 1: 100 steps with same query")
print("=" * 80)

q_base = mx.random.normal((N, Hq, D)).astype(mx.bfloat16)
mx.eval(q_base)

for i in range(100):
    mac(q_base, k, v, req_ids)
    if i % 50 == 49:
        sync()

mac.ring_cache.request_length = mx.array([S], dtype=mx.int32)
mx.eval(mac.ring_cache.request_length)

# Test and extract left_start
mac(q_base, k, v, req_ids)

# Access internal state
if hasattr(mac, "_left_raw") and mac._left_raw is not None:
    left_start = mac._left_raw
    mx.eval(left_start)
    avg_left = mx.mean(left_start.astype(mx.float32)).item()
    min_left = mx.min(left_start).item()
    max_left = mx.max(left_start).item()

    stats = mac.last_stats
    print(f"  Hit rate: {stats.hit_rate:.1%}" if stats else "  No stats")
    print(f"  Left start (avg): {avg_left:.0f} / {S} ({avg_left/S*100:.1f}%)")
    print(f"  Left start (min): {min_left:.0f}")
    print(f"  Left start (max): {max_left:.0f}")
    print(f"  Active tokens: {S - avg_left:.0f} ({(S - avg_left)/S*100:.1f}%)")
    print()

print("Scenario 2: 100 steps with gradual drift")
print("=" * 80)

mac2 = MACDecodeWrapper(
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

q_base = mx.random.normal((N, Hq, D)).astype(mx.bfloat16)
mx.eval(q_base)

for i in range(100):
    noise = mx.random.normal((N, Hq, D)).astype(mx.bfloat16) * 0.05
    q = q_base + noise * (i / 100.0)
    mx.eval(q)
    mac2(q, k, v, req_ids)
    q_base = q
    if i % 50 == 49:
        sync()

mac2.ring_cache.request_length = mx.array([S], dtype=mx.int32)
mx.eval(mac2.ring_cache.request_length)

# Test final state
q_test = q_base + mx.random.normal((N, Hq, D)).astype(mx.bfloat16) * 0.05
mx.eval(q_test)
mac2(q_test, k, v, req_ids)

if hasattr(mac2, "_left_raw") and mac2._left_raw is not None:
    left_start = mac2._left_raw
    mx.eval(left_start)
    avg_left = mx.mean(left_start.astype(mx.float32)).item()
    min_left = mx.min(left_start).item()
    max_left = mx.max(left_start).item()

    stats = mac2.last_stats
    print(f"  Hit rate: {stats.hit_rate:.1%}" if stats else "  No stats")
    print(f"  Left start (avg): {avg_left:.0f} / {S} ({avg_left/S*100:.1f}%)")
    print(f"  Left start (min): {min_left:.0f}")
    print(f"  Left start (max): {max_left:.0f}")
    print(f"  Active tokens: {S - avg_left:.0f} ({(S - avg_left)/S*100:.1f}%)")
    print()

print("=" * 80)
print("结论")
print("=" * 80)
print()
print("如果 active tokens 接近 100% → 虽然命中，但没节省计算")
print("如果 active tokens < 50% → 真正的缓存复用")
print()
print("MAC 论文中 active tokens 应该在 10-30% 左右")
print()
