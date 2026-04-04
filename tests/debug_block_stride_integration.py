#!/usr/bin/env python3
"""
Debug block-stride 集成问题

对比：
1. 单独测试 partial attention kernel
2. 端到端测试

找出为什么单独测试快，但端到端反而慢
"""

from __future__ import annotations

import sys
import time

sys.path.insert(0, "src")
sys.path.insert(0, "mlx-lm-source")

import mlx.core as mx


def sync():
    mx.eval(mx.array(0))


def measure_ms(fn, warmup=10, iters=50):
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
print("Block-Stride 集成问题诊断")
print("=" * 80)
print()

from flashmlx.mac.attention import mac_fused_partial_attention
from flashmlx.mac.attention_opt_block import mac_partial_attention_block_stride

N, H, D = 1, 32, 128
Hkv = 8
S = 32768

mx.random.seed(42)
queries = mx.random.normal((N, H, D)).astype(mx.bfloat16)
keys = mx.random.normal((N, S, Hkv, D)).astype(mx.bfloat16)
values = mx.random.normal((N, S, Hkv, D)).astype(mx.bfloat16)
scale = D**-0.5
mx.eval(queries, keys, values)

print("Test 1: Full attention (start_pos=0)")
print("=" * 80)

start_full = mx.zeros((N, H), dtype=mx.int32)
mx.eval(start_full)

def run_orig_full():
    return mac_fused_partial_attention(queries, keys, values, start_full, scale)

def run_block_full():
    return mac_partial_attention_block_stride(queries, keys, values, start_full, scale)

t_orig_full = measure_ms(run_orig_full)
t_block_full = measure_ms(run_block_full)

print(f"  Original:      {t_orig_full:.3f} ms")
print(f"  Block-stride:  {t_block_full:.3f} ms")
print(f"  Speedup:       {t_orig_full / t_block_full:.2f}×")
print()

print("Test 2: Partial attention (start_pos=90% → 10% active)")
print("=" * 80)

start_partial = mx.full((N, H), int(S * 0.9), dtype=mx.int32)
mx.eval(start_partial)

def run_orig_partial():
    return mac_fused_partial_attention(queries, keys, values, start_partial, scale)

def run_block_partial():
    return mac_partial_attention_block_stride(queries, keys, values, start_partial, scale)

t_orig_partial = measure_ms(run_orig_partial)
t_block_partial = measure_ms(run_block_partial)

print(f"  Original:      {t_orig_partial:.3f} ms")
print(f"  Block-stride:  {t_block_partial:.3f} ms")
print(f"  Speedup:       {t_orig_partial / t_block_partial:.2f}×")
print()

print("Test 3: Highly partial (start_pos=97% → 3% active)")
print("=" * 80)

start_high = mx.full((N, H), int(S * 0.97), dtype=mx.int32)
mx.eval(start_high)

def run_orig_high():
    return mac_fused_partial_attention(queries, keys, values, start_high, scale)

def run_block_high():
    return mac_partial_attention_block_stride(queries, keys, values, start_high, scale)

t_orig_high = measure_ms(run_orig_high)
t_block_high = measure_ms(run_block_high)

print(f"  Original:      {t_orig_high:.3f} ms")
print(f"  Block-stride:  {t_block_high:.3f} ms")
print(f"  Speedup:       {t_orig_high / t_block_high:.2f}×")
print()

print("=" * 80)
print("分析")
print("=" * 80)
print()
print("如果 full attention 时 block-stride 更慢 →")
print("  可能原因：block 划分有额外开销")
print()
print("如果 partial attention 时 block-stride 更快 →")
print("  说明优化对 partial 场景有效")
print()
print("端到端变慢可能是因为 MAC hit 后，大部分时间在 start_pos=90%+ 的场景")
print("这种情况下，原始的 stride 访问可能比 block 更好（workload 不平衡）")
print()
