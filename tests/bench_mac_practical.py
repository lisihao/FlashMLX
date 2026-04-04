#!/usr/bin/env python3
"""
MAC-Attention 实测：质量 + 性能 + 内存

回答四个问题：
  1. 输出质量：MAC 的 attention 输出和标准 full attention 差多少？
  2. TG 影响：每步 decode 快了还是慢了？
  3. TTFT/PP：有没有影响？
  4. 内存：ring cache 额外占多少？省了多少 KV 读取？
"""

from __future__ import annotations

import sys
import time

import mlx.core as mx
import numpy as np


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


# ============================================================================
# 1. 输出质量：MAC attention vs Full attention
# ============================================================================

def test_output_quality():
    from flashmlx.mac import MACDecodeWrapper, mac_partial_attention

    print("=" * 65)
    print("1. 输出质量：MAC output vs Full Attention (ground truth)")
    print("=" * 65)

    configs = [
        # (S, Hq, Hkv, D, label)
        (256,  8, 2, 128, "小 context (256)"),
        (1024, 8, 2, 128, "中 context (1K)"),
        (4096, 8, 2, 128, "长 context (4K)"),
        (1024, 32, 8, 128, "多 head (32Hq/8Hkv)"),
    ]

    for S, Hq, Hkv, D, label in configs:
        N = 1
        M = min(S, 512)

        mac = MACDecodeWrapper(
            max_requests=4, capacity=M,
            num_heads=Hq, num_kv_heads=Hkv, head_dim=D,
            threshold=0.5, band_r=min(256, S // 4), window_left=min(256, S // 4),
        )

        mx.random.seed(42)
        k = mx.random.normal((N, S, Hkv, D)).astype(mx.bfloat16)
        v = mx.random.normal((N, S, Hkv, D)).astype(mx.bfloat16)
        req_ids = mx.array([0], dtype=mx.int32)

        # 预热 cache（模拟已经跑了几步）
        for _ in range(5):
            q_warm = mx.random.normal((N, Hq, D)).astype(mx.bfloat16)
            mac(q_warm, k, v, req_ids)
            mx.eval(mac.ring_cache.query_cache)

        # 实际测试 query
        q = mx.random.normal((N, Hq, D)).astype(mx.bfloat16)

        # Ground truth: 完整 full attention
        start_zero = mx.zeros((N, Hq), dtype=mx.int32)
        full_o, full_lse = mac_partial_attention(q, k, v, start_zero)
        mx.eval(full_o)

        # MAC output
        mac_o = mac(q, k, v, req_ids)
        mx.eval(mac_o)

        # 比较
        full_f = full_o.astype(mx.float32)
        mac_f = mac_o.astype(mx.float32)
        diff = mx.abs(full_f - mac_f)
        max_err = mx.max(diff).item()
        mean_err = mx.mean(diff).item()
        # 相对误差 (对 full_o 的 L2 norm 归一化)
        full_norm = mx.sqrt(mx.sum(full_f * full_f)).item()
        rel_err = mx.sqrt(mx.sum(diff * diff)).item() / max(full_norm, 1e-10)

        # cosine similarity
        dot = mx.sum(full_f * mac_f).item()
        mac_norm = mx.sqrt(mx.sum(mac_f * mac_f)).item()
        cos_sim = dot / max(full_norm * mac_norm, 1e-10)

        hit_rate = mac.last_stats.hit_rate if mac.last_stats else 0

        print(f"\n  [{label}]  S={S}, Hq={Hq}, Hkv={Hkv}, D={D}")
        print(f"    Hit rate:        {hit_rate:.0%}")
        print(f"    Cosine sim:      {cos_sim:.6f}  {'✅' if cos_sim > 0.99 else '⚠️' if cos_sim > 0.95 else '❌'}")
        print(f"    相对误差 (L2):   {rel_err:.6f}  {'✅' if rel_err < 0.01 else '⚠️' if rel_err < 0.05 else '❌'}")
        print(f"    最大绝对误差:    {max_err:.6f}")
        print(f"    平均绝对误差:    {mean_err:.6f}")


# ============================================================================
# 2. TG 速度：MAC decode step vs Full Attention decode step
# ============================================================================

def test_tg_speed():
    from flashmlx.mac import MACDecodeWrapper, mac_partial_attention

    print("\n" + "=" * 65)
    print("2. TG 速度：单步 decode 耗时 (MAC vs Full Attention)")
    print("=" * 65)
    print("   MAC 只影响 decode (TG)，不影响 prefill (PP/TTFT)")
    print()

    configs = [
        # (S, Hq, Hkv, D, label)
        (1024, 32, 8, 128,  "Qwen3-8B + 1K ctx"),
        (4096, 32, 8, 128,  "Qwen3-8B + 4K ctx"),
        (8192, 32, 8, 128,  "Qwen3-8B + 8K ctx"),
        (16384, 32, 8, 128, "Qwen3-8B + 16K ctx"),
        (32768, 32, 8, 128, "Qwen3-8B + 32K ctx"),
    ]

    print(f"  {'场景':<22} {'S':>6} {'SDPA':>8} {'Fused':>8} {'MAC':>8} {'MAC/SDPA':>10}")
    print("  " + "-" * 70)

    for S, Hq, Hkv, D, label in configs:
        N = 1
        M = min(S, 512)

        mac = MACDecodeWrapper(
            max_requests=4, capacity=M,
            num_heads=Hq, num_kv_heads=Hkv, head_dim=D,
            threshold=0.5, band_r=256, window_left=256,
        )

        mx.random.seed(42)
        k = mx.random.normal((N, S, Hkv, D)).astype(mx.bfloat16)
        v = mx.random.normal((N, S, Hkv, D)).astype(mx.bfloat16)
        req_ids = mx.array([0], dtype=mx.int32)
        mx.eval(k, v)

        # SDPA baseline (MLX built-in, no LSE)
        q_sdpa_base = mx.random.normal((N, Hq, D)).astype(mx.bfloat16)
        mx.eval(q_sdpa_base)
        q_sdpa = q_sdpa_base[:, :, None, :]
        groups = Hq // Hkv
        k_sdpa = mx.repeat(k, groups, axis=2).transpose(0, 2, 1, 3)
        v_sdpa = mx.repeat(v, groups, axis=2).transpose(0, 2, 1, 3)
        mx.eval(k_sdpa, v_sdpa)
        scale = D ** -0.5

        def run_sdpa():
            return mx.fast.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, scale=scale)

        t_sdpa = measure_ms(run_sdpa)

        # Fused kernel baseline (our kernel, full attention)
        start_zero = mx.zeros((N, Hq), dtype=mx.int32)

        def run_fused():
            return mac_partial_attention(q_sdpa_base, k, v, start_zero, scale)

        t_fused = measure_ms(run_fused)

        # Warm MAC cache
        for _ in range(5):
            q_warm = mx.random.normal((N, Hq, D)).astype(mx.bfloat16)
            mx.eval(q_warm)
            mac(q_warm, k, v, req_ids)
            _ = mac.last_stats
            sync()

        q = mx.random.normal((N, Hq, D)).astype(mx.bfloat16)
        mx.eval(q)

        # MAC decode step
        def run_mac():
            return mac(q, k, v, req_ids)

        t_mac = measure_ms(run_mac)

        ratio = t_mac / t_sdpa
        verdict = "✅" if ratio < 1.0 else "⚠️" if ratio < 2.0 else "❌"

        print(f"  {label:<22} {S:>6} {t_sdpa:>6.2f}ms {t_fused:>6.2f}ms {t_mac:>6.2f}ms {ratio:>8.2f}x {verdict}")


# ============================================================================
# 3. PP/TTFT 说明
# ============================================================================

def explain_pp_ttft():
    print("\n" + "=" * 65)
    print("3. PP / TTFT 影响")
    print("=" * 65)
    print("""
  MAC-Attention 只在 decode (token generation) 阶段工作。

  ✅ PP (Prompt Processing):  完全不影响，MAC 不参与 prefill
  ✅ TTFT (首 token 延迟):    完全不影响，MAC 不参与 prefill
  ⚠️  TG (Token Generation):  见上面的 benchmark

  MAC 的设计目标：decode 阶段通过匹配历史 query 复用 attention 结果，
  跳过 prefix KV 的读取。对长 context decode 有加速潜力。
""")


# ============================================================================
# 4. 内存开销
# ============================================================================

def test_memory():
    from flashmlx.mac import MACRingCache

    print("=" * 65)
    print("4. 内存开销：Ring Cache 额外内存 vs KV Cache 节省")
    print("=" * 65)

    configs = [
        # (model_label, Hq, Hkv, D, n_layers, typical_ctx)
        ("Qwen3-1.7B",  16,  4,  128, 28, 4096),
        ("Qwen3-8B",    32,  8,  128, 36, 8192),
        ("Llama-3.1-8B", 32, 8,  128, 32, 8192),
    ]

    for model, Hq, Hkv, D, L, ctx in configs:
        R = 1  # 单请求场景 (Apple Silicon 最常见)
        M = 512  # ring cache capacity

        # Ring cache 内存 (per-layer)
        # query_cache: [R, M, Hq, D] bf16
        # attn_cache:  [R, M, Hq, D] bf16
        # lse_cache:   [R, M, Hq]    f32
        # request_length: [R] int32
        q_bytes = R * M * Hq * D * 2  # bf16
        a_bytes = R * M * Hq * D * 2  # bf16
        l_bytes = R * M * Hq * 4       # f32
        rl_bytes = R * 4               # int32
        ring_per_layer = q_bytes + a_bytes + l_bytes + rl_bytes
        ring_total = ring_per_layer * L

        # KV Cache 内存 (per-layer)
        # K: [batch, seq_len, Hkv, D] bf16
        # V: [batch, seq_len, Hkv, D] bf16
        kv_per_layer = R * ctx * Hkv * D * 2 * 2  # K+V, bf16
        kv_total = kv_per_layer * L

        # MAC hit 时跳过的 KV 读取（一步 decode）
        # hit 时只读 [left_start:] 而非 [0:ctx]
        # 假设 hit rate 50%, 平均 left_start = ctx/2
        avg_skip_ratio = 0.5 * 0.5  # 50% hit rate × 50% skip
        kv_read_per_step = R * ctx * Hkv * D * 2 * 2  # 全量 K+V 读取
        kv_saved_per_step = kv_read_per_step * avg_skip_ratio

        print(f"\n  [{model}]  Hq={Hq}, Hkv={Hkv}, D={D}, L={L}, ctx={ctx}")
        print(f"    Ring Cache 额外内存 (全层): {ring_total / 1024 / 1024:>8.1f} MB")
        print(f"    KV Cache 占用 (全层):       {kv_total / 1024 / 1024:>8.1f} MB")
        print(f"    Ring / KV 比例:             {ring_total / kv_total * 100:>8.1f}%")
        print(f"    ---")
        print(f"    每步 KV 读取量 (full):      {kv_read_per_step / 1024 / 1024:>8.1f} MB")
        print(f"    50%hit 时节省的 KV 带宽:    {kv_saved_per_step / 1024 / 1024:>8.1f} MB/step")

        # MAC 实际有多大
        cache = MACRingCache(R, M, Hq, D)
        actual_bytes = (
            cache.query_cache.nbytes +
            cache.attn_cache.nbytes +
            cache.lse_cache.nbytes +
            cache.request_length.nbytes
        )
        print(f"    Ring Cache 实测 (单层):     {actual_bytes / 1024:>8.1f} KB")


# ============================================================================
# 5. 质量专项：hit vs miss 的输出差异
# ============================================================================

def test_hit_miss_quality():
    from flashmlx.mac import (
        MACRingCache, MACDecodeWrapper,
        mac_partial_attention, merge_attention_states,
        mac_ring_match,
    )

    print("\n" + "=" * 65)
    print("5. Hit vs Miss 质量对比")
    print("=" * 65)
    print("   Miss = 完整 attention (ground truth)")
    print("   Hit  = cached prefix + partial suffix (MAC 近似)")
    print()

    N, Hq, Hkv, D, S = 1, 8, 2, 128, 2048
    M = 512

    mac = MACDecodeWrapper(
        max_requests=4, capacity=M,
        num_heads=Hq, num_kv_heads=Hkv, head_dim=D,
        threshold=0.3, band_r=256, window_left=256,
    )

    mx.random.seed(42)
    k = mx.random.normal((N, S, Hkv, D)).astype(mx.bfloat16)
    v = mx.random.normal((N, S, Hkv, D)).astype(mx.bfloat16)
    req_ids = mx.array([0], dtype=mx.int32)

    # 跑 20 步，收集每步的质量数据
    print(f"  {'Step':>4} {'Hit%':>6} {'CosSim':>8} {'RelErr':>8} {'MaxErr':>8}")
    print("  " + "-" * 42)

    for step in range(20):
        # 每隔几步重复 query 来制造 hit
        if step % 3 == 2 and step > 0:
            q = prev_q  # 重复 → 应该 hit
        else:
            q = mx.random.normal((N, Hq, D)).astype(mx.bfloat16)
            prev_q = q

        # Ground truth
        start_zero = mx.zeros((N, Hq), dtype=mx.int32)
        full_o, _ = mac_partial_attention(q, k, v, start_zero)
        mx.eval(full_o)

        # MAC output
        mac_o = mac(q, k, v, req_ids)
        mx.eval(mac_o)

        # 质量
        f = full_o.astype(mx.float32)
        m = mac_o.astype(mx.float32)
        diff = mx.abs(f - m)

        fn = mx.sqrt(mx.sum(f * f)).item()
        mn = mx.sqrt(mx.sum(m * m)).item()
        cos = mx.sum(f * m).item() / max(fn * mn, 1e-10)
        rel = mx.sqrt(mx.sum(diff * diff)).item() / max(fn, 1e-10)
        maxe = mx.max(diff).item()

        hr = mac.last_stats.hit_rate if mac.last_stats else 0

        mark = "✅" if cos > 0.99 else "⚠️" if cos > 0.95 else "❌"
        print(f"  {step:>4} {hr:>5.0%} {cos:>8.5f} {rel:>8.5f} {maxe:>8.5f} {mark}")


def main():
    print("MAC-Attention 实测报告")
    print(f"设备: {mx.default_device()}")
    print()

    test_output_quality()
    test_tg_speed()
    explain_pp_ttft()
    test_memory()
    test_hit_miss_quality()

    print("\n" + "=" * 65)
    print("测试完成")


if __name__ == "__main__":
    sys.path.insert(0, "src")
    sys.path.insert(0, "mlx-lm-source")
    main()
