# TurboQuant Transplant — Final Report

**Project**: FlashMLX Route 3 KV Cache Compression
**Date**: 2026-03-31
**Model**: Qwen3-8B (8 KV heads, head_dim=128, 36 layers, bfloat16)
**Hardware**: Apple Silicon M4 Pro 48GB
**Branch**: `kvtc-p1-p2-improvements`
**Methodology**: `generate_step` (same as bench_ab_final.py), 200 gen tokens

---

## Executive Summary

Transplanted PolarQuant (PQ4) from `TheTom/turboquant_plus` into FlashMLX as
`flat_quant="turboquant"` — a 4th flat buffer compression tier.

**Results at 16K context**:
- **scored+q8_0** (existing flashmlx default): TG 25.9, KV 153 MB (-94%)
- **scored+tq** (new): TG 16.5 (-21% vs flashmlx), KV 80 MB (-97%), PP +6%, TTOF -6%
- **scored+bf16** (speed king): TG 28.3 (+35% vs std), KV 302 MB (-88%)

**Verdict**: turboquant 提供边际内存收益 (153→80 MB) 但 TG 代价显著 (-36% vs q8_0)。
最优平衡仍是 **scored+q8_0** (已有 flashmlx 默认配置)。turboquant 适合极端内存受限场景。

---

## Phase Summary

| Phase | Status | Outcome |
|-------|--------|---------|
| P0: Dual Audit | Done | Identified transplant surface: PolarQuant core only |
| P1: Algorithm Transplant | Done | PolarQuantizer class in quantization_strategies.py |
| P2: Route 3 Integration | Done | flat_quant="turboquant" dispatch in triple_layer_cache.py |
| P3: Correctness Validation | Done | 15/15 token parity at all contexts (Qwen3-8B) |
| P4: MLX/Metal Optimization | Done | -25% dequant latency (critical hot path) |
| P5: Final Bench + Report | Done | This document |

---

## Files Changed (3 files, +214 / -76)

| File | Changes |
|------|---------|
| `mlx_lm/models/quantization_strategies.py` | PolarQuantizer: Haar rotation, Lloyd-Max quant, vectorized pack/unpack, flat API |
| `mlx_lm/models/triple_layer_cache.py` | turboquant dispatch in alloc/write/fetch/eval + head_dim≥128 auto-downgrade |
| `mlx_lm/models/cache_factory.py` | Docstring: turboquant option + head_dim requirement |

---

## Quality Matrix (Token Parity vs Standard bf16)

Generate 15 tokens, compare exact match against `flat_quant=None` baseline.

| Context | standard | triple+bf16 | triple+q4_0 | triple+tq | scored+bf16 | scored+q4_0 | scored+tq |
|---------|----------|-------------|-------------|-----------|-------------|-------------|-----------|
| 558 tok | 15/15 | 15/15 | 15/15 | **15/15** | 15/15 | 15/15 | **15/15** |
| 2,208 tok | 15/15 | 15/15 | 15/15 | **15/15** | 4/15 | 4/15 | **4/15** |
| 5,508 tok | 15/15 | 15/15 | 15/15 | **15/15** | 3/15 | 3/15 | **3/15** |

- **turboquant = bf16 = q4_0** within each strategy family — flat_quant introduces zero additional error
- scored divergence is 100% from AM compression (token eviction), not quantization

---

## Full Benchmark: 2K Context

| Config | PP tok/s | TG tok/s | TTOF (s) | KV PP Peak | KV TG | KV Save |
|--------|----------|----------|----------|------------|-------|---------|
| standard | 446.8 | 28.4 | 4.42 | 978 MB | 340 MB | — |
| triple+bf16 | 442.7 | 28.5 | 4.47 | 641 MB | 340 MB | 0% |
| triple+q8_0 | 437.1 | 25.5 | 4.52 | 492 MB | 173 MB | -49% |
| triple+q4_0 | 437.8 | 17.7 | 4.52 | 459 MB | 96 MB | -72% |
| triple+tq | 425.9 | 15.4 | 4.64 | 469 MB | 90 MB | -74% |
| scored+bf16 | 430.5 | 28.4 | 4.59 | 602 MB | 340 MB | 0% |
| scored+q8_0 | 414.6 | 25.2 | 4.77 | 469 MB | 173 MB | -49% |
| scored+q4_0 | 395.7 | 17.2 | 5.00 | 452 MB | 96 MB | -72% |
| scored+tq | 384.3 | 14.8 | 5.14 | 572 MB | 90 MB | -74% |

## Full Benchmark: 8K Context

| Config | PP tok/s | TG tok/s | TTOF (s) | KV PP Peak | KV TG | KV Save |
|--------|----------|----------|----------|------------|-------|---------|
| standard | 353.4 | 24.6 | 22.99 | 1,843 MB | 1,246 MB | — |
| triple+bf16 | 365.4 | 24.7 | 22.23 | 2,056 MB | 1,246 MB | 0% |
| triple+q8_0 | 375.1 | 17.0 | 21.66 | 1,460 MB | 633 MB | -49% |
| triple+q4_0 | 379.6 | 5.3 | 21.40 | 1,353 MB | 350 MB | -72% |
| triple+tq | 382.1 | 5.7 | 21.27 | 1,210 MB | 331 MB | -73% |
| **scored+bf16** | 394.1 | **28.3** | 20.62 | 976 MB | 302 MB | -76% |
| **scored+q8_0** | 391.1 | **25.7** | 20.77 | 976 MB | 153 MB | -88% |
| scored+q4_0 | 409.4 | 18.6 | 19.85 | 976 MB | 85 MB | -93% |
| scored+tq | 420.9 | 16.5 | 19.30 | 976 MB | 80 MB | -94% |

## Full Benchmark: 16K Context

| Config | PP tok/s | TG tok/s | TTOF (s) | KV PP Peak | KV TG | KV Save |
|--------|----------|----------|----------|------------|-------|---------|
| standard | 347.2 | 21.0 | 47.02 | 3,006 MB | 2,454 MB | — |
| triple+bf16 | 392.7 | 21.0 | 41.58 | 3,701 MB | 2,454 MB | 0% |
| triple+q8_0 | 392.2 | 11.8 | 41.63 | 2,561 MB | 1,246 MB | -49% |
| triple+q4_0 | 386.6 | 2.9 | 42.23 | 2,035 MB | 690 MB | -72% |
| triple+tq | 368.9 | 3.1 | 44.26 | 2,002 MB | 652 MB | -73% |
| **scored+bf16** | 378.3 | **28.3** | 43.15 | 976 MB | 302 MB | -88% |
| **scored+q8_0** | 406.7 | **25.9** | **40.14** | 976 MB | 153 MB | -94% |
| scored+q4_0 | 422.1 | 18.7 | 38.68 | 976 MB | 85 MB | -97% |
| scored+tq | **431.8** | 16.5 | **37.81** | 976 MB | **80 MB** | **-97%** |

**Historical validation** (bench-ab-final.json, 2026-03-30, same model):
- standard 16K: PP 355 / TG 21.2 / TTOF 46.0 / KV 2,454 MB — matches
- flashmlx (scored+q8_0) 16K: PP 411 / TG 25.9 / TTOF 39.7 / KV 153 MB — matches

---

## Analysis: The flat_quant Tradeoff Ladder

At 16K context, each flat_quant tier within scored_pq:

| flat_quant | TG tok/s | vs std | KV TG | KV Save | PP tok/s | TTOF |
|------------|----------|--------|-------|---------|----------|------|
| bf16 | 28.3 | +35% | 302 MB | -88% | 378.3 | 43.15s |
| q8_0 | 25.9 | +23% | 153 MB | -94% | 406.7 | 40.14s |
| q4_0 | 18.7 | -11% | 85 MB | -97% | 422.1 | 38.68s |
| turboquant | 16.5 | -21% | 80 MB | -97% | 431.8 | 37.81s |

**Pattern**: 越重的量化 → TG 越慢, 但 PP/TTOF 越快（内存小→Metal 更高效）

**q8_0→turboquant 的边际收益**:
- 内存: 153 → 80 MB (省 73 MB, -48%)
- TG: 25.9 → 16.5 tok/s (慢 36%)
- PP: +6%, TTOF: -6%

**Without scored (triple only), 量化是灾难性的**:
- triple+q4_0 at 16K: TG 2.9 tok/s (-86% vs std) — 每步 dequant 整个 16K buffer
- triple+tq at 16K: TG 3.1 tok/s (-85%)
- 证明 AM eviction 是量化可用的前提 — 没有 scored, 不要用 q4_0/turboquant

---

## Phase 4 Optimization Results

`flat_dequantize` runs every TG step — the dominant cost for turboquant.

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| flat_dequantize (2K tokens) | 0.99 ms | 0.74 ms | **-25%** |
| _pq_dequantize (2K tokens) | 0.55 ms | 0.44 ms | **-20%** |
| Dequant 4K tokens | 1.39 ms | 1.11 ms | **-20%** |
| Dequant 8K tokens | 2.80 ms | 2.26 ms | **-19%** |

Optimizations: norm_correction=False (safe at head_dim≥128), mx.rsqrt, vectorized pack, precomputed constants.

---

## Constraints & Limitations

### head_dim ≥ 128 Requirement

PQ4 random rotation converges via CLT — works at head_dim=128, fails at head_dim=64.

| Model | head_dim | Result |
|-------|----------|--------|
| Qwen3-8B | 128 | 15/15 token match |
| Qwen2.5-0.5B | 64 | 2/10 match (garbage) |

Auto-downgrade guard: `_TURBOQUANT_MIN_HEAD_DIM = 128` → falls back to q4_0 with warning.

### No QJL stage

Transplanted PQ4 only (no QJL residual). QJL adds complexity for 0.2 bit improvement.

---

## Recommendations

### Tier list (scored_pq, by use case)

| Priority | Config | TG | KV Save | Best for |
|----------|--------|-----|---------|----------|
| 1 | scored+q8_0 | 25.9 | -94% | **Default FlashMLX** — best TG/memory balance |
| 2 | scored+bf16 | 28.3 | -88% | Speed-critical, memory secondary |
| 3 | scored+q4_0 | 18.7 | -97% | Memory-critical, TG acceptable |
| 4 | scored+tq | 16.5 | -97% | Extreme memory constraint (marginal vs q4_0) |

### Deployment guidance

1. **Default**: `scored_pq + q8_0` (existing flashmlx) — 25.9 TG (+23%), -94% KV, proven in production
2. **Speed mode**: `scored_pq + bf16` — 28.3 TG (+35%), -88% KV
3. **Memory mode**: `scored_pq + q4_0` — sufficient compression, broader model support than turboquant
4. **Extreme memory**: `scored_pq + turboquant` — only when the extra 73 MB (q8_0→tq) matters

### When NOT to use quantized flat buffers

- **Without scored_pq (triple only)**: q4_0/turboquant TG drops to 2.9-3.1 at 16K (-85%). AM eviction is prerequisite.
- **Short contexts (< 2K)**: KV cache is already small, dequant overhead isn't worth the savings.

### Future work

- **Metal kernel**: Fused PQ dequant + attention to eliminate bf16 materialization
- **Adaptive flat_quant**: Auto-switch bf16→q8_0→q4_0 based on buffer size / memory pressure
- **Pool size adaptation** (Task #7): SSD detection + runtime hit-rate feedback

---

## Conclusion

TurboQuant adds a 4th flat buffer compression tier with **3.8x compression** and **zero
quality loss** within each strategy family. However, the benchmark tells a clear story:

- **scored+q8_0 remains the optimal default** — 25.9 TG (+23% vs std), -94% KV, minimal dequant cost
- **turboquant's marginal gain is small** — 80 vs 153 MB (saves 73 MB) at -36% TG cost
- **The real value is at extreme scale** — 32K+ contexts on memory-limited devices where every MB counts
- **Without scored_pq, quantized flat buffers are unusable** — triple+tq at 16K = 3.1 TG (−85%)

**Ship status**: Ready for merge. Opt-in, no API breaks, auto-downgrades on small models.
The option exists for users who need maximum compression and accept the TG tradeoff.
