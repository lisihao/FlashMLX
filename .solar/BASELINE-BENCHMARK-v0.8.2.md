# Baseline Benchmark: 全策略真实内存对比

> 版本: v0.8.2 | 模型: Qwen3-8B-MLX | 平台: M4 Pro 48GB
> 测试日期: 2026-03-28
> 内存测量: mx.metal.get_peak_memory() / get_active_memory()，扣除模型参数 8,300 MB

---

## 测试策略

| 策略 | 架构 | 描述 |
|------|------|------|
| standard | bf16 baseline | 无压缩，KVCache |
| triple_pq_am | Pipeline | PolarQuant(4bit) warm + AM cold 压缩 |
| triple_tq_am | Pipeline | TurboQuant(4bit) warm + AM cold 压缩 |
| scored_pq | Architecture D (Scored P2) | AM scored bf16 flat buffer |

---

## 一、16K 性能

| 策略 | PP tok/s | TG tok/s | TTOT ms | 1st Decode | PP Peak | PP Active | TG Mem | Flat |
|------|----------|----------|---------|------------|---------|-----------|--------|------|
| standard | 327.6 | 20.7 | 48,859 | 68ms | 8,399 MB | 6,901 MB | 2,268 MB | — |
| PQ+AM | 232.1 | 23.3 | 69,410 | 542ms | 17,569 MB | 14,389 MB | 1,214 MB | 8,326 |
| TQ+AM | 230.4 | 23.3 | 69,970 | 573ms | 17,848 MB | 14,956 MB | 1,186 MB | 8,326 |
| Scored P2 | 324.0 | 23.9 | 49,548 | 203ms | 8,441 MB | 7,067 MB | 925 MB | 5,746 |

### 16K Delta vs Standard

| 策略 | PP 速度 | TG 速度 | TTOT | PP Peak Δ | TG Mem Δ |
|------|---------|---------|------|-----------|----------|
| PQ+AM | -29.2% | +12.6% | +42.1% | **+109.2%** | **-46.5%** |
| TQ+AM | -29.7% | +12.8% | +43.2% | **+112.5%** | **-47.7%** |
| Scored P2 | -1.1% | +15.6% | +1.4% | **+0.5%** | **-59.2%** |

---

## 二、32K 性能

| 策略 | PP tok/s | TG tok/s | TTOT ms | 1st Decode | PP Peak | PP Active | TG Mem | Flat |
|------|----------|----------|---------|------------|---------|-----------|--------|------|
| standard | 264.6 | 16.2 | 122,425 | 87ms | 17,014 MB | 13,979 MB | 4,597 MB | — |
| PQ+AM | 189.5 | 19.5 | 190,143 | 19,293ms | 35,472 MB | 29,055 MB | 2,313 MB | 16,524 |
| TQ+AM | 181.6 | 19.2 | 197,884 | 19,555ms | 36,080 MB | 29,662 MB | 2,253 MB | 16,524 |
| Scored P2 | 262.2 | 18.5 | 123,954 | 486ms | 17,063 MB | 14,281 MB | 3,225 MB | 21,794 |

### 32K Delta vs Standard

| 策略 | PP 速度 | TG 速度 | TTOT | PP Peak Δ | TG Mem Δ |
|------|---------|---------|------|-----------|----------|
| PQ+AM | -28.4% | +20.6% | +55.3% | **+108.5%** | **-49.7%** |
| TQ+AM | -31.4% | +18.5% | +61.6% | **+112.1%** | **-51.0%** |
| Scored P2 | -0.9% | +14.4% | +1.2% | **+0.3%** | **-29.8%** |

---

## 三、关键发现

### 3.1 PP 内存翻倍问题（PQ/TQ+AM）

**现象**: PQ/TQ+AM 的 PP peak 内存是 standard 的 2 倍

**根因**: Attention 必须收到 bf16 数据，但 Pipeline 同时持有：
1. 量化存储（Warm PQ4 + Cold）— 持久化 cache
2. dequant bf16 结果 — 给 Attention

导致 bf16 + 量化数据并存 → 内存翻倍

**确认**: 查阅 git 历史（commit 664538b → 8bc032f），此行为在 lazy prefill 引入前后一致。
之前文档 "~72% PP savings" 是理论估算（PQ4 vs bf16 数据量），非 GPU 实测。

### 3.2 Scored P2 PP 内存与 Standard 持平

Scored P2 使用 lazy_prefill_threshold=65536，prefill 期间全部保持 bf16（不做量化），
所以 PP 内存 = standard。省内存完全靠 TG 阶段的 AM pruning。

### 3.3 TG 内存表现

| 策略 | 16K TG Mem | 32K TG Mem | 16K 节省 | 32K 节省 |
|------|-----------|-----------|---------|---------|
| standard | 2,268 MB | 4,597 MB | — | — |
| PQ+AM | 1,214 MB | 2,313 MB | -46% | -50% |
| TQ+AM | 1,186 MB | 2,253 MB | -48% | -51% |
| Scored P2 | 925 MB | 3,225 MB | -59% | -30% |

### 3.4 速度对比总结

- **PP 速度**: Scored P2 ≈ Standard >> PQ+AM ≈ TQ+AM (Pipeline 慢 30%)
- **TG 速度**: Scored P2 > TQ+AM ≈ PQ+AM > Standard
- **TTOT**: Scored P2 ≈ Standard << PQ+AM < TQ+AM (Pipeline 慢 40-60%)

---

## 四、综合评价

| 指标 | PQ+AM | TQ+AM | Scored P2 |
|------|-------|-------|-----------|
| PP 速度 | ❌ -30% | ❌ -30% | ✅ 持平 |
| PP 内存 | ❌ +110% (翻倍) | ❌ +112% (翻倍) | ✅ 持平 |
| TG 速度 | ✅ +13-21% | ✅ +13-19% | ✅ +14-16% |
| TG 内存 | ✅ -47~50% | ✅ -48~51% | ✅ -30~59% |
| TTOT | ❌ +42~55% | ❌ +43~62% | ✅ +1% |
| 质量 | ✅ PASS | ✅ PASS | ✅ PASS |

**结论**: Scored P2 在 PP 阶段全面优于 Pipeline（PQ/TQ+AM），TG 阶段各有优劣。
Pipeline 唯一优势是 32K TG 内存更低（2.3 GB vs 3.2 GB），但代价是 PP 内存翻倍 + TTOT 翻倍。

---

## 五、测试脚本

```
/tmp/test_baseline_memory.py
```

使用 `mx.metal.get_active_memory()` / `get_peak_memory()` 测量真实 GPU 内存，
扣除模型参数基线（加载后、创建 cache 前测量）。

---

## 六、优化方向

**PP 内存瓶颈**: Attention 需要 bf16 → prefill 期间无法单纯通过量化省内存

**可能的突破点**:
1. **Chunked Prefill + AM Eviction**: 分块处理 prefill，每块 AM 评分后 prune → 控制 PP 内存
2. **Quantized Attention**: 直接在 PQ4 数据上做 attention（需要 kernel 支持）
3. **Paged KV Cache**: 分页管理，按需加载（需要 MLX 底层支持）

---

*Baseline Benchmark Report v0.8.2*
*Generated: 2026-03-28*
*Model: Qwen3-8B-MLX on Apple M4 Pro 48GB*
