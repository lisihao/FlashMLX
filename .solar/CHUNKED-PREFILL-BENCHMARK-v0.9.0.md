# Chunked Prefill Benchmark: Scored P2 PP Memory Optimization

> 版本: v0.9.0 | 模型: Qwen3-8B-MLX | 平台: M4 Pro 48GB
> 测试日期: 2026-03-28
> 内存测量: mx.get_peak_memory() / get_active_memory()，扣除模型参数 8,300 MB
> 优化: Chunked Prefill (chunk=512) + Streaming AM Eviction (max_cache=4096)

---

## 测试矩阵

| 策略 | 描述 |
|------|------|
| standard | 单次 PP，无压缩 (baseline) |
| std_chunked | 分块 PP (chunk=512)，无压缩 (公平对照) |
| scored_chunked | 分块 PP + AM Eviction (优化目标) |

---

## 一、完整结果

| 策略 | Ctx | PP tok/s | TG tok/s | TTOT ms | PP Peak | PP Active | TG Mem | Evictions | Quality |
|------|-----|----------|----------|---------|---------|-----------|--------|-----------|---------|
| standard | 16K | 330.6 | 21.0 | 48,419 | 8,399 MB | 7,026 MB | 2,268 MB | — | PASS |
| std_chunked | 16K | 299.9 | 19.7 | 53,356 | 2,785 MB | 2,301 MB | 2,268 MB | — | PASS |
| scored_chunked | 16K | 367.5 | 26.9 | 43,559 | 1,131 MB | 576 MB | 602 MB | 5 | PASS |
| standard | 32K | 264.6 | 16.3 | 122,456 | 16,990 MB | 14,207 MB | 4,572 MB | — | PASS |
| std_chunked | 32K | 248.6 | 16.1 | 130,280 | 5,079 MB | 4,608 MB | 4,572 MB | — | PASS |
| scored_chunked | 32K | 369.1 | 27.0 | 87,761 | 1,131 MB | 526 MB | 530 MB | 19 | PASS |

---

## 二、Delta vs Standard (单次 PP)

### 16K

| 策略 | PP 速度 | TG 速度 | TTOT | PP Peak Δ | TG Mem Δ |
|------|---------|---------|------|-----------|----------|
| std_chunked | -9.3% | -6.1% | +10.2% | **-66.8%** | -0.0% |
| scored_chunked | **+11.1%** | **+28.4%** | **-10.0%** | **-86.5%** | **-73.5%** |

### 32K

| 策略 | PP 速度 | TG 速度 | TTOT | PP Peak Δ | TG Mem Δ |
|------|---------|---------|------|-----------|----------|
| std_chunked | -6.0% | -1.1% | +6.4% | **-70.1%** | +0.0% |
| scored_chunked | **+39.5%** | **+66.2%** | **-28.3%** | **-93.3%** | **-88.4%** |

---

## 三、关键发现

### 3.1 PP 内存恒定 — O(1) 内存

PP peak = **1,131 MB**，无论 16K 还是 32K。

原因：`max_cache=4096` + `chunk_size=512` 限制了物理 cache 上限。
每次 cache 超过 4096 tokens → AM eviction 裁剪回 ~2000-3400 tokens。

**这意味着理论上 PP 内存与上下文长度无关 — O(1) 内存复杂度。**

### 3.2 PP 速度反而更快

| Ctx | standard PP | scored_chunked PP | 加速比 |
|-----|-------------|-------------------|--------|
| 16K | 330.6 tok/s | 367.5 tok/s | +11.1% |
| 32K | 264.6 tok/s | 369.1 tok/s | +39.5% |

原因：
1. 小 cache → Attention 计算量 O(chunk × cache) 而非 O(N²)
2. 分块释放中间态 → 减少内存压力 → GPU 运行更高效
3. 32K 加速更明显：standard 的 O(N²) attention 在 32K 严重退化

### 3.3 TG 速度大幅提升

| Ctx | standard TG | scored_chunked TG | 加速比 |
|-----|-------------|-------------------|--------|
| 16K | 21.0 tok/s | 26.9 tok/s | +28.4% |
| 32K | 16.3 tok/s | 27.0 tok/s | +66.2% |

原因：更小的 KV cache → 每步 attention 更快。

32K 特别惊人：standard 的 TG 降到 16.3 (因为 4.6 GB cache)，
而 scored_chunked 维持 27.0 (只有 530 MB cache)。

### 3.4 Adaptive Ratio 正确工作

- 16K: 5 次 eviction，全部 3.0x ratio
- 32K: 前 5 次 3.0x (prefill_tokens_seen ≤ 16K)，后 14 次 1.5x (>16K)

### 3.5 纯 Chunked (无 Eviction) 的局限性

std_chunked 仅节省 Logits 和中间态：
- 16K: PP peak -66.8%（不错，但 TG 内存不变）
- 32K: PP peak -70.1%（同上）

scored_chunked 额外裁剪 KV cache → PP 和 TG 同时受益。

---

## 四、与 v0.8.2 基准对比

### 16K 全策略横向对比

| 策略 | PP Peak | TG Mem | PP tok/s | TG tok/s | TTOT |
|------|---------|--------|----------|----------|------|
| standard (v0.8.2) | 8,399 MB | 2,268 MB | 327.6 | 20.7 | 48.9s |
| PQ+AM (v0.8.2) | 17,569 MB | 1,214 MB | 232.1 | 23.3 | 69.4s |
| Scored P2 原版 (v0.8.2) | 8,441 MB | 925 MB | 324.0 | 23.9 | 49.5s |
| **Scored Chunked (v0.9.0)** | **1,131 MB** | **602 MB** | **367.5** | **26.9** | **43.6s** |

### 32K 全策略横向对比

| 策略 | PP Peak | TG Mem | PP tok/s | TG tok/s | TTOT |
|------|---------|--------|----------|----------|------|
| standard (v0.8.2) | 17,014 MB | 4,597 MB | 264.6 | 16.2 | 122.4s |
| PQ+AM (v0.8.2) | 35,472 MB | 2,313 MB | 189.5 | 19.5 | 190.1s |
| Scored P2 原版 (v0.8.2) | 17,063 MB | 3,225 MB | 262.2 | 18.5 | 124.0s |
| **Scored Chunked (v0.9.0)** | **1,131 MB** | **530 MB** | **369.1** | **27.0** | **87.8s** |

---

## 五、综合评价

| 指标 | vs Standard | vs PQ+AM Pipeline | vs Scored P2 原版 |
|------|-------------|-------------------|-------------------|
| PP Peak (16K) | **-86.5%** | **-93.6%** | **-86.6%** |
| PP Peak (32K) | **-93.3%** | **-96.8%** | **-93.4%** |
| TG Mem (16K) | **-73.5%** | **-50.4%** | **-34.9%** |
| TG Mem (32K) | **-88.4%** | **-77.1%** | **-83.6%** |
| PP Speed (16K) | +11.1% | +58.3% | +13.4% |
| PP Speed (32K) | +39.5% | +94.7% | +40.8% |
| TG Speed (16K) | +28.4% | +15.5% | +12.6% |
| TG Speed (32K) | +66.2% | +38.5% | +45.9% |
| TTOT (16K) | -10.0% | -37.2% | -12.1% |
| TTOT (32K) | -28.3% | -53.8% | -29.2% |
| Quality | PASS | PASS | PASS |

**结论**: Scored Chunked 在所有指标上全面碾压所有其他策略。

---

## 六、技术实现

### 核心改动

1. **triple_layer_cache.py**: +2 新方法, 修改 3 处
   - `_scored_prefill_evict()`: prefill 期间 AM 评分 + 驱逐
   - `_promote_to_flat_buffer()`: 已评分的 cache 直接升级到 flat buffer
   - `offset` property: 返回 `_prefill_tokens_seen` 确保 RoPE 正确

2. **cache_factory.py**: +2 参数
   - `scored_prefill_chunk_evict=True`
   - `scored_prefill_max_cache=4096`

### 内存模型

```
PP peak = max(hot_budget + chunk_size) × bytes_per_token × layers + model_activations(chunk_size)
        ≈ 4608 × 144 KB + chunk_activations
        ≈ 648 MB + ~483 MB
        ≈ 1,131 MB  ← 实测值
```

### 参数

- `CHUNK_SIZE = 512` (prefill 分块大小)
- `scored_prefill_max_cache = 4096` (触发 eviction 阈值)
- Adaptive ratio: 3.0x @ ≤16K, 1.5x @ >16K

---

## 七、Eviction 详情

### 16K (5 次 eviction, 3.0x ratio)

```
#1: 4608 → 1872 tokens (kept 1360 hot + 512 recent)
#2: 4432 → 2038 tokens (kept 1526 hot + 512 recent)
#3: 4598 → 2204 tokens (kept 1692 hot + 512 recent)
#4: 4252 → 1858 tokens (kept 1346 hot + 512 recent)
#5: 4418 → 2024 tokens (kept 1512 hot + 512 recent)
→ Final flat buffer: 3694 tokens
```

### 32K (19 次 eviction, 3.0x → 1.5x)

```
#1-5:  3.0x ratio (prefill_tokens_seen ≤ 16K)
#6-19: 1.5x ratio (prefill_tokens_seen > 16K)
→ Final flat buffer: 3326 tokens
```

---

## 八、测试脚本

```
/tmp/test_scored_chunked.py
```

---

*Chunked Prefill Benchmark Report v0.9.0*
*Generated: 2026-03-28*
*Model: Qwen3-8B-MLX on Apple M4 Pro 48GB*
*Optimization: Chunked Prefill + Streaming AM Eviction*
