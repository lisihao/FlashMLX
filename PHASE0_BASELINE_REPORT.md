# Phase 0 Baseline Report - 文本模型性能基准

**测试日期**: 2026-04-09
**平台**: M4 Max 64GB
**模型**: Qwen3-8B-MLX (4-bit)
**目的**: 建立 VLM 迁移前的性能红线，防止后续迁移导致文本模型性能回退

---

## 执行摘要

✅ **FlashMLX v2.0 文本模型基准已建立**

- 测试了 4 个上下文长度 (4K, 8K, 16K, 32K)
- ✅ 验证了 Route 0 (Density Router) - 4 种产品模式全部正常
- ✅ 验证了 Route 3 (KV Cache 压缩) - scored_pq + Q8 工作正常
- 所有配置输出完全一致 (质量无损)
- 长上下文优化收益显著 (32K: PP +58%, TG +34%, 内存 -89%)
- 产品模式收益: ultra_long (+25% TG, -65% Mem), recall_first (+27% TG, -70% Mem)

---

## 基准性能数据

### Qwen3-8B-MLX-4bit Performance

| Context | Strategy | PP tok/s | TG tok/s | TTFT (s) | PP Peak | TG Mem | 改进 |
|---------|----------|----------|----------|----------|---------|--------|------|
| **4K** | standard | 417.6 | 25.8 | 9.8 | 736M | 619M | baseline |
| 4K | optimal (scored_pq+Q8) | 415.0 | 24.2 | 9.9 | 270M | 273M | TG -6%, Mem -56% |
| **8K** | standard | 364.5 | 23.5 | 22.5 | 1391M | 1193M | baseline |
| 8K | optimal | 379.9 | 24.7 | 21.6 | 272M | 242M | **PP +4%, TG +5%, Mem -80%** |
| **16K** | standard | 311.7 | 19.6 | 52.6 | 2637M | 2344M | baseline |
| 16K | optimal | 398.0 | 24.8 | 41.2 | 264M | 241M | **PP +28%, TG +27%, Mem -90%** |
| **32K** | standard | 257.6 | 15.3 | 127.2 | 4840M | 4647M | baseline |
| 32K | optimal | 407.4 | 20.5 | 80.4 | 526M | 529M | **PP +58%, TG +34%, Mem -89%** |

### 关键指标定义

- **PP tok/s**: Prefill 速度 (tokens/sec) - 首次处理 prompt 的速度
- **TG tok/s**: Token Generation 速度 (tokens/sec) - 生成新 token 的速度
- **TTFT**: Time To First Token (秒) - 从输入到第一个输出 token 的延迟
- **PP Peak**: Prefill 阶段峰值内存 (MB)
- **TG Mem**: Token Generation 阶段稳定内存 (MB)

---

## Route 3 验证 (KV Cache 压缩)

### 工作原理

Route 3 使用 **scored_pq** 策略 + **Q8 量化**：
- AM (Attention Mask) 校准文件: `am_calibration_qwen3-8b_2.0x_onpolicy.pkl`
- 压缩比: 自适应 (compression_ratio=0.0)
- Recent window: 512 tokens
- Budget: 256 tokens (2.0x compression)

### 32K Context Eviction 日志

```
[Scored Prefill Evict #1] 4096 → 1702 tokens (kept 1190 hot + 512 recent, ratio=3.0x)
[Scored Prefill Evict #2] 3750 → 1698 tokens (kept 1186 hot + 512 recent, ratio=3.0x)
...
[Scored Prefill Evict #8] 3726 → 2700 tokens (kept 2188 hot + 512 recent, ratio=1.5x)
...
[Scored Prefill Evict #15] 6775 → 4723 tokens (kept 4211 hot + 512 recent, ratio=1.5x)
```

**验证结果**: ✅ **Route 3 正常工作**
- 15 轮 prefill eviction
- 压缩比从 3.0x 降到 1.5x (自适应)
- 最终保留 4723 tokens (14% of 32K)
- **输出与 standard 完全一致**

---

## 质量验证

### 输出一致性检查

所有 4 个上下文长度、2 种策略 (standard vs optimal) 的输出**完全一致**：

**4K 输出**:
```
" exclusively human. The implications of this technology span across multiple sec"
```

**8K 输出**:
```
" intelligence has been one of the most transformative technological advances of "
```

**16K 输出**:
```
"1st century. From natural language processing to computer vision, AI systems are"
```

**32K 输出**:
```
". The implications of this technology span across multiple sectors including hea"
```

✅ **结论**: scored_pq + Q8 在所有上下文长度下保持质量无损

---

## 性能趋势分析

### 1. 短上下文 (4K)

- **optimal 略慢于 standard** (TG -6%)
- **原因**: KV cache 还不是瓶颈，压缩开销 > 收益
- **内存收益**: -56%
- **建议**: 4K 以下使用 standard 或 no_calibration mode

### 2. 中上下文 (8K)

- **optimal 开始超越 standard** (PP +4%, TG +5%)
- **内存收益**: -80%
- **拐点**: 8K 是 scored_pq 开始有收益的临界点

### 3. 长上下文 (16K, 32K)

- **optimal 大幅超越 standard**:
  - 16K: PP +28%, TG +27%
  - 32K: PP +58%, TG +34%
- **内存收益**: -89% ~ -90%
- **建议**: 16K+ 必须使用 scored_pq

### 4. KV Cache 瓶颈分析

| Context | Standard TG Mem | 占总内存比例 | KV 是否瓶颈 |
|---------|-----------------|-------------|------------|
| 4K | 619M | ~12% (假设总 5GB) | ❌ 不是 |
| 8K | 1193M | ~24% | ⚠️  接近 |
| 16K | 2344M | ~47% | ✅ 是 |
| 32K | 4647M | ~93% | ✅ 严重 |

---

## VLM 迁移质量红线

基于 Phase 0 baseline，为 VLM 迁移设定以下质量红线：

### 文本模型回归测试 (Phase 1)

**必须通过的指标** (使用 Qwen3-8B-MLX):

| 测试场景 | 指标 | 当前值 | 允许回退 | 红线 |
|---------|------|--------|---------|------|
| 4K standard | TG tok/s | 25.8 | ≤5% | **≥24.5** |
| 8K standard | TG tok/s | 23.5 | ≤5% | **≥22.3** |
| 16K standard | TG tok/s | 19.6 | ≤5% | **≥18.6** |
| 32K standard | TG tok/s | 15.3 | ≤5% | **≥14.5** |
| 32K optimal | TG tok/s | 20.5 | ≤5% | **≥19.5** |
| 32K optimal | TG Mem | 529M | ≤10% | **≤582M** |

**输出一致性**:
- ✅ 所有 scored_pq 压缩配置的输出必须与 standard 完全一致
- ✅ 使用相同的 seed 和 prompt，token 序列必须完全匹配

---

## 其他优化路由验证 (待完成)

### ✅ Route 3: Triple Layer KV Cache (已验证)

- **状态**: 正常工作，15 轮 eviction，输出一致
- **配置**: scored_pq + Q8
- **收益**: 32K context 下 TG +34%, Mem -89%

### ✅ Route 0: Density Router (已验证)

**测试命令**:
```bash
python3 benchmarks/bench_density_modes.py /Volumes/toshiba/models/qwen3-8b-mlx \
  --contexts 32768 --tg-tokens 50
```

**测试结果** (32K context):

| Mode | TG tok/s | TG Mem | vs baseline | 压缩比 |
|------|----------|--------|-------------|--------|
| baseline (scored_pq) | 20.6 | 527M | - | 1.5x |
| balanced (scale=0.0) | 20.6 | 527M | identical | 1.5x |
| ultra_long (scale=1.5) | 25.9 | 185M | **+25% TG, -65% Mem** | 5.0x-10.0x |
| recall_first (scale=2.5) | 26.1 | 156M | **+27% TG, -70% Mem** | 10.0x |

**验证点**:
- [x] balanced mode (density_scale=0.0) - ✅ 与 baseline 完全一致
- [x] ultra_long mode (density_scale=1.5) - ✅ 高压缩比 (5-10x), TG +25%
- [x] recall_first mode (density_scale=2.5) - ✅ 极限压缩 (10x), TG +27%
- [x] 输出一致性 - ✅ **ALL IDENTICAL**

**关键发现**:
- Route 0 通过 density_scale 控制压缩级别 (log2 space)
- ultra_long: 压缩比 5x-10x，内存降低 65%，性能提升 25%
- recall_first: 最大压缩 10x，内存降低 70%，性能提升 27%
- 所有模式输出完全一致，质量无损

### ⏳ Route 1: Expert Offloading (待验证)

**适用模型**: Qwen3.5-35B-A3B (MoE)

**测试计划**:
```bash
python3 benchmarks/bench_expert_offload.py /Volumes/toshiba/models/qwen3.5-35b-mlx
```

**验证点**:
- [ ] 专家参数正确 offload 到磁盘
- [ ] TG 速度保持可接受范围 (>10 tok/s)
- [ ] 内存占用降低 (目标: <20GB)

### ⏳ Route 4: Chunked Prefill (待验证)

**测试计划**:
```bash
python3 benchmarks/bench_pp_long.py --contexts 32768,65536
```

**验证点**:
- [ ] PP 峰值内存降低
- [ ] PP 速度不显著下降 (<10%)
- [ ] 输出一致性

### ⏳ Route 5: Context Recall (待验证)

**测试计划**:
```bash
python3 benchmarks/bench_recall_d.py
```

**验证点**:
- [ ] h^(0) capture 开销为零
- [ ] recall_first mode 输出一致性
- [ ] Needle-in-Haystack 召回测试 (≥4/6)

---

## 下一步行动

### 立即可做 (Task #11 后续)

1. **验证剩余优化路由** (Route 0, 1, 4, 5)
   ```bash
   python3 benchmarks/bench_density_modes.py
   python3 benchmarks/bench_recall_d.py
   ```

2. **记录所有路由的基准性能**
   - 更新 PHASE0_BASELINE_REPORT.md
   - 创建回归测试清单

3. **完成 Task #11**
   - 标记为 completed
   - 更新 VLM_MIGRATION_TEST_PLAN.md

### Phase 1 准备

- [ ] 创建 Phase 1 分支: `git checkout -b phase1-mlx-lm-upgrade`
- [ ] Git merge 最新 MLX-LM
- [ ] 运行文本模型回归测试套件 (80 cases)
- [ ] 验证所有性能指标在红线之上

---

## 附录

### 测试环境

```
Platform: M4 Max 64GB
macOS: Darwin 25.3.0
Python: 3.14.3
MLX: (built from mlx-lm-source)
FlashMLX: v2.0 (commit: 待填充)
```

### 测试命令

```bash
# Phase 0 Baseline 完整 benchmark
python3 benchmarks/bench_card.py \
  /Volumes/toshiba/models/qwen3-8b-mlx \
  --contexts 4096,8192,16384,32768 \
  --tg-tokens 100

# 输出日志
tests/phase0_baseline_full.log
```

### Model Card 信息

```json
{
  "model_id": "qwen3-8b-mlx-4bit",
  "optimal": {
    "strategy": "scored_pq",
    "flat_quant": "q8_0",
    "compression_ratio": 0.0,
    "calibration_file": "calibrations/am_calibration_qwen3-8b_2.0x_onpolicy.pkl",
    "recent_size": 512,
    "warm_size": 2048,
    "scored_max_cache": 2048
  }
}
```

### 已知问题

1. **Model Card 验证错误** (已修复):
   - `modes.no_calibration` 缺少 `density_scale` 字段
   - 修复: 添加 `"density_scale": 0.0`

2. **4K context 性能轻微下降**:
   - scored_pq 在短上下文有轻微开销 (TG -6%)
   - 不影响 VLM 迁移（VLM 主要用于长上下文）

---

**报告生成时间**: 2026-04-09 17:49
**生成方式**: 自动化 benchmark + 人工分析
**下次更新**: Phase 1 完成后
