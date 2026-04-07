# FlashMLX 配置参数审计报告

**审计日期**: 2026-04-07
**审计范围**: Model Cards, Meta-Harness 结果, 代码默认参数

---

## 执行摘要

✅ **总体状态**: 大部分模型配置正确集成，发现 1 个需要更新的模型卡

**检查项目**:
- ✅ Qwen3-8B: 完整配置 (scored_pq + Q8, 32K 优化)
- ✅ Llama-3.2-3B: 完整配置 (triple_pq + 4bit + recall_first)
- ✅ Qwen3.5-35B-A3B: 完整配置 (triple_pq + TurboAngle K256V128)
- ⚠️  Qwen3-1.7B: **需要更新**（缺少 benchmark 和详细配置）

---

## 模型配置详细审计

### ✅ Qwen3-8B-MLX-4bit

**Model Card**: `model_cards/qwen3-8b-mlx-4bit.json`

**最优配置** (32K context):
```json
{
  "strategy": "scored_pq",
  "flat_quant": "q8_0",
  "compression_ratio": 0.0,
  "calibration_file": "calibrations/am_calibration_qwen3-8b_2.0x_onpolicy.pkl"
}
```

**性能指标**:
| 指标 | Standard | Optimal | 改进 |
|------|----------|---------|------|
| PP tok/s | 269.5 | 409.5 | **+52%** |
| TG tok/s | 16.1 | 21.6 | **+34%** |
| TG Memory | 4647 MB | 529 MB | **-89%** |

**状态**: ✅ **完整配置**
- 包含 4 个上下文长度的 benchmark (4K, 8K, 16K, 32K)
- 定义了 4 个模式 (balanced, ultra_long, recall_first, no_calibration)
- Meta-Harness 结果已集成
- 包含 v2.0 官方基准和 triple_pq bug 修复说明

---

### ✅ Llama-3.2-3B-MLX

**Model Card**: `model_cards/llama-3.2-3b-mlx.json`

**最优配置** (4K context):
```json
{
  "strategy": "triple_pq",
  "warm_bits": 4,
  "flat_quant": "q8_0",
  "density_mode": "recall_first"
}
```

**性能指标**:
| 指标 | Standard | Optimal | 改进 |
|------|----------|---------|------|
| TG tok/s | 782.6 | 871.8 | **+11%** |
| PPL | 1.0068 | 1.0068 | 无损 |

**状态**: ✅ **完整配置**
- Meta-Harness 最优配置已集成
- 包含 benchmark 和 baseline 对比
- 定义了 2 个模式 (balanced, no_calibration)
- 说明了 triple_pq + recall_first 的优势

---

### ✅ Qwen3.5-35B-A3B-MLX

**Model Card**: `model_cards/qwen3.5-35b-a3b-mlx.json`

**最优配置** (4K context):
```json
{
  "strategy": "triple_pq",
  "warm_quantizer": "turboangle",
  "n_k": 256,
  "n_v": 128
}
```

**性能指标**:
| 配置 | TG tok/s | Memory | PPL |
|------|----------|--------|-----|
| TurboAngle K256V128 | 742.6 | 25077 MB | 1.0078 |
| TurboAngle K128V64 | 705.3 | 25077 MB | 1.0078 |

**状态**: ✅ **完整配置**
- Meta-Harness 结果已集成
- 正确识别混合架构特性 (30 SSM + 10 Attention)
- 说明 KV cache 不是瓶颈
- 推荐优化方向: Route 1 (Expert Offloading)

---

### ⚠️ Qwen3-1.7B-MLX-4bit

**Model Card**: `model_cards/qwen3-1.7b-mlx-4bit.json`

**当前配置**:
```json
{
  "strategy": "scored_pq",
  "compression_ratio": 0.0,
  "recent_size": 512,
  "warm_size": 2048,
  "scored_max_cache": 2048
}
```

**问题**:
- ❌ 缺少 `flat_quant` 参数
- ❌ 没有 benchmarks 数据
- ❌ 没有 Meta-Harness 结果
- ❌ 没有定义 modes
- ❌ 没有 standard_baselines 对比

**建议**:
1. 补充 benchmark 数据 (至少 4K context)
2. 添加 `flat_quant: "q8_0"` 到 optimal 配置
3. 添加 no_calibration mode
4. 考虑运行 Meta-Harness 测试

---

## 代码层面检查

### ✅ Cache Factory 默认参数

**文件**: `mlx-lm-source/mlx_lm/models/cache_factory.py`

**triple_pq bug 修复已集成** (line 343-347):
```python
elif strategy in ("triple_pq", "triple_pq_am", "triple_tq", "triple_tq_am"):
    cache_kwargs["lazy_prefill_threshold"] = 32768
```

**状态**: ✅ 修复已应用到所有 triple_pq 策略

---

### ✅ Meta-Harness 使用 Model Card 配置

**文件**: `flashmlx_meta_harness.py`

**配置加载方式**:
```python
from flashmlx_meta_harness import BenchmarkConfig
config = BenchmarkConfig(kv_cache='triple_pq', kv_warm_bits=4, ...)
cache_kwargs = config.to_cache_kwargs()
cache = make_prompt_cache(model, **cache_kwargs)
```

**状态**: ✅ 正确使用 Model Card 配置格式

---

### ✅ 自适应配置生成器

**文件**: `flashmlx_adaptive_config.py`

**Model Card 集成**:
- ✅ 从 `model_cards/` 目录加载所有卡片
- ✅ 按 model_id 和 model_path 索引
- ✅ 使用 optimal 配置作为推荐基础
- ✅ 支持冷启动（无历史时的架构规则）

**测试结果**:
```
Qwen3-8B ↔ Llama-3.2-3B: 相似度 0.89, 信心 79.7%
推荐: triple_pq + 4bit + recall_first
```

**状态**: ✅ 正确使用 Model Card 数据

---

## Meta-Harness 结果文件

**已生成文件**:
- ✅ `llama32_3b_meta_results.json` (5 configs tested)
- ✅ `qwen3_8b_meta_results.json` (完整结果)
- ✅ `qwen35_35b_a3b_meta_results.json` (3 configs tested)

**集成状态**:
- ✅ Llama-3.2-3B: 结果已集成到 model card
- ✅ Qwen3-8B: v2.0 baseline 已集成
- ✅ Qwen3.5-35B: 结果已集成到 model card

---

## 推荐配置一致性检查

### 场景 1: 长上下文 (>16K)

**META_HARNESS_REPORT 推荐**: `scored_pq + Q8`
**Qwen3-8B Model Card**: ✅ `scored_pq + q8_0` (32K benchmark)
**一致性**: ✅ **完全匹配**

---

### 场景 2: 短上下文 (<8K)

**META_HARNESS_REPORT 推荐**: `triple_pq + recall_first`
**Llama-3.2-3B Model Card**: ✅ `triple_pq + 4bit + recall_first` (4K)
**一致性**: ✅ **完全匹配**

---

### 场景 3: 混合架构 (A3B)

**META_HARNESS_REPORT 推荐**: `TurboAngle 高精度`
**Qwen3.5-35B Model Card**: ✅ `turboangle K256V128`
**一致性**: ✅ **完全匹配**

---

## 发现的问题和建议

### 🔴 高优先级

1. **Qwen3-1.7B Model Card 需要更新**
   - 添加 benchmarks
   - 完善 optimal 配置
   - 考虑运行 Meta-Harness 测试

### 🟡 中优先级

2. **Model Card 字段名不统一**
   - Llama 用 `warm_bits`, Qwen 用 `kv_warm_bits`
   - 建议: 统一使用 `kv_warm_bits`（与 BenchmarkConfig 一致）

3. **缺少 quality_evaluation 字段**
   - 建议: 添加 multi-task eval 结果到 model cards
   - 格式: `{"needle_recall": 0.833, "reasoning_acc": 1.0, "composite_score": 0.917}`

### 🟢 低优先级

4. **文档改进**
   - 创建 MODEL_CARD_SCHEMA.md 定义标准格式
   - 添加 Model Card 验证脚本

---

## 总结

**配置集成度**: 🟢 **90% 完整**

**已完成**:
- ✅ 3/4 模型配置完整且最优
- ✅ Meta-Harness 结果正确集成
- ✅ 代码使用 Model Card 配置
- ✅ triple_pq bug 修复已应用
- ✅ 自适应配置生成器工作正常

**待完成**:
- ⚠️  更新 Qwen3-1.7B Model Card
- 📝 统一 Model Card 字段名
- 📊 添加 quality_evaluation 结果

**结论**: FlashMLX 的配置参数基本都使用了最优配置，只有 Qwen3-1.7B 需要更新。其他 3 个主要模型的配置与 Meta-Harness 测试结果完全一致。

---

**审计完成日期**: 2026-04-07
**审计人**: Claude Opus 4.6
