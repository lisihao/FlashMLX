# FlashMLX Meta-Harness: 端到端优化指南

> 基于 Meta-Harness 论文方法论（arXiv:2603.28052）的 FlashMLX 自动化优化框架

## 核心功能

1. **自动化超参数搜索** — 探索 KV cache 配置空间
2. **多目标优化** — 平衡质量（PPL）、速度（tok/s）、内存（MB）
3. **Pareto 前沿分析** — 找到非支配解集合
4. **可视化工具** — 生成 2D/3D 权衡图

---

## 快速开始

### 1. 运行完整优化

```bash
# 平衡模式（推荐）
python flashmlx_meta_harness.py /path/to/model --target balanced

# 速度优先
python flashmlx_meta_harness.py /path/to/model --target speed

# 内存优先
python flashmlx_meta_harness.py /path/to/model --target memory

# 质量优先
python flashmlx_meta_harness.py /path/to/model --target quality
```

### 2. 快速测试

```bash
# 小规模测试（3个配置）
python test_meta_harness.py
```

### 3. 可视化结果

```bash
# 生成 Pareto 前沿图表
python plot_pareto_frontier.py meta_harness_results.json
```

输出：
- `meta_harness_plots/pareto_ppl_vs_speed.png` — 质量 vs 速度
- `meta_harness_plots/pareto_memory_vs_speed.png` — 内存 vs 速度
- `meta_harness_plots/pareto_memory_vs_quality.png` — 内存 vs 质量
- `meta_harness_plots/pareto_3d.png` — 3D 可视化

---

## 优化目标详解

| 目标 | 搜索空间 | 适用场景 |
|------|----------|----------|
| **balanced** | PolarQuant + TurboAngle + Density | 通用场景，平衡三个指标 |
| **speed** | 低精度配置（2bit） | 推理速度优先，可牺牲内存/质量 |
| **memory** | 高压缩配置（PolarQuant + ultra_long） | 长上下文，内存受限 |
| **quality** | TurboAngle 多精度 | 质量优先，可牺牲速度 |

---

## 配置空间

### 当前支持的配置维度

| 维度 | 可选值 | 说明 |
|------|--------|------|
| `kv_cache` | `standard`, `triple_pq` | 缓存类型 |
| `kv_warm_bits` | `2`, `4`, `8` | PolarQuant 量化位数 |
| `strategy` | `scored_pq`, `turboangle` | 量化策略 |
| `n_k`, `n_v` | `(64,32)`, `(128,64)`, `(256,128)` | TurboAngle 角度bins |
| `density_mode` | `balanced`, `ultra_long`, `recall_first` | Route 0 密度路由 |

### 搜索空间大小

- **balanced**: ~15 配置
- **speed**: ~5 配置
- **memory**: ~10 配置
- **quality**: ~8 配置

---

## Python API

### 基础用法

```python
from flashmlx_meta_harness import FlashMLXMetaHarness

# 初始化
harness = FlashMLXMetaHarness(model_path="/path/to/model")

# 运行优化
best_config = harness.optimize(target='balanced', n_trials=10)

# 打印结果
harness.print_summary()

# 保存结果
harness.save_results("results.json")
```

### 自定义配置

```python
from flashmlx_meta_harness import BenchmarkConfig

# 手动定义配置
custom_configs = [
    BenchmarkConfig(
        kv_cache="triple_pq",
        kv_warm_bits=4,
        strategy="scored_pq",
        density_mode="ultra_long",
    ),
    BenchmarkConfig(
        kv_cache="triple_pq",
        strategy="turboangle",
        n_k=256,
        n_v=128,
    ),
]

# 测试自定义配置
for config in custom_configs:
    result = harness.benchmark_config(config)
    harness.results.append(result)

# 分析 Pareto 前沿
frontier = harness.get_pareto_frontier()
print(f"Pareto frontier: {len(frontier)} configs")
```

### 访问结果

```python
# 遍历所有结果
for result in harness.results:
    print(f"Config: {result.config}")
    print(f"  PPL: {result.perplexity:.4f}")
    print(f"  Speed: {result.tokens_per_sec:.1f} tok/s")
    print(f"  Memory: {result.peak_memory_mb:.1f} MB")
    print(f"  Pareto Score: {result.pareto_score:.4f}")
```

---

## 评分机制

### 单目标评分

```python
# 质量分数（越高越好）
quality_score = 1.0 / (1.0 + perplexity)

# 速度分数（相对于 baseline）
speed_score = tokens_per_sec / baseline_speed

# 内存分数（越低越好）
memory_score = 1.0 - (peak_memory_mb / 10000.0)
```

### Pareto 综合评分

```python
pareto_score = 0.5 * quality_score + 0.3 * speed_score + 0.2 * memory_score
```

**权重说明**：
- 质量（50%）— 最重要，保证输出质量
- 速度（30%）— 次重要，影响用户体验
- 内存（20%）— 相对次要，通常有容忍度

**自定义权重**：
编辑 `flashmlx_meta_harness.py` 中的 `pareto_score` 计算。

---

## Pareto 前沿解读

### 什么是 Pareto 前沿？

- **定义**：没有其他配置在所有目标上都更优的配置集合
- **意义**：这些配置代表了最优的权衡方案
- **选择**：根据具体场景从前沿中挑选

### 示例解读

```
PARETO FRONTIER CONFIGURATIONS
Config                         PPL   Speed      Memory     Pareto
--------------------------------------------------------------------------------
scored_pq 4bit balanced       1.5220   409.5/s    527.0MB    0.8542
turboangle K256V128           1.5220   350.2/s    890.0MB    0.8123
standard                      1.5220   269.5/s   4840.0MB    0.6891
```

**分析**：
- `scored_pq 4bit balanced` — 最优综合选择（高速 + 低内存 + 质量持平）
- `turboangle K256V128` — 质量保证方案（理论保证）
- `standard` — Baseline（无压缩）

**场景匹配**：
- **生产环境** → `scored_pq 4bit balanced`
- **研究/基准测试** → `turboangle K256V128`
- **调试/验证** → `standard`

---

## 实验建议

### 1. 初次运行

```bash
# 先运行快速测试
python test_meta_harness.py

# 验证工具链正常
python plot_pareto_frontier.py test_meta_harness_results.json
```

### 2. 完整优化

```bash
# 运行完整搜索（~20分钟）
python flashmlx_meta_harness.py /Volumes/toshiba/models/qwen3-8b-mlx \
    --target balanced \
    --output qwen3_8b_optimization.json

# 生成报告
python plot_pareto_frontier.py qwen3_8b_optimization.json
```

### 3. 模型对比

```bash
# 对多个模型运行相同优化
for model in qwen3-8b mistral-7b llama2-7b; do
    python flashmlx_meta_harness.py /path/to/$model \
        --target balanced \
        --output ${model}_results.json
done

# 对比分析
python compare_models.py qwen3-8b_results.json mistral-7b_results.json
```

---

## 扩展方向

### 1. 贝叶斯优化（计划中）

当前：网格搜索
未来：使用 `scikit-optimize` 实现贝叶斯优化

```python
from skopt import gp_minimize

def objective(params):
    config = BenchmarkConfig(
        kv_warm_bits=int(params[0]),
        density_scale=params[1],
    )
    result = harness.benchmark_config(config)
    return -result.pareto_score  # Minimize negative score

# Bayesian optimization
res = gp_minimize(objective, [(2, 8), (0.0, 3.0)], n_calls=20)
```

### 2. 多任务评估（计划中）

当前：单一 perplexity 测试
未来：MMLU / HellaSwag / Needle-in-haystack

```python
harness.add_task('mmlu', mmlu_dataset)
harness.add_task('needle', needle_test)
best_config = harness.optimize_multitask()
```

### 3. 自适应配置生成（计划中）

根据模型特性（层数、MLA/MHA、head_dim）自动推荐配置。

---

## 常见问题

### Q1: 为什么我的 Pareto 前沿只有 1 个配置？

**A**: 可能原因：
1. 搜索空间太小 → 增加 `n_trials` 或使用 `balanced` target
2. 所有配置都被某一个支配 → 这个配置就是绝对最优解
3. 评分权重不合理 → 调整 `pareto_score` 权重

### Q2: 内存测量不准确怎么办？

**A**: MLX 的 `get_peak_memory()` 可能不稳定，建议：
1. 运行多次取平均
2. 使用外部工具（如 `nvidia-smi` 等效工具）验证
3. 关注相对趋势而非绝对值

### Q3: 如何添加新的配置维度？

**A**:
1. 修改 `BenchmarkConfig` dataclass
2. 更新 `to_cache_kwargs()` 方法
3. 在 `get_search_space()` 中添加新配置

---

## 致谢

- **方法论来源**: Meta-Harness (arXiv:2603.28052)
- **应用领域**: FlashMLX KV cache optimization
- **实现**: Claude Opus 4.6 + 昊哥

---

**生成时间**: 2026-04-05
**版本**: v1.0
**状态**: ✅ 可用
