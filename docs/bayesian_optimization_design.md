# Bayesian Optimization for FlashMLX Meta-Harness

## 概述

将 Meta-Harness 从网格搜索（Grid Search）升级到贝叶斯优化（Bayesian Optimization），减少搜索次数，提高效率。

**目标**: 用 5-7 次试验达到网格搜索 10+ 次试验的效果

## 背景

### 网格搜索的局限

**当前方法** (flashmlx_meta_harness.py):
- 预定义所有配置组合
- 逐个测试，无智能筛选
- `balanced` target → 10 个配置
- `memory` target → 甚至更多

**问题**:
1. **浪费时间**: 每个配置需要 5-10 秒
2. **无法学习**: 不利用之前试验的结果
3. **维度爆炸**: 参数越多，组合数指数增长

### 贝叶斯优化的优势

**核心思想**:
- 用概率模型 (Gaussian Process / TPE) 拟合 **参数 → 性能** 的函数
- 根据不确定性和潜在收益，智能建议下一个配置
- 利用每次试验的信息，快速收敛到最优解

**预期收益**:
```
网格搜索: 10 trials × 7 sec = 70 sec
贝叶斯优化: 5-7 trials × 7 sec = 35-49 sec

节省时间: 30-50%
```

## 实现方案

### 算法选择: Tree-structured Parzen Estimator (TPE)

**Why TPE**:
1. **无需高斯过程**: 不依赖 `scikit-optimize` 等重型库
2. **处理混合参数**: 支持 categorical + continuous
3. **高效**: scipy + numpy 即可实现
4. **成熟**: Hyperopt, Optuna 等框架都用 TPE

**TPE 原理**:

```
维护两个分布:
- l(x): 导致"好"结果的参数分布
- g(x): 导致"差"结果的参数分布

获取函数 (Acquisition Function):
  EI(x) = (g(x) - l(x)) / g(x)

选择下一个点:
  x* = argmax EI(x)
```

**分割阈值**: γ = 0.25 (前 25% 为 "good")

### 参数空间定义

```python
# Categorical parameters
kv_cache: ['standard', 'triple_pq']
strategy: ['polarquant', 'turboangle', None]
density_mode: [None, 'balanced', 'ultra_long', 'recall_first']
n_k: [None, 64, 128, 256]
n_v: [None, 32, 64, 128]

# Integer parameters
kv_warm_bits: [2, 3, 4]  # PolarQuant only
```

### 采样策略

**1. 冷启动阶段** (前 3 次试验):
- 随机采样
- 目的: 探索参数空间

**2. 学习阶段** (第 4 次试验起):
- 基于 TPE 建议
- Categorical: 从 "good" 配置的经验分布采样
- Continuous/Integer: 从 "good" 配置的高斯分布采样

**3. 约束处理**:
- `strategy='polarquant'` → 需要 `kv_warm_bits`, 忽略 `n_k/n_v`
- `strategy='turboangle'` → 需要 `n_k/n_v`, 忽略 `kv_warm_bits`
- 无效组合 → 跳过，重新采样

## 代码架构

### 新增模块

**flashmlx_bayesian_optimizer.py**:
```
HyperparameterSpace       - 参数空间定义
TreeStructuredParzenEstimator  - TPE 优化器
BayesianMetaHarness       - 包装 FlashMLXMetaHarness
```

**bench_grid_vs_bayesian.py**:
```
run_grid_search()         - 网格搜索基准
run_bayesian_optimization() - 贝叶斯优化
compare_methods()         - 对比分析
```

### 集成方式

**Option 1: 独立使用**
```bash
python3 flashmlx_bayesian_optimizer.py \
  /path/to/model \
  --target balanced \
  --n-trials 7 \
  --output results.json
```

**Option 2: 对比评估**
```bash
python3 bench_grid_vs_bayesian.py \
  /path/to/model \
  --target balanced \
  --bayesian-trials 7
```

**Option 3: 集成到 Meta-Harness**
```python
# Future: add --optimizer flag
python3 flashmlx_meta_harness.py \
  /path/to/model \
  --optimizer bayesian \
  --n-trials 7
```

## 评估指标

### 主要指标

| 指标 | 网格搜索 | 贝叶斯优化 | 目标 |
|------|----------|-----------|------|
| **试验次数** | 10-15 | 5-7 | -40% |
| **总时间** | 70-105 秒 | 35-49 秒 | -50% |
| **Pareto Score** | 基准 | ≥95% 基准 | 质量相当 |

### 成功标准

✅ **效率提升**: 试验次数减少 30% 以上
✅ **质量保持**: Pareto score 不低于网格搜索的 95%
✅ **通用性**: 在 3 种架构上都有效 (Pure Transformer, Hybrid)

## 实验计划

### Phase 1: 概念验证 (已完成)
- [x] 实现 TPE 算法
- [x] 实现 BayesianMetaHarness
- [x] 创建对比脚本

### Phase 2: 单模型测试 (当前)
- [ ] Llama-3.2-3B: 运行贝叶斯优化 (5 trials)
- [ ] 对比网格搜索 vs 贝叶斯优化
- [ ] 分析收敛速度

### Phase 3: 多模型验证
- [ ] Qwen3-8B: 重复测试
- [ ] Qwen3.5-35B-A3B: 混合架构测试
- [ ] 统计显著性检验

### Phase 4: 文档和集成
- [ ] 更新 META_HARNESS_REPORT.md
- [ ] 添加 CLI --optimizer 选项
- [ ] Model Card 集成

## 技术细节

### TPE 实现

**l(x) 和 g(x) 的建模**:

```python
# Categorical: 经验频率分布
good_values = [cfg['param'] for cfg in good_configs]
counts = Counter(good_values)
probs = [counts[val] / len(good_values) for val in choices]
sample = np.random.choice(choices, p=probs)

# Continuous/Integer: 高斯分布
mean = np.mean(good_values)
std = np.std(good_values)
sample = np.random.normal(mean, std)
```

**分割阈值**:
```python
gamma = 0.25  # Top 25%
threshold = np.quantile(scores, 1.0 - gamma)
good_configs = [cfg for cfg, score in observations if score >= threshold]
```

### 依赖项

**必需**:
- numpy
- scipy

**可选**:
- matplotlib (可视化)
- scikit-optimize (未来高级功能)

## 局限性和未来改进

### 当前局限

1. **小数据集**: 5-7 次试验，TPE 优势有限
2. **单目标**: 未充分利用 Pareto 前沿
3. **无并行**: 串行试验，不支持批量采样

### 未来改进方向

1. **多目标贝叶斯优化** (MOBO)
   - 直接优化 Pareto 前沿
   - 使用 NSGA-II 或 MOEA/D

2. **批量采样** (Batch Acquisition)
   - q-EI (q-Expected Improvement)
   - 并行试验，加速搜索

3. **迁移学习** (Transfer Learning)
   - 利用其他模型的优化历史
   - 加速新模型的优化

4. **自适应 γ**
   - 根据收敛情况动态调整分割阈值
   - Early stage: γ=0.5 (探索), Late stage: γ=0.15 (利用)

## 参考文献

1. **Meta-Harness**: arXiv:2603.28052
2. **TPE**: Bergstra et al., "Algorithms for Hyper-Parameter Optimization", NeurIPS 2011
3. **Hyperopt**: http://hyperopt.github.io/hyperopt/
4. **Optuna**: https://optuna.org/

## 总结

贝叶斯优化为 Meta-Harness 提供了智能采样策略，预期可减少 30-50% 的试验次数，同时保持配置质量。TPE 实现简洁，无重型依赖，易于集成和维护。

**下一步**: 完成 Llama-3.2-3B 测试，验证预期收益。
