# SSM State 压缩方法对比测试结果

**日期**: 2026-03-21 14:45
**阶段**: Phase 1 - 原型实现和基础测试

---

## 测试配置

**模拟 Qwen3.5-35B SSM State**:
- Shape: `(B=1, Hv=64, Dv=128, Dk=192)`
- Total elements: 1,572,864
- Memory (FP16): 3.15 MB

**测试数据**: 随机正态分布（模拟真实 SSM state）

---

## 对比结果

| 方法 | 压缩比 | 重建误差 (mean abs) | 速度 | 推荐度 |
|------|--------|---------------------|------|--------|
| **Quantization (8-bit)** | 2.00x | **0.008** ⭐ | 最快 ⚡ | ✅ **推荐优先测试** |
| **Low-Rank (rank=32)** | 2.39x | 0.534 | 慢（SVD） | ⚠️ 备选 |
| **Random Projection (dim=32)** | 6.00x | 0.728 | 快 | ⚠️ 备选 |

---

## 详细分析

### 🥇 Method 1: Quantization (8-bit) - **最佳候选**

**优点**:
- ✅ **最低误差**: 0.008 (比其他方法低 60-90 倍!)
- ✅ **最快速度**: 几乎无开销（直接量化/反量化）
- ✅ **简单可靠**: 无复杂数学运算
- ✅ **质量可控**: Per-head quantization 保持精度

**缺点**:
- ❌ 压缩比较低 (2x)，但对于 30 个 SSM 层来说已经很可观

**推荐场景**:
- 首选方案，质量最优
- 如果 2x 压缩足够，直接使用此方法

**下一步**:
1. 在真实 Qwen3.5 SSM 层上测试
2. 验证生成质量是否保持
3. 如果成功，直接集成到 Heterogeneous Cache

---

### 🥈 Method 2: Low-Rank Approximation (rank=32) - **备选方案**

**优点**:
- ✅ 理论保证（保留最重要的奇异值）
- ✅ 压缩比略高 (2.39x)
- ✅ 可以通过调整 rank 平衡质量和压缩比

**缺点**:
- ❌ 误差较高 (0.534)
- ❌ SVD 计算开销大（每次压缩都需要）
- ❌ 需要在 CPU 上运行（MLX 限制）

**推荐场景**:
- 如果 Quantization 质量不够，尝试此方法
- 如果需要更高压缩比（降低 rank）

**优化方向**:
- 尝试不同的 rank 值（16/24/48）
- 缓存 SVD 结果（如果 state 结构稳定）

---

### 🥉 Method 3: Random Projection (dim=32) - **高压缩比备选**

**优点**:
- ✅ **最高压缩比** (6x)
- ✅ 速度快（只需矩阵乘法）
- ✅ 投影矩阵可共享（所有层使用同一个）

**缺点**:
- ❌ **最高误差** (0.728)
- ❌ 重建不精确（伪逆近似）
- ❌ 可能对 SSM 递推造成累积误差

**推荐场景**:
- 如果需要极高压缩比（6x+）
- 如果发现 SSM state 对误差不敏感

**优化方向**:
- 增加 target_dim (32 → 48/64) 降低误差
- 学习投影矩阵（如果有训练数据）

---

## 误差分析

**为什么 Quantization 误差这么低？**
- Per-head quantization 保留了每个 head 的动态范围
- 8-bit 精度对于 FP16 → INT8 转换足够
- 量化误差均匀分布，不会累积

**为什么 Low-Rank 和 Random Projection 误差高？**
- 随机数据没有低秩结构
- 真实 SSM state 可能有更好的结构（需要实际测试）
- rank=32 和 dim=32 可能太激进（可以增加）

**重要提醒**:
- 这些误差是在**随机数据**上测试的
- **真实 SSM state 可能有低秩或稀疏结构**，误差会更低
- 需要用真实模型验证

---

## Phase 2 计划：真实模型测试

### 🔥 Priority 1: Quantization 方法验证

**目标**: 验证 8-bit quantization 在真实 Qwen3.5 SSM 层上的效果

**步骤**:
1. 创建 layerwise ablation 测试脚本
2. 只压缩单个 SSM 层（如 layer 0）
3. 运行生成测试，对比质量
4. 如果成功，扩展到所有 30 个 SSM 层

**验收标准**:
- ✅ 生成质量 ≥ baseline 90%
- ✅ 无 shape mismatch 或递推错误
- ✅ 压缩开销 < 10% 总时间

### 备选: Low-Rank 和 Random Projection

如果 Quantization 质量不够，再测试其他方法

---

## 代码实现

**文件**: `mlx-lm-source/mlx_lm/compaction/ssm_state_compressor.py`

**类**:
- `LowRankStateCompressor` - Low-Rank SVD 压缩
- `RandomProjectionCompressor` - 随机投影压缩
- `QuantizationCompressor` - 量化压缩

**接口**:
```python
compressor = QuantizationCompressor(bits=8)
compressed = compressor.compress(state)  # Dict
reconstructed = compressor.decompress(compressed)
ratio = compressor.get_compression_ratio(state.shape)
```

**测试**: 运行 `python3 mlx-lm-source/mlx_lm/compaction/ssm_state_compressor.py`

---

## 下一步行动

1. ✅ **已完成**: 实现三种压缩方法并验证基础功能
2. 🔄 **进行中**: 分析结果，选择最优方案
3. 📋 **下一步**: 创建 layerwise ablation 测试，验证 Quantization 方法在真实 SSM 层上的效果
4. 📋 **后续**: 集成到 Heterogeneous Cache Manager

---

*Phase 1 测试报告创建于: 2026-03-21 14:45*
*作者: Solar (Task #53)*
*结论: Quantization (8-bit) 是最优候选，下一步在真实模型上验证*
