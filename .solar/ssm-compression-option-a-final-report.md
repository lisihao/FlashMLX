# SSM State 压缩 Option A 完整报告

**日期**: 2026-03-21 16:00
**任务**: Task #53 - State-Memory 专用压缩算法 (Option A 优化)
**模型**: Qwen3.5-35B-A3B (6-bit)

---

## 背景回顾

### Phase 2 结果 (测试完成于 2026-03-21 15:00)

**唯一可行的方法**: Low-Rank SVD (rank=32)
- ✅ 输出质量正常（完整句子，语义连贯）
- ✅ 压缩比 2.39x (可接受)
- ❌ **速度极慢：1.35 tok/s vs baseline 25.57 tok/s (慢 19 倍)**

**其他方法全部失败**:
- ❌ Quantization (8-bit): 完全乱码（递推误差累积）
- ❌ Random Projection (dim=32): 重复乱码（重建不精确）

**瓶颈分析**:
- SVD 计算 O(Dv × Dk²) = O(128 × 192²) 必须在 CPU 上运行（MLX 限制）
- 每个 token 生成：64 heads × 30 layers = 1920 次 SVD
- 每次 SVD 需要 bfloat16 → float32 → bfloat16 转换

**监护人决策**: 选择 Option A - 优化 Low-Rank 速度

---

## Option A: 优化尝试（三个策略）

### 策略 1: 缓存 SVD 分解 ❌

**实现**: `cached_lowrank_compressor.py`
- **思路**: 如果 SSM state 结构稳定，可以重用 U/Vt，只计算 S
- **测试**: `test_cached_lowrank.py`

**结果**:
```
Original Low-Rank:  1.00 tok/s
Cached Low-Rank:    1.29 tok/s
Speedup:            1.29x
Cache hit rate:     2.5% (75/3060)
```

**失败原因**:
- SSM state 每个 token 都显著变化（递推更新机制）
- 只有 2.5% 的压缩操作命中缓存
- 97.5% 仍需要完整 SVD 计算
- 1.29x 加速远低于目标（需要 >5x 才有意义）

**结论**: ❌ **缓存策略不适用于 SSM state 的动态特性**

---

### 策略 2: 提高 Rank (交换压缩比换速度) ❌

**实现**: `test_rank_tradeoff.py`
- **思路**: rank 越高，U/S/Vt 重建越快（矩阵更大，但 SVD 可能更简单）
- **测试配置**: rank=32, 48, 64, 96

**结果**:

| Rank | 压缩比 | 速度 | 相对变化 |
|------|--------|------|----------|
| 32 | 2.39x | 1.27 tok/s | - |
| 48 | 1.85x | 1.29 tok/s | +1.6% |
| 64 | 1.54x | 1.27 tok/s | 0% |
| 96 | 1.21x | 1.25 tok/s | -1.6% |

**关键发现**:
- **所有 rank 的速度几乎完全相同**（变化 < 2%）
- 速度与 rank 无关，瓶颈不在矩阵乘法

**瓶颈分析**:
- SVD 计算复杂度：O(Dv × Dk²) = O(128 × 192²)
- **与 rank 无关**（SVD 计算不依赖目标 rank）
- 重建复杂度：O(Dv × rank × Dk) 虽然随 rank 增加，但这不是瓶颈

**结论**: ❌ **速度瓶颈在 SVD 计算本身，不是 rank 相关操作**

---

### 策略 3: Randomized SVD (GPU 加速) ❌

**实现**: `randomized_svd.py`
- **思路**: 使用 Randomized SVD 算法，将大部分计算移到 GPU
- **算法**: Halko et al. 2011 - "Finding Structure with Randomness"

**算法步骤**:
1. Random projection (GPU matmul) ✅
2. Power iteration (GPU matmul) ✅
3. QR decomposition (❌ MLX 不支持 GPU)
4. Small matrix projection (GPU matmul) ✅
5. Small matrix SVD (CPU，但矩阵更小) ✅

**测试结果**:

| 方法 | 矩阵大小 | 执行时间 | 设备分布 |
|------|----------|----------|----------|
| Exact SVD | 128×192 | 4.78ms | 100% CPU |
| Randomized SVD | 128×192 | 138.08ms | ~70% GPU + ~30% CPU |

**速度对比**: 慢 **28.9 倍** ❌

**失败原因**:
1. **MLX QR 也不支持 GPU**
   - 错误: `[linalg::qr] This op is not yet supported on the GPU`
   - QR 必须在 CPU 上，需要 CPU-GPU 数据传输

2. **小矩阵不适合 Randomized SVD**
   - (128, 192) 矩阵对于 Randomized SVD 太小
   - 算法开销（random projection + power iteration + QR + SVD）> 直接 SVD
   - Randomized SVD 优势在于大矩阵（10000×10000+）

3. **多次 CPU-GPU 传输**
   - Y (GPU) → CPU (QR) → GPU (matmul) → CPU (SVD)
   - 每次传输都有开销

**结论**: ❌ **Randomized SVD 不适用于 SSM state 的小矩阵场景**

---

## 根本原因分析

### MLX 线性代数限制

| 操作 | CPU 支持 | GPU 支持 | 备注 |
|------|----------|----------|------|
| SVD | ✅ | ❌ | Error: "not yet supported on the GPU" |
| QR | ✅ | ❌ | Error: "not yet supported on the GPU" |
| matmul | ✅ | ✅ | 完全支持 |
| norm | ✅ | ✅ | 完全支持 |

**结论**: MLX 的核心分解算法（SVD, QR）只支持 CPU 执行

### 为什么 Low-Rank 这么慢

**计算量分析** (每个 token):
```
30 SSM layers × 64 heads = 1920 个压缩操作

每次压缩:
1. SVD (Dv=128, Dk=192):    O(128 × 192²) ≈ 4.7M ops  [CPU]
2. dtype conversion:         bfloat16 → float32 → bfloat16
3. CPU-GPU 数据传输:        多次

每次解压缩:
1. 矩阵乘法 (U @ diag(S) @ Vt): O(Dv × rank × Dk) ≈ 0.8M ops  [GPU]
```

**瓶颈**:
- SVD 计算（4.7M ops × 1920 次 = 9.0B ops）全在 CPU
- CPU-GPU 传输频繁
- dtype 转换开销

**为什么其他策略都失败**:
1. **缓存**: SSM state 太动态，缓存命中率只有 2.5%
2. **提高 rank**: SVD 复杂度与 rank 无关
3. **Randomized SVD**: 小矩阵不适合，且 QR 也在 CPU

---

## Amdahl's Law 分析

假设优化各部分的效果：

| 优化方向 | 加速比 S | 占比 P | 总加速 |
|----------|----------|--------|--------|
| 缓存 (2.5% 命中) | 100x | 0.025 | 1.026x (+2.6%) |
| Randomized SVD | 0.03x | 1.0 | 0.03x (慢 30x) ❌ |
| GPU SVD (假设) | 10x | 0.9 | 5.26x (+426%) 🎯 |

**理论极限**:
- 如果 MLX 未来支持 GPU SVD，10x 加速 → 总性能 1.35 → 7.1 tok/s
- 仍然比 baseline 慢 3.6 倍
- 只有 SVD 加速 50x+ 才能接近 baseline 性能

**结论**: 在 MLX 当前架构下，**Option A 已走到死胡同**

---

## 三个策略对比

| 策略 | 理论基础 | 实际效果 | 根本问题 |
|------|----------|----------|----------|
| 缓存 SVD | SSM state 可能相对稳定 | 1.29x (+29%) | State 太动态，缓存命中率低 |
| 提高 Rank | 牺牲压缩比换速度 | 1.00x (无效) | SVD 瓶颈与 rank 无关 |
| Randomized SVD | 移到 GPU 加速 | 0.03x (慢 30x) | MLX QR 不支持 GPU + 小矩阵不适合 |

**共同点**: 都无法解决 **MLX SVD 只能在 CPU 运行** 的根本限制

---

## 最终结论

### Option A 状态: ❌ **失败**

**已尝试策略**: 3 个
**成功策略**: 0 个
**最佳结果**: 1.29x 加速（远低于目标）

### 根本瓶颈

```
┌──────────────────────────────────────────────────┐
│  MLX SVD = CPU-only                              │
│     ↓                                            │
│  Low-Rank 压缩 = 1920 次 SVD/token              │
│     ↓                                            │
│  无法利用 GPU 并行 = 慢 19 倍                     │
│                                                  │
│  除非 MLX 架构级改进，否则无解                    │
└──────────────────────────────────────────────────┘
```

### 为什么 Phase 2 选择了错误的方向

**Phase 1 误导**:
- Quantization 静态误差最低 (0.008)
- Phase 2 才发现递推误差累积导致完全失败

**Phase 2 决策**:
- Low-Rank 是唯一保持质量的方法
- 当时认为速度可以优化

**Option A 教训**:
- 低层库限制（MLX SVD CPU-only）无法通过算法层面解决
- 应该更早分析 MLX 线性代数能力

---

## 下一步建议

### Option B: 探索其他压缩方法 (Task #57)

已创建任务清单：

1. **基于学习的压缩** (研究导向)
   - 训练 Autoencoder 压缩 SSM state
   - 优势：可能找到更好的表示
   - 劣势：需要训练数据、推理开销

2. **低比特量化改进** (工程尝试)
   - Per-tensor 动态量化
   - 混合精度（重要维度 float16，其他 int8）
   - 优势：快速（纯整数/浮点运算）
   - 劣势：Phase 2 已证实量化不稳定

3. **稀疏化** (中风险)
   - Top-k 选择 + 位置编码
   - 结构化稀疏（block sparse）
   - 优势：减少计算量
   - 劣势：可能破坏递推稳定性

4. **混合策略** (保守方案)
   - 只压缩"冷" SSM 层（不常更新）
   - 保留"热" 层原样
   - 优势：部分压缩，风险可控
   - 劣势：压缩比有限

### Option C: 接受现状 (务实选择)

**分析**:
- Low-Rank 压缩比 2.39x，速度慢 19 倍
- **不压缩** 反而更快：25.57 tok/s
- 内存节省 (2.39x) vs 速度损失 (19x) 不划算

**建议**:
- 放弃 SSM state 压缩
- 回到 Task #55/56：诊断 Attention 层压缩问题
- AM (Attention-Memory) 压缩仍有潜力（10.4x 加速已验证）

---

## 技术债务

### 已创建的文件

| 文件 | 用途 | 是否保留 |
|------|------|----------|
| `ssm_state_compressor.py` | Low-Rank 基础实现 | ✅ 保留（Phase 2 唯一可用方法） |
| `cached_lowrank_compressor.py` | SVD 缓存优化 | ⚠️ 存档（失败但有参考价值） |
| `randomized_svd.py` | GPU 加速尝试 | ⚠️ 存档（失败但教育价值高） |
| `test_cached_lowrank.py` | 缓存测试 | ⚠️ 存档 |
| `test_rank_tradeoff.py` | Rank 权衡测试 | ⚠️ 存档 |

### 学到的教训

1. **低层库能力是硬约束**
   - 算法设计前必须验证底层支持
   - MLX SVD CPU-only 是不可逾越的限制

2. **静态误差 ≠ 递推稳定性**
   - Phase 1 静态测试不能预测 Phase 2 质量
   - SSM 的递推特性放大误差

3. **小矩阵优化策略不同**
   - Randomized SVD 适合大矩阵（10000×10000+）
   - (128, 192) 小矩阵直接 SVD 已经是最优

4. **缓存需要稳定性假设**
   - SSM state 每 token 显著变化
   - 动态数据不适合缓存策略

---

## 监护人决策点

**请选择下一步方向**:

- [ ] **Option B**: 探索 Task #57 的其他压缩方法（基于学习/量化改进/稀疏化/混合策略）
- [ ] **Option C**: 放弃 SSM 压缩，回到 Task #55/56（诊断 Attention 层问题）
- [ ] **Option D**: 等待 MLX 支持 GPU SVD（时间不确定，可能永远不支持）
- [ ] **Option E**: 其他方向（请说明）

**关键问题**:
1. 2.39x 压缩比但慢 19 倍是否值得继续？
2. 是否应该回到 AM (Attention-Memory) 优化主线？
3. 如果继续 SSM 压缩，愿意接受多大的速度损失？

---

*Option A 完整报告完成于: 2026-03-21 16:00*
*作者: Solar (Task #53)*
*结论: Option A 在 MLX 架构限制下无法成功，建议切换方向*
