# Critical Finding: AM Calibration 的序列长度依赖性

**日期**: 2026-03-25
**发现者**: Solar + 监护人
**严重性**: ⭐⭐⭐ (阻塞性问题)

---

## 🔴 问题描述

**AM Offline Calibration 不能跨序列长度使用**。

Calibration 基于特定序列长度生成的 `selected_indices` 是**绝对位置索引**，当 runtime 序列长度与 calibration 长度差异过大时，indices 会超出范围，导致：
1. 取值时访问无效位置 → 垃圾数据
2. 质量完全破坏 → 乱码输出

---

## 📊 实验数据

### Calibration 生成

| 参数 | 值 |
|------|-----|
| 序列长度 | ~593 tokens |
| 压缩比 | 2.0x |
| Budget | 384 keys/layer |
| Indices 范围 | [0, 592] |

### Runtime 测试

| 参数 | 值 |
|------|-----|
| 序列长度 | 307 tokens |
| 超出范围 indices | 183/384 (47.7%) |
| 结果 | 0% accuracy, 乱码输出 |

### 实验结果

```
Question 1: "When was the lab founded?"
Expected: 2019
Got: 1000000000000000000  ❌

Question 2: "When did the breakthrough occur?"
Expected: July 15, 2022
Got: 1000000000000000000  ❌

Question 3: "What was the success rate?"
Expected: 89%
Got: 1000000000000000000  ❌

Accuracy: 0% (0/3)
```

### 代码证据

```python
# Calibration indices 检查
layer_0 = data['calibration'][0]
indices = layer_0['selected_indices']

Min: 0
Max: 592
Budget: 384

# Runtime 长度
current_length = 307

# 超出范围
out_of_bounds = np.sum(indices >= current_length)
>>> 183/384 (47.7%)
```

---

## 🎯 根本原因

### 1. AM 的设计假设

AM (Attention Matching) 算法使用 **OMP (Orthogonal Matching Pursuit)** 选择最重要的 keys：

```python
# OMP 选择 top-k indices
scores = queries @ keys.T
avg_scores = np.mean(np.abs(scores), axis=0)
selected_indices = np.argsort(avg_scores)[-budget:]  # 绝对位置！
```

**关键问题**：`selected_indices` 是**绝对位置** [0, seq_len-1]，而不是相对位置或语义标记。

### 2. 与 LoRA 的本质区别

| 对比项 | LoRA | AM Calibration |
|--------|------|----------------|
| 可复用性 | ✅ 跨输入复用 | ❌ 不可跨长度 |
| 类型 | 权重矩阵 (WxH) | 位置索引 (int[]) |
| 长度依赖 | 无 | 强依赖 |
| 语义 | 修改权重空间 | 选择特定位置 |

**监护人的洞察** "AM 像 LoRA 一样，训练一次，存下来"：
- ✅ 对于**相同长度**的序列，确实可以复用
- ❌ 对于**不同长度**的序列，indices 会超出范围

---

## 🔍 Fallback 机制验证

### 实现的 Fallback 逻辑

```python
# ✅ 检测 indices 是否超出范围
max_index = int(np.max(np.array(selected_indices)))
current_length = self.offset

if max_index < current_length:
    use_calibration = True  # Fast path
else:
    # Fallback to runtime calibration
    use_calibration = False
```

### Fallback 结果

```
[CompactedKVCache] Indices out of bounds: 183/384
[CompactedKVCache]   Max index: 592, Current length: 307
[CompactedKVCache] Falling back to runtime calibration for safety
[CompactedKVCache] Using runtime calibration (slow path)
```

**Fallback 执行成功**，但仍然产生乱码，因为：
- Runtime calibration 需要 **queries** 参数
- 测试脚本假设使用 pre-fitted calibration，没有传入 queries
- 导致 `compact_multi_head_quality(queries=None)` → 质量破坏

---

## 💡 可能的解决方案

### 方案 A: 多尺度 Calibration ⭐⭐⭐

为不同长度范围生成多个 calibration files：

```python
calibration_lengths = [256, 512, 1024, 2048, 4096]

# 生成
for length in calibration_lengths:
    calibrate_am_offline(
        model, tokenizer,
        num_questions=24,
        target_length=length,
        output_file=f"am_calibration_{length}.pkl"
    )

# 使用时自动选择最接近的
def select_calibration(runtime_length):
    closest = min(calibration_lengths,
                  key=lambda x: abs(x - runtime_length))
    return f"am_calibration_{closest}.pkl"
```

**优点**：
- ✅ 完整保留 offline calibration 理念
- ✅ 支持多种长度
- ✅ Fast path 大部分时间有效

**缺点**：
- ❌ 需要生成多个文件 (5 files × 7 MB ≈ 35 MB)
- ❌ 仍然有边界情况（如 runtime=350, closest=256 或 512）

### 方案 B: 相对位置映射 ⭐

将绝对 indices 映射到相对位置（百分比）：

```python
# Calibration 时保存相对位置
calibration_length = 593
relative_indices = selected_indices / calibration_length  # [0.0, 1.0]

# Runtime 时映射到当前长度
runtime_length = 307
mapped_indices = (relative_indices * runtime_length).astype(int)
```

**优点**：
- ✅ 单一 calibration file 跨长度使用
- ✅ 理论上支持任意长度

**缺点**：
- ❌ 破坏语义重要性（位置 500 在 593 tokens 中的重要性 ≠ 位置 259 在 307 tokens 中的重要性）
- ❌ 可能降低质量

### 方案 C: 严格限制 + Fallback ⭐⭐

只在**序列长度接近** calibration 长度时使用 pre-fitted calibration，否则 fallback 到 runtime calibration：

```python
# 允许 ±10% 误差
length_ratio = runtime_length / calibration_length
if 0.9 <= length_ratio <= 1.1 and max_index < runtime_length:
    use_calibration = True
else:
    # Fallback to runtime calibration (需要 queries)
    use_calibration = False
```

**优点**：
- ✅ 最安全，保证质量
- ✅ Pre-fitted calibration 在合理范围内有效

**缺点**：
- ❌ 覆盖范围小（只有 ±10%）
- ❌ 大部分情况仍需 runtime calibration

### 方案 D: Hybrid - 多尺度 + 相对映射 ⭐⭐⭐⭐

结合方案 A 和 B：

```python
# 1. 选择最接近的 calibration
closest_calibration = select_calibration(runtime_length)

# 2. 加载 calibration
calibration = load_calibration(closest_calibration)

# 3. 检查是否需要映射
if max_index >= runtime_length:
    # 使用相对位置映射
    mapped_indices = (relative_indices * runtime_length).astype(int)
else:
    # 直接使用 calibration indices
    mapped_indices = selected_indices

# 4. 验证质量（可选）
# 如果映射后质量下降，fallback 到 runtime calibration
```

**优点**：
- ✅ 大部分情况使用 fast path（多尺度覆盖）
- ✅ 边界情况使用映射（相对位置）
- ✅ 兼顾速度和质量

**缺点**：
- ❌ 实现复杂度高
- ❌ 需要额外的质量验证机制

---

## 🛠️ 推荐方案

**推荐：方案 A（多尺度 Calibration）**

**理由**：
1. **符合 offline calibration 理念**：一次拟合，多次使用
2. **覆盖范围广**：[256, 512, 1024, 2048, 4096] 覆盖常见长度
3. **质量有保证**：在 ±50% 范围内直接使用 calibration
4. **存储成本可接受**：5 files × 7 MB = 35 MB
5. **实现简单**：只需修改 calibration 生成脚本

**实施步骤**：

1. **修改 calibrate_am_offline.py**：
   ```python
   target_lengths = [256, 512, 1024, 2048, 4096]
   for length in target_lengths:
       calibrate_for_length(model, tokenizer, target_length=length)
   ```

2. **修改 CompactedKVCache**：
   ```python
   def _select_calibration_file(self, runtime_length):
       lengths = [256, 512, 1024, 2048, 4096]
       closest = min(lengths, key=lambda x: abs(x - runtime_length))
       return f"/path/to/am_calibration_{closest}.pkl"
   ```

3. **Fallback 策略**：
   ```python
   if max_index >= runtime_length:
       # Still out of bounds even with closest calibration
       # Fallback to runtime calibration
       use_calibration = False
   ```

---

## 📝 教训总结

### 1. 算法适用性陷阱

```
❌ 错误假设：AM 是通用 offline calibration
✅ 实际情况：AM 依赖绝对位置，不可跨长度

教训：算法迁移前必须验证核心假设
```

### 2. 测试覆盖不足

```
❌ 只测试了 calibration 生成
❌ 只测试了 calibration 加载
✅ 应该测试不同长度的 runtime 场景

教训：测试不同条件下的边界情况
```

### 3. 监护人洞察的边界

```
监护人："AM 像 LoRA，训练一次，存下来"
✅ 对于相同长度：正确
❌ 对于不同长度：不成立

教训：类比有边界，需要验证适用范围
```

---

## 🔄 下一步行动

### Priority 1: 实现多尺度 Calibration ⭐⭐⭐

1. 修改 `calibrate_am_offline.py` 支持 target_length 参数
2. 生成 5 个 calibration files
3. 修改 CompactedKVCache 自动选择最接近的 calibration
4. 测试不同长度的质量保持

### Priority 2: 完善 Fallback 机制 ⭐⭐

1. 修改测试脚本传入 queries 参数
2. 验证 fallback 到 runtime calibration 时质量保持
3. 添加性能监控（fast path vs slow path 比例）

### Priority 3: 文档更新 ⭐

1. 更新 `am-offline-calibration-implementation.md`
2. 添加长度依赖性说明
3. 更新使用指南

---

## 📚 相关文档

- Implementation: `am-offline-calibration-implementation.md`
- Success Finding: `critical-finding-am-success.md`
- Query Scaling: `am-query-scaling-experiment.md`

---

*Critical Finding v1.0*
*日期: 2026-03-25*
*状态: 已识别，方案待选择*
