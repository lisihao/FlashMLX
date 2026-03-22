# Critical Finding: NNLS 求解器缺失导致质量崩溃

**发现时间**: 2026-03-22 19:25 (重启前 5 分钟)
**严重程度**: CRITICAL - 阻塞真实模型使用
**发现方式**: `test_real_kv_cache.py` 测试真实 Qwen3-8B KV cache

---

## 症状

**压缩成功,质量崩溃**:
```
✅ 压缩: 92 -> 23 tokens (4x)
✅ C1 shape: (23, 128)
✅ beta shape: (23,)
✅ C2 shape: (23, 128)
❌ Cosine similarity: 0.374331 vs baseline 0.931600 (CATASTROPHIC)
```

**对比**:
- 合成 KV cache 测试: ≥80% cosine similarity ✅
- 真实 KV cache 测试: 37% cosine similarity ❌

---

## 根本原因

### 问题 1: Beta 计算用线性化近似,不是 NNLS

**代码位置**: `src/flashmlx/cache/compaction_algorithm.py:153-161`

**错误实现**:
```python
# Linearization: beta ≈ log(target_attn / base_attn)
base_attn = mx.softmax(attn_scores_C1, axis=-1)  # (n, t)
eps = 1e-8
log_ratio = mx.log((target_attn + eps) / (base_attn + eps))  # (n, t)
beta = mx.mean(log_ratio, axis=0)  # (t,)  ← 这是近似,不是真正的 NNLS!
```

**正确实现** (论文要求):
```python
# Goal: 使 softmax(Q@C1^T + beta) ≈ target_attn
# 求解: min_{beta ≥ 0} || exp(Q@C1^T + beta) - unnormalized_target ||^2

# 使用 NNLS 求解器:
from mlx_lm.compaction.solvers import nnls_auto
beta = nnls_auto(M, y, quality='high')
```

### 问题 2: NNLS 求解器缺失

**引用位置**: `tests/compaction/test_nnls.py:15`
```python
from mlx_lm.compaction.solvers import nnls_clamped, nnls_pgd, nnls_auto
```

**现状**:
```bash
$ find . -name "solvers.py"
(没有结果!)
```

**需要实现**:
- `nnls_clamped(M, y, lower_bound)` - Clamped Least Squares (快速,近似)
- `nnls_pgd(M, y, lower_bound, max_iters)` - Projected Gradient Descent (精确)
- `nnls_auto(M, y, quality)` - 自动选择 (根据质量要求)

---

## 影响

### 合成 KV cache 为什么 OK?

合成 cache 是**随机 Gaussian 分布**:
- Attention weights 相对均匀
- log-ratio 近似误差小
- 质量 ≥80% 可接受

### 真实 KV cache 为什么失败?

真实 cache 有**强稀疏性**:
- 少数 token attention 很高 (头部信息、关键词)
- 大部分 token attention 很低 (填充词)
- log-ratio 近似在极值下崩溃 (log(1e-5 / 1e-8) = 大数字)
- Beta 严重过拟合/欠拟合 → 质量 37%

---

## 证据链

### 1. test_real_kv_cache.py (19:25 创建)

```python
# 测试配置: Exp 2.2 最佳配置
MODEL_PATH = "/Volumes/Toshiba/models/qwen3-8b-mlx"
COMPRESSION_RATIO = 4
NUM_QUERIES = 50
TARGET_QUALITY = 0.950000
BASELINE_QUALITY = 0.931600  # 来自合成 cache

# 结果:
❌ FAIL: 0.374331 < 0.931600 (baseline)
```

### 2. quality_test_output_with_qnorm.log (904K)

重复出现的压缩失败错误:
```
Error compressing layer 3: 'Model' object has no attribute 'model'
Error compressing layer 7: name 'head_keys' is not defined
...
```

这些错误可能是重启前尝试修复 NNLS 时引入的 bug。

### 3. attention-matching-fixed-summary.md

文档声称修复完成:
> "质量改善: 13-19% → ≥80% cosine similarity"

但这是基于**合成 cache**测试,未测真实模型!

---

## 重启前进展

**时间线**:
1. 12:26-14:59 - 创建各种测试文件
2. **19:25** - 创建 `test_real_kv_cache.py` (最后修改的文件)
3. 19:25-19:30 - 发现质量崩溃,可能正在调试 NNLS
4. **19:30** - 机器重启,进度丢失

**推测状态**:
- 发现 log-ratio 近似不够
- 查看 test_nnls.py,发现需要 solvers.py
- 可能正在实现 NNLS 求解器 (或查找参考实现)
- 可能遇到了 'Model' object has no attribute 'model' 错误
- 然后重启

---

## 下一步

### Option A: 实现 NNLS 求解器 (治本)

1. 实现 `src/flashmlx/compaction/solvers.py`:
   - `nnls_clamped()` - 先求解 unconstrained LSQ,然后 clamp 负值
   - `nnls_pgd()` - Projected Gradient Descent,迭代优化
   - `nnls_auto()` - 根据质量要求自动选择

2. 修改 `compaction_algorithm.py`:
   ```python
   if self.beta_method == 'nnls':
       from .solvers import nnls_auto
       # 构造 NNLS 问题: min_{beta ≥ 0} || M @ beta - y ||^2
       M = ...  # 设计矩阵
       y = ...  # 目标向量
       beta = nnls_auto(M, y, quality='high')
   ```

3. 重新测试 `test_real_kv_cache.py`

**优点**: 真正修复问题,达到论文质量
**缺点**: 需要实现 NNLS 求解器 (2-3 小时)

### Option B: 使用 SciPy NNLS (快速验证)

1. 临时添加 scipy 依赖:
   ```python
   from scipy.optimize import nnls as scipy_nnls
   ```

2. 在 `compaction_algorithm.py` 中:
   ```python
   if self.beta_method == 'nnls':
       import numpy as np
       from scipy.optimize import nnls as scipy_nnls

       # Convert to NumPy
       M_np = np.array(M.tolist())
       y_np = np.array(y.tolist())

       # Solve NNLS
       beta_np, _ = scipy_nnls(M_np, y_np)
       beta = mx.array(beta_np)
   ```

3. 测试是否能达到 ≥93% 质量

**优点**: 快速验证 NNLS 是否是问题根因 (30 分钟)
**缺点**: 引入 scipy 依赖,格式转换开销

### Option C: 检查 compaction 原始库

1. 克隆原始库:
   ```bash
   git clone https://github.com/adamzweiger/compaction /tmp/compaction-ref
   ```

2. 查找 NNLS 实现:
   ```bash
   grep -r "def nnls" /tmp/compaction-ref/
   ```

3. 复制并移植到 FlashMLX

**优点**: 使用论文作者的实现,最可靠
**缺点**: 可能需要处理 PyTorch ↔ MLX 转换

---

## 教训

### Level 2 失败: 知道规则但忘记验证

**问题**:
- 文档说"修复完成,质量 ≥80%"
- 但只测试了合成 cache,没测真实模型!
- 合成 cache 的成功给了虚假信心

**应该做**:
1. 合成 cache 测试 ✅
2. **真实模型测试** ❌ (漏了!)
3. 端到端质量验证 ❌ (漏了!)

### Level 3 目标: 自动化验证

**解决方案**:
1. ✅ 创建 `test_real_kv_cache.py` (已完成,就是当机前创建的!)
2. ✅ 添加到 CI pipeline (待办)
3. ✅ 质量回归测试 (待办)

---

*最后更新: 2026-03-22 19:40 (重启后恢复)*
*状态: CRITICAL - 需要实现 NNLS 求解器*
*下一步: 监护人决策选择 Option A/B/C*
