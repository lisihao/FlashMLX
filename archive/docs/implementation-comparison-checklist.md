# Attention Matching 实现对照检查清单

**对比对象**：
- 作者实现：https://github.com/adamzweiger/compaction
- 我的实现：FlashMLX AttentionMatchingCompressorV2

---

## ✅ 核心算法（已正确）

| 组件 | 作者实现 | 我的实现 | 状态 |
|------|----------|----------|------|
| **Beta 计算（NNLS）** | ✅ `nnls_solve(M, target)` | ✅ 调用作者代码 | ✅ 正确 |
| **C2 优化（Ridge Regression）** | ✅ `ridge_regression(...)` | ✅ 调用作者代码 | ✅ 正确 |
| **Top-k 选择** | ✅ `score_method='max'` | ✅ 默认 'max' | ✅ 正确 |

---

## ✅ Query 生成（已正确）

| 步骤 | 作者实现 | 我的实现 | 状态 |
|------|----------|----------|------|
| **从 cache keys 采样** | ✅ `cache_keys[:, indices, :]` | ✅ `mx.take(keys, indices)` | ✅ 正确 |
| **应用 q_norm** | ✅ `q_norm(layer_queries)` | ✅ `q_norm(sampled_keys)` | ✅ 正确 |
| **应用 k_norm** | ❌ **没有** | ❌ **没有** | ✅ 一致 |
| **应用 RoPE** | ❌ **没有** | ❌ **没有** | ✅ 一致 |
| **随机采样** | ✅ `torch.randperm(...).sort()` | ✅ `np.random.choice(...).sort()` | ✅ 正确 |

**结论**：Query 生成逻辑完全正确！

---

## ❌ 关键差异 1：attention_bias 参数

| 参数 | 作者实现 | 我的实现 | 状态 |
|------|----------|----------|------|
| **attention_bias** | ✅ 支持（可选参数） | ❌ **完全没有传递** | ❌ **缺失！** |
| **compute_compacted_cache 签名** | `attention_bias: torch.Tensor = None` | 未传递 `attention_bias` | ❌ **错误！** |

**作者代码**：
```python
def compute_compacted_cache(
    self,
    K: torch.Tensor,
    V: torch.Tensor,
    queries: torch.Tensor,
    t: int,
    attention_bias: torch.Tensor = None,  # ❗ 关键参数！
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
```

**我的调用**（wrapper.py）：
```python
C1_torch, beta_torch, C2_torch, indices = self.algorithm.compute_compacted_cache(
    K=head_keys_torch,
    V=head_values_torch,
    queries=sampled_queries_torch,
    t=target_seq_len,
    # ❌ 缺少 attention_bias=None
)
```

**attention_bias 的作用**（从作者代码）：
```python
scores32 = scores_raw.to(torch.float32) * inv_sqrt_d       # (n, T) fp32
if attention_bias is not None:
    bias32 = torch.broadcast_to(
        attention_bias.to(torch.float32),
        scores32.shape
    )
    scores32 = scores32 + bias32  # ❗ 加上 bias！
```

---

## ❌ 关键差异 2：Qwen 的 k_norm

| 组件 | 作者实现 | 我的实现 | 状态 |
|------|----------|----------|------|
| **k_norm 检查** | ✅ 知道 Qwen 有 k_norm | ❓ **没有考虑 k_norm** | ❓ 未知影响 |

**从作者的 Qwen3 模型代码**：
```python
query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
```

**问题**：
- Cache 中的 keys 已经应用了 `k_norm`：`rope(k_norm(k_proj(x)))`
- 我从 keys 采样出来作为 queries
- 然后应用 `q_norm(sampled_keys)` → 得到 `q_norm(rope(k_norm(keys)))`

**但真实的 queries 应该是**：
- `rope(q_norm(q_proj(x)))`

**这是不匹配的！**

---

## 🔍 根因分析

### 问题 1：attention_bias 缺失（确认）

**影响**：
- Qwen 模型可能有 sliding window attention 或其他 bias
- 如果有 bias，没有传递会导致 attention scores 计算错误
- Top-k 选择错误 → Beta 计算错误 → C2 优化错误 → 质量崩溃

**验证方法**：
```python
# 检查 Qwen 是否使用 attention_bias
print(model.config.sliding_window)  # 如果不是 None，就有 sliding window
```

### 问题 2：k_norm 的影响（待确认）

**Cache keys 的真实状态**：
- `keys_in_cache = rope(k_norm(k_proj(x)))`

**我采样的 queries**：
- `sampled_queries = q_norm(keys_in_cache) = q_norm(rope(k_norm(k_proj(x))))`

**真实的 queries**：
- `real_queries = rope(q_norm(q_proj(x)))`

**问题**：
1. `q_norm` 和 `k_norm` 的顺序不同
2. `q_proj` 和 `k_proj` 是不同的线性变换
3. RoPE 被应用了两次（一次在 keys，一次在真实 queries）

**但作者也是这样做的！** 说明这个近似是可接受的。

---

## 🎯 修复方案

### 修复 1：传递 attention_bias（P0 - 必须）

**修改文件**：`src/flashmlx/compaction/wrapper.py`

```python
def compress_kv_cache(
    self,
    keys: mx.array,
    values: mx.array,
    queries: Optional[mx.array] = None,
    num_queries: int = 100,
    attention_bias: Optional[mx.array] = None,  # ✅ 新增参数
) -> Tuple[mx.array, mx.array, mx.array]:
    ...

    # 转换 attention_bias 到 PyTorch（如果提供）
    bias_torch = None
    if attention_bias is not None:
        bias_torch = self.mlx_to_torch(attention_bias)

    # 调用算法，传递 attention_bias
    C1_torch, beta_torch, C2_torch, indices = self.algorithm.compute_compacted_cache(
        K=K_torch,
        V=V_torch,
        queries=queries_torch,
        t=target_size,
        attention_bias=bias_torch,  # ✅ 传递参数
    )
```

**修改文件**：`src/flashmlx/cache/attention_matching_compressor_v2.py`

需要在压缩时传递 attention_bias。**但问题是**：
- Qwen 的 sliding_window 是在 attention_mask 中实现的
- 不是一个独立的 attention_bias tensor
- 需要从模型配置中提取

### 修复 2：检查 Qwen 的 sliding_window（P0 - 必须）

**验证方法**：
```python
# 检查模型配置
print(model.config.sliding_window)  # Qwen3.5-35B: 可能是 4096
print(model.config.layer_types)     # ['sliding_attention', 'global_attention', ...]
```

**如果有 sliding_window**：
- 需要构造 sliding_window_mask
- 传递给 `compute_compacted_cache`

---

## 📋 验证检查清单

### ✅ 已确认正确

- [x] Beta 计算（NNLS）
- [x] C2 优化（Ridge Regression）
- [x] 从 cache keys 采样 queries
- [x] 应用 q_norm
- [x] 批量处理优化

### ❌ 需要修复

- [ ] **传递 attention_bias 参数**
- [ ] **检查 Qwen sliding_window**
- [ ] **构造 sliding_window_mask（如果需要）**

### ⏸️ 待验证

- [ ] Qwen3.5-35B 的 sliding_window 配置
- [ ] Qwen3-8B 的 sliding_window 配置
- [ ] 是否需要传递 causal_mask

---

## 🚀 下一步行动

1. **立即验证**：检查 Qwen 模型的 sliding_window 配置
2. **P0 修复**：添加 attention_bias 参数传递
3. **完整测试**：重新运行质量测试，验证 token overlap ≥ 50%
