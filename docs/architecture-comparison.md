# Attention Matching 架构分析：PyTorch → MLX 移植指南

> **完整对比作者实现**，基于真实代码分析
>
> 来源：/tmp/compaction/
> - models/cache.py
> - models/qwen3/modeling_qwen3.py
> - compaction/algorithms/highest_attention_keys.py

---

## 1. 完整数据流

```
==================== 离线压缩阶段 ====================

原始 KV Cache (T 个 tokens)
    ↓
采样 queries (n 个查询向量)
    ↓
调用 HighestAttentionKeysCompaction.compute_compacted_cache()
    │
    ├─ 计算 attention scores: queries @ K.T
    ├─ Softmax 归一化
    ├─ 选择 top-t keys (highest attention)
    ├─ NNLS 求解 beta (使 C1 + beta ≈ 原始 attention)
    └─ Ridge Regression 求解 C2 (压缩后的 values)
    ↓
生成 compacted_cache = [(C1, beta, C2), ...] (每层一个)
    ↓
创建 CompactedPrefixCache(compacted_cache, original_seq_len)


==================== 在线推理阶段 ====================

用户输入 Prompt
    ↓
Transformer Layer (逐层)
    ├── Input LayerNorm
    ├── Attention.__call__()
    │     ├─ Q, K, V 投影
    │     ├─ Apply RoPE
    │     ├─ cache.update(K, V)  ← 简单 concat，不压缩
    │     │
    │     ├─ 🔥 检测 CompactedPrefixCache
    │     │     if isinstance(cache, CompactedPrefixCache):
    │     │         beta = cache.beta_for_layer(layer_idx)  ← 获取 beta
    │     │         beta_heads = repeat_kv_for_beta(beta, num_groups)
    │     │         modified_mask = attention_mask.clone()
    │     │         modified_mask[:, :, :, :prefix_len] += beta_heads.unsqueeze(2)  ← 应用 beta
    │     │
    │     ├─ Attention(Q, K, V, modified_mask)
    │     └─ Output projection
    │
    ├── Residual + MLP
    └── 下一层
```

**关键点**：
- ⚡ **压缩触发点**：离线阶段，不在 forward 时
- ⚡ **Beta 计算点**：`HighestAttentionKeysCompaction.compute_compacted_cache()`
- ⚡ **Beta 存储点**：`CompactedPrefixCache.__init__()` 构造时
- ⚡ **Beta 应用点**：`Qwen3Attention.forward()` line 227-263
- ⚡ **Keys/Values 替换**：在 CompactedPrefixCache 构造时已替换

---

## 2. CompactedPrefixCache 详细分析

### 继承关系

```
transformers.cache_utils.Cache (基类)
    ↑
CompactedPrefixCache
    │
    └── self.layers: List[CompactedPrefixLayer]
              ↑
        CacheLayerMixin (per-layer cache)
```

### 构造函数逻辑

```python
# models/cache.py: line 130-196
def __init__(
    self,
    compacted_cache: Tuple[Tuple[Tensor, Tensor, Tensor], ...],  # [(C1, beta, C2), ...]
    original_seq_len: int,
    ...
):
    layers = []
    max_compacted_len = 0

    # 为每层创建 CompactedPrefixLayer
    for layer_idx, (C1, beta, C2) in enumerate(compacted_cache):
        layer = CompactedPrefixLayer(C1, beta, C2, clone=clone)
        layers.append(layer)

        # 跟踪最大压缩长度
        max_compacted_len = max(max_compacted_len, C1.shape[-2])

    # 计算 rope_base (用于 RoPE 位置校准)
    self._rope_base = original_seq_len - max_compacted_len

    super().__init__(layers=layers, ...)
```

**关键**：CompactedPrefixCache **不触发压缩**，只存储已压缩的 KV。

### CompactedPrefixLayer.update()

```python
# models/cache.py: line 57-76
def update(self, key_states, value_states, cache_kwargs=None):
    """
    简单 concat，不压缩
    """
    if not self.is_initialized:
        self.lazy_initialization(key_states)

    # 标准动态行为：在 seq 维度 concat
    self.keys = torch.cat([self.keys, key_states], dim=-2)   # (B, KV, t+cur_len, D)
    self.values = torch.cat([self.values, value_states], dim=-2)

    return self.keys, self.values
```

**没有压缩逻辑！** update() 只是 concat 新 tokens 到已压缩的 prefix 后面。

### beta_for_layer() API

```python
# models/cache.py: line 213-222
def beta_for_layer(self, i: int) -> Tensor:
    layer = self.layers[i]
    if isinstance(layer, CompactedPrefixLayer):
        return layer.beta  # (B, KV, t)
    else:
        # Sliding layers don't have beta - return zeros
        return torch.zeros(...)
```

**返回格式**: `(B, num_kv_heads, prefix_length)`

---

## 3. Qwen3 Attention 修改

### 原始代码（标准 Attention）

```python
# 标准 transformers Qwen3Attention (未修改版本)
def forward(self, hidden_states, attention_mask, past_key_values, ...):
    # 投影 Q, K, V
    query_states = self.q_norm(self.q_proj(hidden_states)...).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states)...).transpose(1, 2)
    value_states = self.v_proj(hidden_states)....transpose(1, 2)

    # Apply RoPE
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # 更新 cache
    if past_key_values is not None:
        key_states, value_states = past_key_values.update(key_states, value_states, ...)

    # ❌ 没有 beta 应用

    # Attention
    attn_output, attn_weights = attention_interface(
        self, query_states, key_states, value_states,
        attention_mask,  # ← 原始 mask，没有 beta
        ...
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights
```

### 修改后代码（作者版本）

```python
# models/qwen3/modeling_qwen3.py: line 199-276
def forward(self, hidden_states, attention_mask, past_key_values, ...):
    # ... Q, K, V 投影 (相同) ...
    # ... Apply RoPE (相同) ...
    # ... cache.update() (相同) ...

    # ✅ 新增：Apply beta biases for CompactedPrefixCache
    modified_attention_mask = attention_mask
    if isinstance(past_key_values, CompactedPrefixCache):  # ← 检测 CompactedPrefixCache
        # 获取 beta (B, KV, t)
        beta_kv = past_key_values.beta_for_layer(self.layer_idx)

        # Repeat for GQA: (B, KV, t) → (B, num_heads, t)
        beta_heads = repeat_kv_for_beta(beta_kv, self.num_key_value_groups)

        # 获取维度
        batch_size, num_heads, prefix_length = beta_heads.shape
        query_length = query_states.shape[2]
        kv_length = key_states.shape[2]

        # 处理 nonuniform cache：slice mask 到当前层的 kv_length
        if attention_mask is not None:
            modified_attention_mask = attention_mask[:, :, :, -kv_length:]

            # Expand head dimension
            if modified_attention_mask.shape[1] == 1 and num_heads > 1:
                modified_attention_mask = modified_attention_mask.expand(
                    batch_size, num_heads, query_length, kv_length
                )

            # Clone 避免修改共享 mask
            modified_attention_mask = modified_attention_mask.clone()
        else:
            modified_attention_mask = torch.zeros(
                batch_size, num_heads, query_length, kv_length,
                dtype=query_states.dtype, device=query_states.device
            )

        # ✅ 应用 beta 到 prefix 位置（前 prefix_length 个位置）
        # (B, A, Q, KV) + (B, A, 1, t)
        modified_attention_mask[:, :, :, :prefix_length] += beta_heads.unsqueeze(2)

    # Attention (使用修改后的 mask)
    attn_output, attn_weights = attention_interface(
        self, query_states, key_states, value_states,
        modified_attention_mask,  # ← 修改后的 mask（包含 beta）
        ...
    )

    # ... Output projection (相同) ...
```

### Diff（关键修改）

```diff
  def forward(self, hidden_states, attention_mask, past_key_values, ...):
      # ... Q, K, V projection ...
      # ... RoPE ...
      # ... cache.update() ...

+     # Apply beta biases for CompactedPrefixCache by modifying attention_mask
+     modified_attention_mask = attention_mask
+     if isinstance(past_key_values, CompactedPrefixCache):
+         beta_kv = past_key_values.beta_for_layer(self.layer_idx)
+         beta_heads = repeat_kv_for_beta(beta_kv, self.num_key_value_groups)
+
+         # ... 处理 mask 维度 ...
+
+         # Apply beta to prefix positions
+         modified_attention_mask[:, :, :, :prefix_length] += beta_heads.unsqueeze(2)

      attn_output, attn_weights = attention_interface(
          self, query_states, key_states, value_states,
-         attention_mask,
+         modified_attention_mask,
          ...
      )
```

**代码行号**: line 227-263 in `models/qwen3/modeling_qwen3.py`

---

## 4. 压缩算法详解

### HighestAttentionKeysCompaction

```python
# compaction/algorithms/highest_attention_keys.py
class HighestAttentionKeysCompaction(CompactionAlgorithm):
    def compute_compacted_cache(
        self,
        K: torch.Tensor,     # (T, d) - 原始 keys
        V: torch.Tensor,     # (T, d) - 原始 values
        queries: torch.Tensor,  # (n, d) - 采样的查询向量
        t: int,              # 压缩后大小
        attention_bias: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
        """
        返回:
            C1:   (t, d) - 压缩后的 keys
            beta: (t,)   - 偏置项
            C2:   (t, d) - 压缩后的 values
            indices: List[int] - 选中的 key 索引
        """
        # Step 1: 选择 highest attention keys
        C1, beta, indices = self._select_keys_highest_attention(K, queries, t, attention_bias)

        # Step 2: 计算压缩后的 values
        C2 = self._compute_C2_with_method(C1, beta, K, V, queries, ...)

        return C1, beta, C2, indices
```

### Step 1: 选择 Highest Attention Keys

```python
# line 120-232
def _select_keys_highest_attention(self, K, queries, t, attention_bias):
    """
    1. 计算 attention scores: queries @ K.T
    2. Softmax 归一化
    3. 计算每个 key 的重要性分数（mean/rms/max）
    4. 选择 top-t keys
    5. NNLS 求解 beta
    """
    n, d = queries.shape
    T = K.shape[0]

    # 1. 计算 attention scores (fp32)
    scores_raw = queries @ K.T  # (n, T)
    scores32 = scores_raw.to(torch.float32) * (1.0 / d) ** 0.5

    # 2. 加上 attention_bias (如果有)
    if attention_bias is not None:
        scores32 = scores32 + attention_bias

    # 3. Softmax
    max_scores = scores32.max(dim=1, keepdim=True)[0]
    exp_scores = torch.exp(scores32 - max_scores)
    attention_weights = exp_scores / exp_scores.sum(dim=1, keepdim=True)  # (n, T)

    # 4. 计算 key scores (score_method: 'mean', 'rms', 'max')
    if self.score_method == 'rms':
        key_scores = torch.sqrt((attention_weights ** 2).mean(dim=0))  # (T,)
    elif self.score_method == 'max':
        key_scores = attention_weights.max(dim=0)[0]
    else:  # 'mean'
        key_scores = attention_weights.mean(dim=0)

    # 5. Top-t 选择
    _, indices = torch.topk(key_scores, k=t, largest=True, sorted=False)
    indices = sorted(indices.tolist())

    # 6. 提取选中的 keys
    C1 = K[indices, :]  # (t, d)

    # 7. NNLS 求解 beta (使 C1 + beta ≈ 原始 attention)
    if self.beta_method == 'nnls':
        beta = self._solve_beta_nnls(...)  # (t,)
    else:  # 'zero'
        beta = torch.zeros(t, dtype=K.dtype, device=K.device)

    return C1, beta, indices
```

### Step 2: 计算 C2 (压缩后的 values)

```python
# base.py: _compute_C2_lsq
def _compute_C2_lsq(self, C1, beta, K, V, queries, indices, attention_bias, ridge_lambda, ...):
    """
    Ridge Regression 求解 C2:
        min || P_compacted @ C2 - P_original @ V ||^2 + lambda * ||C2||^2

    where:
        P_compacted = softmax((queries @ C1.T) * scale + beta)
        P_original = softmax((queries @ K.T) * scale + attention_bias)
    """
    n, d = queries.shape
    t = C1.shape[0]

    # 计算压缩后的 attention weights
    scores_C1 = queries @ C1.T  # (n, t)
    scores_C1 = scores_C1.to(torch.float32) * (1.0 / d) ** 0.5
    if beta is not None:
        scores_C1 = scores_C1 + beta.unsqueeze(0)  # (n, t)

    P_compacted = torch.softmax(scores_C1, dim=-1)  # (n, t)

    # 计算原始 attention outputs
    scores_K = queries @ K.T  # (n, T)
    scores_K = scores_K.to(torch.float32) * (1.0 / d) ** 0.5
    if attention_bias is not None:
        scores_K = scores_K + attention_bias

    P_original = torch.softmax(scores_K, dim=-1)  # (n, T)
    target = P_original @ V  # (n, d)

    # Ridge regression: (P^T P + lambda * I) C2 = P^T target
    if ridge_lambda > 0:
        # 计算 lambda (根据 ridge_scale)
        if self.c2_ridge_scale == 'spectral':
            ridge_lambda_scaled = ridge_lambda * spectral_norm(P_compacted)
        # ... 其他缩放方法 ...

        # Cholesky/lstsq solver
        C2 = solve_ridge(P_compacted, target, ridge_lambda_scaled, solver=self.c2_solver)
    else:
        # Least squares without regularization
        C2 = torch.linalg.lstsq(P_compacted, target).solution

    return C2.to(K.dtype)  # (t, d)
```

---

## 5. MLX 移植方案对比

### 方案 A: 继承 MLX KVCache ✅ **推荐**

**实现**:
```python
# src/flashmlx/cache/compacted_kv_cache.py
from mlx_lm.models.cache import KVCache
import mlx.core as mx

class CompactedKVCache(KVCache):
    """
    MLX 版本的 CompactedPrefixCache
    """
    def __init__(
        self,
        compacted_cache: List[Tuple[mx.array, mx.array, mx.array]],  # [(C1, beta, C2), ...]
        original_seq_len: int = None,
    ):
        super().__init__()

        # 为每层初始化 keys/values
        self.keys = None
        self.values = None
        self.beta_cache = {}  # {layer_idx: beta}

        for layer_idx, (C1, beta, C2) in enumerate(compacted_cache):
            # 初始化 keys/values 列表
            if self.keys is None:
                # 第一次初始化
                self.keys = C1
                self.values = C2

            # 存储 beta
            self.beta_cache[layer_idx] = beta  # (B, KV, t)

        self.base_len = C1.shape[-2] if len(compacted_cache) > 0 else 0
        self.offset = self.base_len

    def update_and_fetch(self, keys, values):
        """
        简单 concat，不压缩（与 PyTorch 一致）
        """
        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            self.keys = mx.concatenate([self.keys, keys], axis=-2)
            self.values = mx.concatenate([self.values, values], axis=-2)

        self.offset = self.keys.shape[-2]
        return self.keys, self.values

    def beta_for_layer(self, layer_idx: int) -> mx.array:
        """
        获取指定层的 beta

        Returns:
            beta: (B, KV, t) or None
        """
        return self.beta_cache.get(layer_idx, None)
```

**优点**:
- ✅ 与 MLX-LM 框架集成良好
- ✅ 类型安全，API 一致
- ✅ 可复用现有 cache 管理机制

**缺点**:
- ❌ 需要理解 MLX-LM 内部结构
- ❌ 可能需要调整 cache 创建流程

**实现要点**:
1. MLX `mx.array` 操作与 PyTorch `torch.Tensor` 类似
2. `mx.concatenate` 对应 `torch.cat`
3. 确保 dtype (float16/float32/bfloat16) 一致

---

### 方案 B: Monkey Patch Attention ❌ **不推荐**

**实现**:
```python
# Monkey patch Qwen3Attention.__call__
import types

original_call = layer.self_attn.__call__

def patched_call(self, x, mask=None, cache=None):
    # ... 计算 Q, K, V ...
    # ... Apply RoPE ...
    # ... cache.update() ...

    # 检测 CompactedKVCache 并应用 beta
    if isinstance(cache, CompactedKVCache):
        beta = cache.beta_for_layer(self.layer_idx)
        if beta is not None:
            # 修改 mask
            ...

    # ... Attention ...

layer.self_attn.__call__ = types.MethodType(patched_call, layer.self_attn)
```

**优点**:
- ✅ 无需修改 MLX-LM 源码
- ✅ 快速原型验证

**缺点**:
- ❌ Python `__call__` 特殊方法从类查找，实例级 override 不稳定
- ❌ 可能与 MLX-LM 更新冲突
- ❌ 调试困难

**实现要点**:
- 需要完全复制 Qwen3Attention 的 forward 逻辑
- 确保所有边界情况处理正确

---

## 6. 推荐移植路径

### 🎯 最终推荐：**方案 A（继承 MLX KVCache）**

**理由**:
1. **稳定性**: 不依赖 Monkey Patch hack
2. **可维护性**: 清晰的类继承关系
3. **兼容性**: 与 MLX-LM 生态集成
4. **扩展性**: 便于未来优化（如 on-device quantization）

**分阶段实现**:

#### Phase 2: 创建 CompactedKVCache
```python
# src/flashmlx/cache/compacted_kv_cache.py
class CompactedKVCache(mlx_lm.models.cache.KVCache):
    def __init__(self, compacted_cache, original_seq_len):
        # 存储 C1, beta, C2

    def update_and_fetch(self, keys, values):
        # 简单 concat

    def beta_for_layer(self, layer_idx):
        # 返回 beta
```

#### Phase 3: 移植压缩算法
```python
# src/flashmlx/algorithms/highest_attention_keys.py
class HighestAttentionKeysCompactionMLX:
    def compute_compacted_cache(self, K, V, queries, t):
        # PyTorch → MLX 张量操作
        # torch.cat → mx.concatenate
        # torch.topk → mx.topk
        # torch.linalg.lstsq → mx.linalg.lstsq
```

#### Phase 4: Patch Qwen3 Attention
```python
# src/flashmlx/patch/qwen3_attention.py
def patch_qwen3_attention_for_compaction(model):
    """
    修改 Qwen3Attention，支持 CompactedKVCache
    """
    for layer in model.model.layers:
        # 重写 __call__ 方法（使用装饰器或子类化）
        ...
```

---

## 7. 关键差异总结

| 组件 | PyTorch 实现 | MLX 需要改动 |
|------|-------------|-------------|
| **CompactedPrefixCache** | `transformers.Cache` 子类 | 继承 `mlx_lm.cache.KVCache` |
| **Tensor 操作** | `torch.cat`, `torch.matmul` | `mx.concatenate`, `mx.matmul` (`@`) |
| **NNLS solver** | `torch.linalg.lstsq` | `mx.linalg.lstsq` |
| **Attention mask** | `(B, A, Q, KV)` float tensor | 相同，MLX array |
| **Beta 应用** | `mask += beta.unsqueeze(2)` | `mask + beta[:, :, None, :]` |
| **GQA repeat** | `repeat_kv_for_beta()` | 自定义 `repeat_kv()` for MLX |

---

## 8. 验收标准

### 功能验收
- [ ] ✅ CompactedKVCache 可以构造并存储 (C1, beta, C2)
- [ ] ✅ beta_for_layer() 返回正确格式的 beta
- [ ] ✅ update_and_fetch() 正确 concat 新 tokens
- [ ] ✅ Attention 检测到 CompactedKVCache 并获取 beta
- [ ] ✅ Beta 正确应用到 attention_mask
- [ ] ✅ 输出质量正常（无乱码）

### 性能验证
- [ ] ✅ Token overlap ≥ 50%（论文标准）
- [ ] ✅ 压缩比达到 2.0x
- [ ] ✅ TTFT 改善（首 token 延迟降低）

---

## 9. 移植检查清单

### PyTorch → MLX API 映射

| PyTorch | MLX | 备注 |
|---------|-----|------|
| `torch.cat([a, b], dim=-2)` | `mx.concatenate([a, b], axis=-2)` | dim → axis |
| `torch.matmul(a, b)` | `a @ b` | MLX 推荐用 `@` |
| `a.transpose(0, 2, 1, 3)` | `a.transpose(0, 2, 1, 3)` | 相同 |
| `torch.softmax(x, dim=-1)` | `mx.softmax(x, axis=-1)` | dim → axis |
| `torch.zeros(...)` | `mx.zeros(...)` | 相同 |
| `a.unsqueeze(2)` | `a[:, :, None, :]` 或 `mx.expand_dims(a, 2)` | MLX 推荐 slicing |
| `a.expand(...)` | `mx.broadcast_to(a, ...)` | 不同函数 |
| `torch.topk(a, k)` | `mx.topk(a, k)` | 相同 |
| `torch.linalg.lstsq(A, b)` | `mx.linalg.lstsq(A, b)` | 相同 |
| `a.clone()` | `mx.copy(a)` | 不同函数 |
| `a.to(dtype)` | `a.astype(dtype)` | 不同方法 |

### dtype 映射

| PyTorch | MLX |
|---------|-----|
| `torch.float32` | `mx.float32` |
| `torch.float16` | `mx.float16` |
| `torch.bfloat16` | `mx.bfloat16` |

---

## 附录：代码行号索引

| 文件 | 关键行号 | 内容 |
|------|---------|------|
| `models/cache.py` | 130-196 | CompactedPrefixCache.__init__() |
| `models/cache.py` | 213-222 | beta_for_layer() |
| `models/cache.py` | 57-76 | CompactedPrefixLayer.update() |
| `models/qwen3/modeling_qwen3.py` | 227-263 | Beta 应用到 attention_mask |
| `models/qwen3/modeling_qwen3.py` | 130-140 | repeat_kv_for_beta() |
| `compaction/algorithms/highest_attention_keys.py` | 70-118 | compute_compacted_cache() |
| `compaction/algorithms/highest_attention_keys.py` | 120-232 | _select_keys_highest_attention() |
| `compaction/algorithms/base.py` | ~200-300 | _compute_C2_lsq() |

---

*Architecture Comparison Document v1.0*
*Created: 2026-03-22*
*Based on: /tmp/compaction/ (PyTorch implementation)*
