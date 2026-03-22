# SSM State 压缩方案设计

**日期**: 2026-03-21 14:30
**任务**: Task #53 - State-Memory 专用压缩算法

---

## 1. SSM State 结构分析

### GatedDeltaNet Cache 结构

```python
cache[0] = conv_state  # Shape: (B, 3, conv_dim)
cache[1] = state       # Shape: (B, Hv, Dv, Dk) ← 主要压缩目标
```

### State 维度（Qwen3.5-35B）

```python
B  = batch_size        # 通常 1-4
Hv = num_v_heads      # 64
Dv = head_v_dim       # 128
Dk = head_k_dim       # 192

state.shape = (B, 64, 128, 192)
Total elements per sample = 64 * 128 * 192 = 1,572,864
Memory per sample (fp16) = 1,572,864 * 2 bytes = 3.15 MB
```

### Conv State（较小，暂不压缩）

```python
conv_kernel_size = 4
conv_dim = key_dim * 2 + value_dim
       = (192 * 16) * 2 + (128 * 64)
       = 3,072 * 2 + 8,192
       = 14,336

conv_state.shape = (B, 3, 14,336)
Total elements per sample = 3 * 14,336 = 43,008
Memory per sample (fp16) = 43,008 * 2 bytes = 86 KB (negligible)
```

**结论**: 主要压缩 `state`，`conv_state` 太小可以忽略

---

## 2. 压缩方法设计

### Method 1: Low-Rank State Approximation（推荐优先实现）

**原理**: 对每个 (B, Hv) slice 的 (Dv, Dk) 矩阵进行 SVD 分解，保留 top-k 奇异值

**实现**:
```python
def compress_state_lowrank(state, rank=32):
    """
    Args:
        state: (B, Hv, Dv, Dk) = (B, 64, 128, 192)
        rank: 保留的奇异值数量，default=32

    Returns:
        compressed_state: {
            'U': (B, Hv, Dv, rank),      # 64 * 128 * 32 = 262,144
            'S': (B, Hv, rank),           # 64 * 32 = 2,048
            'Vt': (B, Hv, rank, Dk)       # 64 * 32 * 192 = 393,216
        }
    """
    B, Hv, Dv, Dk = state.shape

    # 对每个 (B, Hv) 进行 SVD
    compressed = {'U': [], 'S': [], 'Vt': []}

    for b in range(B):
        U_batch, S_batch, Vt_batch = [], [], []
        for h in range(Hv):
            # state[b, h]: (Dv, Dk) = (128, 192)
            U, S, Vt = mx.linalg.svd(state[b, h])

            # 保留 top-k
            U_batch.append(U[:, :rank])       # (Dv, rank)
            S_batch.append(S[:rank])          # (rank,)
            Vt_batch.append(Vt[:rank, :])     # (rank, Dk)

        compressed['U'].append(mx.stack(U_batch))
        compressed['S'].append(mx.stack(S_batch))
        compressed['Vt'].append(mx.stack(Vt_batch))

    return {
        'U': mx.stack(compressed['U']),      # (B, Hv, Dv, rank)
        'S': mx.stack(compressed['S']),      # (B, Hv, rank)
        'Vt': mx.stack(compressed['Vt'])     # (B, Hv, rank, Dk)
    }


def decompress_state_lowrank(compressed):
    """
    Reconstruct state from compressed representation

    Args:
        compressed: {'U', 'S', 'Vt'} from compress_state_lowrank

    Returns:
        state: (B, Hv, Dv, Dk)
    """
    U = compressed['U']      # (B, Hv, Dv, rank)
    S = compressed['S']      # (B, Hv, rank)
    Vt = compressed['Vt']    # (B, Hv, rank, Dk)

    B, Hv, Dv, rank = U.shape
    Dk = Vt.shape[-1]

    state = mx.zeros((B, Hv, Dv, Dk), dtype=U.dtype)

    for b in range(B):
        for h in range(Hv):
            # Reconstruct: state[b,h] = U @ diag(S) @ Vt
            state[b, h] = U[b, h] @ mx.diag(S[b, h]) @ Vt[b, h]

    return state
```

**压缩比分析**:
```
原始: B * Hv * Dv * Dk = B * 64 * 128 * 192 = B * 1,572,864

压缩 (rank=32):
  U:  B * Hv * Dv * rank = B * 64 * 128 * 32 = B * 262,144
  S:  B * Hv * rank      = B * 64 * 32      = B * 2,048
  Vt: B * Hv * rank * Dk = B * 64 * 32 * 192 = B * 393,216
  Total:                                       B * 657,408

压缩比 = 1,572,864 / 657,408 = 2.39x

rank=48: 压缩比 = 1.73x
rank=64: 压缩比 = 1.36x
rank=16: 压缩比 = 4.27x (但可能损失质量)
```

**优点**:
- 理论保证（保留最重要的信息）
- 不需要训练数据
- 可以根据奇异值分布动态选择 rank

**缺点**:
- SVD 计算开销（O(Dv * Dk^2)）
- 需要在每次压缩时计算

---

### Method 2: Random Projection（快速替代）

**原理**: 使用固定的随机投影矩阵降维

**实现**:
```python
def compress_state_random_proj(state, target_dim=32, projection_matrix=None):
    """
    Args:
        state: (B, Hv, Dv, Dk)
        target_dim: 目标维度
        projection_matrix: (Dk, target_dim) 固定投影矩阵

    Returns:
        compressed_state: (B, Hv, Dv, target_dim)
    """
    B, Hv, Dv, Dk = state.shape

    if projection_matrix is None:
        # 初始化 Gaussian 随机矩阵（只需一次）
        projection_matrix = mx.random.normal(
            shape=(Dk, target_dim),
            scale=1.0 / (target_dim ** 0.5)
        )

    # state: (B, Hv, Dv, Dk) @ (Dk, target_dim) → (B, Hv, Dv, target_dim)
    compressed = mx.matmul(state, projection_matrix)

    return compressed, projection_matrix


def decompress_state_random_proj(compressed, projection_matrix):
    """
    伪逆重建（不精确，但足够好）

    Args:
        compressed: (B, Hv, Dv, target_dim)
        projection_matrix: (Dk, target_dim)

    Returns:
        state: (B, Hv, Dv, Dk)
    """
    # 计算 Moore-Penrose 伪逆
    proj_pinv = mx.linalg.pinv(projection_matrix)  # (target_dim, Dk)

    # compressed @ proj_pinv
    state = mx.matmul(compressed, proj_pinv)

    return state
```

**压缩比**:
```
原始: B * 64 * 128 * 192 = B * 1,572,864
压缩 (target_dim=32): B * 64 * 128 * 32 = B * 262,144
压缩比 = 6.0x

但需要存储 projection_matrix: (192, 32) = 6,144 elements (shared across all states)
```

**优点**:
- 极快（只需矩阵乘法）
- 投影矩阵可以共享（所有层、所有 batch 使用同一个）
- Johnson-Lindenstrauss 引理保证距离保持

**缺点**:
- 重建不精确（伪逆近似）
- 需要预先计算投影矩阵

---

### Method 3: Quantization（最简单）

**原理**: FP16 → INT8 或 INT4

**实现**:
```python
def compress_state_quantize(state, bits=8):
    """
    Args:
        state: (B, Hv, Dv, Dk) in FP16
        bits: 量化位数 (4 or 8)

    Returns:
        quantized_state: INT array
        scale: (B, Hv) quantization scale
        zero_point: (B, Hv) zero point
    """
    B, Hv, Dv, Dk = state.shape

    # Per-head quantization
    state_flat = state.reshape(B, Hv, -1)  # (B, Hv, Dv*Dk)

    # 计算 scale 和 zero_point
    min_val = state_flat.min(axis=-1, keepdims=True)  # (B, Hv, 1)
    max_val = state_flat.max(axis=-1, keepdims=True)

    qmin = 0
    qmax = (1 << bits) - 1  # 255 for 8-bit, 15 for 4-bit

    scale = (max_val - min_val) / (qmax - qmin)
    zero_point = qmin - min_val / scale

    # 量化
    quantized = mx.clip(
        mx.round((state_flat - min_val) / scale),
        qmin, qmax
    ).astype(mx.uint8 if bits == 8 else mx.uint4)

    return {
        'quantized': quantized.reshape(B, Hv, Dv, Dk),
        'scale': scale.squeeze(-1),        # (B, Hv)
        'zero_point': zero_point.squeeze(-1)
    }


def decompress_state_quantize(compressed):
    """反量化"""
    quantized = compressed['quantized']  # (B, Hv, Dv, Dk)
    scale = compressed['scale'][..., None, None]  # (B, Hv, 1, 1)
    zero_point = compressed['zero_point'][..., None, None]

    state = (quantized.astype(mx.float16) - zero_point) * scale
    return state
```

**压缩比**:
```
INT8: 2x (FP16 → INT8)
INT4: 4x (FP16 → INT4)
```

**优点**:
- 最简单
- 最快（几乎无开销）
- 精度损失可控

**缺点**:
- 压缩比较低

---

## 3. 实验计划

### Phase 1: Baseline 实现（当前）

**目标**: 实现 Method 1 (Low-Rank Approximation) 作为 baseline

**步骤**:
1. ✅ 理解 GatedDeltaNet state 结构
2. 🔄 实现 `compress_state_lowrank` 和 `decompress_state_lowrank`
3. 🔄 单元测试：验证重建误差
4. 🔄 Layerwise ablation: 只压缩单个 SSM 层，测试质量

### Phase 2: 集成测试

**目标**: 集成到 Heterogeneous Cache Manager

**步骤**:
1. 修改 `HeterogeneousCacheManager` 支持 SSM 压缩
2. 测试完整生成流程（Qwen3.5-35B）
3. 验证生成质量

### Phase 3: 优化和对比

**目标**: 对比三种方法，选择最优方案

**步骤**:
1. 实现 Method 2 (Random Projection)
2. 实现 Method 3 (Quantization)
3. 对比表：压缩比 vs 质量 vs 速度
4. 选择最优方案

---

## 4. 验收标准

- [ ] 至少有 1 种方法能压缩 SSM 层且质量 ≥ 85%
- [ ] 压缩比 ≥ 2x
- [ ] 与 Heterogeneous Cache 兼容
- [ ] 无 shape mismatch 或递推错误
- [ ] 压缩/解压缩开销 < 10% 总时间

---

## 5. 风险和缓解

**风险 1**: SVD 计算开销过大
- **缓解**: 如果 SVD 太慢，直接使用 Random Projection

**风险 2**: 压缩误差累积导致质量崩溃
- **缓解**: 从保守的 rank=64 开始，逐步降低到 32/16

**风险 3**: 解压缩后的 state 与原始递推逻辑不兼容
- **缓解**: 在 layerwise ablation 环境中先验证单层

---

*设计文档创建于: 2026-03-21 14:30*
*作者: Solar (Task #53)*
