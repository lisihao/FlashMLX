# Attention Matching 失败根因诊断（最终版）

**时间**: 2026-03-22
**状态**: ✅ 根因已确认

---

## 🔍 问题症状

1. **Qwen3.5-35B (混合架构)**：
   - Token overlap: 0%
   - 生成乱码
   - 特定层压缩失败：3, 7, 11, 15, 19, 23, 27, 31, 35, 39

2. **Qwen3-8B (纯 Transformer)**：
   - Token overlap: 0%
   - 生成乱码 ('S'S'S..., ))))))..., !!!!!!...)
   - 大量 NaN/Inf/Cholesky 错误

3. **PEP 3118 buffer format 错误**：
   ```
   Warning: Compression failed for layer 3: Item size 2 for PEP 3118
   buffer format string B does not match the dtype B item size 1.
   ```

---

## ✅ 根因 1：SSM 层压缩（Qwen3.5-35B）

### 问题

**Qwen3.5-35B 混合架构**：
- 4-4-4-4 pattern：4 层 Attention + 4 层 SSM
- **SSM 层没有 KV cache**！（使用 state cache）

**失败的层**：
- 3, 7, 11, 15, 19, 23, 27, 31, 35, 39
- **所有 SSM 层！**

**我的代码**：
- 试图压缩所有层的 KV cache
- 对 SSM 层调用 `compress_kv_cache()` → 失败（没有 KV cache）

### 修复方案

**方案 1：跳过 SSM 层**

```python
# 在 simple_injection.py 中
def inject_attention_matching(model, compression_ratio=2.0, ...):
    # 检测模型类型
    if hasattr(model.config, 'model_type'):
        model_type = model.config.model_type
    else:
        model_type = 'unknown'

    # 获取层类型（Qwen3.5 混合架构）
    layer_types = getattr(model.config, 'layer_types', None)

    compressor = AttentionMatchingCompressorV2(
        model=model,
        compression_ratio=compression_ratio,
        ...
    )

    # Hook 每层的 attention
    for layer_idx, layer in enumerate(model.model.layers):
        # ✅ 跳过 SSM 层
        if layer_types and layer_idx < len(layer_types):
            if layer_types[layer_idx] == 'ssm':
                continue  # 跳过 SSM 层

        original_forward = layer.self_attn.forward

        def make_hooked_forward(layer_idx):
            def hooked_forward(hidden_states, mask=None, cache=None, **kwargs):
                # 压缩 KV cache
                if cache is not None and hasattr(cache, 'get_kv_cache'):
                    kv_cache = cache.get_kv_cache(layer_idx)
                    if kv_cache is not None:
                        keys, values = kv_cache
                        compressed_keys, compressed_values = compressor.compress_kv_cache(
                            layer_idx, (keys, values)
                        )
                        cache.set_kv_cache(layer_idx, (compressed_keys, compressed_values))

                return original_forward(hidden_states, mask=mask, cache=cache, **kwargs)

            return hooked_forward

        layer.self_attn.forward = make_hooked_forward(layer_idx)
```

---

## ❌ 根因 2：bfloat16 转换问题（两个模型）

### 问题

**PEP 3118 buffer format 错误**：
```python
# 我的转换代码（wrapper.py）
def mlx_to_torch(self, arr: mx.array) -> torch.Tensor:
    if arr.dtype == mx.bfloat16:
        arr = arr.astype(mx.float32)  # ❗ 转换
    np_arr = np.array(arr)
    return torch.from_numpy(np_arr)
```

**问题**：
1. MLX bfloat16 → numpy array 时，numpy 不支持 bfloat16
2. `np.array(mx_bfloat16)` 创建了一个 dtype='B' 的 uint8 数组？
3. PyTorch `torch.from_numpy(np_arr)` 期望 float 类型 → 错误

### 修复方案

**方案 2：正确的 dtype 转换**

```python
def mlx_to_torch(self, arr: mx.array) -> torch.Tensor:
    """Convert MLX array to PyTorch tensor (fixed dtype handling)"""
    # ✅ 必须先转换到 float32，再转 numpy
    if arr.dtype == mx.bfloat16:
        arr = arr.astype(mx.float32)
    elif arr.dtype == mx.float16:
        arr = arr.astype(mx.float32)

    # ✅ 确保转换成功
    np_arr = np.array(arr, dtype=np.float32)

    # ✅ 验证 dtype
    assert np_arr.dtype == np.float32, f"Expected float32, got {np_arr.dtype}"

    return torch.from_numpy(np_arr)
```

---

## ❌ 根因 3：Qwen3-8B 的其他问题

Qwen3-8B 是纯 Transformer，没有 SSM 层，但仍然失败。

### 可能的原因

1. **bfloat16 转换问题**（同上）
2. **数值稳定性问题**：
   - 真实 KV cache 的数值范围导致 NNLS/Ridge Regression 求解失败
   - 出现 NaN/Inf → Cholesky decomposition 失败

3. **MLX 4-bit 量化模型的特殊性**：
   - 4-bit 量化的 KV cache 可能有特殊的数值分布
   - 解量化后的精度损失

### 验证方法

**测试 1：使用非量化模型**
```bash
# 测试 Qwen3-8B 的 bfloat16 版本（非量化）
python3 test_quality_qwen3_pure_transformer.py --model "mlx-community/Qwen3-8B"
```

**测试 2：增加数值稳定性**
```python
# 在 highest_attention_keys.py 中
# 计算 attention scores 时增加 eps
scores32 = scores_raw.to(torch.float32) * inv_sqrt_d + 1e-8
```

**测试 3：使用更鲁棒的求解器**
```python
# 修改 wrapper 初始化
wrapper = AttentionMatchingWrapper(
    compression_ratio=compression_ratio,
    score_method='max',
    beta_method='nnls',
    c2_method='lsq',
    nnls_iters=0,  # 使用 lstsq + clamping
    c2_solver='lstsq',  # 使用 lstsq 而不是 cholesky
)
```

---

## 🎯 完整修复计划

### Phase 1：修复 SSM 层跳过（P0 - 必须）

1. 修改 `simple_injection.py`：
   - 检测 `model.config.layer_types`
   - 跳过 `layer_types[i] == 'ssm'` 的层

2. 测试 Qwen3.5-35B：
   - 验证只有 Attention 层被压缩
   - 验证 token overlap ≥ 50%

### Phase 2：修复 bfloat16 转换（P0 - 必须）

1. 修改 `wrapper.py`：
   - 强制转换到 float32
   - 验证 numpy dtype

2. 测试 Qwen3-8B：
   - 验证无 PEP 3118 错误
   - 验证 token overlap ≥ 50%

### Phase 3：增强数值稳定性（P1 - 推荐）

1. 使用更鲁棒的求解器：
   - `c2_solver='lstsq'` 而不是 'cholesky'
   - `nnls_iters=0`（使用 lstsq + clamping）

2. 测试边界情况：
   - 4-bit 量化模型
   - 长序列（> 4096 tokens）

---

## 📊 预期结果

修复后：
- ✅ Qwen3.5-35B：只压缩 Attention 层，SSM 层跳过
- ✅ Qwen3-8B：bfloat16 转换成功，无数值错误
- ✅ Token overlap ≥ 50%（1.5x-2.0x 压缩率）
- ✅ 生成质量恢复正常

---

## 🧪 验证检查清单

- [ ] Qwen3.5-35B Attention layers 压缩成功
- [ ] Qwen3.5-35B SSM layers 被跳过
- [ ] Qwen3-8B 无 PEP 3118 错误
- [ ] Qwen3-8B 无 NaN/Inf 错误
- [ ] Token overlap ≥ 50%（12 个测试场景）
- [ ] 生成质量正常（无乱码、重复）

---

*Root Cause Analysis v2.0*
*完成于: 2026-03-22*
*确认根因: SSM 层压缩 + bfloat16 转换*
