# Hook 不生效问题分析

**发现时间**: 2026-03-22 19:25（重启前）
**严重程度**: CRITICAL - 阻塞质量验证

---

## 问题描述

在完成 Task #90-92（移植论文正确实现）后，运行 `test_qwen3_8b_v3_long.py` 质量测试时发现：

**症状**:
```
压缩统计:
  - 压缩次数: 0  ❌
  - 压缩前 tokens: 0
  - 压缩后 tokens: 0
  - 实际压缩比: 0.00x
```

**调试输出缺失**:
- 没有看到 `🔥 HOOK CALLED!` 调试信息
- 说明 hook 从未被调用

---

## 根本原因

### Hook 位置

`simple_injection_v3.py:61-62`:
```python
original_attention_call = layer.self_attn.__call__

def make_hooked_attention(layer_idx, verbose_flag):
    def hooked_attention_call(self, x, mask=None, cache=None):
        print(f"🔥 HOOK CALLED! layer={layer_idx}, ...")  # ← 从未打印
```

### 测试调用方式

`test_qwen3_8b_v3_long.py:122`:
```python
output = generate(
    model,
    tokenizer,
    prompt=prompt,
    max_tokens=100,
    verbose=False
)
```

### 根因假设

`mlx_lm.generate()` 可能：
1. 直接调用底层 C++/Metal 实现，绕过 Python hook
2. 使用不同的代码路径（不经过 `layer.self_attn.__call__`）
3. 内部创建了新的 attention 实例，没有使用我们 hook 的实例

---

## 验证实验

### 实验 1: 直接调用 model.forward

如果直接调用 `model(input_ids, cache=cache)`，hook 是否生效？

**参考**: `test_e2e_basic.py:85`
```python
outputs = model(input_ids, cache=cache)
```

### 实验 2: 检查 generate() 源码

查看 `mlx_lm` 的 `generate()` 实现，确认代码路径。

### 实验 3: Hook 更底层的函数

不 hook `__call__`，改为 hook：
- `scaled_dot_product_attention`
- `cache.update_and_fetch`
- 或者直接 hook Metal kernel

---

## 解决方案

### Option A: 调试 hook（治标）

1. 检查 `mlx_lm.generate()` 源码
2. 找到正确的 hook 点
3. 修改 `simple_injection_v3.py`

**优点**: 能在 `generate()` 中使用
**缺点**: 可能很难找到正确的 hook 点

### Option B: 手动 token-by-token 生成（绕过）

1. 实现 `manual_generate()` 函数
2. 逐 token 调用 `model(token, cache=cache)`
3. 直接使用 model forward，不用 `generate()`

**优点**: 完全控制生成流程，hook 肯定生效
**缺点**: 代码复杂，需要重新实现采样逻辑

**参考**: `benchmarks/evaluate_offline_compression_simple.py:55`

### Option C: 离线压缩 + CompactedKVCache（推荐）

1. 预先压缩 KV cache（使用 `offline_compress_kv_cache`）
2. 创建 `CompactedKVCache` 实例
3. 直接用 `model(input_ids, cache=compacted_cache)` 推理

**优点**:
- 不依赖 hook
- 真实的压缩算法
- 可以精确控制压缩过程

**缺点**:
- 需要分两步（压缩 + 推理）
- 不是端到端的在线压缩

**参考**: `test_e2e_basic.py:32-85`

---

## 当前状态

- **BLOCKED**: 无法进行真实模型质量测试
- **下一步**: 等待监护人决策选择哪个方案
- **文件位置**:
  - Hook 代码: `src/flashmlx/cache/simple_injection_v3.py`
  - 失败测试: `test_qwen3_8b_v3_long.py`
  - 手动生成参考: `benchmarks/evaluate_offline_compression_simple.py`
  - 离线压缩参考: `test_e2e_basic.py`

---

## 重启前丢失的进展

**重启时刻**: 2026-03-22 约 19:30
**丢失内容**:
- 正在分析 hook 不生效的问题
- 可能在尝试 Option B 或 Option C
- 未写入 STATE.md（违反规则！）

**教训**:
- ❌ 违反状态持久化规则
- ❌ 违反 Output Persist 规则
- ✅ 现在立即补救（写入 STATE.md + 此文档）

---

*最后更新: 2026-03-22 19:35*
*状态: BLOCKED - 等待监护人决策*
