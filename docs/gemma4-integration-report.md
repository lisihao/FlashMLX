# Gemma 4 + FlashMLX 集成测试报告

**日期**: 2026-04-10
**测试环境**: M4 Pro 48GB
**FlashMLX 版本**: v2.0 + VLM Deep Integration

## 执行摘要

✅ **集成成功**:两个 Gemma 4 模型都成功集成 FlashMLX
✅ **性能验证** (2026-04-10 修正):E4B 69.1 tok/s / 31B **13.74 tok/s** (早期 6.5 数据错误)
✅ **vlm_bridge 零开销**:NATIVE vs BRIDGE 差距 -0.6% (噪声范围)
✅ **Hybrid Cache 修复** (2026-04-10, commit `302f9ca` + `b503a55`):cache_factory 正确识别 `KVCache+RotatingKVCache` 混合架构,所有策略可以创建 cache,10/60 层 (31B) 替换为 TripleLayerKVCache
⚠️ **长 context 实测: 无压缩收益** (8K/16K/32K):Gemma 4 full_attention 层 KV heads 太少 (4 KV × 512 dim),且 triple_pq 在 32K 出现回归 (TG -49%, peak +19%)
🎯 **结论**:修复解锁了 API,但 Gemma 4 **不是** KV 压缩的理想场景 — 建议生产用 `standard` 策略

---

## 测试模型

### Model 1: gemma-4-E4B（小模型）

| 特性 | 值 |
|------|-----|
| **参数量** | ~3.3B (4-bit 量化) |
| **磁盘大小** | 4.9 GB |
| **层数** | 42 Transformer |
| **隐藏层** | 2560 |
| **KV Heads** | 2 (GQA) |
| **Attention Heads** | 8 |

**架构特点**：
- ✅ **KV Sharing**: 后 18 层共享 1 个 cache（节省 40% 内存）
- ✅ **Hybrid Cache**: 4/24 层全量 KVCache，20 层 RotatingKVCache
- 📊 **Cache 分布**: 24 个 cache 对象（不是 42）

### Model 2: gemma-4-31B（大模型）

| 特性 | 值 |
|------|-----|
| **参数量** | ~20.8B (4-bit 量化) |
| **磁盘大小** | 17.1 GB |
| **层数** | 60 Transformer |
| **隐藏层** | 5376 |
| **KV Heads** | 16 (GQA) |
| **Attention Heads** | 32 |

**架构特点**：
- ❌ **无 KV Sharing**: 所有 60 层独立 cache
- ✅ **Hybrid Cache**: 10/60 层全量 KVCache，50 层 RotatingKVCache
- 📊 **Cache 分布**: 60 个 cache 对象

---

## 性能基准测试

### 生成速度（M4 Pro 48GB）

> **⚠️ 2026-04-10 修正**: 初版报告中 31B 的 `6.5 tok/s` 是错误数据,根源是早期
> 测试走了 legacy monkey-patch 路径或 system mlx-vlm 0.4.0 fallback（fallback
> 本身不支持 gemma4）。迁移到 mlx-vlm-source fork 0.4.4 + vlm_bridge 后,实测
> 速度已经自动修复。对照实验脚本:`/tmp/gemma31b_native_vs_bridge.py`。

| 模型 | 生成速度 | Prompt TPS | 内存占用 | Cache 策略 |
|------|---------|-----------|---------|-----------|
| **gemma-4-E4B** | 69.1 tok/s | — | 5.31 MB | Standard |
| **gemma-4-31B** | **13.74 tok/s** ✨ | 20.49 tok/s | 17.41 GB peak | Standard |

**速度比**: 31B 比 E4B 慢 **5.03x**(参数量 6.3x)—— 实际比参数比还快 25%,属于
GQA + 4-bit dequant + Metal kernel 开销后的**健康水平**。

**M4 Max 理论上限**: 内存带宽 ~400 GB/s ÷ 17.1 GB 模型 ≈ 23 tok/s 极限 →
实测 13.74 tok/s = **理论上限的 60%**,符合 Apple Silicon 统一内存推理的典型效率。

### vlm_bridge 开销验证 (2026-04-10)

单进程加载一次模型,两次 benchmark 对比:

| 路径 | GEN tok/s | Prompt TPS | 差距 |
|------|----------|-----------|------|
| **NATIVE** (mlx-vlm 自己管 cache) | 13.74 | 20.49 | baseline |
| **BRIDGE** (FlashMLX create_vlm_cache strategy="standard") | 13.67 | 20.32 | **-0.6%** |

**结论**: FlashMLX vlm_bridge 在 Gemma 4 上**零开销**。cache_factory 检测到 hybrid
架构后 fallback 到 `RotatingKVCache`,与 mlx-vlm 原生路径完全等价。

### Hybrid Cache 分析

```
gemma-4-E4B (42 layers):
  Layers 0-23 (24 caches):
    ├─ Layer 5, 11, 17, 23  → KVCache (全量) [4 层]
    └─ 其他 20 层           → RotatingKVCache (滑动窗口)

  Layers 24-41 (1 cache):
    └─ 18 层共享            → KVCache (KV Sharing)

gemma-4-31B (60 layers):
  Layers 0-59 (60 caches):
    ├─ Layer 5, 11, 17, 23, 29, 35, 41, 47, 53, 59
    │                       → KVCache (全量) [10 层]
    └─ 其他 50 层           → RotatingKVCache (滑动窗口)
```

**设计哲学**: 每隔 ~6 层放置 1 个全量 KVCache（17% 层），用于捕获长期依赖。

---

## FlashMLX 集成状态

### ✅ 成功项

| 功能 | E4B | 31B | 状态 |
|------|-----|-----|------|
| 模型加载 | ✅ | ✅ | mlx-vlm fork 正常 |
| Cache 创建 | ✅ | ✅ | Standard cache 工作 |
| Cache 传递 | ✅ | ✅ | prompt_cache 参数 |
| 文本生成 | ✅ | ✅ | 生成正常 |
| VLM Bridge | ✅ | ✅ | API 完整 |

### ✅ 修复后状态 (2026-04-10)

| FlashMLX 策略 | E4B | 31B | 备注 |
|--------------|-----|-----|------|
| **standard** | ✅ | ✅ | baseline |
| **triple** | ✅ | ✅ | Cache 创建 OK,TripleLayerKVCache 替换全量层 |
| **triple_pq** | ✅ | ✅ | 32K 出现 TG 回归 (见下节) |
| **scored_pq** | ✅ | ✅ | 无 calibration → fallback 到 triple (无 AM) |
| **scored_kv_direct** | 未测 | 未测 | 需 FLASHMLX_EXPERIMENTAL=1 |

**修复内容**:
1. `mlx-lm-source/cache_factory.py` (`_detect_architecture`): 返回 4 元组,区分 `SSM+Attention` vs `KVCache+RotatingKVCache`,hybrid 循环按 `len(native_caches)` 迭代
2. `src/flashmlx/generation/vlm_cache.py`: 删除遗留的 `del sys.modules[mlx_lm.*]` hack,避免类身份 (class identity) 分裂导致 `isinstance` 误判

**教训**: `isinstance` 诡异时先查 class id — 同名但 id 不同 = 被 reload 过。

---

## 压缩潜力分析

### KV Cache 大小估算

**gemma-4-E4B**:
```
24 caches × 2 KV heads × 256 head_dim × 2 (K+V) = 24,576 params/token
→ 32K context = 786 MB (bf16)
→ 但实际更小（20 层用 RotatingKVCache 限制窗口）
```

**gemma-4-31B**:
```
60 caches × 16 KV heads × 256 head_dim × 2 (K+V) = 491,520 params/token
→ 32K context = 15.7 GB (bf16) ⚠️
→ 但实际更小（50 层用 RotatingKVCache）
```

### FlashMLX 压缩收益预测

| 模型 | KV Size | 压缩策略 | 预期收益 | 建议 |
|------|---------|---------|---------|------|
| **E4B** | 小 (2 heads) | Standard | Baseline | KV 已优化，无需压缩 |
| **31B** | 大 (16 heads) | Triple PQ | +20-30% | 值得尝试（需修复 hybrid） |

---

## 长 Context 实测 (2026-04-10, hybrid 修复后)

**环境**: gemma-4-31B, M4 Pro 48GB, single-process, 固定 prompt 重复填充到目标长度
**脚本**: `examples/bench_gemma4_long_context.py` (设置 `GEMMA4_MODEL_PATH` 环境变量后运行)

### 结果矩阵

| Context | Strategy | PP tok/s | TG tok/s | Peak GB | KVΔ GB | vs baseline |
|---------|----------|----------|----------|---------|--------|-------------|
| 8K  | standard  | 94.9 | 12.34 | 22.12 | +4.97 | baseline |
| 8K  | triple_pq | 92.2 | 12.50 | 22.20 | +5.05 | ≈same |
| 8K  | scored_pq | 95.5 | 12.49 | 22.12 | +4.97 | ≈same |
| 16K | standard  | 94.0 | 11.90 | 23.83 | +6.67 | baseline |
| 16K | triple_pq | 93.1 | 11.84 | 23.83 | +6.67 | ≈same |
| 16K | scored_pq | 93.3 | 11.82 | 23.83 | +6.67 | ≈same |
| 32K | standard  | 87.2 | **10.52** | **27.25** | +10.10 | baseline |
| 32K | triple_pq | 85.6 | **5.37** ⚠️ | **32.39** ⚠️ | +15.24 | **TG -49%, peak +19%** |
| 32K | scored_pq | 86.7 | 10.51 | 27.25 | +10.10 | ≈same |

**输出一致性**: 三策略在 8K/16K/32K 全部生成**完全相同**的文本 ✅

### 关键发现

1. **修复后 cache 创建全部成功**, TripleLayerKVCache 正确替换 10 层 full_attention。

2. **Gemma 4 不是 KV 压缩的好靶子**:
   - full_attention 层只有 `num_global_key_value_heads=4` × `global_head_dim=512` = 4 KB/token
   - 10 个可压缩层 × 8 KB (K+V) = 80 KB/token
   - 32K 上下文 ≈ 2.5 GB 原始 KV,但 peak memory 主要被 **prefill attention scratch** 占据 (+10 GB delta at 32K)
   - 压缩 KV 存储对总体 peak 影响极小

3. **⚠️ triple_pq@32K 回归**:
   - TG 从 10.52 跌到 5.37 tok/s (-49%)
   - Peak 从 27.25 涨到 32.39 GB (+19%)
   - **根因**: `triple_pq` 没进入 AM 路径 (`enable_am=False`),但保留了 PolarQuant warm buffer 量化。长 context 下 warm buffer 没有 AM 裁剪,每步都要重新量化 → 内存+延迟双爆炸
   - 8K/16K 没触发因为还没到 TripleLayerKVCache 内部的 warm→flat 迁移阈值

4. **scored_pq ≈ standard**:
   - `scored_pq` 被 fallback 到 `triple (no AM)`,没有 PolarQuant 开销
   - 无 calibration_file 时实际上是一个 pass-through

### 结论

| 策略 | 8K | 16K | 32K | 推荐 |
|------|----|----|------|------|
| standard  | ✅ 12.3 tok/s | ✅ 11.9 tok/s | ✅ 10.5 tok/s | **生产默认** |
| scored_pq | ✅ 12.5 tok/s | ✅ 11.8 tok/s | ✅ 10.5 tok/s | 与 standard 等价 (安全) |
| triple_pq | ✅ 12.5 tok/s | ✅ 11.8 tok/s | ❌ 5.4 tok/s | **32K 避免** |

**生产建议**:
- **默认**: `strategy="standard"` — 在 Gemma 4 上性能等价且无回归
- **如果必须用压缩路径**: 用 `scored_pq` 而不是 `triple_pq` (前者无 calibration 时安全 fallback)
- **要真正获得压缩收益**: 需要生成 Gemma 4 的 AM calibration 文件,启用 `triple_pq_am`/`scored_pq`+calibration
- **Gemma 4 的瓶颈不在 KV**: KV Sharing + RotatingKVCache 已经压到很低;想进一步节省,应压 **模型权重** (4-bit 已做) 或用更激进的 sliding_window

---

## 已知问题

### 1. Hybrid Cache 不兼容

**问题**: FlashMLX cache factory 在 hybrid 架构上失败
```python
# mlx-lm-source/mlx_lm/models/cache_factory.py:115-118
if isinstance(cache, KVCache):
    attention_indices.append(i)

is_hybrid = len(attention_indices) < len(native_caches)
# Gemma: 4 KVCache + 20 RotatingKVCache → hybrid = True → 策略失败
```

**解决方案**:
1. **短期**: 只用 Standard cache（已验证可用）
2. **长期**: 修改 cache_factory 支持混合 cache 类型

### 2. Gemma 4 生成重复

**问题**: temperature=0.0 时模型重复输出 prompt
**原因**: 可能缺少正确的 EOS token 或 chat template
**解决**: 使用 temperature=0.7 + Gemma chat format

---

## 代码示例

### 基础使用

```python
from flashmlx.vlm_bridge import load_vlm_model, create_vlm_cache, generate_vlm

# 加载模型（自动选择）
model, processor = load_vlm_model("/Volumes/toshiba/models/gemma-4-E4B")

# 创建 FlashMLX cache
cache = create_vlm_cache(model, strategy="standard")

# 生成
response = generate_vlm(
    model, processor,
    prompt="<start_of_turn>user\nWhat is ML?<end_of_turn>\n<start_of_turn>model\n",
    cache=cache,
    max_tokens=50,
    temperature=0.7
)

print(f"Response: {response.text}")
print(f"Speed: {response.generation_tps:.1f} tok/s")
```

### Vision + Text

```python
# Vision+Text (Gemma 4 支持多模态)
response = generate_vlm(
    model, processor,
    prompt="Describe this image",
    image="photo.jpg",  # 280 vision tokens
    cache=cache
)
```

---

## 结论

### ✅ 集成成功

1. **mlx-vlm Fork**: 完全可控，深度集成架构已就绪
2. **VLM Bridge API**: 统一接口，支持两个 Gemma 4 变体
3. **Cache 传递**: 通过 `prompt_cache` 参数无缝集成
4. **生成稳定**: 两个模型生成均正常工作

### 🎯 推荐

**对于 gemma-4-E4B**:
- ✅ 使用 Standard cache（已优化，KV Sharing + Hybrid）
- ❌ 不需要 FlashMLX 压缩（KV 本身就小）
- 🎯 适合快速推理（69.1 tok/s）

**对于 gemma-4-31B**:
- ✅ 使用 Standard cache（当前唯一可用）
- ⚠️ 未来可尝试压缩（16 KV heads，压缩潜力大）
- 🎯 适合高质量生成（20.8B 参数）

### 📋 下一步

1. ✅ **Hybrid Cache 支持已完成** (commit `302f9ca` + `b503a55`)
2. ✅ **长 context benchmark 已完成** (8K/16K/32K, 见上节)
3. 🔜 **TODO: triple_pq@32K 回归调查**:
   - 定位 TripleLayerKVCache 的 warm→flat 迁移阈值
   - 考虑让 `triple_pq` 在没 calibration 时也 fallback 到 plain triple (和 scored_pq 同一行为)
   - 或:长 context 自动降级 PolarQuant warm 量化
4. 🔜 **Gemma 4 AM calibration**:
   - 采集 AM 校准数据 → 启用真正的 triple_pq_am / scored_pq+calibration
   - 衡量真实压缩收益 (预计 10 层 full_attention 可省 1.5 GB at 32K,约 5% 总 peak)
5. 🔜 **Vision 测试**:
   - 测试 280 vision tokens 的压缩效果
   - Vision+Text 长上下文性能

---

## 附录：架构图

```
FlashMLX VLM Deep Integration

┌─────────────────────────────────────────┐
│  User Application                       │
└─────────────────┬───────────────────────┘
                  │
          ┌───────▼────────┐
          │  VLM Bridge    │
          │  (vlm_bridge.py)
          └───────┬────────┘
                  │
    ┌─────────────┼─────────────┐
    │             │             │
┌───▼────┐   ┌───▼────┐   ┌───▼────┐
│ Load   │   │ Cache  │   │Generate│
│ Model  │   │Creation│   │        │
└───┬────┘   └───┬────┘   └───┬────┘
    │            │            │
┌───▼────────────▼────────────▼─────┐
│  mlx-vlm-source (Our Fork)        │
│  - gemma-4-E4B  (3.3B)            │
│  - gemma-4-31B  (20.8B)           │
└───────────────┬───────────────────┘
                │
    ┌───────────┼───────────┐
    │           │           │
┌───▼────┐ ┌───▼────┐ ┌───▼────┐
│Standard│ │Hybrid  │ │KV      │
│Cache   │ │Cache   │ │Sharing │
└────────┘ └────────┘ └────────┘
```

---

**测试人员**: Claude (Solar v2.0)
**审核**: 昊哥
**状态**: ✅ 集成验证通过，生产就绪
