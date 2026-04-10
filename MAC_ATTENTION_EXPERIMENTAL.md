# MAC-Attention 实验性实现

⚠️ **不推荐使用** - 仅供研究参考

## 概述

MAC-Attention (Match-Amend-Complete) 是一种通过复用相似 query 的 attention 结果来加速长上下文推理的技术。

本实现是对 [官方 CUDA 版本](https://github.com/YJHMITWEB/MAC-Attention.git) 的 MLX 移植，用于 Apple Silicon。

## 实验结果

### 性能数据（Qwen3-8B, M4 Max 64GB）

| Context | Standard | MAC | 加速比 | Hit Rate | Skip Ratio |
|---------|----------|-----|--------|----------|------------|
| 20K | 0.17 tok/s | 0.16 tok/s | **0.92×** | 82.6% | 65.7% |
| 30K | 0.10 tok/s | 0.10 tok/s | **0.99×** | 82.7% | 66.0% |
| 40K | 0.08 tok/s | 0.08 tok/s | **1.01×** | 82.7% | 66.0% |

**结论**：即使 82% hit rate 和 66% skip ratio，加速比仅 0.92×-1.02×，**几乎无加速**。

### 核心问题分析

#### 1. MLX Attention 对 Partial 输入无线性加速

实测数据（单次 decode, Attn 组件）：

| Full Context | Partial Context | Attn Time | 理论加速 | 实际加速 |
|--------------|-----------------|-----------|----------|----------|
| 10K | 3.4K (66% skip) | 1.32ms | 2.9× | - |
| 20K | 6.8K (66% skip) | 2.37ms | 2.9× | - |
| **增长** | **2×** | **+79%** | - | **1.13×** |

**关键发现**：
- Skip ratio 稳定在 66%（只计算 34% context）
- 但 Attn Time 从 1.32ms → 2.37ms (+79%)
- **时间主要取决于 full context 长度，而非 partial 长度**

#### 2. 根本原因

1. **MLX decode attention 架构**
   - MLX issue 明确提到：decode 的 vector SDPA path (qL ≤ 8) 使用 2-pass chunking
   - 不是简单的 "输入变短 → 时间线性减少" 的模型

2. **Memory bandwidth 瓶颈**
   - Peak Memory: 9.38GB (10K) → 10.11GB (20K) (+7.8%)
   - 需要读取/调度整个 KV cache
   - Partial 输入没有减少内存访问量

3. **额外开销**
   - Match: 1.70ms
   - Merge: 1.99ms
   - Cache update: 0.33ms
   - **总额外开销 ~4ms**，抵消了 partial attention 的收益

#### 3. 与官方 CUDA 实现的对比

| 维度 | 官方 (CUDA) | 本实现 (MLX) |
|------|-------------|--------------|
| 硬件 | NVIDIA H100 | Apple M4 Max |
| Attention kernel | FlashInfer (优化) | MLX SDPA (通用) |
| Partial 加速 | ✅ 线性 | ❌ 非线性 |
| 加速比 | 10-30× (batch serving) | 0.92-1.02× |
| 场景 | 64 并发请求 | 单用户推理 |

**差异来源**：
1. CUDA 有专门的 partial attention kernel
2. 官方实现针对 batch serving（跨请求复用）
3. MLX 的通用 attention 路径没有针对 partial 输入优化

## 实现细节

### 已完成
- ✅ Pre-RoPE matching (hit rate 82%)
- ✅ LSE-based merge (向量化优化)
- ✅ Ring cache (query/attn/lse)
- ✅ Skip ratio 达到 66%

### 未完成（也无需完成）
- ❌ Paged KV cache
- ❌ Per-request ring
- ❌ Async rectification
- ❌ Custom Metal kernel

**原因**：核心瓶颈在 MLX attention kernel，优化其他部分无法解决根本问题。

## 正确的迁移路线（未来参考）

如果要在 MLX 上实现真正有效的 MAC-Attention，需要：

### 第一步：研究版（算法验证）✅ 已完成
- Pre-RoPE matching
- Query ring + Summary ring
- LSE-based merge
- 验证 hit rate / skip ratio

### 第二步：Cache 抽象
- MACLayerState (per-layer ring)
- MACPagedKVCache
- Block table 支持

### 第三步：拆分 Critical Path
- K1 Match (critical)
- K2 Decode (critical)
- K3 Rectify+Append (async, 非 critical)

### 第四步：Custom Metal Kernel ⚠️ **最关键**
- **Match kernel**: 局部窗口 L2 scan
- **Band+tail decode kernel**: 真正的 partial attention（支持线性加速）
- **Rectify+append kernel**: 异步更新

**结论**：没有 custom Metal kernel，MAC-Attention 在 MLX 上无法达到有效加速。

## 使用说明（不推荐）

```python
import flashmlx

# 启用 MAC（会显示警告）
flashmlx.patch_mlx_lm()

# 正常使用 mlx-lm
from mlx_lm import load, generate
model, tokenizer = load("Qwen/Qwen2.5-8B")
generate(model, tokenizer, "你好", max_tokens=100)

# 禁用 MAC
flashmlx.unpatch_mlx_lm()
```

## 文件结构

```
src/flashmlx/
├── mac_simplified.py      # SimplifiedMAC 实现（裁剪自官方）
├── patch.py               # Monkey patch mlx-lm
└── ...

tests/
├── test_mac_final.py      # 端到端性能测试
├── test_real_hit_rate.py  # Hit rate 测试
└── profile_mac_memory.py  # 内存和性能分析
```

## 参考

- 论文：[arXiv:2604.00235](https://arxiv.org/abs/2604.00235) (MLSys 2026)
- 官方实现：https://github.com/YJHMITWEB/MAC-Attention.git
- MLX 文档：https://ml-explore.github.io/mlx/

## 结论

**MAC-Attention 在 MLX/Apple Silicon 上无法达到预期加速效果**。

核心原因是 MLX 的 decode attention 路径对 partial 输入没有线性加速。要实现有效加速，需要编写 custom Metal kernel，这超出了本项目的范围。

**不推荐在生产环境使用**。仅作为研究参考和未来优化的基础。

---

*实验日期：2026-04-05*
*硬件：Apple M4 Max, 64GB*
*软件：MLX 0.x, mlx-lm 0.x*
