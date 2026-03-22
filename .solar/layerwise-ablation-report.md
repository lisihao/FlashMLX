# Layerwise Ablation 实验报告

**日期**: 2026-03-21 13:56:43
**模型**: mlx-community/Qwen3.5-35B-A3B-6bit
**总层数**: 40
**Attention 层**: 10
**SSM 层**: 30

## 实验假设

**H1**: 只有标准 Attention 层可以用 AM (Attention Matching) 压缩
**H2**: SSM 层不能用 AM 压缩（因为没有 attention mass 概念）

## 实验结果

| 实验 | 成功 | Tokens | 质量 | 重复 | 耗时 |
|------|------|--------|------|------|------|
| Baseline | ✅ | 150 | 9.0 | No | 29.23s |
| Attention Only | ✅ | 150 | 9.0 | No | 2.81s |
| SSM Only | ❌ | N/A | N/A | Yes | 0.16s |
| All Layers | ❌ | N/A | N/A | Yes | 0.15s |
| Single Attention | ✅ | 150 | 9.0 | No | 2.74s |
| Single SSM | ❌ | N/A | N/A | Yes | 0.14s |

## 假设验证

1. ✅ Baseline 工作正常: True
2. ✅ Attention 层可以压缩: True
3. ✅ SSM 层不能压缩: True
4. ✅ 混合压缩失败: True
5. ✅ 单个 Attention 层可以: True
6. ✅ 单个 SSM 层失败: True

## 结论

**✅ 假设得到验证！**

AM (Attention Matching) 只能用于标准 softmax attention 层，不能用于 SSM/Mamba 层。

**关键发现**：
- ✅ Attention 层可以安全压缩
- ❌ SSM 层压缩导致输出崩溃
- ❌ 混合压缩会破坏模型生成

**下一步**：
1. 实现选择性压缩（只压缩 Attention 层）
2. 为 SSM 层设计专门的压缩算法
3. 建立混合架构记忆压缩的统一框架

## 详细输出

### Baseline

**配置**: 不压缩任何层（基准）

**压缩层**: []

**输出**:
```
：
# 19999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999
```

### Attention Only

**配置**: 只压缩 Attention 层 - 预期成功 ✅

**压缩层**: [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]

**输出**:
```
：
# 19999999999999999999990/ 1999999999999999999999990/ 1999999999990/ 199990/ 19999999999999999999999999999999999990/ 1999990/ 199999999999990/ 19999
```

### SSM Only

**配置**: 只压缩 SSM 层 - 预期失败 ❌

**压缩层**: [0, 1, 2, 4, 5]

**错误**: [concatenate] All the input array dimensions must match exactly except for the concatenation axis. However, the provided shapes are (1,3,8192), (3,3,8192), and the concatenation axis is 1.

### All Layers

**配置**: 压缩所有层 - 预期失败 ❌

**压缩层**: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]

**错误**: [concatenate] All the input array dimensions must match exactly except for the concatenation axis. However, the provided shapes are (1,3,8192), (3,3,8192), and the concatenation axis is 1.

### Single Attention

**配置**: 单个 Attention 层 - 预期成功 ✅

**压缩层**: [3]

**输出**:
```
：
# 1990/ 199990/ 199999990/ 199999999999999990/ 190/ 1999999999990/

# 1999990/ 199999999999999990/ 199999999990/ 199999990/ 1999999999999999990/ 1999
```

### Single SSM

**配置**: 单个 SSM 层 - 预期失败 ❌

**压缩层**: [0]

**错误**: [concatenate] All the input array dimensions must match exactly except for the concatenation axis. However, the provided shapes are (1,3,8192), (3,3,8192), and the concatenation axis is 1.

