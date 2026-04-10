# MAC-Attention Metal Trace 分析指南

## 快速启动

```bash
open /tmp/mac_trace.gputrace
```

## 在 Instruments 中的关键步骤

### Step 1: 选择正确的 Instrument

1. 打开 trace 后，在左侧选择 **"GPU"** 或 **"Metal System Trace"**
2. 确保时间线视图可见

### Step 2: 定位 Attention Kernel

在顶部搜索框搜索：
- `attention`
- `sdpa` (scaled dot product attention)
- `matmul`

**预期找到**：
- Kernel 名称类似 `attention_kernel` 或 `sdp_attention`
- 每次 decode 应该有 36 次调用（36 层）

### Step 3: 检查 Buffer Bindings（关键证据）

**目标**：验证 attention kernel 读取的 K/V buffer 大小

1. 点击一个 attention kernel 调用
2. 在右侧 Inspector 中查看 **"Buffer Bindings"**
3. 记录：
   - K buffer size (bytes)
   - V buffer size (bytes)
   - 计算：实际读取的 token 数 = buffer_size / (n_kv_heads × head_dim × sizeof(dtype))

**关键对比**：
- Full context: 1000 tokens
- Skip ratio: 66%
- 如果只读 partial: 应该约 340 tokens
- 如果读 full: 1000 tokens

**铁证条件**：
- 如果 buffer size 对应 1000 tokens → **证明** MLX 读取完整 KV cache
- 如果 buffer size 对应 340 tokens → **推翻** 我们的假设（但概率极低）

### Step 4: 分析 Memory Bandwidth

1. 在 Instruments 左侧选择 **"Memory"** instrument
2. 查看时间线上的内存带宽使用
3. 对比：
   - Attention kernel 执行期间的带宽峰值
   - 理论带宽（M4 Max: ~400 GB/s）

**关键问题**：
- Bandwidth utilization 是否接近峰值？
- 是否存在 memory stall？

### Step 5: Kernel 耗时分布

1. 选择 **"Time Profiler"** 视图
2. 按 kernel 分组统计
3. 记录：
   - `attention_kernel`: X ms
   - `match_kernel` (如果有): X ms
   - `merge_kernel` (如果有): X ms

**对比我们的 Python 统计**：
- Attn Time: 1152.07ms (Python 统计)
- 对比 Metal trace 的实际 kernel 时间

### Step 6: 2-Pass Chunking 证据

**寻找**：
- 同一个 attention 调用是否分成 2 个 Metal kernel？
- 或者一个 kernel 内部有 2 个 pass？

**方法**：
- 放大时间线，看单次 decode 的 attention kernel
- 查看是否有连续的两个相似 kernel 调用

## 预期发现总结

| 证据 | 预期结果 | 如果符合 → 结论 |
|------|----------|----------------|
| K/V buffer size | ~1000 tokens | MLX 读取完整 cache，不支持 partial |
| Memory bandwidth | 接近峰值 | Memory-bound，不是 compute-bound |
| Kernel count | 每层 1-2 个 | 确认 2-pass 架构 |
| Time 对比 | Metal ≈ Python | 统计数据可信 |

## 关键截图建议

如果要写最终报告，截图这些：
1. Buffer bindings 显示完整 K/V size
2. Memory bandwidth timeline（峰值使用）
3. Kernel duration breakdown
4. 2-pass 的证据（如果找到）

---

**分析完成后，更新到 `MAC_ATTENTION_EXPERIMENTAL.md`**
