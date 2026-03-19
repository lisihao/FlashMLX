# FlashMLX Profiling 工具规划

**日期**: 2026-03-18
**状态**: 📝 规划阶段
**优先级**: 🔥 最高（优化前置依赖）

---

## ⚡ 铁律：Profiler First

```
┌─────────────────────────────────────────────────────────────────┐
│                   PROFILER FIRST PRINCIPLE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   任何优化前必须先用 Profiler 分析                              │
│                                                                 │
│   1. 优化前 → 先 Profile → 找瓶颈 → 再优化                     │
│   2. 优化后 → 再 Profile → 验证效果                            │
│   3. 分析不到位 → 先优化 Profiler → 再继续优化系统             │
│                                                                 │
│   ❌ 禁止: 凭感觉/猜测进行优化                                  │
│   ❌ 禁止: 没有数据支撑的优化方案                               │
│   ✅ 必须: 基于 Profiler 数据做决策                            │
│   ✅ 必须: 有 before/after 对比数据                            │
│                                                                 │
│   Profiler 不够精确 = 优先优化 Profiler                        │
│   而不是盲目优化系统                                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 铁律详解

**1. 优化前强制 Profile**
- 任何优化 PR 必须附带 profiling 数据
- 必须明确指出优化目标函数及其占比
- 占比 < 5% 的函数不值得优化

**2. 优化后强制验证**
- 提交优化 PR 时必须附带对比数据
- 格式: `baseline.json` vs `optimized.json`
- 必须证明提升 > 3%（否则不值得引入复杂度）

**3. Profiler 迭代优化**
- 发现 Profiler 盲区 → 立即停止系统优化
- 优先完善 Profiler → 再继续系统优化
- Profiler 的准确性 > 系统优化的速度

**4. 反例：禁止模式**
```
❌ "我觉得这里可能慢" → 直接优化
❌ "理论上这个算法更快" → 直接替换
❌ "其他项目这么做的" → 直接照搬

✅ Profile → 发现瓶颈 → 设计优化 → Profile 验证
```

### 教训来源

**ThunderLLAMA 教训**:
- 尝试了 10+ 个优化，只有 2 个有效
- 原因：没有精确的 profiling 数据
- 浪费时间在占比 < 1% 的操作上

**正确流程**:
```
1. Profile baseline
2. 识别 Top 3 瓶颈（占比 > 80%）
3. 设计针对性优化
4. Profile 验证
5. 如果提升不明显 → 回到步骤 1，优化 Profiler
```

---

## 🎯 目标

构建一个完整的、深入的、全面的 profiling 工具，用于：
1. **性能分析** - 找出 MLX-LM 和 MLX 的性能瓶颈
2. **优化验证** - 验证优化效果是否显著
3. **回归检测** - 防止性能回退

**核心理念**: 证据驱动优化 (Evidence-Driven Optimization)

---

## 📐 架构设计

### 三层架构

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 3: Analysis & Visualization (分析层)                 │
│  - Flame Graph 生成                                         │
│  - 瓶颈识别                                                  │
│  - 优化建议                                                  │
│  - 对比分析                                                  │
└─────────────────────────────────────────────────────────────┘
                            ↑
┌─────────────────────────────────────────────────────────────┐
│  Layer 2: Instrumentation & Tracing (插桩层)               │
│  - Python 函数 Hook                                         │
│  - Metal Kernel Hook                                        │
│  - 内存分配追踪                                              │
│  - 调用栈捕获                                                │
└─────────────────────────────────────────────────────────────┘
                            ↑
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: Data Collection (数据收集层)                      │
│  - 时间戳记录                                                │
│  - 参数捕获                                                  │
│  - 性能计数器                                                │
│  - 日志输出                                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔍 核心功能模块

### 模块 1: Python 层打桩 (Python Instrumentation)

**目标**: 在不修改 MLX 源码的情况下，hook 所有关键函数

**实现方式**:
```python
# 方式 1: Decorator 装饰器
@profile("flash_attention")
def flash_attention(q, k, v):
    ...

# 方式 2: Monkey Patching
original_matmul = mx.matmul
mx.matmul = profiled_matmul(original_matmul)

# 方式 3: Context Manager
with Profiler("inference"):
    output = model.generate(prompt)
```

**捕获数据**:
- 函数名
- 输入参数形状和类型
- 执行时间 (wall time, CPU time)
- 内存占用 (before/after)
- 调用次数
- 调用栈

**关键函数列表**:
```python
# MLX Core
- mx.matmul
- mx.fast.scaled_dot_product_attention
- mx.fast.rope
- mx.fast.rms_norm
- mx.quantize / mx.dequantize

# MLX-LM
- generate_step()
- apply_repetition_penalty()
- sample()
```

---

### 模块 2: Metal Kernel 追踪 (Metal Kernel Tracing)

**目标**: 追踪 Metal GPU 执行细节

**实现方式**:
```python
# 方式 1: Metal Performance HUD (系统工具)
# MTL_HUD_ENABLED=1 python script.py

# 方式 2: Metal Performance Shaders Profiling
# 插入 MTLCaptureScope

# 方式 3: 自定义 Metal 计时器
# 在 kernel 入口/出口插入时间戳
```

**捕获数据**:
- Kernel 名称 (gemv_4bit, flash_attention, etc.)
- 启动次数
- 线程组配置 (threadgroup size)
- 执行时间 (GPU time)
- 内存带宽使用
- 寄存器压力

**关键 Kernel 列表**:
```
- scaled_dot_product_attention
- gemv (量化)
- matmul_4bit
- rms_norm
- rope
- quantize/dequantize
```

---

### 模块 3: 内存追踪 (Memory Profiling)

**目标**: 追踪内存分配和释放，找出内存瓶颈

**实现方式**:
```python
# 方式 1: tracemalloc (Python 标准库)
import tracemalloc
tracemalloc.start()
# ... run code ...
snapshot = tracemalloc.take_snapshot()

# 方式 2: Metal Memory Tracking
# 追踪 MTLBuffer 分配

# 方式 3: mx.metal.set_memory_limit() 监控
```

**捕获数据**:
- 峰值内存占用
- 内存分配次数
- 内存碎片化
- KV Cache 占用
- Activation 占用

---

### 模块 4: 日志分析 (Log Analysis)

**目标**: 从日志中提取性能数据并生成报告

**实现方式**:
```python
# 日志格式 (JSON Lines)
{
  "timestamp": 1234567890.123,
  "event": "function_call",
  "name": "flash_attention",
  "duration_ms": 1.23,
  "input_shapes": [[1, 128, 32, 64]],
  "memory_mb": 512
}
```

**分析功能**:
- 解析 JSON 日志
- 聚合统计 (min/max/avg/p50/p99)
- 时间线可视化
- 热点识别 (占比 > 5% 的操作)
- 对比分析 (baseline vs optimized)

---

### 模块 5: 可视化 (Visualization)

**目标**: 直观展示性能数据

**实现方式**:

#### 5.1 Flame Graph (火焰图)
```
┌────────────────────────────────────────────┐
│ generate()              100%               │
├────────────────────────────────────────────┤
│ flash_attention() 45% │ matmul() 30% │... │
├───────────────────────┼──────────────┼─────┤
│ gemv() 20% │ rope() │ rms_norm()    │     │
└────────────┴────────┴───────────────┴─────┘
```

工具: `py-spy`, `flamegraph.pl`, 或自定义生成器

#### 5.2 Timeline (时间线)
```
Time (ms)  0    10   20   30   40   50   60
────────┬────┬────┬────┬────┬────┬────┬────
Flash    ████████               ████████
Attention
MatMul       ████████████   ████████
RMSNorm           ██     ██     ██
```

工具: Chrome Trace Format (chrome://tracing)

#### 5.3 Hotspot Table (热点表格)
```
┌────────────────────┬──────────┬──────────┬──────────┐
│ Function           │ Time (ms)│ Calls    │ % Total  │
├────────────────────┼──────────┼──────────┼──────────┤
│ flash_attention    │ 45.2     │ 32       │ 45%      │
│ matmul_4bit        │ 30.1     │ 128      │ 30%      │
│ gemv_quantized     │ 12.3     │ 256      │ 12%      │
│ rms_norm           │ 8.4      │ 64       │ 8%       │
│ Others             │ 4.0      │ ...      │ 5%       │
└────────────────────┴──────────┴──────────┴──────────┘
```

---

## 📂 目录结构

```
FlashMLX/
├── src/flashmlx/
│   └── profiler/
│       ├── __init__.py
│       ├── instrumentation.py      # Python 层打桩
│       ├── metal_trace.py          # Metal kernel 追踪
│       ├── memory.py               # 内存追踪
│       ├── logger.py               # 日志输出
│       ├── analyzer.py             # 日志分析
│       └── visualizer.py           # 可视化生成
├── benchmarks/
│   └── profile_model.py            # 端到端 profiling 脚本
└── tools/
    ├── generate_flamegraph.py      # 火焰图生成器
    ├── generate_timeline.py        # 时间线生成器
    └── compare_profiles.py         # 对比分析工具
```

---

## 🔧 API 设计

### 1. 简单模式 (Context Manager)

```python
from flashmlx.profiler import Profiler

# 最简单的用法
with Profiler("my_experiment"):
    output = model.generate(prompt, max_tokens=100)

# 自动生成报告: my_experiment_profile.json
```

### 2. 精细模式 (Manual Control)

```python
from flashmlx.profiler import Profiler, InstrumentationLevel

profiler = Profiler(
    name="detailed_analysis",
    level=InstrumentationLevel.DETAILED,  # BASIC, DETAILED, FULL
    capture_memory=True,
    capture_kernels=True,
    output_dir="./profiling_data"
)

profiler.start()

# 手动标记区域
with profiler.region("prefill"):
    model.forward(tokens)

with profiler.region("decode"):
    for _ in range(100):
        next_token = model.generate_step()

profiler.stop()

# 生成报告
profiler.generate_report()
profiler.generate_flamegraph()
profiler.generate_timeline()
```

### 3. 函数装饰器

```python
from flashmlx.profiler import profile

@profile("forward_pass", capture_args=True)
def forward(self, x):
    return self.model(x)
```

### 4. 对比分析

```python
from flashmlx.profiler import compare_profiles

baseline = "baseline_profile.json"
optimized = "optimized_profile.json"

report = compare_profiles(baseline, optimized)
print(report.summary())
# Output:
# Total time: 100ms -> 85ms (-15%)
# flash_attention: 45ms -> 38ms (-15.5%)
# matmul: 30ms -> 28ms (-6.7%)
```

---

## 📊 输出格式

### 1. JSON 日志 (原始数据)

```json
{
  "metadata": {
    "experiment_name": "baseline_qwen_7b",
    "timestamp": "2026-03-18T21:00:00Z",
    "device": "Apple M4 Pro",
    "mlx_version": "0.31.1"
  },
  "events": [
    {
      "event_type": "function_call",
      "name": "flash_attention",
      "timestamp": 1234567890.123,
      "duration_ms": 1.23,
      "input_shapes": [[1, 128, 32, 64]],
      "memory_allocated_mb": 512,
      "gpu_utilization": 0.85
    }
  ]
}
```

### 2. Markdown 报告

```markdown
# Profiling Report: baseline_qwen_7b

## Summary
- Total Time: 100.5 ms
- Total Calls: 512
- Peak Memory: 2.3 GB

## Top 10 Hotspots
1. flash_attention - 45.2ms (45%)
2. matmul_4bit - 30.1ms (30%)
...

## Optimization Opportunities
🔥 flash_attention 占 45% 时间 - 建议优化
⚠️  gemv_quantized 调用 256 次 - 考虑 batching
```

### 3. Chrome Trace Format (时间线)

```json
{
  "traceEvents": [
    {
      "name": "flash_attention",
      "cat": "compute",
      "ph": "X",
      "ts": 1234567890123,
      "dur": 1230,
      "pid": 1,
      "tid": 1
    }
  ]
}
```

可用 `chrome://tracing` 查看

---

## 🎯 实现阶段

### Phase 1: 基础框架 (2-3 天)
- [ ] 创建 `profiler/` 目录结构
- [ ] 实现 `Profiler` 类 (context manager)
- [ ] 实现 Python 函数 hook (装饰器 + monkey patching)
- [ ] 实现基础日志输出 (JSON)
- [ ] 单元测试

**交付物**:
- 可以 profile Python 层函数
- 生成 JSON 日志
- 基础单元测试通过

### Phase 2: Metal Kernel 追踪 (2-3 天)
- [ ] 集成 Metal Performance HUD
- [ ] 实现 Metal kernel 计时
- [ ] 捕获 kernel 启动参数
- [ ] 记录 GPU 利用率

**交付物**:
- 可以追踪 Metal kernel 执行
- 记录 GPU 时间和参数

### Phase 3: 日志分析 & 可视化 (2-3 天)
- [ ] 实现 JSON 日志解析器
- [ ] 实现聚合统计 (min/max/avg/p99)
- [ ] 生成火焰图 (Flame Graph)
- [ ] 生成时间线 (Timeline)
- [ ] 生成 Markdown 报告

**交付物**:
- 完整的日志分析工具
- 火焰图、时间线、报告生成

### Phase 4: 高级功能 (1-2 天)
- [ ] 内存追踪
- [ ] 对比分析
- [ ] 优化建议生成
- [ ] 端到端测试

**交付物**:
- 完整的 profiling 工具链
- 对比分析功能
- 自动优化建议

---

## 🔍 关键技术点

### 1. 零开销 (Zero Overhead when Disabled)

```python
# 使用环境变量控制
FLASHMLX_PROFILE=0  # 禁用 (zero overhead)
FLASHMLX_PROFILE=1  # 基础 profiling
FLASHMLX_PROFILE=2  # 详细 profiling
```

### 2. 最小侵入性 (Non-Invasive)

- 不修改 MLX 源码
- 使用 Python 标准 hook 机制
- 可以随时启用/禁用

### 3. 线程安全 (Thread-Safe)

```python
import threading
lock = threading.Lock()

def log_event(event):
    with lock:
        events.append(event)
```

### 4. 异步日志 (Async Logging)

```python
# 避免日志写入影响性能
from queue import Queue
log_queue = Queue()

def async_logger():
    while True:
        event = log_queue.get()
        write_to_file(event)
```

---

## 📈 验收标准

| 标准 | 要求 |
|------|------|
| **功能完整性** | 支持 Python 层、Metal 层、内存追踪 |
| **性能开销** | Profiling 开启时性能损失 < 5% |
| **易用性** | 一行代码启用: `with Profiler():` |
| **输出丰富** | JSON 日志 + Markdown 报告 + 火焰图 |
| **准确性** | 时间测量误差 < 1% |
| **可扩展性** | 易于添加新的 hook 点 |

---

## 🎓 参考资料

### 类似工具
1. **py-spy** - Python profiler
2. **cProfile** - Python 标准库
3. **torch.profiler** - PyTorch profiler
4. **Instruments.app** - Apple 性能分析工具
5. **Chrome Tracing** - 时间线可视化

### Metal Profiling
1. **Metal Performance HUD** - 实时 GPU 监控
2. **Metal System Trace** - Instruments 模板
3. **MTLCaptureManager** - 编程式捕获

---

## 🚀 下一步

**等待批准后开始实施 Phase 1**

预计总时长: 7-10 天
- Phase 1: 2-3 天
- Phase 2: 2-3 天
- Phase 3: 2-3 天
- Phase 4: 1-2 天

**关键问题**:
1. 是否需要支持分布式 profiling？
2. 是否需要实时可视化（Web UI）？
3. 日志存储格式偏好？(JSON Lines vs SQLite vs Parquet)

---

*规划文档版本: v1.0*
*作者: Solar + Claude Sonnet 4.5*
*日期: 2026-03-18*
