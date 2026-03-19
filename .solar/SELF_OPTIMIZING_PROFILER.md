# 自我优化 Profiler 设计

**日期**: 2026-03-18
**状态**: 📝 设计阶段
**优先级**: 🔥 高（Profiler 质量决定优化效果）

---

## 🎯 核心理念

**让 Profiler 具有自我诊断和自我优化能力**

```
传统 Profiler:
    User → Profiler → 发现盲区 → 手动改进 Profiler

自我优化 Profiler:
    User → Profiler → 自动发现盲区 → 自动优化自己 → 重新 Profile
```

---

## 🧠 四大自我优化机制

### 机制 1: 盲区检测 (Blind Spot Detection)

**问题**: Profiler 可能遗漏关键热点

**检测方法**:
```python
# 方法 1: 总时间验证
total_profiled_time = sum(all_function_times)
total_wallclock_time = end_time - start_time

if total_profiled_time < total_wallclock_time * 0.8:
    # 有 20% 以上的时间未被追踪
    blind_spot_detected = True
```

```python
# 方法 2: 函数调用链完整性检查
if model.generate() 总时间 != sum(子函数时间):
    # 存在未被 hook 的子函数
    missing_functions_detected = True
```

```python
# 方法 3: Metal GPU 利用率检查
if CPU_profiled_time >> GPU_time:
    # 可能遗漏了 GPU kernel 追踪
    kernel_tracking_missing = True
```

**自动修复**:
1. 扫描调用栈，找出未 hook 的函数
2. 自动添加 hook
3. 重新 profile

---

### 机制 2: 粒度自适应 (Adaptive Granularity)

**问题**: 粒度太粗 → 找不到瓶颈；粒度太细 → 开销太大

**自适应策略**:
```python
class AdaptiveProfiler:
    def __init__(self):
        self.granularity = "coarse"  # 初始粗粒度

    def optimize_granularity(self, results):
        """根据结果自动调整粒度"""
        for func in results.top_hotspots():
            if func.time_percent > 20%:
                if func.has_children:
                    # 热点函数 → 增加子函数追踪
                    self.add_detailed_hooks(func)
                    self.granularity = "fine"
            else:
                # 非热点函数 → 移除详细追踪
                self.remove_hooks(func)
```

**示例**:
```
Pass 1 (粗粒度):
- generate() 占 100%
- 无法定位瓶颈

自动优化: 增加 generate() 内部函数追踪

Pass 2 (细粒度):
- flash_attention() 占 45%
- matmul() 占 30%
- 成功定位瓶颈！
```

---

### 机制 3: 准确性自验证 (Accuracy Self-Verification)

**问题**: Profiler 本身可能有误差

**验证方法**:
```python
# 方法 1: 多次运行一致性检查
run1 = profile(model, "run1")
run2 = profile(model, "run2")
run3 = profile(model, "run3")

variance = calculate_variance([run1, run2, run3])
if variance > 10%:
    # 结果不稳定 → 增加 warmup 次数或采样率
    self.increase_stability()
```

```python
# 方法 2: 跨层验证
python_layer_time = sum(all_python_functions)
metal_layer_time = sum(all_metal_kernels)

if abs(python_layer_time - metal_layer_time) > 5%:
    # Python 层和 Metal 层时间不一致
    self.add_synchronization_points()
```

```python
# 方法 3: 开销测量
with_profiler_time = profile_and_run()
without_profiler_time = run_only()

overhead = (with_profiler_time - without_profiler_time) / without_profiler_time

if overhead > 5%:
    # 开销太大 → 减少插桩点
    self.reduce_instrumentation()
```

---

### 机制 4: 智能插桩点选择 (Smart Hook Selection)

**问题**: 不知道应该 hook 哪些函数

**自动选择策略**:
```python
class SmartHookSelector:
    def select_hooks(self, codebase):
        """自动选择应该 hook 的函数"""

        # 规则 1: Hook 所有外部调用
        hooks = find_all_mlx_calls(codebase)

        # 规则 2: Hook 所有循环内的操作
        for loop in find_loops(codebase):
            hooks.extend(find_calls_in_loop(loop))

        # 规则 3: Hook 所有 Metal kernel 启动
        hooks.extend(find_metal_kernel_calls())

        # 规则 4: 基于历史数据优化
        if has_previous_profile():
            # 只 hook 上次占比 > 1% 的函数
            hooks = filter_by_historical_importance(hooks)

        return hooks
```

**动态调整**:
```python
# Profile 后分析
results = profile_with_hooks(hooks)

for func in results:
    if func.time_percent > 5%:
        # 热点函数 → 增加更详细的 hook
        add_detailed_hooks(func)
    elif func.time_percent < 0.1%:
        # 冷函数 → 移除 hook（减少开销）
        remove_hooks(func)

# 重新 profile（更精确）
```

---

## 🔄 自我优化循环

```
┌─────────────────────────────────────────────────────────────┐
│ Pass 0: 初始状态                                            │
│ - 粗粒度插桩                                                │
│ - 只 hook 主要函数                                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Pass 1: 初次 Profiling                                     │
│ with Profiler() as p:                                       │
│     model.generate()                                        │
│ → 生成 profile_pass1.json                                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 自我诊断                                                    │
│ issues = p.self_diagnose()                                  │
│ → 发现: 20% 时间未追踪到                                    │
│ → 发现: flash_attention 占 45% 但没有细节                  │
│ → 发现: 开销 < 2% (正常)                                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 自我优化                                                    │
│ p.self_optimize(issues)                                     │
│ → 添加 Metal kernel 追踪 (解决 20% 盲区)                   │
│ → 增加 flash_attention 内部 hook (获取细节)                │
│ → 移除冷函数 hook (减少开销)                                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Pass 2: 重新 Profiling                                     │
│ with Profiler() as p:  # 已优化配置                        │
│     model.generate()                                        │
│ → 生成 profile_pass2.json                                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 验证改进                                                    │
│ improvement = compare(pass1, pass2)                         │
│ → 盲区: 20% → 5% (✓ 改进)                                  │
│ → 细节: 无 → 26 个 Metal kernel (✓ 改进)                   │
│ → 开销: 2% → 3% (✓ 可接受)                                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 决策                                                        │
│ if improvement.is_significant():                            │
│     保存优化后的配置                                        │
│     p.save_optimized_config()                               │
│ else:                                                       │
│     回滚到上一版本                                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ API 设计

### 1. 自动模式 (全自动)

```python
from flashmlx.profiler import SelfOptimizingProfiler

# 最简单：全自动优化
profiler = SelfOptimizingProfiler(
    auto_optimize=True,    # 自动优化
    max_passes=3           # 最多迭代 3 次
)

with profiler:
    model.generate(prompt)

# Profiler 自动完成:
# 1. 初次 profile
# 2. 发现盲区
# 3. 自动优化配置
# 4. 重新 profile
# 5. 验证改进
# 6. 生成最终报告
```

### 2. 半自动模式 (需确认)

```python
profiler = SelfOptimizingProfiler(auto_optimize=False)

# Pass 1: 初次 profile
with profiler.pass_1():
    model.generate(prompt)

# 自我诊断
issues = profiler.self_diagnose()
print(issues)
# Output:
# ⚠️  Blind spot: 20% time untracked
# ⚠️  flash_attention lacks detail
# ✓ Overhead: 2% (acceptable)

# 生成优化建议
suggestions = profiler.suggest_optimizations(issues)
print(suggestions)
# Output:
# 1. Add Metal kernel tracing (+15% coverage, +1% overhead)
# 2. Add flash_attention hooks (+detail, +0.5% overhead)
# 3. Remove cold function hooks (-0.3% overhead)

# 用户确认后应用
profiler.apply_optimizations(suggestions)

# Pass 2: 重新 profile
with profiler.pass_2():
    model.generate(prompt)

# 验证改进
improvement = profiler.verify_improvement()
```

### 3. 手动模式 (完全控制)

```python
profiler = SelfOptimizingProfiler()

# 手动配置优化策略
profiler.set_blind_spot_threshold(0.1)  # 10% 盲区即优化
profiler.set_max_overhead(0.05)         # 最大 5% 开销
profiler.enable_adaptive_granularity()

# 手动触发诊断
with profiler:
    model.generate(prompt)

diagnostics = profiler.diagnose()
if diagnostics.has_issues():
    profiler.optimize()
    profiler.rerun()
```

---

## 📊 自我诊断指标

### 1. 覆盖率 (Coverage)

```python
coverage = sum(profiled_functions_time) / total_wallclock_time

✓ coverage >= 95%  # 优秀
⚠️  coverage 80-95%  # 可接受
❌ coverage < 80%   # 需要优化
```

### 2. 粒度评分 (Granularity Score)

```python
# 热点函数应该有详细追踪
for hotspot in top_10_hotspots:
    if hotspot.time_percent > 20% and not hotspot.has_detailed_trace:
        granularity_score -= 10

✓ score >= 80  # 粒度合适
⚠️  score 60-80  # 需要调整
❌ score < 60   # 粒度不足
```

### 3. 准确性评分 (Accuracy Score)

```python
# 多次运行的一致性
runs = [profile() for _ in range(3)]
variance = calculate_variance(runs)

accuracy_score = 100 - (variance * 10)

✓ score >= 90  # 准确
⚠️  score 80-90  # 可接受
❌ score < 80   # 不稳定
```

### 4. 开销评分 (Overhead Score)

```python
overhead = (with_profiler - without_profiler) / without_profiler

✓ overhead < 5%   # 优秀
⚠️  overhead 5-10%  # 可接受
❌ overhead > 10%  # 太高
```

### 5. 综合健康分数

```
health_score = (
    coverage_score * 0.4 +
    granularity_score * 0.3 +
    accuracy_score * 0.2 +
    overhead_score * 0.1
)

✓ health >= 85  # 健康
⚠️  health 70-85  # 需要优化
❌ health < 70   # 不健康
```

---

## 🎯 自我优化决策树

```
┌─────────────────────────────────────────────────────────────┐
│ 开始 Profiling                                              │
└─────────────────────────────────────────────────────────────┘
                            ↓
                    [ Coverage < 80% ? ]
                     /              \
                   Yes               No
                    ↓                ↓
        ┌───────────────────┐    [ Granularity < 70 ? ]
        │ 增加插桩点         │     /              \
        │ - 扫描调用栈      │   Yes               No
        │ - 添加 Metal hook │    ↓                ↓
        └───────────────────┘  ┌────────────┐  [ Accuracy < 80 ? ]
                                │ 增加细粒度  │   /           \
                                │ - Hook 子函数│ Yes           No
                                └────────────┘  ↓             ↓
                                            ┌──────────┐  [ Overhead > 10% ? ]
                                            │ 增加采样  │   /           \
                                            │ - Warmup  │ Yes           No
                                            │ - 多次运行│  ↓             ↓
                                            └──────────┘ ┌─────────┐  ✓ 健康
                                                        │ 减少插桩 │
                                                        │ - 移除冷函│
                                                        └─────────┘
                                                             ↓
                                                      重新 Profile
```

---

## 🔍 实现技术

### 1. 调用栈扫描

```python
import inspect
import sys

def scan_call_stack():
    """扫描调用栈，找出未 hook 的函数"""
    frame = sys._getframe()
    unhooked_functions = []

    while frame is not None:
        function_name = frame.f_code.co_name
        if not is_hooked(function_name):
            unhooked_functions.append(function_name)
        frame = frame.f_back

    return unhooked_functions
```

### 2. 动态 Hook 注入

```python
def inject_hook_dynamically(function_name):
    """运行时动态注入 hook"""
    module = sys.modules[get_module(function_name)]
    original_func = getattr(module, function_name)

    def wrapped_func(*args, **kwargs):
        start = time.perf_counter()
        result = original_func(*args, **kwargs)
        duration = time.perf_counter() - start
        log_event(function_name, duration)
        return result

    setattr(module, function_name, wrapped_func)
```

### 3. Metal Kernel 自动追踪

```python
def auto_trace_metal_kernels():
    """自动检测并追踪所有 Metal kernel 调用"""

    # Hook mx.fast.* 所有函数
    for attr in dir(mx.fast):
        if not attr.startswith('_'):
            original = getattr(mx.fast, attr)
            setattr(mx.fast, attr, wrap_with_metal_trace(original))
```

### 4. 自适应采样

```python
class AdaptiveSampler:
    def __init__(self):
        self.hot_functions = set()
        self.cold_functions = set()

    def should_sample(self, function_name):
        """决定是否采样此函数调用"""
        if function_name in self.hot_functions:
            return True  # 热点函数：每次都采样
        elif function_name in self.cold_functions:
            return random.random() < 0.01  # 冷函数：1% 采样
        else:
            return True  # 未知函数：初次采样

    def update_from_results(self, results):
        """根据结果更新采样策略"""
        for func, time_percent in results:
            if time_percent > 5%:
                self.hot_functions.add(func)
            elif time_percent < 0.1%:
                self.cold_functions.add(func)
```

---

## 📈 优化效果预期

### 场景 1: 发现盲区

**问题**: 初次 profile 显示 generate() 占 100%，无细节

**自动优化**:
- Pass 1: 粗粒度，coverage = 60%
- 自我诊断: 40% 盲区
- 自动添加子函数 hook
- Pass 2: 细粒度，coverage = 95%
- 结果: 成功定位 flash_attention (45%), matmul (30%)

**收益**: 从"无法定位"到"精确定位"

### 场景 2: 平衡精度和开销

**问题**: 开启全部 hook 后开销 15%，太高

**自动优化**:
- Pass 1: 全部 hook，overhead = 15%
- 自我诊断: 开销过高
- 移除占比 < 0.1% 的函数 hook
- Pass 2: 优化后，overhead = 4%
- 结果: 保持 90% 准确性，开销降到 4%

**收益**: 精度损失 < 5%，开销降低 73%

### 场景 3: 自动增加 Metal 追踪

**问题**: Python 层显示 30% 时间在 matmul，但不知道具体哪个 kernel

**自动优化**:
- Pass 1: 只有 Python 层，无 Metal 细节
- 自我诊断: 缺少 GPU 追踪
- 自动启用 Metal Performance HUD
- Pass 2: 获得 26 个 kernel 的详细数据
- 结果: 发现 gemv_4bit 占 matmul 的 80%

**收益**: 从"模糊"到"精确到 kernel"

---

## 🎓 配置持久化

```python
# 自动保存优化后的配置
profiler.save_config("optimized_config.json")

# 下次直接使用优化配置
profiler = SelfOptimizingProfiler.from_config("optimized_config.json")
```

配置文件示例:
```json
{
  "version": "1.0",
  "optimized_at": "2026-03-18T21:00:00Z",
  "hooks": {
    "flash_attention": {"level": "detailed", "sample_rate": 1.0},
    "matmul": {"level": "detailed", "sample_rate": 1.0},
    "gemv": {"level": "basic", "sample_rate": 1.0},
    "rope": {"level": "none", "sample_rate": 0.0}
  },
  "metal_tracing": true,
  "memory_tracking": true,
  "diagnostics": {
    "coverage": 0.95,
    "granularity_score": 85,
    "accuracy_score": 92,
    "overhead": 0.04
  }
}
```

---

## 📊 自我优化报告

```markdown
# Profiler Self-Optimization Report

## Summary
- Passes: 2
- Optimization Time: 12.3s
- Final Health Score: 92/100

## Pass 1: Initial Profile
- Coverage: 60% ⚠️
- Granularity: 45 ❌
- Accuracy: 88 ✓
- Overhead: 2% ✓
- Health: 62 ❌

### Issues Detected
1. 40% blind spot (untracked time)
2. Insufficient granularity for hotspots
3. Missing Metal kernel tracking

## Auto-Optimization Applied
1. Added 15 new hooks (covering 35% more code)
2. Enabled Metal Performance HUD
3. Increased granularity for top 3 hotspots

## Pass 2: Optimized Profile
- Coverage: 95% ✓
- Granularity: 85 ✓
- Accuracy: 90 ✓
- Overhead: 4% ✓
- Health: 92 ✓

### Improvements
- Coverage: +35% (60% → 95%)
- Granularity: +40 (45 → 85)
- Overhead: +2% (2% → 4%, acceptable)

## Conclusion
✅ Profiler successfully self-optimized
✅ Ready for accurate system profiling
```

---

## 🚀 实现阶段

### Phase 1: 基础自我诊断 (1-2 天)
- [ ] 实现 coverage 检测
- [ ] 实现 overhead 测量
- [ ] 实现基础诊断报告

### Phase 2: 自动优化 (2-3 天)
- [ ] 实现动态 hook 注入
- [ ] 实现自适应粒度调整
- [ ] 实现自动重新 profile

### Phase 3: 智能决策 (1-2 天)
- [ ] 实现优化决策树
- [ ] 实现配置持久化
- [ ] 实现改进验证

### Phase 4: 高级功能 (1-2 天)
- [ ] 实现 Metal 自动追踪
- [ ] 实现多次运行一致性检查
- [ ] 实现自我优化报告生成

---

## ✅ 验收标准

| 标准 | 要求 |
|------|------|
| **自动发现盲区** | 覆盖率 < 80% 时自动检测 |
| **自动优化** | 无需人工干预即可优化配置 |
| **改进验证** | 自动对比优化前后效果 |
| **配置持久化** | 保存优化配置供后续使用 |
| **健康评分** | 综合评分 >= 85 |

---

*Self-Optimizing Profiler Design v1.0*
*设计于: 2026-03-18*
*核心理念: 让工具具有自我改进能力*
