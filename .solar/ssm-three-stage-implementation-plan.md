# SSM 三段式压缩实现计划

**日期**: 2026-03-21 17:00
**任务**: Task #53 - State-Memory 专用压缩算法 (新方案)
**方案来源**: 监护人提供的"三段式保守压缩"方案

---

## 核心思想

**不要每 token 压缩/解压，只压缩不活跃的 state**

```
Hot State (live path)  → 不压缩，精确执行
Warm State (idle)      → 压缩存储（critical channels + low-rank bulk）
Cold State (archive)   → Snapshot/spill
```

**关键突破**：
- 之前失败原因：每 token 解压→执行→压缩 (19x 慢)
- 新方案：只在事件边界解压，hot path 不压缩

---

## Phase 1: Critical Channels Profiling (1-2 天)

### 目标

识别每层 SSM state 中的关键通道（控制中文/英文、<think>、格式等功能）

### 输入

- 模型：Qwen3.5-35B-A3B
- 测试 prompt：包含中文、<think>、格式控制
- SSM layers: 30 layers (layer 0-39 中的 SSM 层)

### 输出

每层一个 calibration file：

```json
{
  "layer": 12,
  "rank": 32,
  "critical_channels": [1, 7, 22, 41],
  "basis_file": "layer12_basis.npy",
  "safe": true,
  "profiling_metadata": {
    "test_prompt": "...",
    "num_channels": 128,
    "critical_ratio": 0.05,
    "importance_scores": [...]
  }
}
```

### 实现步骤

#### 1.1 设计测试 Prompt

```python
TEST_PROMPT = """请用中文回答以下问题：

<think>
首先，我需要理解这个问题...
</think>

问题：什么是机器学习？

回答：
"""
```

**关键要素**：
- 中文 tokens
- `<think>` 标签
- 格式控制（换行、列表等）

#### 1.2 实现 Channel Perturbation

```python
def perturb_channel(state, layer_idx, channel_idx, strength=0.1):
    """
    微扰单个 channel

    Args:
        state: (B, Hv, Dv, Dk) - SSM state
        layer_idx: 层索引
        channel_idx: 通道索引 (0-127 for Dv dimension)
        strength: 扰动强度

    Returns:
        perturbed_state: 扰动后的 state
    """
    # Clone state
    perturbed = state.copy()

    # 对该层该通道所有 heads 添加噪声
    # channel_idx 对应 Dv 维度的第 channel_idx 行
    perturbed[:, :, channel_idx, :] += strength * mx.random.normal(
        shape=perturbed[:, :, channel_idx, :].shape
    )

    return perturbed
```

#### 1.3 实现影响度量

```python
def measure_impact(
    model, tokenizer,
    original_state, perturbed_state,
    layer_idx, test_prompt
):
    """
    测量扰动对输出的影响

    Returns:
        {
            'chinese_prob_change': float,   # 中文 token 概率变化
            'think_tag_change': float,      # <think> tag 概率变化
            'logits_kl': float,             # Top-k logits KL divergence
            'format_change': float,         # 格式 token 概率变化
            'overall_score': float          # 综合分数
        }
    """
    # 1. 用原始 state 生成
    original_logits = generate_with_state(
        model, tokenizer, test_prompt, original_state, layer_idx
    )

    # 2. 用扰动 state 生成
    perturbed_logits = generate_with_state(
        model, tokenizer, test_prompt, perturbed_state, layer_idx
    )

    # 3. 测量各项指标
    chinese_prob_change = measure_chinese_prob_change(
        original_logits, perturbed_logits, tokenizer
    )

    think_tag_change = measure_special_token_change(
        original_logits, perturbed_logits,
        special_tokens=['<think>', '</think>']
    )

    logits_kl = compute_kl_divergence(original_logits, perturbed_logits)

    format_change = measure_format_token_change(
        original_logits, perturbed_logits,
        format_tokens=['\n', '：', '、', '1.', '2.']
    )

    # 4. 综合打分
    overall_score = (
        0.3 * chinese_prob_change +
        0.3 * think_tag_change +
        0.2 * logits_kl +
        0.2 * format_change
    )

    return {
        'chinese_prob_change': chinese_prob_change,
        'think_tag_change': think_tag_change,
        'logits_kl': logits_kl,
        'format_change': format_change,
        'overall_score': overall_score
    }
```

#### 1.4 实现 Profiling 主流程

```python
def profile_layer_critical_channels(
    model, tokenizer,
    layer_idx,
    test_prompt,
    num_channels=128,
    critical_ratio=0.05
):
    """
    Profile 单层的 critical channels

    Returns:
        critical_channels: List[int] - 关键通道索引
        importance_scores: List[float] - 每个通道的重要性分数
    """
    print(f"Profiling layer {layer_idx}...")

    # 1. 获取 baseline state (运行一次前向传播)
    baseline_state = capture_ssm_state(model, tokenizer, test_prompt, layer_idx)

    importance_scores = []

    # 2. 对每个 channel 进行 profiling
    for channel_idx in range(num_channels):
        print(f"  Channel {channel_idx}/{num_channels}...", end='\r')

        # 微扰该 channel
        perturbed_state = perturb_channel(
            baseline_state, layer_idx, channel_idx
        )

        # 测量影响
        impact = measure_impact(
            model, tokenizer,
            baseline_state, perturbed_state,
            layer_idx, test_prompt
        )

        importance_scores.append(impact['overall_score'])

    # 3. 排序并选出 top p%
    sorted_indices = np.argsort(importance_scores)[::-1]
    num_critical = int(num_channels * critical_ratio)
    critical_channels = sorted_indices[:num_critical].tolist()

    print(f"\n✅ Layer {layer_idx}: {num_critical} critical channels identified")
    print(f"   Top 5 channels: {critical_channels[:5]}")
    print(f"   Scores: {[importance_scores[i] for i in critical_channels[:5]]}")

    return critical_channels, importance_scores
```

#### 1.5 批量 Profiling 所有 SSM 层

```python
def profile_all_ssm_layers(
    model, tokenizer,
    ssm_layer_indices,
    output_dir=".solar/calibration"
):
    """
    Profile 所有 SSM 层
    """
    os.makedirs(output_dir, exist_ok=True)

    for layer_idx in ssm_layer_indices:
        # Profile 该层
        critical_channels, importance_scores = profile_layer_critical_channels(
            model, tokenizer, layer_idx, TEST_PROMPT
        )

        # 保存 calibration file
        calibration = {
            "layer": layer_idx,
            "rank": 32,  # 默认 rank
            "critical_channels": critical_channels,
            "safe": True,
            "profiling_metadata": {
                "test_prompt": TEST_PROMPT,
                "num_channels": len(importance_scores),
                "critical_ratio": 0.05,
                "importance_scores": importance_scores
            }
        }

        output_path = os.path.join(output_dir, f"layer_{layer_idx}_calibration.json")
        with open(output_path, 'w') as f:
            json.dump(calibration, f, indent=2)

        print(f"✅ Saved: {output_path}\n")

    print(f"✅ All {len(ssm_layer_indices)} layers profiled!")
```

### 实现文件

- `mlx-lm-source/mlx_lm/compaction/critical_channels_profiler.py` - Profiling 工具
- `benchmarks/profile_critical_channels.py` - 运行 profiling 的脚本
- `.solar/calibration/layer_X_calibration.json` - 输出的 calibration files

### 预期耗时

- 每层 profiling：~5-10 分钟（128 channels × 前向传播）
- 30 个 SSM 层：~2.5-5 小时
- **可并行优化**：batch 多个 channels 一起测试

---

## Phase 2: 三段式缓存实现 (1-2 天)

### 目标

实现 Hot/Warm/Cold 三段式状态管理

### 核心组件

#### 2.1 ThreeStageSSMCache

```python
class ThreeStageSSMCache:
    """
    三段式 SSM State 管理

    - Hot: 最近 M steps，不压缩
    - Warm: 中期状态，压缩（critical + low-rank）
    - Cold: 归档状态，snapshot
    """

    def __init__(self, calibration_configs, hot_window_size=10):
        self.calibration = calibration_configs  # 每层的 calibration
        self.hot_window = deque(maxlen=hot_window_size)
        self.warm_store = {}
        self.cold_archive = {}

    def update(self, step_idx, state_dict):
        """更新 cache（每个 token 调用）"""
        # 新 state 进入 hot
        self.hot_window.append(state_dict)

        # Hot 满了 → 最老的进入 Warm
        if len(self.hot_window) == self.hot_window.maxlen:
            old_state = self.hot_window[0]
            self.warm_store[step_idx - self.hot_window.maxlen] = \
                self._compress_to_warm(old_state)

        # Warm 太多 → 最老的进入 Cold
        if len(self.warm_store) > 100:
            oldest_idx = min(self.warm_store.keys())
            self.cold_archive[oldest_idx] = \
                self._snapshot_to_cold(self.warm_store.pop(oldest_idx))

    def get_live_state(self, layer_idx):
        """获取 live decode 使用的 state（只返回 hot）"""
        return self.hot_window[-1][layer_idx]

    def _compress_to_warm(self, state_dict):
        """压缩 state 到 warm store"""
        compressed = {}
        for layer_idx, state in state_dict.items():
            config = self.calibration[layer_idx]

            # 分离 critical 和 bulk
            critical = state[:, :, config['critical_channels'], :]
            bulk_mask = np.ones(state.shape[2], dtype=bool)
            bulk_mask[config['critical_channels']] = False
            bulk = state[:, :, bulk_mask, :]

            # Low-rank 压缩 bulk
            # TODO: 使用预计算的 basis
            U, S, Vt = np.linalg.svd(bulk.reshape(-1, bulk.shape[-1]))
            z = U[:, :config['rank']] @ np.diag(S[:config['rank']])

            compressed[layer_idx] = {
                'critical': critical,
                'z': z,
                'Vt': Vt[:config['rank'], :]
            }

        return compressed
```

#### 2.2 集成到 generate() 流程

修改 `mlx_lm/utils.py` 中的 `generate()` 函数，使用 `ThreeStageSSMCache`

### 实现文件

- `mlx-lm-source/mlx_lm/compaction/three_stage_cache.py` - 三段式缓存
- `mlx-lm-source/mlx_lm/compaction/calibration_loader.py` - 加载 calibration files

---

## Phase 3: 集成测试 (1 天)

### 测试目标

1. **速度测试**：验证 hot path 速度不变（~25 tok/s）
2. **质量测试**：验证输出质量正常（中文、<think>、格式）
3. **内存测试**：验证累积内存节省

### 测试脚本

- `benchmarks/three_stage_performance_test.py` - 性能测试
- `benchmarks/three_stage_quality_test.py` - 质量测试
- `benchmarks/three_stage_memory_test.py` - 内存测试

---

## 成功标准

| 指标 | 目标 | 测试方法 |
|------|------|----------|
| **速度** | ≥ 24 tok/s (不低于 baseline 的 95%) | 生成 100 tokens 测速 |
| **质量** | 输出正常（无乱码、无 <think> 泄漏） | 对比 baseline 输出 |
| **内存** | 累积节省 > 30% (Warm + Cold) | 长文本生成内存跟踪 |

---

## 风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| Profiling 耗时太长 | Batch 多个 channels 一起测试；减少测试 tokens |
| Critical channels 识别不准 | 多种测试 prompts；调整权重 |
| Warm 压缩仍影响速度 | 只在真正 idle 时压缩；增大 hot window |

---

*Phase 1 开始时间: 2026-03-21 17:00*
*预计完成时间: 2026-03-22*
