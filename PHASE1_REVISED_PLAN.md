# Phase 1 修订计划：轻量级分析 (不 merge)

**修订日期**: 2026-04-09
**原因**: 发现 MLX-VLM 提供完整 Vision 实现，MLX-LM 只有半成品

---

## 目标调整

**原目标**: Git merge 最新 MLX-LM + 文本模型回归测试
**新目标**: 分析 VLM 接口 + 验证文本性能 + 制定移植计划

---

## 任务清单 (3 天)

### Day 1: VLM 接口分析

**任务 1.1**: 深入分析 MLX-LM VLM wrapper
```bash
# 读取并理解
cd mlx-lm-source
cat mlx_lm/models/qwen3_vl.py
cat mlx_lm/models/qwen2_vl.py

# 记录:
# - ModelArgs 结构
# - __call__ 签名
# - sanitize() 逻辑
# - layers 属性
```

**任务 1.2**: Clone 并分析 MLX-VLM
```bash
cd /tmp
git clone https://github.com/Blaizzy/mlx-vlm.git
cd mlx-vlm

# 重点阅读:
cat mlx_vlm/models/qwen2_vl/qwen2_vl.py
cat mlx_vlm/models/qwen2_vl/vision.py
cat mlx_vlm/models/qwen2_vl/language.py

# 记录:
# - VisionTransformer 实现
# - 权重加载逻辑
# - 图像预处理流程
```

**输出**: `VLM_INTERFACE_ANALYSIS.md`

---

### Day 2: 文本模型回归测试

**任务 2.1**: 重跑 Phase 0 baseline
```bash
# 确保 Phase 0 结果可复现
python3 benchmarks/bench_card.py \
  /Volumes/toshiba/models/qwen3-8b-mlx \
  --contexts 32768 \
  --tg-tokens 100

# 对比 PHASE0_BASELINE_REPORT.md
# 允许 ±3% 波动
```

**任务 2.2**: 测试 FlashMLX 优化稳定性
```bash
# Route 0
python3 benchmarks/bench_density_modes.py \
  /Volumes/toshiba/models/qwen3-8b-mlx \
  --contexts 32768

# Route 3
python3 benchmarks/bench_card.py \
  /Volumes/toshiba/models/qwen3-8b-mlx \
  --contexts 4096,8192,16384,32768
```

**验证标准**:
- 32K TG tok/s ≥ 19.5 (Phase 0 红线)
- 输出完全一致
- 无 regression

**输出**: `PHASE1_REGRESSION_REPORT.md`

---

### Day 3: 移植计划制定

**任务 3.1**: 设计 FlashMLX VLM 架构
```python
# 草图设计
class FlashMLXQwen3VL:
    """
    混合架构:
    - Wrapper: 参考 MLX-LM (57 行)
    - Vision: 移植 MLX-VLM (~150 行)
    - 优化: FlashMLX Routes (现有)
    """
    pass
```

**任务 3.2**: 评估移植工作量
```markdown
| 组件 | 来源 | 代码量 | 依赖 | 难度 |
|------|------|--------|------|------|
| VisionTransformer | MLX-VLM | 150 行 | mlx.nn | ⭐⭐ |
| 权重加载 | 自研 | 50 行 | safetensors | ⭐⭐⭐ |
| KV 优化适配 | 自研 | 100 行 | FlashMLX | ⭐⭐⭐⭐ |
```

**任务 3.3**: 制定 Phase 2 详细计划
```markdown
Week 1: 移植 VisionTransformer
Week 2: 实现权重加载
Week 3: 基础推理测试
Week 4: FlashMLX 优化集成
```

**输出**: `PHASE2_MIGRATION_PLAN.md`

---

## 成功标准

- [ ] MLX-LM VLM 接口完全理解
- [ ] MLX-VLM Vision 实现分析完成
- [ ] 文本模型性能无 regression (≥ Phase 0 红线)
- [ ] Phase 2 详细计划制定
- [ ] 工作量评估 (person-days)

---

## 时间估算

| 任务 | 时间 |
|------|------|
| Day 1: VLM 接口分析 | 8 小时 |
| Day 2: 回归测试 | 6 小时 |
| Day 3: 移植计划 | 6 小时 |
| **总计** | **3 天** |

vs 原计划 2 周 (merge + 解决冲突)

---

## 下一步

完成 Phase 1 后:
1. Review Phase 2 计划
2. 获得批准
3. 开始移植 VisionTransformer
