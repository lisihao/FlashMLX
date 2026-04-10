# Phase 2: VLM 架构移植详细计划

**制定日期**: 2026-04-09
**预计时间**: 3-4 周
**目标**: 移植 MLX-VLM Vision Encoder，实现端到端 VLM 推理

---

## 目标

### 功能目标
- ✅ 支持图像输入 (Qwen2-VL, Qwen3-VL)
- ✅ 端到端推理 (image + text → text)
- ✅ 保留 FlashMLX 所有优化 (Route 0-5)

### 性能目标
- ✅ 文本模型性能不回退 (≥ Phase 0 红线)
- ✅ VLM 推理可用 (TTFT < 20s)
- ⚠️  暂不优化 Vision KV (Phase 3 处理)

---

## 架构设计

### 整体架构

```python
# src/flashmlx/models/qwen3_vlm.py (新建)

class FlashMLXQwen3VL(nn.Module):
    """
    FlashMLX VLM = MLX-VLM (Vision) + FlashMLX (Language + Optimization)
    """
    def __init__(self, config):
        # 1. Vision Encoder (从 MLX-VLM 移植)
        self.vision_tower = VisionModel(config.vision_config)

        # 2. Visual Projection
        self.visual_projection = nn.Linear(
            config.vision_config.hidden_size,  # 3584 (Vision)
            config.text_config.hidden_size,    # 4096 (Language)
        )

        # 3. Language Model (FlashMLX 现有，带 Route 0-5)
        from mlx_lm.models import qwen3
        self.language_model = qwen3.Model(config.text_config)

    def __call__(self, input_ids, pixel_values=None, cache=None):
        # 1. 获取 embeddings (文本 + 图像)
        inputs_embeds = self.get_input_embeddings(input_ids, pixel_values)

        # 2. 应用 FlashMLX KV cache
        if cache is None and pixel_values is not None:
            vision_token_count = self._calculate_vision_tokens(pixel_values)
            cache = make_prompt_cache(
                self.language_model,
                kv_cache="scored_pq",
                recent_size=512 + vision_token_count,  # 动态 L0
            )

        # 3. Language Model 推理
        return self.language_model(
            None,  # 不用 input_ids
            cache=cache,
            input_embeddings=inputs_embeds
        )
```

---

## 移植任务分解

### Week 1: Vision Encoder 移植 (5 天)

#### Task 1.1: 创建 Vision 模块 (2 天)

**文件**: `src/flashmlx/models/vision.py` (新建)

**内容**:
```python
# 从 MLX-VLM 复制以下类:
- VisionRotaryEmbedding
- PatchEmbed
- PatchMerger
- Attention
- MLP
- Qwen2VLVisionBlock
- VisionModel

# 代码量: 310 行
```

**验证**:
```python
# 单元测试
import mlx.core as mx
from flashmlx.models.vision import VisionModel

config = VisionConfig(...)
model = VisionModel(config)
pixel_values = mx.random.normal([1, 3, 448, 448])
output = model(pixel_values, grid_thw)

assert output.shape == (256, 3584)  # 256 vision tokens
```

#### Task 1.2: 图像预处理 (1 天)

**文件**: `src/flashmlx/processors/image_processing.py` (新建)

**内容**:
```python
from PIL import Image
import numpy as np

class ImageProcessor:
    def __init__(self, image_size=448):
        self.image_size = image_size

    def preprocess(self, image: Image.Image):
        # 1. Resize
        image = image.resize((self.image_size, self.image_size))

        # 2. To numpy
        pixel_values = np.array(image).astype(np.float32)

        # 3. Normalize (ImageNet stats)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        pixel_values = (pixel_values / 255.0 - mean) / std

        # 4. To MLX array
        pixel_values = mx.array(pixel_values).transpose(2, 0, 1)  # HWC → CHW
        return pixel_values

# 代码量: ~100 行
```

**验证**:
```python
from PIL import Image
from flashmlx.processors import ImageProcessor

processor = ImageProcessor()
image = Image.open("cat.jpg")
pixel_values = processor.preprocess(image)

assert pixel_values.shape == (3, 448, 448)
```

#### Task 1.3: Vision 单元测试 (2 天)

**文件**: `tests/test_vision_encoder.py`

**测试用例**:
```python
def test_patch_embed():
    """测试 PatchEmbed"""
    pass

def test_vision_block():
    """测试 VisionBlock"""
    pass

def test_vision_model_output_shape():
    """测试 VisionModel 输出形状"""
    pass

def test_vision_model_vs_mlx_vlm():
    """对比 FlashMLX vs MLX-VLM 输出一致性"""
    # 加载相同权重，对比输出
    pass
```

---

### Week 2: 模型集成 (5 天)

#### Task 2.1: 创建 VLM 模型入口 (2 天)

**文件**: `src/flashmlx/models/qwen3_vlm.py` (新建)

**内容**:
```python
class FlashMLXQwen3VL(nn.Module):
    def __init__(self, config):
        # Vision
        self.vision_tower = VisionModel(config.vision_config)
        self.visual_projection = nn.Linear(...)

        # Language (复用 FlashMLX)
        from mlx_lm.models import qwen3
        self.language_model = qwen3.Model(config.text_config)

    def get_input_embeddings(self, input_ids, pixel_values):
        # 实现图像 + 文本融合逻辑
        pass

    def sanitize(self, weights):
        # 权重加载和映射
        pass

# 代码量: ~200 行
```

#### Task 2.2: 权重加载实现 (2 天)

**挑战**: VLM 权重结构复杂

**步骤**:
1. 分析 Hugging Face 权重格式
2. 实现 `sanitize()` 方法
3. 测试权重加载

**测试**:
```python
from flashmlx.models import FlashMLXQwen3VL

model = FlashMLXQwen3VL.from_pretrained(
    "mlx-community/Qwen2-VL-7B-Instruct-4bit"
)

# 验证权重加载成功
assert hasattr(model, 'vision_tower')
assert hasattr(model, 'language_model')
```

#### Task 2.3: 端到端推理测试 (1 天)

**测试脚本**: `tests/test_vlm_e2e.py`

```python
def test_text_only():
    """测试纯文本推理 (确保不回退)"""
    model = FlashMLXQwen3VL.from_pretrained(...)
    response = generate(model, tokenizer, prompt="Hello")
    assert len(response) > 0

def test_image_text():
    """测试图像 + 文本推理"""
    model = FlashMLXQwen3VL.from_pretrained(...)
    image = Image.open("cat.jpg")
    response = generate(
        model, tokenizer,
        prompt="描述这张图片",
        image=image
    )
    assert len(response) > 0
    assert "cat" in response.lower()  # 简单验证
```

---

### Week 3: FlashMLX 优化集成 (5 天)

#### Task 3.1: 动态 L0 boundary (2 天)

**目标**: 调整 Route 3 支持 Vision tokens

**修改文件**: `src/flashmlx/triple_layer_cache.py`

```python
class TripleLayerKVCache:
    def __init__(
        self,
        recent_size=512,
        vision_token_count=0,  # ✅ 新增参数
        ...
    ):
        # 动态调整 L0
        effective_recent_size = recent_size + vision_token_count
        self.keys_recent = mx.zeros([n_kv_heads, effective_recent_size, head_dim])
        ...
```

**cache_factory 适配**:
```python
def make_prompt_cache(model, vision_token_count=0, **kwargs):
    if kwargs.get("kv_cache") == "scored_pq":
        return TripleLayerKVCache(
            recent_size=512,
            vision_token_count=vision_token_count,  # ✅ 传入
            ...
        )
```

#### Task 3.2: H0Store 分离 (1 天)

**修改文件**: `src/flashmlx/h0_store.py`

```python
class H0Store:
    def __init__(self, max_tokens, hidden_size, split_vision=False):
        if split_vision:
            # 分离存储
            self._h0_text = mx.zeros([max_text_tokens, hidden_size])
            self._h0_vision = mx.zeros([max_vision_tokens, hidden_size])
        else:
            # 统一存储 (向后兼容)
            self._h0 = mx.zeros([max_tokens, hidden_size])

    def capture(self, h0, token_range, token_type="text"):
        if hasattr(self, '_h0_text'):
            if token_type == "vision":
                self._h0_vision[token_range] = h0
            else:
                self._h0_text[token_range] = h0
        else:
            self._h0[token_range] = h0
```

#### Task 3.3: 优化回归测试 (2 天)

**验证**: FlashMLX 优化在 VLM 下仍然有效

**测试脚本**: `tests/test_vlm_optimization.py`

```python
def test_route0_vlm():
    """测试 Route 0 (Density Router) 在 VLM 下工作"""
    model = FlashMLXQwen3VL.from_pretrained(...)

    # balanced mode
    cache = make_prompt_cache(model, density_scale=0.0)
    response1 = generate(model, ..., cache=cache)

    # ultra_long mode
    cache = make_prompt_cache(model, density_scale=1.5)
    response2 = generate(model, ..., cache=cache)

    # 输出应一致
    assert response1 == response2

def test_route3_vlm():
    """测试 Route 3 (KV Cache) 在 VLM 下工作"""
    model = FlashMLXQwen3VL.from_pretrained(...)

    # standard
    cache = make_prompt_cache(model, kv_cache=None)
    response1 = generate(model, ..., cache=cache)

    # scored_pq
    cache = make_prompt_cache(model, kv_cache="scored_pq",
                               vision_token_count=256)
    response2 = generate(model, ..., cache=cache)

    # 输出应一致
    assert response1 == response2
```

---

### Week 4: 质量验证 (5 天)

#### Task 4.1: VQA 测试 (2 天)

**使用 Phase 0 准备的数据**: 1,100 VQA 样本

**测试脚本**: `tests/test_vlm_vqa.py`

```python
def test_vqav2():
    """VQAV2 样本测试"""
    model = FlashMLXQwen3VL.from_pretrained(...)

    dataset = load_vqav2_samples("tests/datasets/vqav2/val_samples.json")

    correct = 0
    for sample in dataset[:100]:  # 前 100 个
        image = Image.open(sample["image"])
        response = generate(model, tokenizer,
                            prompt=sample["question"],
                            image=image)

        # 简单匹配
        if any(ans["answer"] in response.lower()
               for ans in sample["answers"]):
            correct += 1

    accuracy = correct / 100
    print(f"VQAV2 Accuracy: {accuracy:.2%}")

    # 期望: >60% (基本可用)
    assert accuracy > 0.6
```

#### Task 4.2: 性能 Benchmark (2 天)

**测试脚本**: `benchmarks/bench_vlm.py`

```python
def bench_vlm_ttft():
    """测试 VLM TTFT"""
    model = FlashMLXQwen3VL.from_pretrained(...)
    image = Image.open("tests/fixtures/images/cat.jpg")

    start = time.time()
    response = generate(model, tokenizer,
                        prompt="What is in this image?",
                        image=image,
                        max_tokens=50)
    ttft = time.time() - start

    print(f"TTFT: {ttft:.2f}s")

    # 期望: <20s (Phase 2 目标)
    assert ttft < 20.0

def bench_vlm_throughput():
    """测试 VLM TG tok/s"""
    # 类似测试
    pass
```

#### Task 4.3: 文档和示例 (1 天)

**文件**: `examples/vlm_inference.py`

```python
"""FlashMLX VLM 推理示例"""

from flashmlx.models import FlashMLXQwen3VL
from flashmlx import generate
from PIL import Image

# 加载模型
model = FlashMLXQwen3VL.from_pretrained(
    "mlx-community/Qwen2-VL-7B-Instruct-4bit"
)

# 加载图像
image = Image.open("cat.jpg")

# 推理
response = generate(
    model,
    tokenizer,
    prompt="Describe this image in detail",
    image=image,
    max_tokens=100,
    # FlashMLX 优化
    kv_cache="scored_pq",
    density_mode="balanced",
)

print(response)
```

---

## 成功标准

### 功能标准
- [ ] 支持图像输入 (PIL Image)
- [ ] 支持纯文本推理 (向后兼容)
- [ ] 支持图像 + 文本推理
- [ ] 权重加载成功 (Qwen2-VL, Qwen3-VL)

### 性能标准
- [ ] 文本模型性能无回退 (32K TG ≥ 19.5 tok/s)
- [ ] VLM TTFT < 20s
- [ ] VLM TG tok/s ≥ 15 (可接受)
- [ ] 内存占用 < 12GB (Vision + Text)

### 质量标准
- [ ] VQAV2 准确率 > 60%
- [ ] 输出一致性 (standard vs optimized)
- [ ] 无 crash 或 NaN

---

## 风险和缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| **权重加载失败** | 高 | 高 | 参考 MLX-VLM，逐步调试 |
| **Vision 输出不一致** | 中 | 高 | 与 MLX-VLM 逐层对比 |
| **KV Cache 溢出** | 中 | 中 | 动态 L0，充分测试 |
| **性能回退** | 低 | 高 | 保留 Phase 0 baseline |

---

## 时间线

```
Week 1: Vision Encoder 移植       [=========>          ] 45% (5 天)
Week 2: 模型集成                  [=========>          ] 45% (5 天)
Week 3: FlashMLX 优化集成         [======>             ] 30% (5 天)
Week 4: 质量验证                  [====>               ] 20% (5 天)

总计: 20 工作日 (4 周)
```

---

## 交付物

1. **代码**:
   - `src/flashmlx/models/vision.py` (310 行)
   - `src/flashmlx/models/qwen3_vlm.py` (200 行)
   - `src/flashmlx/processors/image_processing.py` (100 行)
   - 优化适配修改 (~150 行)

2. **测试**:
   - `tests/test_vision_encoder.py`
   - `tests/test_vlm_e2e.py`
   - `tests/test_vlm_optimization.py`
   - `tests/test_vlm_vqa.py`

3. **文档**:
   - `examples/vlm_inference.py`
   - `PHASE2_COMPLETION_REPORT.md`

4. **Benchmark**:
   - `benchmarks/bench_vlm.py`
   - VLM 性能报告

---

**下一步**: 完成 Day 2 回归测试，验证文本性能基线稳定
