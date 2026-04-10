# VLM 接口深度分析

**分析日期**: 2026-04-09
**目标**: 理解 MLX-VLM 架构，为 FlashMLX 移植做准备

---

## 执行摘要

**核心发现**:
1. ✅ MLX-VLM 架构清晰，组件分离良好
2. ✅ Vision Encoder 只有 **310 行**，标准 Transformer
3. ✅ 与 FlashMLX 集成点明确（5 个接口）
4. ✅ 移植难度: **中等** (⭐⭐⭐)

---

## 1. MLX-VLM 架构全景

### 整体结构

```
mlx-vlm/mlx_vlm/models/qwen2_vl/
├── config.py              (86 行)  - 配置管理
├── vision.py              (310 行) - Vision Encoder ← 核心
├── language.py            (539 行) - Language Model (可复用)
├── qwen2_vl.py            (185 行) - 模型入口
└── processing_qwen2_vl.py (195 行) - 图像预处理
```

### 数据流

```
输入图像 (PIL/numpy)
    ↓
[ImageProcessor] - 预处理 (resize, normalize)
    ↓
pixel_values: [B, C, H, W]
    ↓
[VisionModel] - Vision Encoder
    ↓ 包含:
    - PatchEmbed: 图像 → patches
    - VisionBlocks: Transformer layers
    - PatchMerger: 合并 patches
    ↓
vision_features: [num_patches, hidden_size]
    ↓
[merge_input_ids_with_image_features] - 融合
    ↓
combined_embeds: [seq_len, hidden_size]
    ↓
[LanguageModel] - 文本生成
    ↓
logits: [seq_len, vocab_size]
```

---

## 2. Vision Encoder 详细分析

### 核心组件 (vision.py, 310 行)

#### 2.1 PatchEmbed (图像 → Patches)

```python
class PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 14,           # 每个 patch 大小
        temporal_patch_size: int = 2,   # 视频时间维度
        in_channels: int = 3,           # RGB
        embed_dim: int = 1152,          # 输出维度
    ):
        # 3D Conv: [B, 3, T, H, W] → [B, 1152, T', H', W']
        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=[temporal_patch_size, patch_size, patch_size],
            stride=[temporal_patch_size, patch_size, patch_size],
        )

    def __call__(self, hidden_states):
        # 输入: [B, 3, T, H, W]
        # 输出: [num_patches, embed_dim]
        hidden_states = self.proj(hidden_states)
        hidden_states = hidden_states.reshape(-1, self.embed_dim)
        return hidden_states
```

**关键参数**:
- 图像大小: 448x448 (Qwen2-VL)
- Patch 大小: 14x14
- Patches 数量: (448/14) × (448/14) = 32 × 32 = **1,024 patches**
- 输出: [1024, 1152]

#### 2.2 VisionRotaryEmbedding (位置编码)

```python
class VisionRotaryEmbedding(nn.Module):
    """2D RoPE for vision tokens"""
    def __init__(self, dim: int, theta: float = 10000.0):
        self.dim = dim
        self.theta = theta

    def __call__(self, seqlen: int):
        # 生成位置频率
        inv_freq = 1.0 / (self.theta ** (mx.arange(0, self.dim, 2) / self.dim))
        freqs = mx.outer(mx.arange(seqlen), inv_freq)
        return freqs
```

**特点**:
- 使用 2D RoPE (与文本的 1D RoPE 不同)
- 包含 h 和 w 两个维度的位置信息
- 对每个 patch 计算独立的位置编码

#### 2.3 Qwen2VLVisionBlock (Transformer Layer)

```python
class Qwen2VLVisionBlock(nn.Module):
    def __init__(self, config: VisionConfig):
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = Attention(dim=embed_dim, num_heads=16)
        self.mlp = MLP(dim=embed_dim, hidden_dim=embed_dim * mlp_ratio)

    def __call__(self, x, cu_seqlens, rotary_pos_emb):
        # Pre-norm Transformer
        x = x + self.attn(self.norm1(x), cu_seqlens, rotary_pos_emb)
        x = x + self.mlp(self.norm2(x))
        return x
```

**架构**: 标准 Pre-norm Transformer
- LayerNorm → Attention → Residual
- LayerNorm → MLP → Residual

#### 2.4 PatchMerger (Patch 合并)

```python
class PatchMerger(nn.Module):
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2):
        # 合并 2x2 patches → 1 token
        self.hidden_size = context_dim * (spatial_merge_size ** 2)
        self.ln_q = nn.LayerNorm(context_dim)
        self.mlp = [
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        ]

    def __call__(self, x):
        # 输入: [1024, 1152] (32x32 patches)
        # 输出: [256, 3584] (16x16 merged tokens)
        x = self.ln_q(x).reshape(-1, self.hidden_size)
        for layer in self.mlp:
            x = layer(x)
        return x
```

**压缩**:
- 1024 patches → **256 vision tokens**
- 压缩比: 4x (2×2 spatial merge)

#### 2.5 VisionModel (完整 Vision Encoder)

```python
class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        self.patch_embed = PatchEmbed(...)
        self.rotary_pos_emb = VisionRotaryEmbedding(...)
        self.blocks = [Qwen2VLVisionBlock(config) for _ in range(depth)]
        self.merger = PatchMerger(...)

    def __call__(self, pixel_values, grid_thw):
        # 1. Patch Embedding
        hidden_states = self.patch_embed(pixel_values)  # [1024, 1152]

        # 2. Position Embedding
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        # 3. Transformer Blocks
        for block in self.blocks:
            hidden_states = block(hidden_states, cu_seqlens, rotary_pos_emb)

        # 4. Patch Merging
        hidden_states = self.merger(hidden_states)  # [256, 3584]

        return hidden_states
```

**输出**:
- Vision tokens: **256 tokens** (16×16 grid)
- 每个 token: 3584 维 (Qwen2-VL hidden_size)

---

## 3. 模型入口接口 (qwen2_vl.py)

### 核心接口

```python
class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config, config)

    def get_input_embeddings(
        self,
        input_ids: mx.array,           # [B, seq_len]
        pixel_values: mx.array = None, # [B, C, H, W]
        image_grid_thw: mx.array = None,
        **kwargs
    ):
        # 1. 纯文本模式
        if pixel_values is None:
            return self.language_model.embed_tokens(input_ids)

        # 2. 图像 + 文本模式
        # 2.1 编码图像
        vision_features = self.vision_tower(pixel_values, image_grid_thw)
        # → [256, 3584]

        # 2.2 编码文本
        text_embeds = self.language_model.embed_tokens(input_ids)
        # → [text_len, 3584]

        # 2.3 融合 (替换 <image> token)
        final_embeds = self.merge_input_ids_with_image_features(
            image_token_id=151655,  # Qwen2-VL 特殊 token
            image_features=vision_features,
            inputs_embeds=text_embeds,
            input_ids=input_ids,
        )
        # → [text_len + 256, 3584]

        return final_embeds
```

### 融合逻辑

```python
@staticmethod
def merge_input_ids_with_image_features(
    image_token_id,
    image_features,    # [256, 3584]
    inputs_embeds,     # [text_len, 3584]
    input_ids,         # [text_len]
):
    """
    输入示例:
        input_ids = [1, 2, 3, 151655, 4, 5, 6]
                              ↑
                         <image> token

    步骤:
        1. 找到 <image> token 位置 (index=3)
        2. 分割: [1,2,3] + [image_features] + [4,5,6]
        3. 拼接: [1,2,3] + [256 vision tokens] + [4,5,6]

    输出:
        final_embeds: [3 + 256 + 3 = 262, 3584]
    """
    image_positions = (input_ids == image_token_id)

    # 替换 <image> token 为 vision features
    # (实际实现更复杂，支持多张图像)
    ...

    return final_embeds
```

---

## 4. 与 FlashMLX 集成点分析

### 集成点 1: Vision Encoder (新增)

**需要移植**:
- VisionModel (310 行)
- 依赖: mlx.nn (Conv3d, LayerNorm, Linear)

**FlashMLX 适配**:
```python
# src/flashmlx/models/vision.py (新建)
class FlashMLXVisionModel(nn.Module):
    """从 MLX-VLM 移植"""
    def __init__(self, config):
        # 直接复制 MLX-VLM 实现
        self.patch_embed = PatchEmbed(...)
        self.blocks = [Qwen2VLVisionBlock(...) ...]
        self.merger = PatchMerger(...)
```

### 集成点 2: 模型包装 (修改)

**MLX-LM 原实现** (57 行):
```python
# mlx-lm/mlx_lm/models/qwen3_vl.py
class Model(nn.Module):
    def __init__(self, args):
        self.language_model = qwen3.Model(args.text_config)
        # ❌ 没有 vision_tower

    def sanitize(self, weights):
        weights.pop("vision_tower", None)  # ❌ 删除 Vision
```

**FlashMLX 新实现**:
```python
# src/flashmlx/models/qwen3_vlm.py (新建)
class FlashMLXQwen3VL(nn.Module):
    def __init__(self, args):
        # 复用 FlashMLX qwen3.py
        self.language_model = qwen3.Model(args.text_config)

        # ✅ 添加 Vision Encoder
        self.vision_tower = FlashMLXVisionModel(args.vision_config)
        self.visual_projection = nn.Linear(
            args.vision_config.hidden_size,  # 3584
            args.text_config.hidden_size,    # 4096
        )

    def sanitize(self, weights):
        # ✅ 保留 Vision 权重
        return weights  # 不删除

    def get_input_embeddings(self, input_ids, pixel_values=None):
        # 参考 MLX-VLM 实现
        ...
```

### 集成点 3: KV Cache 适配 (关键!)

**挑战**: Vision tokens (256) 超出 L0 boundary (512)

**FlashMLX Route 3 适配**:
```python
# src/flashmlx/triple_layer_cache.py (修改)

class TripleLayerKVCache:
    def __init__(self, recent_size=512, ...):
        # ❌ 原实现: 固定 L0=512
        self.recent_size = recent_size

    # ✅ 新实现: 动态 L0
    def __init__(self, recent_size=512, vision_tokens=0, ...):
        # 如果有 vision tokens，调整 L0
        self.recent_size = recent_size + vision_tokens  # 512 + 256 = 768
```

**适配策略**:
1. **Prefill 阶段**: 检测 vision tokens
2. **动态调整 L0**: `recent_size = 512 + vision_token_count`
3. **Eviction 策略**: Vision tokens 固定在 L0，文本 tokens 可压缩

### 集成点 4: H0Store 分离 (Route 5)

**挑战**: Vision 和 Text embeddings 混合

**解决方案**:
```python
# src/flashmlx/h0_store.py (修改)

class H0Store:
    def __init__(self, max_tokens, hidden_size, dtype):
        # ✅ 分离存储
        self._h0_text = mx.zeros([max_text_tokens, hidden_size], dtype)
        self._h0_vision = mx.zeros([max_vision_tokens, hidden_size], dtype)

    def capture(self, h0, token_type="text"):
        if token_type == "vision":
            self._h0_vision[:len(h0)] = h0
        else:
            self._h0_text[:len(h0)] = h0
```

### 集成点 5: 权重加载 (困难)

**挑战**: VLM 权重结构复杂

**权重映射**:
```python
# Hugging Face 权重结构
{
    "visual.patch_embed.proj.weight": [...],
    "visual.blocks.0.attn.qkv.weight": [...],
    ...
    "language_model.model.embed_tokens.weight": [...],
    "language_model.model.layers.0.self_attn.q_proj.weight": [...],
    ...
}

# FlashMLX 期望结构
{
    "vision_tower.patch_embed.proj.weight": [...],
    "vision_tower.blocks.0.attn.qkv.weight": [...],
    ...
    "language_model.model.embed_tokens.weight": [...],
    ...
}
```

**实现**:
```python
def sanitize(self, weights):
    sanitized = {}
    for key, value in weights.items():
        # 重映射 vision 权重
        if key.startswith("visual."):
            new_key = key.replace("visual.", "vision_tower.")
            sanitized[new_key] = value
        # 重映射 language 权重
        elif key.startswith("language_model."):
            sanitized[key] = value
        else:
            # 兼容旧格式
            if not key.startswith("language_model."):
                key = "language_model." + key
            sanitized[key] = value
    return sanitized
```

---

## 5. 移植优先级

| 组件 | 优先级 | 代码量 | 难度 | 依赖 |
|------|--------|--------|------|------|
| **VisionModel** | P0 | 310 行 | ⭐⭐ | mlx.nn |
| **模型包装** | P0 | 50 行 | ⭐⭐⭐ | VisionModel |
| **权重加载** | P0 | 50 行 | ⭐⭐⭐⭐ | safetensors |
| **图像预处理** | P1 | 195 行 | ⭐ | PIL |
| **KV Cache 适配** | P1 | 100 行 | ⭐⭐⭐⭐ | FlashMLX |
| **H0Store 分离** | P2 | 50 行 | ⭐⭐⭐ | FlashMLX |

**总计**: ~755 行新代码

---

## 6. 风险评估

| 风险 | 等级 | 缓解措施 |
|------|------|---------|
| **Vision 权重加载失败** | ⚠️⚠️⚠️ | 参考 MLX-VLM 实现，逐步调试 |
| **KV Cache 溢出** | ⚠️⚠️ | 动态 L0 boundary，充分测试 |
| **性能回退** | ⚠️ | 保留 Phase 0 baseline，对比验证 |
| **Vision 输出不一致** | ⚠️⚠️ | 与 MLX-VLM 逐层对比 |
| **内存溢出** | ⚠️ | 监控内存，优化 batch size |

---

## 7. 下一步行动

### Day 1 完成 ✅
- [x] 分析 MLX-VLM VisionModel 实现
- [x] 分析模型入口接口
- [x] 识别 5 个关键集成点

### Day 2 计划
- [ ] 运行 Phase 0 回归测试
- [ ] 验证 FlashMLX 优化稳定性
- [ ] 确认文本性能红线有效

### Day 3 计划
- [ ] 设计 FlashMLX VLM 详细架构
- [ ] 制定 Phase 2 移植计划
- [ ] 评估工作量和时间线

---

**结论**: MLX-VLM Vision Encoder 架构清晰，移植可行性高。关键挑战在权重加载和 KV Cache 适配，预计 2-3 周完成。
