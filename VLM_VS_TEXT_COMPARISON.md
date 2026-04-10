# VLM vs 文本模型对比分析

## 核心差异总结

| 维度 | 文本模型 (Qwen3) | VLM 模型 (Qwen3-VL) |
|------|-----------------|-------------------|
| **代码复杂度** | 220 行（完整实现） | **58 行（薄包装层）** |
| **架构** | 完整 Transformer | **文本模型 + Vision Encoder** |
| **输入** | 文本 token IDs | 文本 + 图像 embeddings |
| **权重** | 纯文本权重 | 文本权重 + Vision Tower |
| **推理模式** | 标准 forward | **支持 input_embeddings** |

---

## 1. 架构对比

### 文本模型 (qwen3.py)

```python
class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        self.model = TransformerModel(args)  # 完整 Transformer
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def __call__(self, inputs: mx.array, cache=None):
        # inputs: [batch, seq_len] - token IDs
        x = self.embed_tokens(inputs)  # → [batch, seq_len, hidden_size]
        x = self.model(x, cache=cache)
        return self.lm_head(x)  # → [batch, seq_len, vocab_size]
```

**特点**:
- ✅ 完整的 220 行实现
- ✅ 包含 Attention, MLP, RMSNorm 等所有层
- ✅ 处理纯文本输入

### VLM 模型 (qwen3_vl.py)

```python
class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        # ⚠️  关键: 内部包装一个完整的 Qwen3 文本模型
        self.language_model = qwen3.Model(
            qwen3.ModelArgs.from_dict(args.text_config)
        )

    def __call__(self, inputs: mx.array, cache=None,
                 input_embeddings: Optional[mx.array] = None):
        # ⚠️  支持两种输入:
        # 1. inputs: token IDs (纯文本)
        # 2. input_embeddings: 预处理的 embeddings (文本+视觉)
        return self.language_model(
            inputs, cache=cache, input_embeddings=input_embeddings
        )

    def sanitize(self, weights):
        # ⚠️  删除 vision_tower 权重
        weights.pop("vision_tower", None)
        # 重命名: model.xxx → language_model.model.xxx
        ...
```

**特点**:
- ⚠️  **只有 58 行** - 薄包装层
- ⚠️  **内部复用** `qwen3.Model` (文本模型)
- ⚠️  **删除** Vision Encoder 权重
- ✅ 支持 `input_embeddings` 参数

---

## 2. 权重结构对比

### 文本模型权重

```python
{
    "model.embed_tokens.weight": [vocab_size, hidden_size],
    "model.layers.0.self_attn.q_proj.weight": [...],
    "model.layers.0.self_attn.k_proj.weight": [...],
    ...
    "lm_head.weight": [vocab_size, hidden_size]
}
```

**总大小**: ~8GB (Qwen3-8B 4-bit)

### VLM 模型权重 (原始)

```python
{
    # 视觉编码器 (被 sanitize() 删除!)
    "vision_tower.patch_embed.weight": [...],
    "vision_tower.encoder.layers.0.xxx": [...],
    ...

    # 语言模型 (保留)
    "language_model.model.embed_tokens.weight": [...],
    "language_model.model.layers.0.xxx": [...],
    ...
}
```

**总大小**: ~10GB (8GB 文本 + 2GB Vision Encoder)

### VLM 模型权重 (sanitize 后)

```python
{
    # ❌ vision_tower 被删除

    # ✅ 只保留 language_model 部分
    "language_model.model.embed_tokens.weight": [...],
    "language_model.model.layers.0.xxx": [...],
    ...
}
```

**最终大小**: ~8GB (只有文本部分)

---

## 3. 推理流程对比

### 文本模型推理

```python
# 1. 输入: 文本 token IDs
input_ids = tokenizer.encode("Hello world")  # [1, 2, 3, 4, 5]

# 2. 直接推理
logits = model(input_ids, cache=cache)

# 3. 输出: 下一个 token 的概率分布
next_token = logits.argmax(dim=-1)
```

**流程**: Token IDs → Embedding → Transformer → Logits

### VLM 模型推理 (当前实现)

```python
# 1. 输入: 文本 + 图像
text = "描述这张图片"
image = PIL.Image.open("cat.jpg")

# 2. 图像编码 (需要外部完成!)
vision_embeddings = vision_encoder(image)  # [1, 1024, 4096]
# ⚠️  问题: vision_encoder 权重被删除了!

# 3. 文本编码
text_embeddings = model.embed_tokens(text_ids)  # [1, 5, 4096]

# 4. 合并 embeddings
combined = mx.concatenate([vision_embeddings, text_embeddings], axis=1)
# → [1, 1029, 4096]  (1024 vision + 5 text)

# 5. 推理
logits = model(inputs=None, input_embeddings=combined)
```

**问题**:
- ❌ Vision Encoder 权重被 `sanitize()` 删除
- ❌ 需要外部提供 `vision_embeddings`
- ❌ MLX-LM 没有提供 Vision Encoder 实现

---

## 4. KV Cache 差异

### 文本模型 KV Cache

```python
cache_entry = {
    "keys": [batch, n_kv_heads, seq_len, head_dim],
    "values": [batch, n_kv_heads, seq_len, head_dim]
}
```

**特点**:
- ✅ seq_len 固定增长
- ✅ 所有 token 平等对待
- ✅ FlashMLX 优化适用

### VLM 模型 KV Cache

```python
# Prefill 阶段
cache_entry = {
    # Vision tokens (1024)
    "keys": [batch, n_kv_heads, 1024, head_dim],
    "values": [batch, n_kv_heads, 1024, head_dim]
}

# Generation 阶段 (追加文本)
cache_entry = {
    # Vision (1024) + Text (growing)
    "keys": [batch, n_kv_heads, 1024+N, head_dim],
    "values": [batch, n_kv_heads, 1024+N, head_dim]
}
```

**挑战**:
- ⚠️  **Vision tokens 数量动态** (1K-10K)
- ⚠️  **超过 FlashMLX L0 boundary** (512)
- ⚠️  **需要调整压缩策略**

---

## 5. FlashMLX 优化适配性

| 优化路由 | 文本模型 | VLM 模型 | 需要修改 |
|---------|---------|---------|---------|
| Route 0 (Density Router) | ✅ 完美支持 | ⚠️  需调整 density_scale | 轻微 |
| Route 1 (Expert Offload) | ✅ MoE 适用 | ✅ MoE 适用 | 无 |
| Route 3 (KV Compress) | ✅ 完美支持 | ❌ **L0 溢出** | **重大** |
| Route 4 (Chunked PP) | ✅ 适用 | ⚠️  Vision chunk 特殊处理 | 中等 |
| Route 5 (Context Recall) | ✅ 适用 | ❌ **双 H0Store** | **重大** |

---

## 6. 迁移到 VLM 的关键问题

### 问题 1: Vision Encoder 缺失

**现状**:
```python
# qwen3_vl.py:45
weights.pop("vision_tower", None)  # ❌ 直接删除
```

**解决方案** (Phase 2):
```python
# 保留 Vision Encoder
self.vision_tower = VisionTransformer(vision_config)

def encode_image(self, pixel_values):
    return self.vision_tower(pixel_values)
```

### 问题 2: KV Cache 溢出

**现状**:
```python
# Route 3: L0 boundary = 512
# Vision tokens = 1024-10240  # ❌ 超出 L0
```

**解决方案** (Phase 3):
```python
# 动态调整 L0
recent_size = vision_token_count + 512  # ✅ 自适应
```

### 问题 3: H0Store 单一来源

**现状**:
```python
# Route 5: 单一 H0Store
self._h0_store = mx.zeros([max_tokens, hidden_size])
# ❌ 无法区分 text vs vision
```

**解决方案** (Phase 3):
```python
# 双 H0Store
self._h0_store_text = mx.zeros([max_text, hidden_size])
self._h0_store_vision = mx.zeros([max_vision, hidden_size])  # ✅ 分离存储
```

---

## 7. 代码量对比

```bash
$ wc -l mlx_lm/models/qwen3.py mlx_lm/models/qwen3_vl.py
     220 mlx_lm/models/qwen3.py      # 完整实现
      58 mlx_lm/models/qwen3_vl.py   # 薄包装层 (74% 代码复用)
```

**结论**:
- VLM 本质是 **文本模型 + Vision Encoder + 输入融合**
- 代码复用率 **74%**
- 主要差异在 **权重管理** 和 **输入处理**

---

## 8. 性能对比 (预估)

| 指标 | 文本模型 | VLM 模型 | 差异原因 |
|------|---------|---------|---------|
| **PP tok/s** | 400 | 50-100 | Vision encoding 慢 |
| **TG tok/s** | 25 | 20-25 | KV cache 更大 |
| **内存 (4K)** | 600M | 800M | +Vision KV |
| **内存 (32K)** | 4.6GB | 5.2GB | +Vision KV (固定) |
| **TTFT** | 10s | 15-20s | +Vision encoding |

---

## 9. 为什么需要 Phase 1-4？

| Phase | 目标 | 原因 |
|-------|------|------|
| Phase 1 | 文本回归测试 | 确保 merge 不破坏现有优化 |
| Phase 2 | VLM 架构适配 | 添加 Vision Encoder + 输入融合 |
| Phase 3 | VLM KV 优化 | 解决 L0 溢出 + H0Store 分离 |
| Phase 4 | 生产就绪 | 质量验证 + 文档 + 高级特性 |

---

## 10. 最小可行 VLM (MVP) 实现

如果只是为了"能跑"，最小修改：

```python
class Qwen3VL(nn.Module):
    def __init__(self, args):
        self.language_model = qwen3.Model(args.text_config)
        # ⚠️  暂时不加载 Vision Encoder

    def __call__(self, inputs, input_embeddings=None):
        # ⚠️  要求外部提供 input_embeddings
        return self.language_model(inputs, input_embeddings=input_embeddings)
```

**缺点**:
- ❌ 需要外部 Vision Encoder (MLX-VLM 项目)
- ❌ 无法端到端推理
- ❌ FlashMLX 优化可能失效

**完整实现** (Phase 2-4):
```python
class Qwen3VL(nn.Module):
    def __init__(self, args):
        self.vision_tower = VisionTransformer(args.vision_config)  # ✅ 保留
        self.language_model = qwen3.Model(args.text_config)
        self.visual_projection = nn.Linear(vision_dim, hidden_size)  # ✅ 投影层

    def encode_image(self, pixel_values):
        vision_features = self.vision_tower(pixel_values)
        return self.visual_projection(vision_features)

    def __call__(self, inputs, pixel_values=None, cache=None):
        if pixel_values is not None:
            vision_embeds = self.encode_image(pixel_values)
            text_embeds = self.language_model.embed_tokens(inputs)
            combined = mx.concatenate([vision_embeds, text_embeds], axis=1)
            return self.language_model(None, input_embeddings=combined, cache=cache)
        else:
            return self.language_model(inputs, cache=cache)  # 纯文本模式
```

---

## 总结

**VLM vs 文本模型**:

1. **架构**: VLM = 文本模型 (复用) + Vision Encoder (新增)
2. **代码**: 74% 复用，26% 新增
3. **权重**: +2GB Vision Tower
4. **输入**: Token IDs → Token IDs + Pixel Values
5. **KV Cache**: +1K-10K vision tokens (挑战!)
6. **性能**: TTFT +50%, 内存 +15%

**FlashMLX 迁移关键**:
- Phase 1: 确保文本模型不回退 ✅
- Phase 2: 添加 Vision Encoder (核心)
- Phase 3: 调整 KV 优化 (Route 3/5)
- Phase 4: 质量验证

**当前 MLX-LM 实现的局限**:
- ❌ Vision Encoder 权重被删除
- ❌ 需要外部提供 vision embeddings
- ❌ 不是端到端方案

**FlashMLX 需要做的**:
- ✅ 保留 Vision Encoder
- ✅ 实现端到端推理
- ✅ 优化 Vision KV Cache
