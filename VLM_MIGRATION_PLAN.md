# FlashMLX → MLX-LM VLM 版本迁移计划

**目标**: 将 FlashMLX 迁移到最新 MLX-LM，支持 Vision-Language Models (VLM)

**当前基线**: MLX-LM origin/main (f8019f7)，落后 ~25 commits

**总工作量**: **12 周，37 人日**

---

## 执行摘要

| 维度 | 现状 | 挑战 | 风险 |
|------|------|------|------|
| **代码规模** | 50+ commits 深度改造 | 18 个关键 patch 点需要适配 | 中 |
| **VLM 支持** | 5 个 VLM 模型（但 vision 权重被删除） | Vision encoder KV cache 优化缺失 | **高** |
| **核心冲突** | Route 3/5 假设纯文本 | Vision tokens 动态数量、无法从 h^(0) 重建 | **高** |

**核心风险**:
1. ❌ **Vision Token 序列处理**: Vision tokens (1K-10K) 可能超载 L0 边界 (512)
2. ❌ **KV-Direct 重建失效**: Vision encoder 权重被删除，无法从 h^(0) 重建 Vision K/V
3. ❌ **三层缓存复杂化**: Vision + Text 混合时，L0/L1/L2 转移逻辑复杂

**推荐策略**: **渐进式迁移 + Feature Flag**（分 4 个阶段，每阶段可独立回滚）

---

## 四阶段迁移路线图

### 📍 Phase 1: 基础适配（周 1-2，8 人日）

**目标**: 确保文本模型不退化，建立 CI baseline

**关键任务**:
1. Git merge origin/main 最新代码（冲突解决）
2. API 签名兼容性适配
   - `make_prompt_cache()` 新参数支持
   - `Model.__call__()` vision embedding 占位
3. Text-only 全量回归测试
   - Mistral-7B, Qwen3-8B, Llama-3.2-3B
   - 验证 Route 3/4/5 性能无退化
4. CI pipeline 更新 + 基准线建立

**交付物**:
- ✅ mlx-lm-source 与 origin/main 同步
- ✅ 文本模型性能无退化（TG speed ±5%）
- ✅ 基准线数据（TG tok/s, Memory peak）

**风险**: 低（可随时回滚到 origin/main）

---

### 📍 Phase 2: VLM 架构适配（周 3-6，12 人日）

**目标**: VLM 模型能加载 + 推理（暂不优化性能）

**关键任务**:

#### 1. Vision Encoder 权重加载恢复 (**CRITICAL**)
```python
# 修改: qwen2_vl.py, qwen3_vl.py, kimi_vl.py
def sanitize(self, weights):
    weights = tree_unflatten(list(weights.items()))
    # ❌ 删除这行: weights.pop("visual", None)
    # ✅ 保留 vision encoder 权重
    return weights
```
**工作量**: 1 人日

#### 2. Vision Token 序列处理
```python
# 修改: cache_factory.py, triple_layer_cache.py

def _estimate_vision_tokens(model):
    """动态检测 vision encoder 输出 token 数量"""
    if hasattr(model, 'visual'):
        # 估算: Qwen3-VL ≈ 576-2048 tokens per image
        return 2048  # 保守估计
    return 0

# TripleLayerKVCache 调整 L0 边界
if vision_token_count > 0:
    self.recent_size = vision_token_count + 512  # 保守方案
```
**工作量**: 3 人日

#### 3. H0Store 多源嵌入支持
```python
# 修改: kv_direct_cache.py

class H0Store:
    def __init__(self, source_type='text'):  # NEW: 'text' | 'vision'
        self._source_type = source_type

# 安装时创建两个独立 store
model._h0_store_text = H0Store(source_type='text')
model._h0_store_vision = H0Store(source_type='vision')
```
**工作量**: 3 人日

#### 4. Generate Pipeline 多模态预处理
```python
# 修改: generate.py

def generate(..., images=None):
    # Vision encoder 前置调用
    if images is not None and hasattr(model, 'visual'):
        vision_tokens = model.visual(images)
        # 拼接 vision_tokens + text_tokens
```
**工作量**: 2 人日

#### 5. VLM 集成测试
- Qwen2-VL + 单张图像
- Qwen3-VL-MOE + 多张图像
- 验证: 推理完成，输出合理

**工作量**: 3 人日

**交付物**:
- ✅ Vision encoder 权重可加载
- ✅ VLM 推理可完成（可能很慢，内存占用高）
- ✅ Vision tokens 被正确处理

**风险**: 中（可通过 `ENABLE_VLM_SUPPORT=False` 回退）

---

### 📍 Phase 3: VLM KV Cache 优化（周 7-10，10 人日）

**目标**: Vision tokens 享受压缩，但策略差异化

**关键任务**:

#### 1. Vision-aware Quantization Strategy
```python
# 修改: quantization_strategies.py, config.py

class CacheConfig:
    vision_quantizer: str = "bf16"  # NEW: "bf16" | "q8_0" | "q4_0"

# Vision tokens 用 Q8_0, Text tokens 用 Q4_0
if token_type == 'vision':
    quantizer = vision_quantizer  # 保守 4x 压缩
else:
    quantizer = text_quantizer    # 激进 8x 压缩
```
**工作量**: 2 人日

#### 2. Vision-specific Density Router
```python
# 修改: config.py

class CacheConfig:
    density_scale_vision: float = 0.0  # 保守，不压缩
    density_scale_text: float = 2.5    # 激进，recall_first
```
**工作量**: 1 人日

#### 3. Vision Token AM Scoring（跳过）
```python
# 修改: triple_layer_cache.py

# Vision tokens 不参与 AM 评分（避免关键信息丢失）
if enable_am_compression:
    text_indices = [i for i, t in token_types if t == 'text']
    am_scores = compute_am_scores(keys[..., text_indices, :])
```
**工作量**: 1 人日

#### 4. Vision H0Store 重建验证
- 验证 vision h^(0) 重建精度
- 确保 vision_encoder 权重可访问

**工作量**: 2 人日

#### 5. VLM Benchmark Suite
- 多长度: 2K, 4K, 8K tokens (含 vision)
- 多配置: no-opt, Route-3-only, Route-3+5
- 质量: VQAV2, GQA 评估

**工作量**: 4 人日

**交付物**:
- ✅ Vision tokens 被压缩（Q8_0, 4x）
- ✅ VLM 长上下文性能可接受（<50% 退化）
- ✅ 质量无损失（VQAV2 baseline）

**风险**: 中（可回退到 Phase 2 无压缩配置）

---

### 📍 Phase 4: 高级特性（周 11-12，7 人日）

**目标**: 生产就绪 + 性能突破

**关键任务**:

1. **Vision Token Batching (Route 1 适配)**
   - Vision encoder output 缓冲 + 批处理
   - Expert offload 对 vision weights
   - 工作量: 2 人日

2. **Vision Chunked Prefill (Route 4 适配)**
   - Vision tokens 流式淘汰
   - 避免 PP 峰值 explosion
   - 工作量: 2 人日

3. **Model Card 完善**
   - 各 VLM 的 optimal 配置卡片
   - 基准数据记录
   - 工作量: 1 人日

4. **文档 + 示例**
   - VLM 使用指南
   - 常见问题排查
   - 工作量: 2 人日

**交付物**:
- ✅ VLM 生产就绪
- ✅ 完整文档和示例
- ✅ Model Cards 完善

---

## 关键 Patch 清单（18 处）

| 优先级 | Patch ID | 文件 | 改动 | 工作量 | Phase |
|--------|---------|------|------|--------|-------|
| **P0** | P1 | qwen2_vl.py, qwen3_vl.py | 恢复 visual 权重 | 0.5 人日 | 2 |
| **P0** | P2 | cache_factory.py | Vision 检测 + 参数计算 | 1.5 人日 | 2 |
| **P0** | P3 | triple_layer_cache.py | Vision-aware L0/L1/L2 | 2.5 人日 | 2 |
| **P0** | P4 | kv_direct_cache.py | 多源 H0Store | 2.5 人日 | 2 |
| **P0** | P5 | generate.py | 多模态预处理 pipeline | 1.5 人日 | 2 |
| **P1** | P6 | quantization_strategies.py | Vision-aware quantizer | 1.5 人日 | 3 |
| **P1** | P7 | config.py | vision_quantizer 配置 | 0.5 人日 | 3 |
| **P1** | P8 | triple_layer_cache.py | Vision AM scoring 跳过 | 1 人日 | 3 |
| **P2** | P9 | expert_offload.py | Vision weights offload | 2 人日 | 4 |
| **P2** | P10 | generate.py | Vision chunked prefill | 2 人日 | 4 |

**总计**: ~1,000 行代码修改 + ~500 行新增

---

## 技术风险矩阵

### 高风险区域

#### 1. Vision Token 序列处理
**问题**: Vision tokens (1K-10K) 可能超载 L0 边界 (512)

**影响**:
- Vision tokens 被强制进入 L2 (Cold)
- L2 使用 AM 压缩 → 信息丢失
- 输出质量下降

**缓解**:
```
Phase 2: Vision tokens 全部进 L0 (无压缩)
         recent_size = vision_count + 512
         内存开销: +4.8GB (6K vision, Qwen3-8B)

Phase 3: Vision tokens 用 Q8_0 进 L0
         内存开销: +1.2GB (6K vision, Qwen3-8B)
```

#### 2. KV-Direct 重建失效
**问题**: Vision encoder 权重被删除，无法从 h^(0) 重建 Vision K/V

**影响**:
- Route 5 重建失效
- 长上下文无法从 h^(0) 恢复视觉信息

**缓解**:
```
1. 恢复 vision_encoder 权重加载 (P1)
2. 维护独立 H0Store (vision vs text)
3. 重建时: 调用 vision_encoder.attn(h^(0)_vision)
```

#### 3. 三层缓存状态机复杂化
**问题**: Vision + Text 混合时，L0/L1/L2 转移逻辑复杂

**缓解**:
```
Phase 2: Vision tokens 永不进 L1/L2 (简化)
Phase 3: Token Type Tag 追踪 (优化)
```

### 中风险区域

#### 4. Expert Offloading 与 Vision Weights
**问题**: Vision encoder 权重规模 (2-10B)

**缓解**: Vision weights 加入 CPUWarmCache / SSD Tier (Phase 4)

#### 5. Chunked Prefill 的 Vision Token Burst
**问题**: Vision 编码瞬间产生大量 tokens

**缓解**: Vision encoder 前置 + 缓冲管理 (Phase 4)

---

## Feature Flag 设计

```python
# mlx_lm/models/cache_factory.py

ENABLE_VLM_SUPPORT = True  # Phase 2 开启

def make_optimized_cache(..., enable_vlm=None):
    if enable_vlm is None:
        enable_vlm = _detect_architecture(model)[3] > 0

    if enable_vlm and not ENABLE_VLM_SUPPORT:
        print("[WARNING] VLM detected but disabled, fallback to text-only")
        enable_vlm = False

    if enable_vlm:
        return _make_vlm_cache(model, ...)  # Phase 2+ 路径
    else:
        return _make_text_cache(model, ...) # 原始路径
```

**回滚策略**:
- Phase 1 失败 → revert Git merge (1 小时)
- Phase 2 失败 → `ENABLE_VLM_SUPPORT=False` (5 分钟)
- Phase 3 失败 → 关闭激进量化 (5 分钟)
- Phase 4 失败 → 禁用 offload (5 分钟)

---

## 测试策略

### Phase 1: 文本模型回归
```
Model          Context  Strategy      Metric           Target
──────────────────────────────────────────────────────────────
Mistral-7B     4K       triple_pq     TG: 20-25 tok/s  无退化
Qwen3-8B       8K       scored_pq     TG: 18-22 tok/s  无退化
Llama-3.2-3B   16K      triple_pq_am  TG: 50+ tok/s    无退化
```

### Phase 2: VLM 功能
```
Model          Images  Context  Metric              Target
──────────────────────────────────────────────────────────
Qwen2-VL       1       2K       推理完成，合理输出    ✓
Qwen3-VL       3       4K       推理完成             ✓
Qwen3-VL-MOE   1       8K       推理完成             ✓
```

### Phase 3: VLM 性能
```
Model          Images  Context  期望改进  目标
───────────────────────────────────────────
Qwen3-VL       1       4K       <50%     <400ms PP
Qwen3-VL       3       8K       <50%     <600ms PP
Qwen3-VL-MOE   1       16K      <30%     <1000ms PP
```

### Phase 4: VLM 质量
```
Benchmark      Baseline  Phase 3  Phase 4  Target
────────────────────────────────────────────────
VQAV2          72.3%     >70%     >72%     无损
GQA            65.1%     >63%     >65%     无损
TextVQA        58.4%     >56%     >58%     无损
```

---

## 资源需求

### 人力
- **Phase 1**: 1 人 × 2 周 = 8 人日
- **Phase 2**: 2 人 × 4 周 = 12 人日（并行开发）
- **Phase 3**: 2 人 × 4 周 = 10 人日（并行开发）
- **Phase 4**: 1 人 × 2 周 = 7 人日

**总计**: 2-3 人团队，12 周，37 人日

### 算力
- **开发环境**: M4 Max 64GB（当前硬件）
- **测试环境**: M4 Max 128GB（推荐，用于 VLM 长上下文）
- **CI/CD**: GitHub Actions (文本模型回归)

---

## 里程碑和交付物

| 里程碑 | 时间点 | 交付物 | 验收标准 |
|--------|--------|--------|----------|
| **M1: 基础适配** | 周 2 | mlx-lm 同步 + 文本回归通过 | 文本模型性能无退化 |
| **M2: VLM 功能** | 周 6 | VLM 推理可完成 | 3 个 VLM 模型推理通过 |
| **M3: VLM 优化** | 周 10 | VLM 性能可接受 | PP/TG <50% 退化 |
| **M4: 生产就绪** | 周 12 | 文档 + Model Cards | 质量基准达标 |

---

## 决策点

### 决策 1: Vision Weights 管理
**选项 A**: 恢复常驻加载（推荐）
- ✅ Route 5 重建需要
- ❌ 内存 +2-10GB

**选项 B**: 动态加载
- ✅ 参数内存节省
- ❌ 首次推理延迟

**最终**: **选项 A + Route 1 offload (Phase 4)**

### 决策 2: Vision Token 压缩激进度
**Phase 2**: `vision_quantizer="bf16"` (无压缩)
- 内存: +4.8GB
- 质量: 100% 保证

**Phase 3**: `vision_quantizer="q8_0"` (4x 压缩)
- 内存: +1.2GB
- 质量: 需 benchmark

**Phase 4**: `vision_quantizer="q4_0"` (8x 压缩，可选)
- 内存: +0.6GB
- 质量: 高风险

### 决策 3: AM 评分与 Vision Tokens
**决定**: Vision tokens **跳过** AM 评分

**理由**:
- AM 基于文本 token importance
- Vision tokens 的 importance 分布不同
- 盲目评分可能丢失关键视觉信息

---

## 最终建议

### 短期（1-3 个月）
**立即启动 Phase 1-2**:
1. Week 1-2: Git merge + 文本回归
2. Week 3-6: VLM 架构适配
3. 交付: VLM 推理可完成，文本模型无退化

### 中期（3-6 个月）
**Phase 3-4**:
1. Week 7-10: VLM 优化（量化 + benchmark）
2. Week 11-12: 生产就绪（文档 + Model Cards）
3. 交付: VLM 性能可接受，质量无损

### 长期（6-12 个月）
**多模态平台扩展**:
- Audio 支持（Phase 4+ 扩展）
- 企业特性（SSD Tier 2/3, 批处理）
- 开源贡献（upstream 到 MLX-LM）

---

## 附录：关键代码框架

### Vision Token 检测
```python
def _estimate_vision_tokens(model):
    if not hasattr(model, 'visual'):
        return 0

    try:
        dummy_image = mx.zeros((3, 336, 336))
        vision_output = model.visual(dummy_image)
        return vision_output.shape[1]  # num_tokens
    except:
        return 2048  # 保守估计
```

### Vision-aware TripleLayerKVCache
```python
class TripleLayerKVCache(_BaseCache):
    def __init__(self, vision_token_count=0, vision_quantizer="bf16"):
        # 调整 L0 边界以容纳 vision tokens
        if vision_token_count > 0:
            self.recent_size = max(512, vision_token_count + 512)
```

### 多源 H0Store
```python
def _install_h0_capture(model, vision_aware=True):
    # Text embedding
    model._h0_store_text = H0Store(source_type='text')

    # Vision embedding
    if vision_aware and hasattr(model, 'visual'):
        model._h0_store_vision = H0Store(source_type='vision')
```

---

**创建日期**: 2026-04-09
**审核人**: Claude Opus 4.6
**版本**: v1.0
