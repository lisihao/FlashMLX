# FlashMLX ↔ ThunderOMLX 集成指导

> 本文档供另一个 Claude Code 实例在 ThunderOMLX 项目中使用。
> 目标：让 ThunderOMLX 消费 FlashMLX SDK 的模型层优化能力，两个项目独立开发、协同工作。

---

## 1. 架构关系

```
┌─────────────────────────────────────────────────┐
│  ThunderOMLX (服务层)                             │
│  - OpenAI-compatible API server                  │
│  - Continuous batching (BatchGenerator)           │
│  - L1 RAM + L2 SSD 两级缓存                       │
│  - Prefix cache, KVTC codec                      │
│  - 多请求调度 (Scheduler)                          │
│                                                   │
│  ┌─────────────────────────────────────────────┐ │
│  │  FlashMLX SDK (模型层, 可选依赖)               │ │
│  │  - KV cache 压缩 (scored_pq + flat_quant)    │ │
│  │  - Expert offloading (MoE 三层管理)           │ │
│  │  - 模型能力探测 + 自动配置推荐                  │ │
│  │  - AM 自动校准                                │ │
│  └─────────────────────────────────────────────┘ │
│                                                   │
│  ┌─────────────────────────────────────────────┐ │
│  │  mlx-lm (增强版, FlashMLX 子模块)             │ │
│  │  - mlx_lm.generate / stream_generate         │ │
│  │  - mlx_lm.generate.BatchGenerator            │ │
│  │  - mlx_lm.models.cache (make_prompt_cache)   │ │
│  │  - mlx_lm.models.cache_factory               │ │
│  │  - mlx_lm.models.triple_layer_cache          │ │
│  │  - mlx_lm.models.quantization_strategies     │ │
│  │  - mlx_lm.models.expert_offload              │ │
│  └─────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

**关键点**：
- FlashMLX 是增强版 mlx-lm 的 SDK 封装（`pip install -e FlashMLX/mlx-lm-source` 替代 vanilla mlx-lm）
- ThunderOMLX 当前依赖 vanilla `mlx-lm`，集成后改为依赖 FlashMLX 的增强版
- FlashMLX 是**可选依赖** — ThunderOMLX 没装 FlashMLX 时应 fallback 到 vanilla 行为

---

## 2. 安装方式

```bash
# Step 1: 安装增强版 mlx-lm（替代 vanilla mlx-lm）
cd /path/to/FlashMLX/mlx-lm-source && pip install -e .

# Step 2: 安装 FlashMLX SDK
cd /path/to/FlashMLX && pip install -e .

# 验证
python3 -c "import flashmlx; print(flashmlx.__version__)"  # → 1.0.0
```

增强版 mlx-lm 与 vanilla mlx-lm 100% 向后兼容 — 所有 `from mlx_lm import ...` 不需要任何修改。增强版只是新增了 `cache_factory`, `triple_layer_cache`, `quantization_strategies`, `expert_offload`, `am_calibrator` 等模块。

---

## 3. 需要修改的 ThunderOMLX 文件

### 3.1 `src/omlx/settings_v2.py` — 新增 FlashMLX 配置节

在 `GlobalSettingsV2` 中嵌入 FlashMLX 配置。

**改动**：

```python
# === 新增 import（文件头部）===
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    pass

# === 新增配置类（放在 CacheSettingsV2 之后）===

class FlashMLXSettingsV2(BaseModel):
    """FlashMLX 模型层优化配置。

    控制 KV cache 压缩和 expert offloading。
    仅在安装了 flashmlx 包时生效。
    """
    enabled: bool = Field(
        default=False,
        description="启用 FlashMLX 模型层优化。需要安装 flashmlx 包。",
    )

    # KV Cache 策略
    cache_strategy: str = Field(
        default="auto",
        description=(
            "KV cache 策略: 'standard' (无优化), 'scored_pq' (推荐, AM压缩+量化), "
            "'triple' (三层缓存), 'auto' (自动选择)"
        ),
    )
    flat_quant: Optional[str] = Field(
        default="q8_0",
        description="Flat buffer 量化: None (bf16), 'q8_0' (推荐), 'q4_0', 'turboquant'",
    )
    compression_ratio: float = Field(
        default=2.0,
        description="AM 压缩比 (0 = 自适应)",
    )

    # Expert Offloading (MoE 模型)
    expert_offload: bool = Field(
        default=False,
        description="启用 expert offloading (仅 MoE 模型)",
    )
    expert_pool_size: int = Field(
        default=0,
        description="GPU expert pool 大小 (0 = 自动检测)",
    )

    @field_validator("cache_strategy")
    @classmethod
    def validate_cache_strategy(cls, v: str) -> str:
        valid = ("standard", "triple", "triple_am", "triple_pq",
                 "scored_pq", "auto")
        if v not in valid:
            raise ValueError(f"Unknown cache_strategy: {v!r}. Use one of: {valid}")
        return v

    @field_validator("flat_quant")
    @classmethod
    def validate_flat_quant(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in ("q8_0", "q4_0", "turboquant"):
            raise ValueError(f"Unknown flat_quant: {v!r}")
        return v


# === 在 GlobalSettingsV2 中新增一行 ===

class GlobalSettingsV2(BaseSettings):
    # ... 现有字段 ...
    cache: CacheSettingsV2 = Field(default_factory=CacheSettingsV2)
    flashmlx: FlashMLXSettingsV2 = Field(default_factory=FlashMLXSettingsV2)  # ← 新增
    # ... 其余字段 ...
```

**对应的 settings.json 配置**：

```json
{
  "flashmlx": {
    "enabled": true,
    "cache_strategy": "scored_pq",
    "flat_quant": "q8_0",
    "expert_offload": false
  }
}
```

**环境变量覆盖**：

```bash
OMLX_FLASHMLX__ENABLED=true
OMLX_FLASHMLX__CACHE_STRATEGY=scored_pq
OMLX_FLASHMLX__FLAT_QUANT=q8_0
```

---

### 3.2 `src/omlx/engine/batched.py` — 模型加载时应用 FlashMLX

这是模型加载的入口。当 `flashmlx.enabled=True` 时，加载后自动应用优化。

**改动**（在 `_load_model_sync()` 函数内，model/tokenizer load 之后）：

```python
def _load_model_sync():
    from mlx_lm import load
    model, tokenizer = load(model_path, ...)

    # === 新增: FlashMLX 模型层优化 ===
    flashmlx_ctx = None
    if settings.flashmlx.enabled:
        flashmlx_ctx = _apply_flashmlx(model, tokenizer, model_path, settings)

    return model, tokenizer, flashmlx_ctx


def _apply_flashmlx(model, tokenizer, model_path, settings):
    """尝试应用 FlashMLX 优化。FlashMLX 未安装时静默跳过。"""
    try:
        from flashmlx import FlashMLXConfig, CacheConfig, OffloadConfig
        from flashmlx.integration import setup_flashmlx
    except ImportError:
        import logging
        logging.getLogger(__name__).warning(
            "flashmlx.enabled=True but flashmlx package not installed. Skipping."
        )
        return None

    config = FlashMLXConfig(
        cache=CacheConfig(
            strategy=settings.flashmlx.cache_strategy,
            flat_quant=settings.flashmlx.flat_quant,
            compression_ratio=settings.flashmlx.compression_ratio,
        ),
        offload=OffloadConfig(
            enabled=settings.flashmlx.expert_offload,
            pool_size=settings.flashmlx.expert_pool_size,
        ),
    )

    cache_list, offload_ctx = setup_flashmlx(model, tokenizer, model_path, config)

    # cache_list 在后续 BatchGenerator 创建时使用
    # offload_ctx 必须保持存活（prevent GC）
    return {
        "cache_list": cache_list,
        "offload_ctx": offload_ctx,
        "config": config,
    }
```

---

### 3.3 `src/omlx/scheduler.py` — BatchGenerator 使用 FlashMLX 缓存

ThunderOMLX 的 Scheduler 通过 `BatchGenerator` 管理推理。关键集成点在 `_create_batch_generator()` 方法（约 L2142）。

**当前代码**（简化）：

```python
def _create_batch_generator(self, sampling_params):
    bg = _BoundarySnapshotBatchGenerator(
        model=self.model,
        tokenizer=self.tokenizer,
        # ... sampling params ...
    )
    return bg
```

**改动思路**：

BatchGenerator 内部调用 `make_prompt_cache()` 创建 per-request KV cache。FlashMLX 的优化通过 `make_prompt_cache()` 的参数传入，所以有两种集成方式：

**方式 A（推荐 — 透明传参）**：

增强版 mlx-lm 的 `BatchGenerator` 已经支持 `kv_cache`, `kv_flat_quant` 等参数。只需在创建时传入：

```python
def _create_batch_generator(self, sampling_params):
    kv_kwargs = {}
    if self._flashmlx_ctx:
        cfg = self._flashmlx_ctx["config"]
        kv_kwargs = cfg.cache.to_cache_kwargs()  # → {"kv_cache": "scored_pq", "kv_flat_quant": "q8_0", ...}

    bg = _BoundarySnapshotBatchGenerator(
        model=self.model,
        tokenizer=self.tokenizer,
        **kv_kwargs,
        # ... 其余参数 ...
    )
    return bg
```

**方式 B（预创建 cache）**：

如果 BatchGenerator 不直接支持 kv_cache 参数，可以预创建 cache 并注入：

```python
from mlx_lm.models.cache import make_prompt_cache

cache_list = make_prompt_cache(self.model, **kv_kwargs)
# 然后将 cache_list 传给 BatchGenerator 或直接用于 generate_step
```

---

### 3.4 与现有 KVTC 的关系

ThunderOMLX 已有 `src/omlx/cache/kvtc_codec.py`（从 FlashMLX 手动移植的 KVTC 编解码器），用于 L2 SSD 缓存压缩。

**这两者不冲突**：
- **KVTC** = SSD 持久化压缩（存盘/读盘时压缩 KV blocks）
- **FlashMLX scored_pq** = GPU 内存压缩（推理过程中 KV cache 压缩）

两者可以叠加：FlashMLX 在 GPU 内存中压缩 KV cache（-94%），KVTC 在写入 SSD 时进一步压缩。

**不需要修改** KVTC 相关代码。

---

## 4. FlashMLX SDK 完整 API 参考

### 4.1 顶层导出 (`import flashmlx`)

```python
import flashmlx

flashmlx.__version__           # "1.0.0"

# 模型加载（等同 from mlx_lm import load）
flashmlx.load(model_path)     # → (model, tokenizer)
flashmlx.generate(...)         # → str
flashmlx.stream_generate(...)  # → generator

# 配置
flashmlx.FlashMLXConfig        # 完整配置 (cache + offload)
flashmlx.CacheConfig            # KV cache 配置
flashmlx.OffloadConfig          # Expert offloading 配置

# 能力探测
flashmlx.detect_capabilities(model, model_path)  # → ModelCapabilities
flashmlx.recommend_config(model, model_path)       # → FlashMLXConfig

# 缓存创建
flashmlx.make_prompt_cache(model, **kwargs)        # → list[cache]
flashmlx.make_optimized_cache(model, **kwargs)     # → list[cache]
flashmlx.VALID_STRATEGIES                          # tuple of strategy names

# 量化
flashmlx.get_quantizer(name)                       # → QuantizationStrategy
```

### 4.2 CacheConfig 字段

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `strategy` | str | "standard" | 缓存策略 |
| `flat_quant` | str\|None | None | Flat buffer 量化方式 |
| `compression_ratio` | float | 2.0 | AM 压缩比 |
| `scored_max_cache` | int | 2048 | scored_pq 最大缓存 token 数 |
| `calibration_file` | str\|None | None | AM 校准文件路径 |
| `warm_bits` | int | 4 | 暖层量化位数 (2/3/4) |
| `recent_size` | int | 512 | Recent 层大小 |
| `warm_size` | int | 2048 | Warm 层大小 |

**转换方法**：
- `config.cache.to_cache_kwargs()` → `{"kv_cache": "scored_pq", "kv_flat_quant": "q8_0", ...}`
  用于 `make_prompt_cache(model, **kwargs)`
- `config.cache.to_factory_kwargs()` → `{"strategy": "scored_pq", "flat_quant": "q8_0", ...}`
  用于 `make_optimized_cache(model, **kwargs)`

### 4.3 OffloadConfig 字段

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | bool | False | 是否启用 |
| `pool_size` | int | 0 | GPU pool 大小 (0=auto) |
| `max_workers` | int | 4 | SSD 加载线程数 |
| `cpu_cache_gb` | float\|None | None | CPU 缓存大小 |

### 4.4 FlashMLXConfig.apply()

一键应用配置：

```python
config = FlashMLXConfig(
    cache=CacheConfig(strategy="scored_pq", flat_quant="q8_0"),
    offload=OffloadConfig(enabled=True),
)
cache_list, offload_ctx = config.apply(model, model_path="/path/to/model")

# cache_list: list[TripleLayerKVCache], 每层一个
# offload_ctx: OffloadContext 或 None（必须保持引用防止 GC）
```

### 4.5 setup_flashmlx() — 一键集成

```python
from flashmlx.integration import setup_flashmlx

# 自动: 探测能力 → 校准（如需要）→ 创建缓存 → 应用 offload
cache_list, offload_ctx = setup_flashmlx(model, tokenizer, model_path, config=None)
# config=None 时自动用 recommend_config() 生成最优配置
```

### 4.6 ModelCapabilities

```python
caps = flashmlx.detect_capabilities(model, model_path)

caps.model_type             # "transformer" | "hybrid" | "moe"
caps.num_layers             # 36
caps.head_dim               # 128
caps.is_moe                 # False
caps.is_hybrid              # False
caps.supports_scored_pq     # True
caps.supports_turboquant    # True (head_dim >= 128)
caps.supports_expert_offload # False
caps.has_calibration        # True
caps.calibration_path       # "/path/to/am_calibration.pkl"
caps.recommended_strategy   # "scored_pq"
caps.recommended_flat_quant # "q8_0"
caps.warnings               # []
```

### 4.7 FlashMLXProvider Protocol（松耦合）

```python
from flashmlx.integration.protocol import FlashMLXProvider
from flashmlx.integration.thunderomlx import ThunderOMLXAdapter

adapter = ThunderOMLXAdapter()
assert isinstance(adapter, FlashMLXProvider)  # runtime checkable

# 通过 protocol 接口使用
cache = adapter.create_cache(model, strategy="scored_pq", flat_quant="q8_0")
info = adapter.get_cache_info(cache)
caps = adapter.detect_capabilities(model, model_path)
cal_path = adapter.auto_calibrate(model, tokenizer, compression_ratio=2.0)
```

---

## 5. 性能数据参考

Qwen3-8B, M4 Pro 48GB, 16K context, 200 gen tokens:

| 配置 | TG tok/s | KV MB | 节省 |
|------|----------|-------|------|
| standard (vanilla mlx-lm) | 21.0 | 2,454 | — |
| **scored+q8_0 (推荐)** | **25.9** | **153** | **-94%** |
| scored+bf16 (最快) | 28.3 | 302 | -88% |
| scored+q4_0 | 18.7 | 85 | -97% |
| scored+turboquant | 16.5 | 80 | -97% |

**scored+q8_0 是推荐默认配置**：TG +23% 且 KV 内存 -94%。

---

## 6. 集成注意事项

### 6.1 FlashMLX 是可选依赖

ThunderOMLX 必须在 FlashMLX 未安装时正常工作。所有 `import flashmlx` 必须包在 try/except 中：

```python
try:
    from flashmlx import FlashMLXConfig, CacheConfig
    from flashmlx.integration import setup_flashmlx
    HAS_FLASHMLX = True
except ImportError:
    HAS_FLASHMLX = False
```

### 6.2 增强版 mlx-lm 向后兼容

增强版 mlx-lm 安装后，所有 `from mlx_lm import ...` 继续工作。ThunderOMLX 现有的 BatchGenerator、generate_step、make_sampler 等导入不受影响。

### 6.3 scored_pq 对 Hybrid 模型自动禁用

FlashMLX 内部会检测 hybrid 架构（如 Qwen3.5-35B-A3B 的 SSM+Attention 混合）并自动退回 standard 缓存。ThunderOMLX 不需要额外处理。

### 6.4 Expert Offloading 与 BatchGenerator 的关系

Expert offloading 通过 `patch_model_for_offload()` 修改模型的 forward pass（替换 MoE 层为 FlashMoeSwitchGLU）。这对 BatchGenerator 透明 — 不需要修改 BatchGenerator 代码。

但 `offload_ctx`（OffloadContext）必须在推理期间保持存活。建议把它存在 Scheduler 或 EngineCore 的实例变量上。

### 6.5 AM 校准文件管理

FlashMLX 的 `auto_calibrate()` 会自动在 `~/.cache/flashmlx/calibrations/` 下生成和缓存校准文件。首次使用某个模型的 scored_pq 策略时会自动触发（约 10-30 秒），之后复用缓存。

ThunderOMLX 也可以显式指定校准文件路径（通过 `CacheConfig.calibration_file`），比如指向预生成的校准文件。

### 6.6 不要复制代码

FlashMLX SDK 通过 re-export 提供所有 API。ThunderOMLX 不应从 `mlx_lm.models.triple_layer_cache` 等内部模块直接 import — 始终通过 `flashmlx` 或 `flashmlx.cache` 导入。

```python
# Good
from flashmlx.cache import TripleLayerKVCache, make_optimized_cache

# Bad — 绕过 SDK，直接耦合内部模块
from mlx_lm.models.triple_layer_cache import TripleLayerKVCache
```

---

## 7. 文件清单

### FlashMLX SDK 文件结构（只读参考）

```
FlashMLX/
├── src/flashmlx/
│   ├── __init__.py              # 公共 API 导出
│   ├── config.py                # Pydantic 配置 (CacheConfig, OffloadConfig, FlashMLXConfig)
│   ├── capabilities.py          # 模型能力探测 + 推荐配置
│   ├── cache/__init__.py        # 从 mlx-lm 重导出缓存 API
│   ├── offload/__init__.py      # 从 mlx-lm 重导出 offload API
│   ├── integration/
│   │   ├── __init__.py
│   │   ├── protocol.py          # FlashMLXProvider 协议接口
│   │   └── thunderomlx.py       # ThunderOMLX 适配器 + setup_flashmlx()
│   ├── profiler/                # 性能分析工具
│   └── utils.py                 # timer/benchmark 工具
├── mlx-lm-source/               # 增强版 mlx-lm（git submodule）
│   └── mlx_lm/models/
│       ├── cache.py             # make_prompt_cache()
│       ├── cache_factory.py     # make_optimized_cache(), VALID_STRATEGIES
│       ├── triple_layer_cache.py # TripleLayerKVCache (1,610 行)
│       ├── quantization_strategies.py # Q4_0, Q8_0, PolarQuant, TurboQuant
│       ├── expert_offload.py    # patch_model_for_offload, FlashBatchGenerator
│       └── am_calibrator.py     # auto_calibrate()
├── pyproject.toml               # v1.0.0, pydantic>=2.0
└── INTEGRATION_GUIDE.md         # 本文档
```

### ThunderOMLX 需要改的文件

| 文件 | 改动 |
|------|------|
| `src/omlx/settings_v2.py` | 新增 `FlashMLXSettingsV2` 类 + GlobalSettingsV2 中嵌入 |
| `src/omlx/engine/batched.py` | 模型加载后调用 `setup_flashmlx()` |
| `src/omlx/scheduler.py` | `_create_batch_generator()` 传入 kv_cache 参数 |
| `pyproject.toml` / `setup.py` | 可选依赖: `flashmlx` |

### ThunderOMLX 不需要改的文件

| 文件 | 原因 |
|------|------|
| `src/omlx/cache/kvtc_codec.py` | KVTC 是 SSD 层压缩，与 FlashMLX GPU 层压缩互补，不冲突 |
| `src/omlx/models/llm.py` | 继续用 `from mlx_lm import load, generate`（增强版兼容） |
| `src/omlx/server.py` | API 层不变 |

---

## 8. 最小可行集成（MVP）

如果只想用最少改动获得最大收益，只需：

1. 安装增强版 mlx-lm（替代 vanilla）
2. 在模型加载后加一行：

```python
from mlx_lm.models.cache import make_prompt_cache

# 创建优化缓存（替代默认的 KVCache）
cache_list = make_prompt_cache(
    model,
    kv_cache="scored_pq",
    kv_flat_quant="q8_0",
    kv_calibration="/path/to/calibration.pkl",  # 或 None（自动校准）
)
```

这就够了。不需要 `import flashmlx`，不需要改 settings。增强版 mlx-lm 的 `make_prompt_cache()` 直接支持所有策略参数。

FlashMLX SDK 的价值在于：配置管理、能力探测、自动校准、Protocol 松耦合。当你需要这些时再接入完整 SDK。
