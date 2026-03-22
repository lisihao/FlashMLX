# FlashMLX Hybrid Cache API Reference

## Table of Contents

1. [High-Level API](#high-level-api)
   - [inject_hybrid_cache_manager()](#inject_hybrid_cache_manager)
   - [restore_original_cache()](#restore_original_cache)
   - [create_layer_types_from_model()](#create_layer_types_from_model)
2. [Configuration](#configuration)
   - [HybridCacheConfig](#hybridcacheconfig)
   - [LayerType](#layertype)
3. [Cache Wrapper](#cache-wrapper)
   - [HybridCacheWrapper](#hybridcachewrapper)
4. [Low-Level Components](#low-level-components)

---

## High-Level API

### inject_hybrid_cache_manager()

Inject hybrid cache into MLX-LM model (non-invasive monkey patching).

**Signature**:
```python
def inject_hybrid_cache_manager(
    model: Any,
    config: HybridCacheConfig,
    layer_types: Dict[int, LayerType],
    auto_inject: bool = True
) -> HybridCacheWrapper
```

**Parameters**:
- `model` (Any): MLX-LM model instance (e.g., from `mlx_lm.load()`)
- `config` (HybridCacheConfig): Cache configuration
- `layer_types` (Dict[int, LayerType]): Mapping of layer index to layer type
- `auto_inject` (bool, default=True): If True, automatically replace `model.cache`

**Returns**:
- `HybridCacheWrapper`: Wrapper instance managing the hybrid cache

**Example**:
```python
from mlx_lm import load
from flashmlx.cache import (
    inject_hybrid_cache_manager,
    create_layer_types_from_model,
    HybridCacheConfig
)

# Load model
model, tokenizer = load("mlx-community/Qwen3.5-35B-Instruct-4bit")

# Detect layer types
layer_types = create_layer_types_from_model(
    model,
    attention_layer_pattern="every 4th"
)

# Configure cache
config = HybridCacheConfig(
    total_budget_bytes=64 * 1024 * 1024,  # 64MB
    compression_ratio=4.0,
    beta_calibration=True
)

# Inject hybrid cache
cache_wrapper = inject_hybrid_cache_manager(
    model=model,
    config=config,
    layer_types=layer_types,
    auto_inject=True
)

# Now use model normally with mlx_lm.generate()
```

**Notes**:
- Automatically saves original cache to `model.cache._original_cache`
- Works with any MLX-LM compatible model
- Non-invasive (can be reversed with `restore_original_cache()`)

---

### restore_original_cache()

Restore model's original cache (reverse monkey patching).

**Signature**:
```python
def restore_original_cache(
    model: Any,
    wrapper: HybridCacheWrapper
) -> None
```

**Parameters**:
- `model` (Any): MLX-LM model instance
- `wrapper` (HybridCacheWrapper): Cache wrapper returned by `inject_hybrid_cache_manager()`

**Returns**:
- None

**Example**:
```python
# Inject hybrid cache
cache_wrapper = inject_hybrid_cache_manager(model, config, layer_types)

# ... use model ...

# Restore original cache
restore_original_cache(model, cache_wrapper)

# Model now uses original cache again
```

---

### create_layer_types_from_model()

Automatically detect layer types from model architecture.

**Signature**:
```python
def create_layer_types_from_model(
    model: Any,
    attention_layer_indices: Optional[List[int]] = None,
    attention_layer_pattern: Optional[str] = None
) -> Dict[int, LayerType]
```

**Parameters**:
- `model` (Any): MLX-LM model instance
- `attention_layer_indices` (Optional[List[int]]): Explicit list of Attention layer indices
- `attention_layer_pattern` (Optional[str]): Pattern string (e.g., "every 4th")

**Returns**:
- `Dict[int, LayerType]`: Mapping of layer index to LayerType.SSM or LayerType.ATTENTION

**Detection Methods**:

1. **Explicit indices** (highest priority):
```python
layer_types = create_layer_types_from_model(
    model,
    attention_layer_indices=[3, 7, 11, 15, 19, 23, 27, 31, 35, 39]
)
```

2. **Pattern-based**:
```python
layer_types = create_layer_types_from_model(
    model,
    attention_layer_pattern="every 4th"  # Layers 3, 7, 11, ...
)
```

3. **Auto-detection** (fallback):
```python
layer_types = create_layer_types_from_model(model)
# Inspects model.layers[i].self_attn existence
```

**Supported Patterns**:
- `"every 4th"`: Layers 3, 7, 11, 15, ... (Qwen3.5 default)
- `"every 2nd"`: Layers 1, 3, 5, 7, ...
- `"every Nth"`: Custom pattern

---

## Configuration

### HybridCacheConfig

Configuration dataclass for hybrid cache.

**Signature**:
```python
@dataclass
class HybridCacheConfig:
    total_budget_bytes: int = 128 * 1024 * 1024
    compression_ratio: float = 4.0
    beta_calibration: bool = True
    hot_budget_ratio: float = 0.15
    warm_budget_ratio: float = 0.25
    cold_budget_ratio: float = 0.55
    pinned_budget_ratio: float = 0.05
    hot_high_waterline: float = 0.85
    warm_high_waterline: float = 0.85
    warm_low_waterline: float = 0.25
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `total_budget_bytes` | int | 128MB | Total memory budget for cache |
| `compression_ratio` | float | 4.0 | Attention layer compression ratio (2.0-5.0) |
| `beta_calibration` | bool | True | Enable β calibration for Attention Matching |
| `hot_budget_ratio` | float | 0.15 | Hot tier budget (% of total) |
| `warm_budget_ratio` | float | 0.25 | Warm tier budget (% of total) |
| `cold_budget_ratio` | float | 0.55 | Cold tier budget (% of total) |
| `pinned_budget_ratio` | float | 0.05 | Pinned tier budget (% of total) |
| `hot_high_waterline` | float | 0.85 | Hot tier eviction threshold |
| `warm_high_waterline` | float | 0.85 | Warm tier eviction threshold |
| `warm_low_waterline` | float | 0.25 | Warm tier revival threshold |

**Validation**:
- `hot + warm + cold + pinned` ratios must sum to 1.0
- `compression_ratio` must be ≥ 1.0
- `total_budget_bytes` must be > 0

**Recommended Configurations**:

**Long Context** (recommended):
```python
config = HybridCacheConfig(
    total_budget_bytes=64 * 1024 * 1024,
    compression_ratio=4.0,
    beta_calibration=True
)
```

**Medium Context**:
```python
config = HybridCacheConfig(
    total_budget_bytes=64 * 1024 * 1024,
    compression_ratio=3.0,
    beta_calibration=True
)
```

**Short Context**:
```python
config = HybridCacheConfig(
    total_budget_bytes=64 * 1024 * 1024,
    compression_ratio=2.0,
    beta_calibration=True
)
```

---

### LayerType

Enum defining layer types.

**Definition**:
```python
class LayerType(Enum):
    SSM = "ssm"
    ATTENTION = "attention"
```

**Usage**:
```python
from flashmlx.cache import LayerType

layer_types = {
    0: LayerType.SSM,
    1: LayerType.SSM,
    2: LayerType.SSM,
    3: LayerType.ATTENTION,
    # ...
}
```

---

## Cache Wrapper

### HybridCacheWrapper

Unified cache wrapper providing MLX-LM compatible interface.

**Attributes**:
- `scheduler` (LayerScheduler): Layer routing scheduler
- `ssm_cache` (ManagedArraysCache): SSM layer cache backend
- `attention_cache` (CompressedKVCache): Attention layer cache backend

**Methods**:

#### update_and_fetch_ssm()

Update and retrieve SSM layer cache.

**Signature**:
```python
def update_and_fetch_ssm(
    self,
    layer_idx: int,
    state: Tuple[mx.array, mx.array],
    priority: float = 1.0
) -> mx.array
```

**Parameters**:
- `layer_idx` (int): Layer index
- `state` (Tuple[mx.array, mx.array]): SSM state tuple (h, c)
- `priority` (float): Access priority (default 1.0)

**Returns**:
- `mx.array`: Retrieved/updated SSM state

---

#### update_and_fetch_attention()

Update and retrieve Attention layer cache.

**Signature**:
```python
def update_and_fetch_attention(
    self,
    layer_idx: int,
    keys: mx.array,
    values: mx.array,
    query: Optional[mx.array] = None
) -> Tuple[mx.array, mx.array]
```

**Parameters**:
- `layer_idx` (int): Layer index
- `keys` (mx.array): Key tensor
- `values` (mx.array): Value tensor
- `query` (Optional[mx.array]): Query tensor (for Attention Matching)

**Returns**:
- `Tuple[mx.array, mx.array]`: (compressed_keys, compressed_values)

---

#### get_statistics()

Get cache statistics.

**Signature**:
```python
def get_statistics(self) -> Dict[str, Any]
```

**Returns**:
```python
{
    "ssm": {
        "local_cache": {
            "size": 30,
            "total_updates": 1500,
            "total_retrievals": 1500,
            "hit_rate": 0.85
        },
        "tiered_cache": {
            "hot_size": 10,
            "warm_size": 15,
            "cold_size": 5,
            "pinned_size": 2
        }
    },
    "attention": {
        "local_cache": {
            "size": 10,
            "avg_compression_ratio": 3.85,
            "total_compressions": 500
        }
    },
    "scheduler": {
        "ssm_layer_count": 30,
        "attention_layer_count": 10
    }
}
```

---

#### clear()

Clear cache (all or specific layer).

**Signature**:
```python
def clear(self, layer_idx: Optional[int] = None) -> None
```

**Parameters**:
- `layer_idx` (Optional[int]): Layer index to clear (None = clear all)

**Example**:
```python
# Clear all caches
cache_wrapper.clear()

# Clear specific layer
cache_wrapper.clear(layer_idx=5)
```

---

## Low-Level Components

### AttentionMatchingCompressor

Core Attention Matching compression algorithm.

**Key Methods**:

#### compress()
```python
def compress(
    keys: mx.array,
    values: mx.array,
    query: mx.array,
    compression_ratio: float
) -> Tuple[mx.array, mx.array, float]:
    """
    Compress KV cache using Attention Matching.

    Returns:
        (compressed_keys, compressed_values, beta)
    """
```

#### calibrate_beta()
```python
def calibrate_beta(
    attention_weights: mx.array,
    selected_indices: mx.array
) -> float:
    """
    Calibrate β parameter for distribution compensation.

    Returns:
        beta value
    """
```

---

### ManagedArraysCache

SSM layer cache with tiered management.

**Key Methods**:

#### update_and_fetch()
```python
def update_and_fetch(
    layer_idx: int,
    state: Tuple[mx.array, mx.array],
    priority: float = 1.0
) -> mx.array:
    """Update and fetch SSM state"""
```

#### retrieve()
```python
def retrieve(layer_idx: int) -> Optional[mx.array]:
    """Retrieve SSM state (None if not found)"""
```

#### contains()
```python
def contains(layer_idx: int) -> bool:
    """Check if layer is cached"""
```

---

### CompressedKVCache

Attention layer cache with compression.

**Key Methods**:

#### update_and_fetch()
```python
def update_and_fetch(
    layer_idx: int,
    keys: mx.array,
    values: mx.array,
    query: Optional[mx.array] = None
) -> Tuple[mx.array, mx.array]:
    """Update and fetch compressed KV"""
```

---

## Error Handling

### Common Errors

**TypeError: Expected tuple for SSM layer**
```python
# Wrong: passing single array to SSM layer
cache.update_and_fetch_ssm(0, state_array)  # ❌

# Correct: passing tuple
cache.update_and_fetch_ssm(0, (h, c))  # ✅
```

**TypeError: Expected mx.array for Attention layer**
```python
# Wrong: passing tuple to Attention layer
cache.update_and_fetch_attention(3, (keys, values))  # ❌

# Correct: passing arrays separately
cache.update_and_fetch_attention(3, keys, values, query)  # ✅
```

**ValueError: Budget ratios don't sum to 1.0**
```python
# Wrong: ratios sum to 0.9
config = HybridCacheConfig(
    hot_budget_ratio=0.15,
    warm_budget_ratio=0.25,
    cold_budget_ratio=0.45,  # Should be 0.55
    pinned_budget_ratio=0.05
)

# Correct: ratios sum to 1.0
config = HybridCacheConfig(
    hot_budget_ratio=0.15,
    warm_budget_ratio=0.25,
    cold_budget_ratio=0.55,
    pinned_budget_ratio=0.05
)
```

---

## Type Hints

All public APIs include complete type hints for IDE support:

```python
from typing import Dict, List, Optional, Tuple, Any
import mlx.core as mx

def inject_hybrid_cache_manager(
    model: Any,
    config: HybridCacheConfig,
    layer_types: Dict[int, LayerType],
    auto_inject: bool = True
) -> HybridCacheWrapper: ...
```

---

## Thread Safety

**Current Status**: Not thread-safe

**Recommendation**: Use single-threaded generation or implement external locking

**Future Work**: Thread-safe cache implementation planned for v2.0

---

## Performance Tips

1. **Use recommended configurations**: Load pre-tuned configs from `tuning_results/config_templates/`

2. **Disable for short contexts**: For prompts <1000 tokens, hybrid cache overhead may exceed benefits

3. **Monitor statistics**: Call `get_statistics()` periodically to track cache performance

4. **Adjust compression ratio**: Higher ratio = more memory savings but lower quality

5. **Budget allocation**: 64MB sufficient for most single-session use cases

---

*Last Updated: 2026-03-21*
*Version: 1.0*
