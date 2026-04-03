# FlashMLX ↔ ThunderOMLX 缓存桥接集成方案

## THUNDEROMLX-CACHE-BRIDGE-SPEC v1.0

> **目标读者**: ThunderOMLX 侧开发者 (Solar 实例)
>
> **前提**: 已了解 ThunderOMLX 的 PagedCacheManager、PagedSSDCacheManager、BlockAwarePrefixCache、CacheTypeHandler 体系。
>
> **文档性质**: 自包含技术规格，FlashMLX 侧 API 已全部实现。

---

## 1. 问题陈述

### 1.1 现状

ThunderOMLX 的 paged cache 体系 (block_size=64, chain-hash dedup, LRU-2 SSD eviction) 正常工作于标准 `KVCache` 和 `RotatingKVCache`。

FlashMLX 引入了 `TripleLayerKVCache`，在 prefill→TG 转换时将 K/V 压缩为 flat buffer (Q8/Q4/TurboQuant)，同时维护 `H0Store` (h^(0) 残差检查点)。

### 1.2 根因

`TripleLayerKVCache` **没有** override `_BaseCache.state` property。`_BaseCache.state` 返回 `[]` (cache.py:320)。

这导致 `scheduler.py:2885` 的 `_extract_cache_states()` 对 FlashMLX 缓存层拿到的是空 tuple，所有下游操作失效：

```
_extract_cache_states()
  → layer_cache.state     # 返回 () ← 根因
  → extracted = {'state': (), ...}
  → store_cache()         # 存空块到 SSD
  → fetch_cache()         # 永远 miss
```

### 1.3 后果

1. FlashMLX 缓存无法存入 paged SSD → prefix cache hit 不可能
2. 每次请求重新 prefill + AM scoring + 压缩 → 浪费算力
3. H0Store 的 18× 压缩优势无法被 SSD 缓存利用

### 1.4 修复目标

三层递进集成，让 FlashMLX 的压缩产物融入 ThunderOMLX 的 paged block 体系：

| 层级 | 名称 | 效果 |
|------|------|------|
| **Tier 1** | 压缩 Block 存储 | SSD 省 50-75%，热路径无退化 |
| **Tier 2** | H0Store SSD 集成 | H0 落盘，KV 淘汰后保留 H0 |
| **Tier 3** | 3PIR 冷缓存恢复 | 从 H0 重建 KV，冷 hit ~200ms (vs full miss ~2-10s) |

---

## 2. FlashMLX 侧已完成 API

> 以下所有方法均已实现并通过语法验证。ThunderOMLX 侧可直接调用。

### 2.1 TripleLayerKVCache 新增接口

**文件**: `mlx_lm/models/triple_layer_cache.py`

#### `state` property (Tier 1 兼容层)

```python
@property
def state(self) -> tuple:
    """返回 dequant bf16 (keys, values)，兼容现有提取逻辑。

    Returns:
        - flat_mode=True: 调用 _fetch_flat() 返回 dequant bf16
        - flat_mode=False: 返回 (recent_keys, recent_values)
        - 无数据: 返回 ()

    Shape: (B, n_kv_heads, seq_len, head_dim) bf16
    """

@state.setter
def state(self, v):
    """从 bf16 K/V 恢复 flat buffer（会重新量化）。

    Args:
        v: (keys, values) tuple, shape (B, n_kv_heads, seq_len, head_dim)
    """
```

**行为**: getter 返回 dequant bf16，现有 `_extract_cache_states()` 不需要任何修改即可工作。setter 将 bf16 重新量化写入 flat buffer (round-trip 有量化误差)。

#### `meta_state` property

```python
@property
def meta_state(self) -> tuple:
    """返回序列化元数据 tuple。

    Returns:
        (flat_quant, flat_offset, true_offset,
         flat_prefix_token_count, flat_mode, pq_head_dim)
        所有值为 str 类型（兼容 safetensors metadata）。
    """

@meta_state.setter
def meta_state(self, v: tuple):
    """从 tuple 恢复元数据。"""
```

#### `export_flat_state()` → dict (Tier 1 核心)

```python
def export_flat_state(self) -> Optional[Dict]:
    """导出原始压缩 arrays，不做 dequant。

    Returns:
        {
            'flat_keys': mx.array,       # (B, n_heads, offset, packed_dim) int8/uint8/uint32
            'flat_values': mx.array,     # 同上
            'flat_keys_scales': mx.array | None,   # (B, n_heads, offset, scale_dim)
            'flat_values_scales': mx.array | None,
            'flat_quant': str | None,    # 'q8_0' | 'q4_0' | 'turboquant' | None
            'flat_offset': int,
            'true_offset': int,
            'flat_prefix_token_count': int,
            'head_dim': int | None,      # turboquant only
        }
        只包含 [0:flat_offset] 的有效数据，不导出空余空间。
        返回 None 如果不在 flat_mode。
    """
```

#### `import_flat_state(state)` → bool (Tier 1 核心)

```python
def import_flat_state(self, state: Dict) -> bool:
    """从压缩 arrays 恢复 flat buffer，不做 requant。

    round-trip: export_flat_state() → import_flat_state() 是 bit-exact 的。
    不经过 dequant/requant，直接复制压缩数据。

    Args:
        state: export_flat_state() 的返回值。

    Returns:
        True if import succeeded.
    """
```

### 2.2 H0Store 新增接口

**文件**: `mlx_lm/models/kv_direct_cache.py`

#### `export_blocks(block_size=64)` → list (Tier 2)

```python
def export_blocks(self, block_size: int = 64) -> list:
    """按 block_size 切割 h^(0)，每 block 一个 dict。

    Args:
        block_size: 必须匹配 ThunderOMLX 的 paged cache block_size。

    Returns:
        [{
            'h0': mx.array,          # (B, <=block_size, dim) 原始 dtype
            'scales': mx.array|None, # (B, <=block_size, 1) if quantized
            'block_idx': int,
            'token_start': int,
            'token_end': int,
            'quant': str,            # 'bf16' | 'q8' | 'q4'
        }, ...]
        最后一个 block 可能不满 block_size。
    """
```

#### `import_blocks(blocks)` → int (Tier 2)

```python
def import_blocks(self, blocks: list) -> int:
    """从 block 列表重建 H0Store。

    Blocks 可以乱序传入（按 block_idx 排序后 concat）。

    Returns:
        Total token count restored.
    """
```

#### `block_hash_key(parent_hash, block_idx)` → bytes (Tier 2)

```python
@staticmethod
def block_hash_key(parent_hash: bytes, block_idx: int) -> bytes:
    """SHA-256("h0:" + parent_hash + block_idx)

    h0: 前缀确保不与 KV block hash 碰撞。
    """
```

### 2.3 ThunderOMLXAdapter 新增方法

**文件**: `flashmlx/integration/thunderomlx.py`

```python
class ThunderOMLXAdapter:
    # Tier 1
    def export_compressed_cache_state(self, cache_list: list) -> Optional[Dict]:
        """遍历 cache_list，每层调用 export_flat_state()。

        Returns:
            {
                'layers': [{
                    'layer_idx': int,
                    'type': 'TripleLayerKVCache' | other,
                    'flat_state': dict | None,  # export_flat_state() 结果
                    'meta_state': tuple,
                }, ...],
                'h0_store': {count, quant, nbytes} | None,
                'format': 'flashmlx_compressed_v1',
            }
        """

    def import_compressed_cache_state(
        self, cache_list: list, state: Dict
    ) -> bool:
        """恢复压缩缓存状态（Tier 1 逆操作）。

        Validates format == 'flashmlx_compressed_v1'。
        每层调用 import_flat_state()。
        """

    # Tier 2
    def export_h0_blocks(
        self, cache_list: list, block_size: int = 64
    ) -> Optional[Dict]:
        """从 cache_list 发现 H0Store，调用 export_blocks()。

        Returns:
            {
                'blocks': list,           # export_blocks() 结果
                'total_tokens': int,
                'quant': str,
                'nbytes_per_token': float,
                'format': 'h0_blocks_v1',
            }
        """
```

### 2.4 RCEngine 新增方法

**文件**: `flashmlx/rc_engine.py`

```python
class RCEngine:
    def register_from_h0_blocks(
        self,
        seq_id: str,
        h0_blocks: list,
        inner_model: Any,
        target_cache_list: List[Any],
        h0_quant: Optional[str] = None,
        importance_scores: Optional[Any] = None,
        min_coverage: float = 0.95,
    ) -> RCSequenceState:
        """Tier 3: 从 SSD 加载的 H0 blocks 注册重建序列。

        内部：import_blocks → H0Store → register_sequence()
        之后通过 process_chunk() 逐 chunk 重建。
        """
```

### 2.5 ReconstructionController 新增方法

**文件**: `flashmlx/reconstruction.py`

```python
class ReconstructionController:
    def reconstruct_from_h0_blocks(
        self,
        h0_blocks: list,
        h0_quant: Optional[str] = None,
        strategy: str = "full",
        coverage: float = 0.95,
        chunk_size: int = 512,
        eval_every: int = 8,
    ) -> ReconResult:
        """Tier 3: 阻塞式从 H0 blocks 重建 KV（简单场景用）。

        非阻塞版本走 RCEngine.register_from_h0_blocks() + process_chunk()。
        """
```

### 2.6 CacheConfig 新增字段

**文件**: `flashmlx/config.py`

```python
class CacheConfig(BaseModel):
    # ... existing fields ...

    # ThunderOMLX SSD Cache Bridge
    enable_compressed_ssd: bool = True     # Tier 1
    enable_h0_ssd: bool = True             # Tier 2
    h0_ssd_quant: Optional[str] = "q8"     # Tier 2: None | 'q8' | 'q4'
    enable_cold_restoration: bool = True    # Tier 3
```

---

## 3. ThunderOMLX 侧修改方案

### 3.1 新增 CacheType: TRIPLE_LAYER_KVCACHE

**文件**: `src/omlx/cache/type_handlers.py`

```python
class CacheType(Enum):
    # ... existing ...
    TRIPLE_LAYER_KVCACHE = "TripleLayerKVCache"
```

### 3.2 新增 TripleLayerKVCacheHandler

**文件**: `src/omlx/cache/type_handlers.py`

```python
class TripleLayerKVCacheHandler(CacheTypeHandler):
    """Handler for FlashMLX TripleLayerKVCache.

    Two storage paths:
      - Path A (compressed): export_flat_state() → 压缩 safetensors (Tier 1)
      - Path B (fallback):   .state → dequant bf16 → 标准 safetensors
    """

    @property
    def cache_type(self) -> CacheType:
        return CacheType.TRIPLE_LAYER_KVCACHE

    @property
    def supports_block_slicing(self) -> bool:
        return True  # flat buffer 是连续的，可以按 seq dim 切

    def extract_state(self, cache_obj) -> Dict[str, Any]:
        """尝试 compressed 路径，fallback 到 dequant bf16。

        Returns:
            {
                'keys': mx.array,          # (B, heads, T, dim)
                'values': mx.array,
                'compressed': bool,        # True 表示以下字段有效
                'flat_state': dict | None, # export_flat_state() raw
            }
        """
        flat_state = None
        if hasattr(cache_obj, 'export_flat_state'):
            flat_state = cache_obj.export_flat_state()

        if flat_state is not None:
            # Compressed path: 同时存 dequant bf16 (兼容) + raw flat state
            state = cache_obj.state  # dequant bf16 for standard path
            return {
                'keys': state[0] if state else None,
                'values': state[1] if state else None,
                'compressed': True,
                'flat_state': flat_state,
            }
        else:
            # Fallback: 标准 bf16 (flat_mode=False, 可能在 prefill 阶段)
            state = cache_obj.state
            if state and len(state) >= 2:
                return {'keys': state[0], 'values': state[1], 'compressed': False}
            return {'keys': None, 'values': None, 'compressed': False}

    def get_seq_len(self, state: Dict[str, Any]) -> int:
        if state.get('flat_state'):
            return state['flat_state'].get('flat_offset', 0)
        keys = state.get('keys')
        if keys is not None:
            return keys.shape[2]
        return 0

    def slice_state(
        self, state: Dict[str, Any], start_idx: int, end_idx: int
    ) -> Optional[Dict[str, Any]]:
        """Block-level 切片。

        compressed=True 时切压缩 arrays（保持压缩 dtype），
        compressed=False 时切 bf16 keys/values。
        """
        if state.get('compressed') and state.get('flat_state'):
            fs = state['flat_state']
            sliced_fs = {
                'flat_keys': fs['flat_keys'][..., start_idx:end_idx, :],
                'flat_values': fs['flat_values'][..., start_idx:end_idx, :],
                'flat_quant': fs['flat_quant'],
                'flat_offset': end_idx - start_idx,
                'true_offset': end_idx - start_idx,
                'flat_prefix_token_count': fs.get('flat_prefix_token_count', 0),
            }
            if fs.get('flat_keys_scales') is not None:
                sliced_fs['flat_keys_scales'] = fs['flat_keys_scales'][..., start_idx:end_idx, :]
                sliced_fs['flat_values_scales'] = fs['flat_values_scales'][..., start_idx:end_idx, :]
            if fs.get('head_dim') is not None:
                sliced_fs['head_dim'] = fs['head_dim']

            # 同时切 bf16 fallback (如果存在)
            sliced = {'compressed': True, 'flat_state': sliced_fs}
            if state.get('keys') is not None:
                sliced['keys'] = state['keys'][:, :, start_idx:end_idx, :]
                sliced['values'] = state['values'][:, :, start_idx:end_idx, :]
            return sliced
        else:
            # bf16 path
            if state.get('keys') is None:
                return None
            return {
                'keys': state['keys'][:, :, start_idx:end_idx, :],
                'values': state['values'][:, :, start_idx:end_idx, :],
                'compressed': False,
            }

    def concatenate_states(self, states: List[Dict[str, Any]]) -> Dict[str, Any]:
        """合并多个 block 的状态。

        如果全部是 compressed，合并压缩 arrays。
        否则 fallback 到 bf16 合并。
        """
        all_compressed = all(s.get('compressed') for s in states)

        if all_compressed:
            # 合并 flat_state
            flat_states = [s['flat_state'] for s in states]
            merged_fs = {
                'flat_keys': mx.concatenate(
                    [fs['flat_keys'] for fs in flat_states], axis=2
                ),
                'flat_values': mx.concatenate(
                    [fs['flat_values'] for fs in flat_states], axis=2
                ),
                'flat_quant': flat_states[0]['flat_quant'],
                'flat_offset': sum(fs['flat_offset'] for fs in flat_states),
                'true_offset': sum(fs['true_offset'] for fs in flat_states),
                'flat_prefix_token_count': flat_states[0].get(
                    'flat_prefix_token_count', 0
                ),
            }
            if flat_states[0].get('flat_keys_scales') is not None:
                merged_fs['flat_keys_scales'] = mx.concatenate(
                    [fs['flat_keys_scales'] for fs in flat_states], axis=2
                )
                merged_fs['flat_values_scales'] = mx.concatenate(
                    [fs['flat_values_scales'] for fs in flat_states], axis=2
                )
            if flat_states[0].get('head_dim') is not None:
                merged_fs['head_dim'] = flat_states[0]['head_dim']

            result = {'compressed': True, 'flat_state': merged_fs}

            # 合并 bf16 fallback
            if states[0].get('keys') is not None:
                result['keys'] = mx.concatenate(
                    [s['keys'] for s in states], axis=2
                )
                result['values'] = mx.concatenate(
                    [s['values'] for s in states], axis=2
                )
            return result
        else:
            # bf16 fallback
            keys_list = [s['keys'] for s in states if s.get('keys') is not None]
            values_list = [s['values'] for s in states if s.get('values') is not None]
            if not keys_list:
                return {'keys': None, 'values': None, 'compressed': False}
            return {
                'keys': mx.concatenate(keys_list, axis=2),
                'values': mx.concatenate(values_list, axis=2),
                'compressed': False,
            }

    def reconstruct_cache(
        self, state: Dict[str, Any], meta_state: Optional[Tuple] = None
    ) -> Any:
        """重建 TripleLayerKVCache。

        compressed=True → import_flat_state() (bit-exact, 无 requant)
        compressed=False → state setter (会 requant)
        """
        from mlx_lm.models.triple_layer_cache import TripleLayerKVCache

        # 需要 TripleLayerKVCache 的构造参数
        # 这些从 meta_state 或全局 model config 获取
        # 这里创建一个 minimal cache 然后恢复状态
        cache = TripleLayerKVCache.__new__(TripleLayerKVCache)
        # ... 初始化必要字段 (见下方 §3.3) ...

        if state.get('compressed') and state.get('flat_state'):
            cache.import_flat_state(state['flat_state'])
        elif state.get('keys') is not None:
            cache.state = (state['keys'], state['values'])

        if meta_state:
            cache.meta_state = meta_state

        return cache
```

### 3.3 TripleLayerKVCache 重建的初始化问题

`TripleLayerKVCache` 的 `__init__` 需要 `head_dim`, `n_kv_heads`, `max_size` 等参数。两种解决方案：

**方案 A (推荐): 从 import_flat_state 推断**

`import_flat_state()` 已经从 flat_state dict 恢复所有必要的 buffer 和 metadata。只需要确保 cache 对象的 `_flat_step` 和 `_flat_mode` 等字段被正确设置。

```python
# 在 reconstruct_cache() 中:
cache = TripleLayerKVCache.__new__(TripleLayerKVCache)
# 设置 minimal defaults
cache._flat_step = 256
cache._flat_keys = None
cache._flat_values = None
cache._flat_keys_scales = None
cache._flat_values_scales = None
cache._flat_offset = 0
cache._true_offset = 0
cache._flat_mode = False
cache._flat_quant = None
cache._flat_pq = None
cache._flat_pq_head_dim = None
cache._flat_prefix_token_count = 0
cache.recent_keys = None
cache.recent_values = None
# import_flat_state 会覆盖以上字段
cache.import_flat_state(state['flat_state'])
```

**方案 B: 从 ModelCacheConfig 获取**

在 `_extract_cache_states()` 中记录每层 cache 的构造参数到 `ModelCacheConfig`，重建时使用。

### 3.4 _extract_cache_states() 修改

**文件**: `src/omlx/scheduler.py` (line 2885)

现有逻辑已经调用 `layer_cache.state`，由于 FlashMLX 侧已添加 `state` property，**不需要修改这个函数**即可获得基本功能。

但如果要启用压缩存储路径（Tier 1），需要：

```python
def _extract_cache_states(raw_cache):
    # ... existing code ...
    for layer_idx, layer_cache in enumerate(raw_cache):
        class_name = type(layer_cache).__name__

        if class_name == 'TripleLayerKVCache' and HAS_CACHE_TYPE_HANDLERS:
            handler = CacheTypeRegistry.get_handler_by_class_name('TripleLayerKVCache')
            state_dict = handler.extract_state(layer_cache)
            extracted.append({
                'state': state_dict,  # 包含 compressed flat_state
                'meta_state': layer_cache.meta_state,
                'class_name': class_name,
                'cache_type': 'TripleLayerKVCache',
            })
        else:
            # ... existing path ...
```

### 3.5 SSD 存储格式扩展 — 压缩 Tensor

**文件**: `src/omlx/cache/paged_ssd_cache.py`

现有 `_write_safetensors_no_mx()` 已支持 int8, uint8 dtype (见 `_MX_TO_ST_DTYPE` 映射)。

需要在存储压缩 flat_state 时，使用以下 tensor 命名约定：

```python
# Standard KVCache tensors (existing):
f"layer_{layer_idx}_keys"     # (B, heads, block_size, dim) bf16
f"layer_{layer_idx}_values"

# TripleLayerKVCache compressed tensors (Tier 1 new):
f"layer_{layer_idx}_flat_keys"          # int8/uint8 (compressed)
f"layer_{layer_idx}_flat_values"        # int8/uint8 (compressed)
f"layer_{layer_idx}_flat_keys_scales"   # float16 (scales)
f"layer_{layer_idx}_flat_values_scales" # float16 (scales)
```

**Metadata 扩展** (safetensors `__metadata__`):

```json
{
  "__metadata__": {
    "block_hash": "...",
    "token_count": "64",
    "layer_cache_types": "[\"TripleLayerKVCache\", ...]",
    "layer_meta_states": "[(\"q8_0\", \"64\", \"64\", \"0\", \"1\", \"0\"), ...]",
    "flashmlx_compressed": "true",
    "flashmlx_flat_quant": "q8_0"
  }
}
```

### 3.6 H0 Block SSD 存储 (Tier 2)

**新增子目录**: `cache_dir/h0/`

```
cache_dir/
├── 0/ block_<kv_hash>.safetensors    # KV blocks (existing)
├── ...
├── f/ block_<kv_hash>.safetensors
├── h0/
│   ├── 0/ block_<h0_hash>.safetensors  # H0 blocks (Tier 2)
│   ├── ...
│   └── f/ block_<h0_hash>.safetensors
└── .index.json
```

**H0 Block Format** (safetensors):

```python
# Tensors:
"h0"     # (B, block_size, d_hidden) — bf16/int8/uint8
"scales" # (B, block_size, 1) — float16 (if quantized)

# Metadata:
{
    "block_hash": "<h0_hash_hex>",
    "parent_kv_hash": "<parent_kv_block_hash_hex>",
    "block_idx": "0",
    "token_start": "0",
    "token_end": "64",
    "quant": "q8",
    "format": "h0_blocks_v1",
}
```

**H0 Hash 计算**:

```python
from mlx_lm.models.kv_direct_cache import H0Store
h0_hash = H0Store.block_hash_key(parent_kv_hash, block_idx)
# SHA-256("h0:" + parent_kv_hash + block_idx_le_bytes)
```

### 3.7 PagedSSDCacheManager 修改

**文件**: `src/omlx/cache/paged_ssd_cache.py`

#### 3.7.1 save_block() 扩展

```python
def save_block(self, block_hash, tensors, metadata, extra_tensors=None):
    """
    extra_tensors: Optional dict of additional tensors to save.
    用于存储 flat_keys_scales, flat_values_scales 等。
    """
    all_tensors = {**tensors}
    if extra_tensors:
        all_tensors.update(extra_tensors)
    # ... existing safetensors write ...
```

#### 3.7.2 新增 H0 Block 方法

```python
def save_h0_block(self, h0_hash, h0_array, scales_array, metadata):
    """Save H0 block to h0/ subdirectory."""
    h0_dir = self.cache_dir / "h0" / h0_hash[:1].hex()
    h0_dir.mkdir(parents=True, exist_ok=True)
    path = h0_dir / f"block_{h0_hash.hex()}.safetensors"

    tensors = {"h0": h0_array}
    if scales_array is not None:
        tensors["scales"] = scales_array

    _write_safetensors_no_mx(path, tensors, metadata)
    # Update H0 index (separate from KV index)
    self._h0_index.add(h0_hash, path, metadata)

def load_h0_block(self, h0_hash):
    """Load H0 block from SSD."""
    meta = self._h0_index.get(h0_hash)
    if meta is None:
        return None
    return _read_safetensors(meta.file_path)

def has_h0_blocks(self, parent_kv_hash, n_blocks):
    """Check if all H0 blocks exist for a KV prefix."""
    for i in range(n_blocks):
        h0_hash = H0Store.block_hash_key(parent_kv_hash, i)
        if not self._h0_index.has(h0_hash):
            return False
    return True
```

### 3.8 淘汰策略修改 (Tier 2 核心)

**当前逻辑**: KV block 被 LRU 淘汰 → 从 SSD 删除

**新逻辑**: KV block 被淘汰时，对应的 H0 blocks **保留**

```python
def evict_block(self, block_hash):
    """Modified eviction: keep H0 blocks."""
    # 删除 KV block
    self._delete_kv_block(block_hash)

    # H0 blocks 不删除 — 它们大小只有 KV 的 ~0.7%
    # H0 blocks 有自己的 LRU 淘汰（容量极大，几乎不需要）
```

**H0 独立淘汰**:

```python
class H0SSDIndex:
    """H0 blocks 的独立索引。

    H0 blocks 很小（KV 的 ~0.7%），默认不淘汰。
    只在 h0/ 总大小超过阈值时淘汰最老的 orphan blocks。
    """
    max_h0_size_bytes: int  # 默认 10GB — 可存 ~500 万 tokens 的 H0
```

---

## 4. 数据流图

### 4.1 Store 流程 (请求完成后)

```
Request completes
    ↓
_extract_cache_states(raw_cache)
    ↓ TripleLayerKVCache detected
    ↓
TripleLayerKVCacheHandler.extract_state()
    ├→ export_flat_state() → compressed arrays
    └→ .state → dequant bf16 (fallback)
    ↓
store_cache(tokens, extracted_cache)
    ↓ split into blocks (block_size=64)
    ↓
┌─────────────────────────────────────────┐
│ For each block:                         │
│                                         │
│ 1. compute_block_hash(chain)            │
│ 2. save_block() to SSD                  │
│    ├── KV tensors: compressed int8/uint8│
│    └── scales: float16                  │
│                                         │
│ 3. H0Store.export_blocks(block_size=64) │
│    → H0Store.block_hash_key()           │
│    → save_h0_block() to h0/ dir         │
└─────────────────────────────────────────┘
```

### 4.2 Fetch 流程 (新请求到达)

```
New request arrives with tokens
    ↓
fetch_cache(tokens)
    ↓ chain-hash prefix matching
    ↓
┌─ CASE A: Full KV hit ──────────────────┐
│ load_block() × N                        │
│ TripleLayerKVCacheHandler.reconstruct() │
│   → import_flat_state() (bit-exact)     │
│ Cost: ~5ms (SSD read, no compute)       │
└─────────────────────────────────────────┘

┌─ CASE B: H0-only hit (cold) ───────────┐
│ KV blocks evicted, H0 blocks remain     │
│ has_h0_blocks(parent_hash, N) → True    │
│ load_h0_block() × N                     │
│ RCEngine.register_from_h0_blocks()      │
│   → process_chunk() × M (3PIR)         │
│ Cost: ~200ms (H0 read + recompute)      │
│ (vs full miss: ~2-10s)                  │
└─────────────────────────────────────────┘

┌─ CASE C: Full miss ────────────────────┐
│ No cached blocks found                  │
│ Full prefill + AM scoring + compression │
│ Cost: ~2-10s                            │
└─────────────────────────────────────────┘
```

### 4.3 Eviction 流程

```
LRU-2 eviction triggered
    ↓
evict_until_size(target)
    ↓
┌─ COLD queue (evict first) ──────────┐
│ For each evicted KV block:           │
│   delete KV safetensors from SSD     │
│   KEEP corresponding H0 blocks ←←←  │ ← Tier 2 核心
│   Mark H0 blocks as "orphan"         │
└──────────────────────────────────────┘
    ↓ if still over target
┌─ HOT queue (evict second) ──────────┐
│ Same as above                        │
└──────────────────────────────────────┘
    ↓ (optional, separate LRU)
┌─ H0 orphan eviction ───────────────┐
│ Only when h0/ exceeds max_h0_size   │
│ Evict oldest orphan H0 blocks       │
│ (rarely needed — H0 is ~0.7% of KV)│
└──────────────────────────────────────┘
```

---

## 5. SSD 空间分析

### 5.1 Per-Token 空间对比 (Qwen3-8B, 36 layers)

| 格式 | 每 token 每层 | 每 token 总量 | 相对 |
|------|--------------|--------------|------|
| bf16 KV (标准) | 8 heads × 128 dim × 2 × 2B = 4,096B | 147,456B | 1.0× |
| Q8 flat (Tier 1) | 8 × 128 × 2 × 1B + scales = 2,080B | 74,880B | **0.51×** |
| Q4 flat (Tier 1) | 8 × 64 × 2 × 1B + scales = 1,056B | 38,016B | **0.26×** |
| H0 only (Tier 2) | — | 4,096 × 2B = 8,192B | **0.056×** |
| H0 Q8 (Tier 2) | — | 4,096 + 1 = 4,097B | **0.028×** |

### 5.2 SSD Cache 容量对比 (100GB SSD quota)

| 模式 | 可缓存 tokens | 相当于 |
|------|--------------|--------|
| bf16 KV | ~680K tokens | ~10 conversations |
| Q8 flat KV | ~1.3M tokens | ~20 conversations |
| H0 Q8 only | ~24M tokens | ~360 conversations |

### 5.3 冷 hit 延迟对比

| 场景 | 延迟 | 说明 |
|------|------|------|
| Full KV hit | ~5ms | SSD read + import_flat_state |
| H0 cold hit (3PIR) | ~200ms | SSD read + reconstruct 4K tokens |
| Full miss | 2-10s | Full prefill + scoring + compress |

---

## 6. 配置参数

### 6.1 FlashMLX 侧 (CacheConfig)

```python
CacheConfig(
    strategy="scored_pq",
    flat_quant="q8_0",
    enable_compressed_ssd=True,      # Tier 1
    enable_h0_ssd=True,              # Tier 2
    h0_ssd_quant="q8",              # Tier 2
    enable_cold_restoration=True,    # Tier 3
)
```

### 6.2 ThunderOMLX 侧 (建议新增)

```python
# settings_v2.py
class CacheSettings(BaseSettings):
    # ... existing ...
    flashmlx_compressed_ssd: bool = True     # 启用压缩 SSD 存储
    flashmlx_h0_ssd: bool = True             # 启用 H0 SSD 存储
    h0_ssd_max_bytes: int = 10 * 1024**3     # H0 SSD quota (10GB)
    cold_restore_enabled: bool = True         # 启用冷恢复
    cold_restore_async: bool = True           # 异步冷恢复 (3PIR)
    cold_restore_chunk_size: int = 512        # RC chunk size
```

---

## 7. 测试策略

### 7.1 单元测试

1. **TripleLayerKVCacheHandler round-trip**:
   - extract_state → slice_state → concatenate_states → reconstruct_cache
   - Verify: compressed=True path preserves dtype (int8/uint8)
   - Verify: compressed=False fallback path works

2. **H0 Block SSD round-trip**:
   - export_blocks → save_h0_block → load_h0_block → import_blocks
   - Verify: data integrity after SSD round-trip
   - Verify: block_hash_key uniqueness (no collision with KV hashes)

3. **Eviction with H0 preservation**:
   - Store KV + H0 blocks → evict KV → verify H0 still exists
   - Load H0 blocks → verify data integrity

### 7.2 集成测试

4. **End-to-end prefix caching**:
   - Request A: prefill → store → verify SSD has compressed blocks
   - Request B (same prefix): fetch → verify prefix hit → verify compressed restore

5. **Cold restoration path**:
   - Store KV + H0 → evict KV → new request with same prefix
   - Verify: H0-only hit detected → 3PIR triggered → KV reconstructed
   - Verify: reconstruction quality (within Q8 tolerance)

6. **Performance benchmarks**:
   - SSD save/load latency with compression
   - Cold restore latency vs full miss
   - Memory usage comparison

---

## 8. 实现时序

```
Phase 0: FlashMLX 侧已完成 ✅
  - TripleLayerKVCache: state, meta_state, export/import_flat_state
  - H0Store: export_blocks, import_blocks, block_hash_key
  - ThunderOMLXAdapter: export/import compressed, export_h0_blocks
  - RCEngine: register_from_h0_blocks
  - ReconstructionController: reconstruct_from_h0_blocks
  - CacheConfig: SSD bridge fields

Phase 1: ThunderOMLX Tier 1 (可独立开发)
  1. CacheType.TRIPLE_LAYER_KVCACHE enum
  2. TripleLayerKVCacheHandler 完整实现
  3. CacheTypeRegistry 注册
  4. _extract_cache_states() 识别 TripleLayerKVCache
  5. save_block() extra_tensors 支持
  6. 验证: prefix cache hit with compressed blocks

Phase 2: ThunderOMLX Tier 2 (依赖 Phase 1)
  7. h0/ 子目录结构
  8. save_h0_block() / load_h0_block()
  9. H0SSDIndex
  10. 淘汰策略: KV eviction 保留 H0
  11. has_h0_blocks() 检测
  12. 验证: H0 blocks 存活于 KV eviction

Phase 3: ThunderOMLX Tier 3 (依赖 Phase 2)
  13. fetch_cache() 增加 H0-only hit 检测
  14. _reconstruct_from_h0() 调用 FlashMLX API
  15. 调度器集成: cold_restore 请求类型
  16. 验证: 冷 hit → 3PIR 重建 → 正确生成
```

---

## 9. 关键依赖

| ThunderOMLX 需要导入 | 来源 |
|---------------------|------|
| `TripleLayerKVCache` | `mlx_lm.models.triple_layer_cache` |
| `H0Store` | `mlx_lm.models.kv_direct_cache` |
| `ThunderOMLXAdapter` | `flashmlx.integration` |
| `RCEngine` | `flashmlx.rc_engine` |
| `FlashMLXConfig`, `CacheConfig` | `flashmlx.config` |

所有导入都应用 try/except 包裹，FlashMLX 不可用时 graceful fallback:

```python
try:
    from mlx_lm.models.triple_layer_cache import TripleLayerKVCache
    HAS_TRIPLE_LAYER = True
except ImportError:
    HAS_TRIPLE_LAYER = False
```

---

## 10. 数据库类比总结

| 数据库概念 | ThunderOMLX | FlashMLX |
|-----------|-------------|----------|
| Buffer Pool | Paged Cache Blocks | TripleLayerKVCache flat buffer |
| 压缩页 | Tier 1: 压缩 safetensors | Q8/Q4 flat arrays |
| WAL | H0Store on SSD | h^(0) = embed_tokens(x) |
| REDO Log Replay | 3PIR cold restoration | reconstruct_prefix_kv |
| 页淘汰 | LRU-2 eviction | KV eviction, H0 保留 |
| 预读 (prefetch) | Smart prefetch | importance-guided coverage |
| 物化视图 | prefix cache hit | cached compressed KV |
