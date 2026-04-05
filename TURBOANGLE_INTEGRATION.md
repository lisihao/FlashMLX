# TurboAngle Per-Layer Integration - Complete ✅

## Summary

Successfully integrated TurboAngle per-layer quantization into FlashMLX cache factory. Users can now use TurboAngle with preset configurations or custom per-layer quantizers.

## Files Modified

1. **mlx-lm-source/mlx_lm/models/cache_factory.py**
   - Added `layer_quantizers` parameter to `make_optimized_cache()`
   - Implemented preset loading and dict mapping support
   - Modified cache creation loops (hybrid + pure transformer paths)
   - Per-layer quantizers now passed to each `TripleLayerKVCache`

2. **mlx-lm-source/mlx_lm/models/cache.py**
   - Added `kv_layer_quantizers` parameter to `make_prompt_cache()`
   - Added documentation for new parameter
   - Pass-through to `make_optimized_cache()`

3. **test_perlayer_integration.py** (NEW)
   - Integration tests for preset loading, dict quantizers, and inference
   - All tests passed ✅

## Usage

### Method 1: Use TurboAngle Preset (Recommended)

```python
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

model, tokenizer = load("path/to/model")

# Use Mistral-7B preset (E4 boost: layers 0-3, baseline: layers 4-31)
cache = make_prompt_cache(
    model,
    kv_cache="triple_pq",
    kv_layer_quantizers="mistral-7b",
)

# Run inference
logits = model(tokens, cache=cache)
```

**Available Presets** (from paper Table 3):
- `"tinyllama"` - TinyLlama-1.1B (V-dominated, E4)
- `"mistral-7b"` - Mistral-7B-v0.1 (K-dominated, E4)
- `"smollm2"` - SmolLM2-1.7B (K+V, E20)
- `"phi-1.5"` - phi-1.5 (K-selective, skip 8-15)
- `"stablelm-2"` - StableLM-2-1.6B (K+V, E24)
- `"starcoder2"` - StarCoder2-3B (K+V, E16)
- `"olmo-1b"` - OLMo-1B (K-dominated, E4)

### Method 2: Custom Per-Layer Quantizers

```python
from mlx_lm.models.turboangle import TurboAngleQuantizer

# Define custom quantizers
layer_quantizers = {}

# Layers 0-5: Aggressive V boost
for i in range(6):
    layer_quantizers[i] = TurboAngleQuantizer(n_k=128, n_v=256, head_dim=128)

# Layers 6-35: Baseline
for i in range(6, 36):
    layer_quantizers[i] = TurboAngleQuantizer(n_k=128, n_v=64, head_dim=128)

# Create cache
cache = make_prompt_cache(
    model,
    kv_cache="triple_pq",
    kv_layer_quantizers=layer_quantizers,
)
```

### Method 3: Programmatic Preset Loading

```python
from mlx_lm.models.turboangle_config import get_preset, create_layer_quantizers

# Load preset
preset = get_preset("mistral-7b")

# Create quantizers
layer_quantizers = create_layer_quantizers(preset)

# Create cache
cache = make_prompt_cache(
    model,
    kv_cache="triple_pq",
    kv_layer_quantizers=layer_quantizers,
)
```

## Test Results

**Test 1: Preset String Through Cache Factory** ✅
- Successfully loaded Mistral-7B preset for Qwen3-8B (36 layers)
- Layers 0-3 correctly use K256V128 (E4 boost)
- Layers 4-30 correctly use K128V64 (baseline)
- Layer 35 falls back to Q4_0 (preset only defines 32 layers)

**Test 2: Dict Quantizers** ✅
- Successfully created custom quantizer mapping
- Layers 0-5 correctly use K128V256 (V-boost)
- Layers 6-35 correctly use K128V64 (baseline)

**Test 3: Inference with Per-Layer Quantizers** ✅
- Forward pass completed successfully
- Logits shape: (1, 10, 151936)

## Implementation Details

### Cache Factory Resolution Logic

```python
# In make_optimized_cache()

# Step 1: Resolve layer_quantizers parameter
layer_quantizer_map = None
if layer_quantizers is not None:
    if isinstance(layer_quantizers, str):
        # Load TurboAngle preset by name
        preset = get_preset(layer_quantizers)
        layer_quantizer_map = create_layer_quantizers(preset)
    elif isinstance(layer_quantizers, dict):
        # Direct dict mapping
        layer_quantizer_map = layer_quantizers

# Step 2: Create caches with per-layer quantizers
for i in range(num_layers):
    layer_kwargs = cache_kwargs.copy()
    # Use per-layer quantizer if available
    if layer_quantizer_map is not None and i in layer_quantizer_map:
        layer_kwargs["warm_quantizer"] = layer_quantizer_map[i]
    caches.append(TripleLayerKVCache(**layer_kwargs, layer_idx=i))
```

### Fallback Behavior

- If preset doesn't cover all layers, undefined layers fall back to `warm_quantizer` in `cache_kwargs`
- If no preset or dict provided, all layers use uniform `warm_quantizer`
- Compatible with all cache strategies: `triple_pq`, `triple_pq_am`, `triple_tq`, etc.

## Compatibility

**Compatible with:**
- ✅ Pure transformer models (Qwen3-8B, Mistral-7B, etc.)
- ✅ Hybrid SSM+Attention models (Qwen3.5-35B-A3B, etc.)
- ✅ All triple cache strategies (`triple_pq`, `triple_pq_am`, `triple_tq`, etc.)
- ✅ Route 5 (Scored KV-Direct) with h^(0) capture
- ✅ Route 0 (Density Router) with adaptive compression

**Not compatible with:**
- ❌ `kv_cache="standard"` (no quantization)
- ❌ `max_kv_size` mode (RotatingKVCache)

## Benchmarked Quality (from bench_turboangle_simple.py)

| Method | K Similarity | V Similarity | Compression |
|--------|-------------|-------------|-------------|
| TurboAngle Baseline (K128V64) | 0.999542 | 0.999524 | 4.0× |
| TurboAngle E4 (K256V128) | 0.999844 | 0.999842 | 2.0× |
| Q4_0 (baseline) | 0.995245 | 0.995124 | 4.0× |
| PolarQuant 4-bit | 0.999125 | 0.999087 | 4.0× |

**TurboAngle achieves 0.9995-0.9998 cosine similarity (near-lossless)**

## Expected Perplexity Impact (from Paper Table 3)

| Model | Pattern | Expected ΔPPL |
|-------|---------|---------------|
| Mistral-7B | E4 (layers 0-3) | +0.0002 |
| TinyLlama | E4 (layers 0-3) | -0.0022 |
| SmolLM2 | E20 (layers 0-19) | -0.0003 |
| phi-1.5 | Selective (skip 8-15) | 0.0000 |
| StableLM-2 | E24 (layers 0-23) | +0.0012 |
| StarCoder2 | E16 (layers 0-15) | -0.0007 |
| OLMo-1B | E4 K-only (layers 0-3) | +0.0063 |

**Near-lossless: |ΔPPL| < 0.01 for all models**

## Next Steps (Optional)

1. **WikiText-2 Perplexity Testing**
   - Validate expected ΔPPL claims from paper
   - Run `bench_turboangle.py` with full perplexity evaluation

2. **Memory Profiling**
   - Measure actual memory savings vs theoretical compression ratios
   - Compare with scored_pq Q8 flat buffer

3. **Preset Extension**
   - Add more model presets (Qwen3, Llama-3, etc.)
   - Auto-detect model and suggest preset

4. **Auto-Selection**
   - Analyze layer sensitivity at runtime
   - Dynamically assign E4 boost to sensitive layers

## References

- **Paper**: TurboAngle: Optimal KV Cache Compression via Angle Quantization (arXiv:2603.27467)
- **Core Implementation**: `mlx-lm-source/mlx_lm/models/turboangle.py`
- **Presets**: `mlx-lm-source/mlx_lm/models/turboangle_config.py`
- **Tests**: `test_turboangle.py`, `test_turboangle_perlayer.py`, `test_perlayer_integration.py`
- **Benchmarks**: `bench_turboangle_simple.py`, `bench_turboangle.py`
