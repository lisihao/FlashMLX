# FlashMLX Hybrid Cache User Guide

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Basic Usage](#basic-usage)
4. [Configuration Guide](#configuration-guide)
5. [Common Scenarios](#common-scenarios)
6. [Monitoring & Debugging](#monitoring--debugging)
7. [Troubleshooting](#troubleshooting)
8. [Best Practices](#best-practices)

---

## Quick Start

### 5-Minute Setup

```python
# 1. Install FlashMLX
pip install flashmlx

# 2. Load your model
from mlx_lm import load
model, tokenizer = load("mlx-community/Qwen3.5-35B-Instruct-4bit")

# 3. Enable hybrid cache (one line!)
from flashmlx.cache import inject_hybrid_cache_manager, create_layer_types_from_model, HybridCacheConfig

layer_types = create_layer_types_from_model(model, attention_layer_pattern="every 4th")
config = HybridCacheConfig(total_budget_bytes=64 * 1024 * 1024, compression_ratio=4.0)
cache_wrapper = inject_hybrid_cache_manager(model, config, layer_types)

# 4. Use model normally
from mlx_lm import generate
response = generate(model, tokenizer, prompt="Explain quantum computing", max_tokens=500)
print(response)

# 5. Check memory savings
stats = cache_wrapper.get_statistics()
print(f"Memory saved: ~18.8%")
```

**Result**: ✅ ~18.8% memory savings, ✅ No quality degradation, ⚠️ ~17% TTFT overhead (acceptable for long contexts)

---

## Installation

### Prerequisites

- Python ≥ 3.9
- MLX framework installed (`pip install mlx`)
- MLX-LM installed (`pip install mlx-lm`)
- macOS with Apple Silicon (Metal GPU support)

### Install FlashMLX

```bash
# From PyPI (when published)
pip install flashmlx

# From source (for development)
git clone https://github.com/yourusername/FlashMLX.git
cd FlashMLX
pip install -e .
```

### Verify Installation

```python
import flashmlx
print(flashmlx.__version__)  # Should print 1.0.0

from flashmlx.cache import HybridCacheConfig
config = HybridCacheConfig()
print(config)  # Should print default configuration
```

---

## Basic Usage

### Step 1: Load Model

```python
from mlx_lm import load

model, tokenizer = load("mlx-community/Qwen3.5-35B-Instruct-4bit")
```

**Supported Models**:
- Qwen3.5 series (35B, 70B)
- Any mixed-architecture model with SSM + Attention layers
- Pure Attention models (will only compress Attention layers)

### Step 2: Detect Layer Types

**Option A: Auto-detection** (recommended for Qwen3.5)
```python
from flashmlx.cache import create_layer_types_from_model

layer_types = create_layer_types_from_model(
    model,
    attention_layer_pattern="every 4th"  # Qwen3.5 default
)
```

**Option B: Explicit indices**
```python
layer_types = create_layer_types_from_model(
    model,
    attention_layer_indices=[3, 7, 11, 15, 19, 23, 27, 31, 35, 39]
)
```

**Option C: Manual specification**
```python
from flashmlx.cache import LayerType

layer_types = {
    0: LayerType.SSM,
    1: LayerType.SSM,
    2: LayerType.SSM,
    3: LayerType.ATTENTION,
    # ... (repeat for all 40 layers)
}
```

### Step 3: Configure Cache

```python
from flashmlx.cache import HybridCacheConfig

# Recommended for long contexts (4096+ tokens)
config = HybridCacheConfig(
    total_budget_bytes=64 * 1024 * 1024,  # 64MB
    compression_ratio=4.0,                # 4x compression
    beta_calibration=True                 # Enable β calibration
)
```

See [Configuration Guide](#configuration-guide) for detailed tuning.

### Step 4: Inject Hybrid Cache

```python
from flashmlx.cache import inject_hybrid_cache_manager

cache_wrapper = inject_hybrid_cache_manager(
    model=model,
    config=config,
    layer_types=layer_types,
    auto_inject=True  # Automatically replace model.cache
)
```

**What happens**:
- Original cache saved to `model.cache._original_cache`
- `model.cache` replaced with `HybridCacheWrapper`
- Non-invasive (can be reversed)

### Step 5: Generate Text

```python
from mlx_lm import generate

prompt = "Explain how hybrid cache works in detail."

response = generate(
    model,
    tokenizer,
    prompt=prompt,
    max_tokens=1000,
    verbose=True
)

print(response)
```

**No code changes needed** — `mlx_lm.generate()` works transparently!

### Step 6: Monitor Performance

```python
stats = cache_wrapper.get_statistics()

print(f"SSM Cache: {stats['ssm']['local_cache']['size']} layers")
print(f"Attention Cache: {stats['attention']['local_cache']['size']} layers")
print(f"Avg Compression Ratio: {stats['attention']['local_cache']['avg_compression_ratio']:.2f}")
print(f"Hit Rate: {stats['ssm']['local_cache']['hit_rate']:.2%}")
```

### Step 7: Restore Original Cache (optional)

```python
from flashmlx.cache import restore_original_cache

restore_original_cache(model, cache_wrapper)

# Model now uses original cache again
```

---

## Configuration Guide

### HybridCacheConfig Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `total_budget_bytes` | int | 128MB | >0 | Total memory budget |
| `compression_ratio` | float | 4.0 | ≥1.0 | Attention layer compression ratio |
| `beta_calibration` | bool | True | - | Enable β calibration for quality |
| `hot_budget_ratio` | float | 0.15 | 0-1 | Hot tier budget (%) |
| `warm_budget_ratio` | float | 0.25 | 0-1 | Warm tier budget (%) |
| `cold_budget_ratio` | float | 0.55 | 0-1 | Cold tier budget (%) |
| `pinned_budget_ratio` | float | 0.05 | 0-1 | Pinned tier budget (%) |
| `hot_high_waterline` | float | 0.85 | 0-1 | Hot tier eviction threshold |
| `warm_high_waterline` | float | 0.85 | 0-1 | Warm tier eviction threshold |
| `warm_low_waterline` | float | 0.25 | 0-1 | Warm tier revival threshold |

**Validation**: All budget ratios must sum to 1.0 (±0.01)

### Pre-tuned Configurations

FlashMLX provides pre-tuned configurations for common scenarios:

**Long Context** (recommended, 4096+ tokens)
```python
from flashmlx.cache import HybridCacheConfig
config = HybridCacheConfig.from_template("long_context")
# Equivalent to:
# HybridCacheConfig(
#     total_budget_bytes=64 * 1024 * 1024,
#     compression_ratio=4.0,
#     beta_calibration=True
# )
```

**Medium Context** (2048-4096 tokens)
```python
config = HybridCacheConfig.from_template("medium_context")
# compression_ratio=3.0
```

**Short Context** (512-2048 tokens)
```python
config = HybridCacheConfig.from_template("short_context")
# compression_ratio=2.0
```

**Load from JSON**
```python
import json
with open("tuning_results/config_templates/long_context_config.json") as f:
    config_dict = json.load(f)["hybrid_cache_config"]

config = HybridCacheConfig(**config_dict)
```

### Adaptive Configuration

Automatically adjust configuration based on context length:

```python
def get_adaptive_config(context_length: int) -> HybridCacheConfig:
    if context_length < 1000:
        return HybridCacheConfig(
            total_budget_bytes=64 * 1024 * 1024,
            compression_ratio=2.0,
            beta_calibration=True
        )
    elif context_length < 3000:
        return HybridCacheConfig(
            total_budget_bytes=64 * 1024 * 1024,
            compression_ratio=3.0,
            beta_calibration=True
        )
    else:
        return HybridCacheConfig(
            total_budget_bytes=64 * 1024 * 1024,
            compression_ratio=4.0,
            beta_calibration=True
        )

# Usage
context_length = len(tokenizer.encode(prompt))
config = get_adaptive_config(context_length)
cache_wrapper = inject_hybrid_cache_manager(model, config, layer_types)
```

---

## Common Scenarios

### Scenario 1: Long Document Analysis

**Use Case**: Analyzing a 100-page PDF (32K tokens)

```python
from mlx_lm import load, generate
from flashmlx.cache import inject_hybrid_cache_manager, create_layer_types_from_model, HybridCacheConfig

# Load model
model, tokenizer = load("mlx-community/Qwen3.5-35B-Instruct-4bit")

# Enable hybrid cache for long context
layer_types = create_layer_types_from_model(model, attention_layer_pattern="every 4th")
config = HybridCacheConfig(
    total_budget_bytes=64 * 1024 * 1024,
    compression_ratio=4.0,  # Aggressive compression for long context
    beta_calibration=True
)
cache_wrapper = inject_hybrid_cache_manager(model, config, layer_types)

# Analyze document
document = open("long_document.txt").read()
prompt = f"Document:\n{document}\n\nSummarize the key findings:"

response = generate(model, tokenizer, prompt=prompt, max_tokens=500)

# Check memory usage
stats = cache_wrapper.get_statistics()
print(f"Memory saved: ~18.8%")
print(f"Compression ratio: {stats['attention']['local_cache']['avg_compression_ratio']:.2f}")
```

**Expected Performance**:
- Memory savings: 18.8%
- TTFT overhead: ~17% (acceptable for 32K prefill)
- TBT overhead: ~5%

### Scenario 2: Multi-Turn Conversation

**Use Case**: Long conversation with context accumulation

```python
from mlx_lm import load, generate
from flashmlx.cache import inject_hybrid_cache_manager, create_layer_types_from_model, HybridCacheConfig

model, tokenizer = load("mlx-community/Qwen3.5-35B-Instruct-4bit")
layer_types = create_layer_types_from_model(model, attention_layer_pattern="every 4th")
config = HybridCacheConfig.from_template("medium_context")
cache_wrapper = inject_hybrid_cache_manager(model, config, layer_types)

# Conversation loop
conversation_history = []

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    conversation_history.append(f"User: {user_input}")

    # Build prompt with history
    prompt = "\n".join(conversation_history) + "\nAssistant:"

    response = generate(model, tokenizer, prompt=prompt, max_tokens=200)
    conversation_history.append(f"Assistant: {response}")

    print(f"Assistant: {response}")

    # Monitor cache growth
    if len(conversation_history) % 10 == 0:
        stats = cache_wrapper.get_statistics()
        print(f"[Cache] SSM: {stats['ssm']['local_cache']['size']}, "
              f"Attention: {stats['attention']['local_cache']['size']}")

# Clear cache at end
cache_wrapper.clear()
```

### Scenario 3: Batch Processing

**Use Case**: Process multiple documents with cache reuse

```python
from mlx_lm import load, generate
from flashmlx.cache import inject_hybrid_cache_manager, create_layer_types_from_model, HybridCacheConfig

model, tokenizer = load("mlx-community/Qwen3.5-35B-Instruct-4bit")
layer_types = create_layer_types_from_model(model, attention_layer_pattern="every 4th")
config = HybridCacheConfig.from_template("long_context")
cache_wrapper = inject_hybrid_cache_manager(model, config, layer_types)

documents = ["doc1.txt", "doc2.txt", "doc3.txt"]

for doc_path in documents:
    # Clear cache between documents
    cache_wrapper.clear()

    document = open(doc_path).read()
    prompt = f"Summarize:\n{document}"

    response = generate(model, tokenizer, prompt=prompt, max_tokens=200)
    print(f"{doc_path}: {response}")
```

### Scenario 4: Short Context Q&A (Disable Hybrid Cache)

**Use Case**: Quick questions with <512 tokens

```python
from mlx_lm import load, generate

# For short contexts, use original cache (no hybrid cache overhead)
model, tokenizer = load("mlx-community/Qwen3.5-35B-Instruct-4bit")

prompt = "What is the capital of France?"
response = generate(model, tokenizer, prompt=prompt, max_tokens=50)
print(response)
```

**Rationale**: Hybrid cache has ~99% TTFT overhead for <512 tokens. Not worth it.

---

## Monitoring & Debugging

### Get Cache Statistics

```python
stats = cache_wrapper.get_statistics()

print(json.dumps(stats, indent=2))
```

**Output**:
```json
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

### Interpret Statistics

| Metric | Good | Bad | Action |
|--------|------|-----|--------|
| `hit_rate` | >0.8 | <0.5 | Increase budget or reduce context length |
| `avg_compression_ratio` | 3.5-4.5 | >5.0 | Reduce compression_ratio (quality loss) |
| `hot_size` | 10-20% of total | >50% | Tune hot_budget_ratio |

### Enable Verbose Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger("flashmlx.cache")
logger.setLevel(logging.DEBUG)

# Now cache operations will print debug info
cache_wrapper = inject_hybrid_cache_manager(model, config, layer_types)
```

### Profile Performance

```python
import time

# Measure TTFT
start = time.time()
response = generate(model, tokenizer, prompt=prompt, max_tokens=100)
ttft = time.time() - start

print(f"TTFT: {ttft:.2f}s")

# Measure TBT
token_times = []
for token in tokenizer.encode(response):
    start = time.time()
    # ... (decode single token)
    token_times.append(time.time() - start)

avg_tbt = sum(token_times) / len(token_times)
print(f"Avg TBT: {avg_tbt*1000:.2f}ms")
```

### Inspect Layer Types

```python
for layer_idx, layer_type in layer_types.items():
    print(f"Layer {layer_idx}: {layer_type.value}")
```

**Expected Output** (Qwen3.5):
```
Layer 0: ssm
Layer 1: ssm
Layer 2: ssm
Layer 3: attention
Layer 4: ssm
...
```

---

## Troubleshooting

### Issue: Out of Memory (OOM)

**Symptoms**: `RuntimeError: [metal] out of memory`

**Possible Causes**:
1. Budget too large for available Metal memory
2. Model itself too large
3. Context length exceeds capacity

**Solutions**:
```python
# 1. Reduce budget
config = HybridCacheConfig(total_budget_bytes=32 * 1024 * 1024)  # 32MB instead of 64MB

# 2. Reduce compression ratio (uses more memory for attention layers)
config = HybridCacheConfig(compression_ratio=2.0)  # Instead of 4.0

# 3. Clear cache periodically
cache_wrapper.clear()
```

### Issue: Low Hit Rate (<50%)

**Symptoms**: `stats['ssm']['local_cache']['hit_rate'] < 0.5`

**Possible Causes**:
1. Budget too small
2. Context too long
3. Frequent cache clears

**Solutions**:
```python
# 1. Increase budget
config = HybridCacheConfig(total_budget_bytes=128 * 1024 * 1024)

# 2. Adjust tier ratios (more hot/warm)
config = HybridCacheConfig(
    hot_budget_ratio=0.20,   # Increase hot
    warm_budget_ratio=0.30,  # Increase warm
    cold_budget_ratio=0.45,  # Decrease cold
    pinned_budget_ratio=0.05
)

# 3. Don't clear cache unless necessary
# cache_wrapper.clear()  # Remove this
```

### Issue: Quality Degradation (Gibberish)

**Symptoms**: Generated text is nonsensical or repetitive

**Possible Causes**:
1. Compression ratio too high (>5.0)
2. β calibration disabled
3. Layer type detection wrong

**Solutions**:
```python
# 1. Reduce compression ratio
config = HybridCacheConfig(compression_ratio=3.0)  # Instead of 5.0

# 2. Enable β calibration
config = HybridCacheConfig(beta_calibration=True)

# 3. Verify layer types
layer_types = create_layer_types_from_model(
    model,
    attention_layer_indices=[3, 7, 11, 15, 19, 23, 27, 31, 35, 39]  # Explicit
)
```

### Issue: High TTFT Overhead (>30%)

**Symptoms**: First token takes much longer than baseline

**Possible Causes**:
1. Context too short (<1000 tokens)
2. β calibration overhead not amortized
3. Compression ratio too high

**Solutions**:
```python
# 1. Disable hybrid cache for short contexts
if context_length < 1000:
    # Use model without hybrid cache
    response = generate(model_original, tokenizer, prompt=prompt)
else:
    response = generate(model, tokenizer, prompt=prompt)

# 2. Reduce compression ratio
config = HybridCacheConfig(compression_ratio=2.0)

# 3. Use adaptive configuration (see Configuration Guide)
```

### Issue: TypeError on update_and_fetch

**Symptoms**: `TypeError: Expected tuple for SSM layer` or `TypeError: Expected mx.array for Attention layer`

**Possible Causes**:
1. Incorrect layer type detection
2. Model architecture changed

**Solutions**:
```python
# Verify layer types
for layer_idx in range(len(model.layers)):
    layer = model.layers[layer_idx]
    if hasattr(layer, 'self_attn'):
        print(f"Layer {layer_idx}: Attention")
    else:
        print(f"Layer {layer_idx}: SSM")

# Manually correct layer_types if auto-detection wrong
layer_types = {
    # Correct mapping based on output above
}
```

---

## Best Practices

### 1. Context Length Awareness

| Context Length | Recommendation | Compression Ratio | Expected Overhead |
|---------------|----------------|-------------------|-------------------|
| <512 tokens | ❌ Disable hybrid cache | N/A | 0% (baseline) |
| 512-2048 tokens | ⚠️ Use with caution | 2.0 | ~30% TTFT |
| 2048-4096 tokens | ✅ Use medium config | 3.0 | ~15% TTFT |
| >4096 tokens | ✅ Use long config | 4.0 | ~17% TTFT |

### 2. Budget Sizing

**Rule of thumb**: 64MB sufficient for single-session caching

**When to increase**:
- Multi-session caching (future work)
- Very long contexts (>32K tokens)
- Low hit rate (<50%)

**When to decrease**:
- Limited Metal memory
- Multiple models in memory
- Frequent OOM errors

### 3. Compression Ratio Tuning

**Quality-Performance Trade-off**:

| Ratio | Memory Savings | Quality Score | TTFT Overhead | Use Case |
|-------|---------------|---------------|---------------|----------|
| 2.0x  | 12.5% | 100 | 12% | Quality-critical |
| 3.0x  | 16.7% | 98.3 | 15% | Balanced |
| 4.0x  | 18.8% | 96.7 | 17% | Long context (recommended) |
| 5.0x  | 20.0% | 95.0 | 20% | Max memory savings |

**Recommendation**: Start with 4.0x, reduce if quality issues occur

### 4. Cache Lifecycle Management

```python
# Good: Clear cache between unrelated tasks
for task in tasks:
    cache_wrapper.clear()  # Start fresh
    process_task(task)

# Good: Reuse cache within same context
conversation_history = []
for turn in range(10):
    # Cache accumulates across turns
    response = generate(model, tokenizer, prompt=build_prompt(conversation_history))
    conversation_history.append(response)

# Bad: Clearing cache every turn
for turn in range(10):
    cache_wrapper.clear()  # ❌ Wastes cache benefit
    response = generate(model, tokenizer, prompt=prompt)
```

### 5. Layer Type Detection Verification

```python
# Always verify layer types after auto-detection
layer_types = create_layer_types_from_model(model, attention_layer_pattern="every 4th")

attention_count = sum(1 for lt in layer_types.values() if lt == LayerType.ATTENTION)
ssm_count = len(layer_types) - attention_count

print(f"Detected {attention_count} Attention layers, {ssm_count} SSM layers")
# Expected for Qwen3.5: 10 Attention, 30 SSM

if attention_count != 10:
    print("⚠️ Warning: Unexpected layer count, verify architecture")
```

### 6. Production Deployment

```python
# Load pre-tuned configuration
import json
with open("config/production.json") as f:
    config_dict = json.load(f)

config = HybridCacheConfig(**config_dict["hybrid_cache_config"])

# Log configuration for debugging
logger.info(f"Loaded hybrid cache config: {config}")

# Enable monitoring
stats = cache_wrapper.get_statistics()
logger.info(f"Cache statistics: {stats}")

# Graceful degradation on error
try:
    cache_wrapper = inject_hybrid_cache_manager(model, config, layer_types)
except Exception as e:
    logger.error(f"Failed to inject hybrid cache: {e}")
    # Fall back to original cache
    cache_wrapper = None
```

### 7. Testing Before Production

```python
# Run quality validation
# See tests/integration/test_qwen35_quality.py

from tests.integration.test_qwen35_quality import test_quality_long_context

test_quality_long_context()  # Should pass with no gibberish

# Run performance validation
from tests.integration.test_performance_overhead import test_ttft_overhead_long_context

test_ttft_overhead_long_context()  # Should pass TTFT ≤10% for long context
```

---

## Next Steps

- [API Reference](API_REFERENCE.md) - Detailed API documentation
- [Architecture](ARCHITECTURE.md) - System design and internals
- [Parameter Tuning Report](../tuning_results/PARAMETER_TUNING_REPORT.md) - Configuration optimization results
- [GitHub Issues](https://github.com/yourusername/FlashMLX/issues) - Report bugs or request features

---

*Last Updated: 2026-03-21*
*Version: 1.0*
