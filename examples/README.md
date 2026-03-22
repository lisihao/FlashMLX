# FlashMLX Examples

Practical examples demonstrating FlashMLX hybrid cache usage.

## Quick Start

```bash
# Install FlashMLX
pip install flashmlx

# Run basic usage example
python examples/basic_usage.py
```

## Available Examples

### 1. Basic Usage (`basic_usage.py`)

**What**: Simplest way to enable hybrid cache

**When to use**: First-time users, quick start

**Demonstrates**:
- Loading Qwen3.5 model
- Detecting layer types automatically
- Configuring hybrid cache
- Generating text
- Checking cache statistics

**Run**:
```bash
python examples/basic_usage.py
```

**Expected output**:
```
[1/5] Loading Qwen3.5-35B model...
✓ Model loaded
[2/5] Detecting layer types...
✓ Detected 10 Attention layers, 30 SSM layers
[3/5] Configuring hybrid cache...
✓ Configuration: Budget: 64MB, Compression: 4x
[4/5] Injecting hybrid cache...
✓ Hybrid cache injected
[5/5] Generating text...
Response: [generated text]

Expected Performance:
  - Memory savings: ~18.8%
  - TTFT overhead: ~17.3%
  - TBT overhead: ~5.0%
```

---

### 2. Custom Configuration (`custom_config.py`)

**What**: Customize cache configuration for different requirements

**When to use**: Advanced users, specific performance requirements

**Demonstrates**:
- Maximum memory savings (5x compression)
- Balanced quality and performance (3x compression)
- Quality-first approach (2x compression)
- Loading pre-tuned templates

**Run**:
```bash
python examples/custom_config.py
```

**Interactive menu**:
```
Select configuration:
1. Maximum Memory Savings (5x compression)
2. Balanced Quality and Performance (3x compression)
3. Quality-First (2x compression)
4. Pre-tuned Template (4x compression)
```

**Key configurations**:

| Configuration | Compression | Memory Savings | Quality | TTFT Overhead |
|--------------|-------------|----------------|---------|---------------|
| Max Savings | 5.0x | 20% | 95 | 20% |
| Balanced | 3.0x | 16.7% | 98.3 | 15% |
| Quality-First | 2.0x | 12.5% | 100 | 12% |
| Pre-tuned | 4.0x | 18.8% | 96.7 | 17.3% |

---

### 3. Monitoring (`monitoring.py`)

**What**: Monitor cache performance and health

**When to use**: Production deployments, performance tuning

**Demonstrates**:
- Taking cache snapshots
- Tracking statistics over time
- Health checks and warnings
- Comparing snapshots
- Detecting potential issues

**Run**:
```bash
python examples/monitoring.py
```

**Sample output**:
```
Cache Snapshot @ 14:32:15
==================================================

SSM Cache:
  Total Size:    30 layers
  Updates:       1500
  Retrievals:    1500
  Hit Rate:      85.3%

  Tiered Distribution:
    Hot:     10 layers
    Warm:    15 layers
    Cold:     5 layers
    Pinned:   2 layers

Attention Cache:
  Total Size:         10 layers
  Avg Compression:    3.85x
  Total Compressions: 500

Health Report
==================================================
✓ All metrics healthy
```

**Health warnings**:
- Low hit rate (<50%)
- High compression (>5x, may affect quality)
- Hot tier bloat (>50% of cache)

---

### 4. Profiling (`profiling.py`)

**What**: Benchmark hybrid cache vs baseline performance

**When to use**: Performance analysis, validation

**Demonstrates**:
- Measuring TTFT (Time to First Token)
- Measuring TBT (Time Between Tokens)
- Comparing baseline vs hybrid cache
- Profiling different compression ratios

**Run**:
```bash
python examples/profiling.py
```

**Sample results**:
```
Baseline vs Hybrid Cache Comparison
==================================================

TTFT:
  Baseline:    2456.00 ms
  Hybrid:      2881.00 ms
  Overhead:    +17.3%
  Status:      ⚠️ Exceeds target (>10%)

TBT:
  Baseline:      17.00 ms
  Hybrid:        17.85 ms
  Overhead:      +5.0%
  Status:        ✓ Within target (≤10%)

Overall Assessment:
  ✓ Acceptable trade-off for long contexts
  Memory savings: ~18.8%
```

---

### 5. Adaptive Configuration (`adaptive_config.py`)

**What**: Automatically adjust configuration based on context length

**When to use**: Variable context lengths, production systems

**Demonstrates**:
- Context length detection
- Automatic configuration selection
- Use case-specific strategies (general/quality/memory/performance)
- Disabling hybrid cache for short contexts

**Run**:
```bash
python examples/adaptive_config.py
```

**Adaptive rules**:

| Context Length | Recommendation | Compression | Reason |
|---------------|----------------|-------------|---------|
| <1000 tokens | ❌ Disable | N/A | TTFT overhead too high |
| 1000-2000 | 2x-3x | Moderate | Balanced approach |
| 2000-4000 | 3x-4x | Recommended | Good trade-off |
| 4000+ | 4x | Optimal | Best ROI |

**Use case strategies**:
- **General**: Balanced configuration (default)
- **Quality**: Conservative compression, prioritize accuracy
- **Memory**: Aggressive compression, maximize savings
- **Performance**: Minimal overhead, moderate compression

---

## Running Examples

### Prerequisites

```bash
# Install FlashMLX
pip install flashmlx

# Verify installation
python -c "import flashmlx; print(flashmlx.__version__)"
```

### Model Requirements

All examples use `Qwen3.5-35B-Instruct-4bit`:
- Size: ~20GB (quantized)
- Memory: ~64GB+ unified memory recommended
- Download: Automatic via `mlx_lm.load()`

### Running All Examples

```bash
# Basic usage
python examples/basic_usage.py

# Custom config (interactive)
python examples/custom_config.py

# Monitoring
python examples/monitoring.py

# Profiling (takes ~5 minutes)
python examples/profiling.py

# Adaptive config (interactive)
python examples/adaptive_config.py
```

---

## Example Output Files

Some examples generate output files:

| Example | Output File | Description |
|---------|------------|-------------|
| `monitoring.py` | `examples/cache_history.json` | Cache statistics history |
| `monitoring.py` | `examples/conversation_history.json` | Multi-turn conversation stats |

---

## Troubleshooting

### Issue: Out of Memory

**Solution**:
```python
# Reduce budget
config = HybridCacheConfig(total_budget_bytes=32 * 1024 * 1024)  # 32MB

# Or reduce compression ratio
config = HybridCacheConfig(compression_ratio=2.0)
```

### Issue: Model Not Found

**Solution**:
```bash
# Download model manually
mlx_lm.convert --hf-path Qwen/Qwen3.5-35B-Instruct --mlx-path mlx-community/Qwen3.5-35B-Instruct-4bit
```

### Issue: Import Error

**Solution**:
```bash
# Reinstall FlashMLX
pip uninstall flashmlx
pip install flashmlx

# Verify
python -c "from flashmlx.cache import HybridCacheConfig"
```

---

## Next Steps

- Read [User Guide](../docs/USER_GUIDE.md) for detailed documentation
- Read [API Reference](../docs/API_REFERENCE.md) for API details
- Read [Architecture](../docs/ARCHITECTURE.md) for system internals
- Check [Parameter Tuning Report](../tuning_results/PARAMETER_TUNING_REPORT.md) for optimization results

---

*Last Updated: 2026-03-21*
*FlashMLX Version: 1.0*
