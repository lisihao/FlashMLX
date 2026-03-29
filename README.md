# FlashMLX

**Three Routes to Eat Every Byte of LLM Inference Memory on Apple Silicon**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![MLX](https://img.shields.io/badge/MLX-0.31+-green.svg)](https://github.com/ml-explore/mlx)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

LLM inference memory has three parts. Most projects optimize one. FlashMLX attacks all three simultaneously:

| Memory Type | What It Is | Before | After | Reduction |
|-------------|-----------|--------|-------|-----------|
| **Parameters** | MoE expert weights on GPU | 18.21 GB | 9.77 GB | **-46%** |
| **PP (Prefill)** | KV Cache + activations during prompt processing | 5,079 MB | 774 MB | **-85%** |
| **TG (Decode)** | KV Cache during token generation | 4,572 MB | 147 MB | **-97%** |

> And it's **faster**: PP +74%, TG +54%, zero quality loss.

## The Three Routes

### Route 1: Parameter Memory — Expert Offloading Compact Pool

MoE models (like Qwen3.5-35B-A3B) have 256 experts per layer but only activate top-8 per token. The other 248 waste GPU memory.

**Two-Phase Compact Pool**:
- **PP Phase**: Keep all 256 experts on GPU (identity path, zero overhead). Record activation counts.
- **Compact**: After PP, select top-K hot experts. Demote cold experts to CPU cache (numpy, UMA fast). Rebuild pool tensor.
- **TG Phase**: Use compact pool with pre-clamp remap table. Zero `.item()`, zero GPU→CPU sync, full lazy evaluation.

```
pool=256 (identity):  90.0 tok/s, 18.21 GB
pool=128 (compact):   92.8 tok/s,  9.77 GB  ← faster (better cache locality)
```

**Key insight**: The killer optimization was eliminating GPU→CPU synchronization. First version used `.item()` for miss detection → 5.6 tok/s (40x GPU→CPU syncs per token). Final version uses speculative execution with pre-clamp remap → **92.8 tok/s** (16x improvement, zero sync).

### Route 2: PP Memory — Chunked Prefill + Streaming Eviction

Standard attention is O(N²). 32K tokens = 5 GB+ peak memory.

**Solution**: Process input in 512-token chunks. When cache exceeds threshold, use AM importance scoring to evict cold tokens while keeping hot ones. Memory becomes O(1) regardless of input length.

```
Standard 32K PP:  213.6 tok/s, 5,079 MB peak
Chunked 32K PP:   369.1 tok/s,   774 MB peak  ← O(1) memory, +73% speed
```

### Route 3: TG Memory — Scored P2 + Q8 Flat Buffer

KV Cache grows linearly with context. 32K = 4.6 GB, and it slows down TG (each token reads all history).

**Scored P2**: Instead of complex pipeline compression (L0→L1→L2 aging), do one-shot AM scoring at PP→TG transition. Hot tokens go to Q8 flat buffer, cold tokens are evicted. PP memory = standard (no double buffering).

```
Standard 32K TG:  16.0 tok/s, 4,572 MB KV
Scored Q8 32K TG: 24.7 tok/s,   147 MB KV  ← +54% speed, -97% memory
```

## Architecture

```
                     FlashMLX v1.0 — Three-Dimensional Memory Optimization
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  ┌─── Route 1: Parameter Memory ────────────────────────────────────┐   │
│  │  MoE Expert Offloading + Compact Pool                            │   │
│  │  Full Pool (256) → PP → Compact to hot-K → TG (remap+clamp)     │   │
│  │  18.21 GB → 9.77 GB (-46%), TG: 92.8 tok/s (zero loss)          │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌─── Route 2: PP Memory ──────────────────────────────────────────┐   │
│  │  Chunked Prefill + Streaming AM Eviction                         │   │
│  │  chunk=512, max_cache=2048, O(chunk×cache) per chunk → O(1)      │   │
│  │  PP peak 5,079 → 774 MB (-85%), PP speed +73% @ 32K              │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌─── Route 3: TG Memory ─────────────────────────────────────────┐   │
│  │  Scored P2 + Q8_0 Flat Buffer Quantization                      │   │
│  │  PP: bf16 (no quant overhead) → Promotion: AM score → Q8 flat   │   │
│  │  KV 4,572 → 147 MB (-97%), TG speed +54% @ 32K                  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌─── Support Systems ─────────────────────────────────────────────┐   │
│  │  Auto-Calibration: first use ~26s, cached <1ms                   │   │
│  │  Pluggable Quantizers: Q4_0, Q8_0, PolarQuant, TurboQuant       │   │
│  │  On-Policy Calibration: stage-wise for deep networks              │   │
│  │  UMA-Aware: Apple Silicon shared memory, numpy→mx.array ~6μs     │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  Platform: Apple M4 Pro 48GB                                             │
└──────────────────────────────────────────────────────────────────────────┘
```

## Performance

### Combined Results (PP + TG, Qwen3-8B, 32K Context)

| Metric | Standard | FlashMLX | Change |
|--------|----------|----------|--------|
| **PP Speed** | 213.6 tok/s | 372.8 tok/s | **+74.5%** |
| **PP Peak Memory** | 5,079 MB | 774 MB | **-84.8%** |
| **TG Speed** | 16.0 tok/s | 24.7 tok/s | **+54.4%** |
| **TG KV Memory** | 4,572 MB | 147 MB | **-96.8%** |
| **TTOF** | 151.7s | 86.9s | **-42.7%** |
| **Quality** | PASS | PASS | lossless |

### Parameter Memory — Qwen3.5-35B-A3B (Q4, 256 experts/layer, MoE)

| Config | TG (steady) | Memory | Saved |
|--------|------------|--------|-------|
| No offload | 90.0 tok/s | 18.21 GB | — |
| Compact pool=192 | 90.9 tok/s | 13.99 GB | **-4.23 GB (-23%)** |
| Compact pool=128 | **92.8 tok/s** | **9.77 GB** | **-8.44 GB (-46%)** |

### All KV Cache Configurations — Qwen3-8B-MLX (Q8)

| Config | Ctx | PP tok/s | TG tok/s | TTOF | KV PP Peak | KV TG | Quality |
|--------|-----|----------|----------|------|------------|-------|---------|
| standard | 16K | 275.2 | 18.9 | 58.4s | 2,785 MB | 2,268 MB | PASS |
| scored_bf16 | 16K | 362.4 | 26.4 | 44.2s | 773 MB | 252 MB | PASS |
| **scored_q8** | **16K** | **361.8** | **24.7** | **44.3s** | **773 MB** | **129 MB** | **PASS** |
| scored_q4 | 16K | 378.9 | 19.4 | 42.3s | 773 MB | 72 MB | PASS |
| standard | 32K | 213.6 | 16.0 | 151.7s | 5,079 MB | 4,572 MB | PASS |
| scored_bf16 | 32K | 369.5 | 26.2 | 87.7s | 774 MB | 288 MB | PASS |
| **scored_q8** | **32K** | **372.8** | **24.7** | **86.9s** | **774 MB** | **147 MB** | **PASS** |
| scored_q4 | 32K | 376.9 | 16.1 | 86.0s | 774 MB | 81 MB | PASS |

## Quick Start

### KV Cache Compression (Dense Transformer)

```python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Qwen3-8B-Instruct")

# One line — auto-calibration on first use (~26s), cached afterwards (<1ms)
result = generate(model, tokenizer, prompt="Your long prompt here...",
                  kv_cache="scored_pq", kv_flat_quant="q8_0")
```

### Expert Offloading (MoE Model)

```python
from mlx_lm import load
from mlx_lm.models.expert_offload import patch_model_for_offload

model, tokenizer = load("qwen3.5-35b-mlx")
ctx = patch_model_for_offload(model, model_path, max_workers=4, cpu_cache_gb=2.0)

# PP with full pool → compact → TG with compact pool
# ... generate tokens ...
ctx.compact(pool_size=128)  # 18.21 → 9.77 GB, TG: 92.8 tok/s
```

### Advanced Options

```python
result = generate(
    model, tokenizer, prompt=prompt,
    kv_cache="scored_pq",        # AM-scored chunked prefill + streaming eviction
    kv_flat_quant="q8_0",        # Flat buffer: None (bf16), "q8_0", "q4_0"
    kv_scored_max_cache=2048,    # Max tokens retained after eviction (default: 2048)
    kv_calibration="/path/to/custom.pkl",  # Optional: custom calibration file
)
```

## 25 Technical Innovations

This project contains **10 design-level innovations** and **15 system-engineering optimizations**. The design decisions solve *what* problem with *what* approach. The engineering optimizations solve "making it actually fast on real GPU hardware."

### Design Decisions (10)

| # | Innovation | Route | What It Solves |
|---|-----------|-------|---------------|
| 1 | Two-Phase Compact Pool | Params | 46% parameter memory savings, zero TG penalty |
| 2 | Speculative Execution (clamp, no sentinel) | Params | Eliminate GPU→CPU sync in MoE layers (5.6→92.8 tok/s) |
| 3 | UMA-Aware CPU Cache | Params | Apple Silicon unified memory for numpy→mx fast transfer |
| 4 | Chunked Prefill + Streaming Eviction | PP | O(N²) → O(1) PP memory |
| 5 | On-Policy Stage-wise Calibration | TG | Deep network error accumulation (18/36→36/36 layers) |
| 6 | Bounded β Optimization | TG | Paper's implicit assumption (β flew to [-171, +221]) |
| 7 | Scored P2 One-Shot Promotion | TG | Avoid Pipeline's PP memory doubling |
| 8 | Q8_0 Flat Buffer Quantization | TG | 50% memory with 6% speed cost (sweet spot) |
| 9 | Pluggable Quantization Strategies | TG | Q4_0 / Q8_0 / PolarQuant / TurboQuant unified interface |
| 10 | Auto-Calibration System | TG | Zero-config for new models |

### System Engineering (15)

| # | Optimization | Route | Remove It And... |
|---|-------------|-------|-----------------|
| S1 | Gather-Sort Cache Locality | Params | gather_qmm cache miss rate spikes, TG slows |
| S2 | Three-Tier Hierarchical Cache (GPU/CPU/SSD) | Params | Every miss hits SSD (240μs vs 6μs) |
| S3 | Telemetry-Driven Expert Prediction | Params | Wrong experts in compact pool, frequent misses |
| S4 | Dynamic Pool Self-Optimization | Params | Long conversations degrade as expert distribution drifts |
| S5 | Background Prefetch Engine | Params | Miss recovery jumps from 6μs to 240μs |
| S6 | Regime Auto-Detection | Params | Users must manually choose streaming/three-tier/full-gpu |
| S7 | Identity Path Detection | Params | PP wastes mx.take remap on every layer |
| S8 | Async SSD→CPU Population | Params | CPU cache fill blocks GPU during compact |
| S9 | Deferred PP Index Collection | Params | 2560 GPU→CPU syncs during PP (12.8ms wasted) |
| S10 | Pool Miss Mini-Pool Fallback | Params | No precise recovery for non-clamp misses |
| S11 | Lazy Prefill Threshold | KV | Short contexts waste precision on unnecessary quantization |
| S12 | Adaptive Compression Ratio | KV | 32K with 3.0x compression = quality collapse |
| S13 | Chunk-Aware Eviction | KV | TurboQuant re-quantize amplifies QJL noise |
| S14 | Vectorized Single-Gather | KV | 30 GPU kernel dispatches vs 1 |
| S15 | RoPE Position Correction | KV | Position encoding breaks after eviction |

## The Graveyard: 9 Abandoned Approaches

Every dead approach taught something the working approach couldn't:

| # | What Died | Why | Lesson |
|---|----------|-----|--------|
| 1 | Pipeline L0→L1→L2 cache | PP memory **doubled** (bf16 + Q4 simultaneously) | Simpler (one-shot) beat complex (three-layer) |
| 2 | AM compress-and-reconstruct | β compensation unstable, error accumulation | AM is great for **scoring**, terrible for **reconstruction** |
| 3 | Unbounded β solver | β flew to [-171, +221] on long sequences | Always bound numerical optimization in production |
| 4 | Off-policy calibration | Layer 18+ calibrated on wrong distribution | On-policy is mandatory for multi-stage pipelines |
| 5 | `.item()` → `mx.minimum` → pre-clamp remap | 5.6 → 28.1 → **92.8 tok/s** (3 generations) | Move work from hot path to cold path |
| 6 | AM on hybrid architecture (Qwen3.5) | SSM layers amplify Attention compression error | Architecture-level incompatibility, not a tuning problem |
| 7 | Q4_0 flat buffer | -39% TG speed for -45% more compression | KV is 6% of bandwidth; nibble unpack cost > bandwidth savings |
| 8 | Discovery phase `.tolist()` | Per-token GPU→CPU sync during PP | Preload everything then trim > discover one by one |
| 9 | Precise miss handling on hot path | `.item()` check on every token for 0.1% case | Don't penalize 99.9% path for 0.1% edge case |

**Four ways approaches die**:
1. *Looks elegant but actually slower* — Pipeline, Q4_0
2. *Paper's unstated constraints* — Unbounded β, off-policy calibration
3. *Architecture-level incompatibility* — AM on hybrid, discovery on large memory
4. *Cold operation on hot path* — `.item()` sentinel, precise miss handling

## How It Works

### Why is Chunked Prefill FASTER than Standard?

Standard attention is O(N²). FlashMLX bounds the cache at 2048 tokens, so attention becomes O(chunk × 2048) = O(1) per chunk. At 32K, standard PP drops to 213 tok/s while FlashMLX maintains 373 tok/s.

### Why is Q8 the Sweet Spot?

KV Cache reads are only ~6% of TG total bandwidth (94% is model parameters). Q4's nibble unpack is compute-bound, but the bandwidth it saves is just 3% (half of 6%). Compute cost >> bandwidth savings = net negative. Q8 needs one multiply instruction to dequant — negligible.

### Why does Compact Pool get FASTER?

pool=128 is 3% faster than pool=256 because smaller pool tensor = more compact memory layout = better GPU L2 cache locality for `gather_qmm`.

### The Speculative Execution Story

```
Version          TG Speed    Problem
.item() check    5.6 tok/s   40× GPU→CPU sync per token (kills MLX lazy eval)
mx.minimum       28.1 tok/s  Extra MLX op per layer
Pre-clamp remap  92.8 tok/s  Zero extra ops, all work done at compact time
```

16x speedup from "move the check from hot path to cold path."

## Configuration Guide

| Config | KV Memory | TG Speed | Use Case |
|--------|-----------|----------|----------|
| `scored_pq` (bf16) | -89% @ 16K | +40% | Maximum speed |
| `scored_pq` + `q8_0` | -94% @ 16K | +31% | **Recommended default** |
| `scored_pq` + `q4_0` | -97% @ 16K | +3% | Maximum compression |

**Recommendation**: `kv_flat_quant="q8_0"` — Q4_0's additional savings (147→81 MB at 32K) aren't worth the 30% TG speed penalty.

## Core Files

```
FlashMLX/mlx-lm-source/mlx_lm/
├── generate.py                      # Entry point — kv_cache, kv_flat_quant params
├── models/
│   ├── cache.py                     # make_prompt_cache() routing
│   ├── cache_factory.py             # Strategy factory + adaptive params
│   ├── triple_layer_cache.py        # Scored P2 + Chunked Prefill + Q8/Q4 quantization
│   ├── am_calibrator.py             # Auto-calibration system
│   ├── quantization_strategies.py   # Pluggable quantizers (Q4_0, Q8_0, PolarQuant)
│   └── expert_offload.py            # Two-Phase Compact Pool + Speculative Execution
```

## Development Journey

This project started as a paper reproduction and evolved through systematic failure:

- **Day 1-2**: AM works single-layer, 36-layer = gibberish. Found rank-deficient beta matrices.
- **Day 3**: Almost gave up. Found two critical bugs (unbounded β, biased query sampling).
- **Day 3-4**: On-Policy calibration — the key insight no paper discusses.
- **Day 5-6**: Triple-Layer Cache architecture born from accumulated failures.
- **Day 6-7**: Chunked Prefill + Streaming Eviction = O(1) PP memory.
- **Day 7+**: Auto-calibration + Q8_0 flat buffer = production-ready.
- **Day 10-12**: Expert Offloading — from `.item()` sentinel (5.6 tok/s) to speculative clamp (92.8 tok/s).

Full technical article with 25 innovations, 15 system optimizations, and 9 graveyard entries: **[From Paper to Production (Chinese)](.solar/ARTICLE-week-in-review.md)**

## Research References

- **Attention Matching**: [Fast KV Cache Compaction](https://arxiv.org/abs/2602.16284) — Core scoring algorithm
- **H2O**: [Heavy-Hitter Oracle](https://arxiv.org/abs/2306.14048) — Token eviction concept
- **StreamingLLM**: [Efficient Streaming LLMs](https://arxiv.org/abs/2309.17453) — Attention sink preservation
- **TurboQuant**: [ICLR 2026](https://arxiv.org/abs/2502.02631) — PolarQuant + QJL residual correction
- **MLX**: https://github.com/ml-explore/mlx — Apple's ML framework
- **MLX-LM**: https://github.com/ml-explore/mlx-lm — LLM inference

## License

MIT License — see [LICENSE](LICENSE)

---

*FlashMLX v1.0 — Built for Apple Silicon, March 2026*
*KV Cache data: Qwen3-8B-MLX (Q8) on Apple M4 Pro 24GB*
*Expert Offloading data: Qwen3.5-35B-A3B (Q4) on Apple M4 Pro 48GB*
*All benchmarks run in isolated subprocesses, serial execution*
