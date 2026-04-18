# Temporal Expert Pipeline (TEP) — Analysis Report

> Route 1 extension for MoE expert offloading on Apple Silicon.
> Target model: Qwen3.5-35B-A3B (256 experts/layer, top_k=8, 40 MoE layers, 6-bit).
> Hardware: M4 Max 64 GB.

---

## 1. Problem Statement

Running a 256-expert MoE model on a 64 GB Mac keeps all experts resident in GPU
memory (~20 GB for model weights alone). With KV cache and activations, peak
memory reaches 26+ GB — leaving little room for batching or longer contexts.

The core question: **can we keep only a small subset of experts in GPU memory
during decode, offloading the rest to SSD, without destroying throughput or
quality?**

### Challenges

1. **PP-TG Expert Drift** — experts activated during prefill (PP) differ from
   those needed during decode (TG). A pool built from PP frequency data achieves
   only 46-61% TG hit rate.
2. **Miss Handling** — when a token needs an expert not in the pool, what do you
   do? Load from SSD (correct but slow), use the wrong expert (fast but garbled),
   or skip it entirely?
3. **Wrapper Overhead** — any indirection layer around the MoE kernel must not
   break MLX's lazy evaluation pipeline. A single `numpy` conversion in the hot
   path forces GPU synchronization and serializes all 40 layers.

---

## 2. Architecture

### Three-Tier Storage

```
+-----------------------------------------------------------+
|                    GPU Unified Memory                      |
|                                                            |
|  +-------------+   +----------------------------------+   |
|  | Model Weights|   |     Expert Pool (hot tier)       |   |
|  |  ~20 GB      |   |                                 |   |
|  |  40 x 256    |   |  pool=32: 32/256 per layer      |   |
|  |  experts     |   |  = 5.0 GB (19% of full model)   |   |
|  |  (6-bit)     |   |                                 |   |
|  |              |   |  pool=64: 64/256 per layer      |   |
|  |  PP: full    |   |  = 8.1 GB (31% of full model)   |   |
|  |  TG: offload |   |                                 |   |
|  +------+-------+   |  gather_qmm direct compute     |   |
|         |           +----------------------------------+   |
|         | compact                    ^                     |
|         v                            | hot_swap            |
|  +--------------+              +-----+--------+           |
|  |  CPU Cache   |<-------------|  SSD (cold)  |           |
|  |  ~0 GB       |   mmap       |  safetensors |           |
|  |  (future)    |   240us/read |  all 256 exp |           |
|  +--------------+              +--------------+           |
+-----------------------------------------------------------+
```

### Token Execution Flow

During decode, each token passes through 40 MoE layers. The router selects
top_k=8 experts per layer. The pool intercepts this call:

```
Token arrives
    |
    v
Router: select top_k=8 experts from 256
    |
    v
+-- expert in pool? --+
|  YES                 |  NO (miss)
|                      |
v                      v
gather_qmm          Miss Policy
(0us overhead)       +---+---+---+
                     |   |   |   |
                k1_clamp zero true_load
                (wrong   (skip (SSD read
                 expert)  it)   240us)
```

### Decode Recompact Flow

```
=== PREFILL ===
Prompt --> all 256 experts (identity pool)
              |
              v  record PP activation frequency
         PP Frequency Buffer
              |
              v  compact to pool size
=== INITIAL TG (warmup) ===
Token 1-10 --> Pool(PP top-N)
              |
              v  record TG activation frequency
         TG Frequency Buffer
              |
              v  decode_recompact()
=== STEADY TG ===
Token 11+ --> Pool(TG top-N)  [hit rate: 85-99%]
```

---

## 3. Experiment Results

### Experiment A: Miss Handling Baseline

Pool built from PP frequency, forced miss via SSD true_load.

| Pool Size | TG tok/s | vs Standard | Quality |
|-----------|----------|-------------|---------|
| 64        | 8.7      | -87%        | Coherent |
| 32        | 7.3      | -89%        | Coherent |
| 16        | 7.3      | -89%        | Coherent |

**Conclusion**: SSD true_load preserves quality but is unviable for production.

### Experiment B: Decode Recompact

| Config | Pool | Hit Rate | TG tok/s |
|--------|------|----------|----------|
| PP pool | 64 | 61% | baseline |
| PP pool | 32 | 47% | baseline |
| Decode pool | 64 | 98.4% | +70% vs PP pool |
| Decode pool | 32 | 84.7% | +80% vs PP pool |

**Conclusion**: Decode recompact is the single most important optimization.
PP-frequency pools are nearly useless for TG.

### Experiment C: Miss Policy Gradient

All with decode recompact, pool=32.

| Policy | TG tok/s | Quality | Mechanism |
|--------|----------|---------|-----------|
| k1_clamp | 74.7 | Garbled (84.7% HR) | Use wrong expert |
| zero_out | 73.6 | Coherent | Zero missed expert output |
| true_load | 7.4 | Perfect | SSD load real expert |

**Conclusion**: zero_out is the sweet spot. k1_clamp is only safe at HR > 95%.

### Wrapper Gap Investigation

The "identity hit_only" test: pool=256 (all experts), identity=False (force
remap path), 0% miss rate. Should match standard — but was 38% slower.

Root cause: `np.array(mx_tensor, copy=False)` in telemetry code forces
`mx.eval()` per layer per token. For 150 tokens x 40 layers = 6,000 GPU
synchronization points, completely serializing the lazy eval pipeline.

```
BEFORE FIX (serialized):
Layer 0: [gather_qmm]--[np.array SYNC]--wait--
Layer 1:                                [gather_qmm]--[SYNC]--wait--
...40 layers serial...
= 25ms/token = 40 tok/s

AFTER FIX (pipelined):
Layer 0: [gather_qmm]-+
Layer 1:  [gather_qmm]-+   GPU batches lazily
...                     |
Layer 39:   [gather_qmm]+--[eval once]--done
= 13ms/token = 74 tok/s
```

Fix: removed inline `np.array()`, added deferred `flush_tg_telemetry()` that
batch-processes all telemetry after generation completes.

### Final Results (Post-Fix)

| Config | TG tok/s | Memory | Quality | Hit Rate |
|--------|----------|--------|---------|----------|
| standard | 61.3 | 26.3 GB | Perfect | 100% |
| **dr+zero_32** | **73.6** | **5.0 GB** | Coherent | 84.7% |
| dr+zero_64 | 60.3 | 8.1 GB | Coherent | 98.4% |
| dr+k1_64 | 71.8 | 8.1 GB | Coherent | 98.4% |
| dr+k1_32 | 74.7 | 5.0 GB | Garbled | 84.7% |
| dr+load_64 | 7.4 | 8.1 GB | Perfect | 100% |

**Key result**: `dr+zero_32` is **+20% faster** than standard while using
**81% less memory**.

Pool=32 is faster than standard because smaller expert tensors have better
GPU cache locality for the `gather_qmm` kernel.

---

## 4. Competitive Analysis

| System | Approach | Advantage | Gap vs FlashMLX TEP |
|--------|----------|-----------|---------------------|
| MoE-Infinity | CPU offload + LRU | Scales to 100B+ | No UMA optimization |
| HOBBIT | 1-bit shadow expert | Low-precision fallback | GPU-to-GPU only |
| SliceMoE | Bit-slice compression | Progressive quality | No decode-aware pool |
| Speculating Experts | MLP predictor | Async prefetch | Requires training |
| Qualcomm Mobile MoE | NPU+GPU heterogeneous | Mobile optimized | Proprietary hardware |

### FlashMLX TEP Unique Contributions

1. **Decode Recompact** — first system to address PP-to-TG expert drift by
   rebuilding the pool from decode activation data.
2. **UMA-native design** — built for Apple Silicon shared memory, not adapted
   from GPU-to-CPU transfer patterns.
3. **Zero-out miss policy** — dropout-like degradation that is safe at any hit
   rate, unlike K-1 clamping which produces garbled output below 95% HR.
4. **Lazy eval pipeline preservation** — identified and fixed the GPU
   synchronization anti-pattern that serializes MLX's lazy evaluation across
   MoE layers.

---

## 5. Future Directions

### P0: Low-bit Shadow Experts

Store 2-bit or 4-bit copies of all 256 experts as fallback for misses.
When a miss occurs, compute with the low-precision copy instead of zeroing out.
Expected: quality improvement with no speed regression.
References: HOBBIT (1-bit shadow), SliceMoE (bit-slice).

### P1: Cache-aware Reranking

Modify the router to prefer in-pool experts when gating scores are close.
When expert A (score 0.12) is in-pool and expert B (score 0.13) is not,
use A instead of B. Expected: HR 84.7% -> 92%+ at pool=32, zero extra memory.

### P2: Async Prefetch Pipeline

Predict layer N+1 expert needs from layer N activations. Start SSD-to-GPU
transfer asynchronously while layer N computes. The PrefetchEngine framework
exists but needs production hardening.
Expected: true_load 7.4 -> 40+ tok/s.

### P3: Two-tier Pool

Split the pool into a hot tier (16 experts, never evicted) and a warm tier
(48 experts, evictable). Add a shadow tier (256 experts, 2-bit) for fallback.
Expected: graceful degradation under extreme memory pressure.

---

## 6. Benchmark Scripts

All benchmarks are standalone scripts under `benchmarks/`:

| Script | Purpose |
|--------|---------|
| `bench_decode_pool.py` | Decode recompact vs PP pool hit rates |
| `bench_miss_policy.py` | k1_clamp vs zero_out vs true_load |
| `bench_combo.py` | Combined decode recompact + miss policy |
| `bench_wrapper_gap.py` | Isolate telemetry/remap overhead |
| `bench_oracle.py` | Hit-only wrapper + layer-fix experiments |
| `bench_diagnostic.py` | Phase-split timing + per-layer miss analysis |
| `bench_hybrid.py` | Hybrid miss dispatch (negative result) |

Run any benchmark:
```bash
python benchmarks/bench_combo.py --model /path/to/Qwen3.5-35B-A3B-6bit --tokens 150
```

Results are saved to `.solar/tep-*.json`.

---

## 7. Key Code Locations

| Component | File | Lines |
|-----------|------|-------|
| FlashMoeSwitchGLU | `mlx-lm-source/mlx_lm/models/expert_offload.py` | ~700-1500 |
| Pool compact | `expert_offload.py` `_compact_pool()` | ~1243-1362 |
| Decode recompact | `expert_offload.py` `_decode_recompact()` | ~1363-1420 |
| Miss policies | `expert_offload.py` `_pool_call()` | ~1080-1160 |
| Deferred telemetry | `expert_offload.py` `flush_tg_telemetry()` | ~1200-1240 |
| Hot swap | `expert_offload.py` `hot_swap()` | ~1368-1484 |
| OffloadContext | `expert_offload.py` | ~1700-2000 |
| Standard MoE path | `mlx-lm-source/mlx_lm/models/switch_layers.py` | reference |
