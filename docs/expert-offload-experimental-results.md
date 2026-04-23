# Expert Offloading: Desktop & Mobile Experimental Results

> FlashMLX v2.0 | 2026-04-21 | M4 Pro 48GB
> Model: Qwen3.5-35B-A3B-6bit (20.8B params, 40 layers, 256 experts/layer, top-8 routing)

---

## Executive Summary

We present a dual-track expert offloading system for MoE models on Apple Silicon.
The Desktop track uses a 2-bit quantized shadow (all 256 experts) for lossless prefill
and fast decode fallback. The Mobile track eliminates the shadow entirely, streaming
experts from NVMe on-demand during prefill and using frequency-aware cache eviction
during decode.

**Key results (all measured on M4 Pro 48GB):**

| Config | Description | Memory | TG tok/s | PP tok/s | Quality |
|--------|------------|--------|----------|----------|---------|
| A (standard) | No offloading | 26.2 GB | 44.5 | 0.4 | 85% |
| G (desktop) | Pool-32 + Shadow-2bit | **14.3 GB** | **62.9** | 8.1 | 85% |
| R (mobile) | Streaming + Freq-Aware | **1.85-16.8 GB** | **15.0** | ~12 | 85% |

Desktop achieves **-45% memory and +41% TG speed** simultaneously.
Mobile starts at **1.85 GB** (93% reduction) with a configurable cache cap.

---

## 1. Desktop Track: Pool-32 + Shadow-2bit

### Architecture

```
GPU Memory Layout (14.3 GB total):
  Non-expert weights (Attention/SSM/Embed/LM Head):  1.85 GB
  Pool (32 hot experts, 6-bit original precision):    3.05 GB
  Shadow (all 256 experts, 2-bit re-quantized):       9.38 GB
```

### Execution Model

| Phase | Mechanism | Code Path |
|-------|-----------|-----------|
| PP (seq_len > 1) | Shadow bypass — all tokens route through 2-bit shadow | expert_offload.py:1368 |
| TG (seq_len == 1) | Pool hit (~95%) → 6-bit; miss → shadow 2-bit fallback via `mx.where` | expert_offload.py:1434 |

### Measured Performance

| Metric | Standard (A) | Desktop (G) | Delta |
|--------|-------------|-------------|-------|
| Model memory | 26.2 GB | 14.3 GB | **-45%** |
| PP throughput | 0.4 tok/s | 8.1 tok/s | **+20x** |
| TG throughput | 44.5 tok/s | 62.9 tok/s | **+41%** |
| Peak memory | 26.3 GB | 14.4 GB | **-45%** |
| MATH-500 accuracy | 17/20 = 85% | 17/20 = 85% | No loss |

**Why is TG faster at lower memory?** The standard model occupies 26.2 GB — 41% of the
64 GB physical memory. Memory bandwidth contention limits throughput. At 14.3 GB, the
GPU has more bandwidth headroom, and the smaller pool tensor has better cache locality
for `gather_qmm`.

### Shadow Precision: 6-bit vs 2-bit

| Shadow bits | Shadow size | Total memory | TG tok/s | Note |
|------------|-------------|-------------|----------|------|
| 6-bit (old) | 24.4 GB | 29.3 GB | 44.9 | Redundant — same precision as model |
| **2-bit (fixed)** | **9.4 GB** | **14.3 GB** | **62.9** | **-15 GB, +40% TG** |

The 6-bit shadow was a full-precision duplicate of the model weights — wasteful.
2-bit re-quantization cuts shadow size by 61% while the pool (6-bit) still handles
95%+ of TG tokens at full precision. Shadow is only used for the remaining ~5% misses.

**Precision note:** PP routes through the 2-bit shadow, which means KV cache values
are computed at lower precision than standard. In practice this has not shown measurable
quality degradation on MATH-500, but it is technically lossy. The Mobile track
(streaming 6-bit originals) avoids this tradeoff entirely.

### Batch Support (FlashBatchGenerator)

Shadow works in batch mode without modification. The `FlashBatchGenerator` uses
chunked interleaved scheduling:

```
Each tick:
  Phase 1: Decode ALL active requests (seq_len=1) → pool + shadow TG path
  Phase 2: Prefill 1 chunk of 1 pending request (seq_len=512) → shadow PP bypass
```

Each `__call__` is homogeneous (all-decode or all-prefill), so the seq_len-based
dispatch works correctly. `gather_qmm` and `_switchglu` support arbitrary batch sizes.

**Batch memory model:**

| Concurrent requests | Total memory | Note |
|--------------------|-------------|------|
| 1 | 14.3 GB | Baseline |
| 4 | ~14.7 GB | +0.1 GB/req KV cache (only 10/40 layers have attention) |
| 8 | ~15.1 GB | Excellent scaling due to minimal KV overhead |

Safety mechanisms: memory-gated backpressure (skip prefill if headroom < 2 GB),
adaptive pool resize (shrink/expand based on hit rate), dynamic k-pruning.

---

## 2. Mobile Track: Streaming + Frequency-Aware Eviction

### Architecture

No shadow, no pre-built pool. Three innovations ("three axes"):

### Innovation 1: Zero-Shadow Streaming PP

During prefill, `_streaming_pp=True` forces all tokens through `_discovery_call`,
which loads experts from NVMe per-layer on demand.

| Property | Desktop PP | Mobile PP |
|----------|-----------|-----------|
| Mechanism | Shadow bypass (2-bit GPU tensor) | NVMe streaming (per-layer load) |
| Extra memory | +9.4 GB (shadow) | **+0 GB** |
| Expert precision | 2-bit (re-quantized) | **6-bit (original)** |
| PP throughput | 8.1 tok/s | ~12 tok/s |
| Quality | Lossless (within 2-bit precision) | **Lossless (original precision)** |

### Innovation 2: PP-as-Profiling (Zero-Cost Frequency Collection)

`ExpertTelemetry` collects per-expert activation frequency during PP as a side effect
of `_discovery_call` — no additional computation or synchronization required.

After PP completes:
- Hot experts (high frequency) → retained in GPU discovery cache
- Cold experts (low frequency) → eviction candidates

**Cost: Zero.** Telemetry collection is a byproduct of existing code paths.

### Innovation 3: Frequency-Aware Eviction

**The problem with LRU:** In MoE models, hot experts may be temporarily inactive
(a few tokens route elsewhere), causing LRU to evict them. The next token that needs
them triggers an NVMe reload → cache thrashing.

**Frequency-aware solution:** Evict the expert with the lowest global activation
frequency (from PP telemetry), not the least recently used.

```
PP → TG transition:
  1. Prune discovery cache to top-K by telemetry frequency
  2. Set 2x soft limit (e.g., hotset=32 → limit=64)

During TG when cache is full:
  3. Sort candidates by global frequency, evict coldest (not current request's experts)
```

**Measured impact (same memory budget, limit=64):**

| Eviction Strategy | TG tok/s | Cache Hit Rate | Improvement |
|------------------|----------|---------------|-------------|
| LRU | 7.09 | ~35% | baseline |
| **Frequency-Aware** | **15.0** | **70.8%** | **+111%** |

### Complete Mobile Performance

| Metric | Standard (A) | Mobile (R) | Delta |
|--------|-------------|-----------|-------|
| Startup memory | 26.2 GB | 1.85 GB | **-93%** |
| PP throughput | 0.4 tok/s | ~12 tok/s | **+30x** |
| TG throughput | 44.5 tok/s | 15.0 tok/s | -66% |
| TG steady-state memory | 26.3 GB | 16.8 GB | **-36%** |
| MATH-500 accuracy | 17/20 = 85% | 17/20 = 85% | **No loss** |

### Memory Timeline

```
t=0   Load model (non-expert weights):      1.85 GB
      |
t=PP  Streaming (load on demand, cache):    1.85 → ~4 GB (single-layer peak)
      |
t=TG  Discovery cache grows:               4 → 16.8 GB (limit=64 cap)
      |                                      ↕ freq-aware eviction maintains steady state
t=end Steady state:                          ~16.8 GB
```

### Cache Limit Tuning

| Cache Limit | TG Memory | TG Speed | Target Device |
|------------|-----------|----------|---------------|
| Unlimited | 26.2 GB | 21.3 tok/s | Mac Studio (64 GB) |
| limit=64 | 16.8 GB | 15.0 tok/s | Mac mini (24 GB) |
| limit=32 | ~9 GB | ~10 tok/s (est.) | iPad Pro (16 GB) |
| limit=16 | ~5 GB | ~7 tok/s (est.) | iPhone (8 GB) |

---

## 3. Cross-Config Comparison

| | Desktop Single | Desktop Batch | Mobile Single |
|--|---------------|---------------|--------------|
| **Config** | Pool-32 + Shadow-2bit | + Interleaved scheduling | Zero-Shadow + Freq-Aware |
| **PP mechanism** | Shadow bypass (2-bit) | Shadow bypass (per-chunk) | NVMe streaming (6-bit) |
| **TG mechanism** | Pool + shadow fallback | Same | Discovery cache + freq eviction |
| **Startup memory** | 14.3 GB | 14.3 GB | 1.85 GB |
| **TG steady-state** | 14.3 GB | ~15 GB (8 concurrent) | 16.8 GB (limit=64) |
| **PP speed** | 8.1 tok/s | ~8 tok/s | ~12 tok/s |
| **TG speed** | 62.9 tok/s | ~55-60 tok/s (est.) | 15.0 tok/s |
| **Quality** | 85% (lossless*) | 85% (lossless*) | 85% (lossless) |
| **Concurrency** | 1 | 4-8+ | 1 |
| **Target** | Mac Studio/Pro (32-64 GB) | Server/Workstation | iPad/iPhone (8-16 GB) |

*Desktop PP uses 2-bit shadow — technically lossy vs 6-bit but no measurable quality difference.

---

## 4. Precision Analysis

| Phase | Standard (A) | Desktop (G) | Mobile (R) |
|-------|-------------|-------------|-----------|
| PP expert precision | 6-bit | 2-bit shadow | **6-bit original** |
| TG hit precision | 6-bit | 6-bit pool | 6-bit cache |
| TG miss precision | N/A | 2-bit shadow | 6-bit NVMe load |

**Precision ranking: A = R > G** (Mobile streaming is the highest fidelity option)

The Desktop track trades PP precision (2-bit shadow) for constant-time pool+shadow
execution. The Mobile track maintains full 6-bit precision throughout at the cost
of I/O-bound PP latency.

---

## 5. Technical Innovation Summary

### 5.1 Shadow Precision Scaling (Desktop)

- **Problem:** Original 6-bit shadow = 24.4 GB (redundant full copy of model weights)
- **Solution:** 2-bit re-quantization → 9.4 GB (-61%)
- **Unexpected benefit:** Less memory pressure → more bandwidth → TG 44.9 → 62.9 tok/s (+40%)
- **Tradeoff:** PP computed at 2-bit; no measurable quality impact in practice

### 5.2 PP-as-Profiling + Frequency-Aware Eviction (Mobile)

- **Problem:** Mobile cannot afford shadow memory; LRU cache eviction causes thrashing
- **Solution:** Free frequency profiling during PP → frequency-ranked eviction during TG
- **Result:** +111% TG speed at identical memory budget (7.09 → 15.0 tok/s)
- **Insight:** LRU fails for MoE — hot experts evicted during brief inactivity cause reload storms

### 5.3 Zero-Shadow Streaming PP (Mobile)

- **Problem:** Shadow bypass requires 9+ GB shadow in GPU; impossible on mobile
- **Solution:** PP streams experts from NVMe per-layer; 0 extra resident memory
- **Result:** Startup memory 14.3 GB → 1.85 GB (-87%); PP quality BETTER (6-bit vs 2-bit)
- **Tradeoff:** PP is I/O-bound (~12 tok/s vs 8.1 tok/s shadow — actually faster due to smaller working set)

---

## 6. Reproduction

```bash
cd /Users/lisihao/FlashMLX

# Desktop single request (Config G)
python benchmarks/bench_quality.py --configs G --n 20 --max-tokens 2048

# Mobile streaming (Config R)
python benchmarks/bench_quality.py --configs R --n 20 --max-tokens 2048

# Direct A vs G vs R comparison
python benchmarks/bench_quality.py --configs A,G,R --n 20 --max-tokens 2048
```

---

## 7. Commits

| Hash | Message |
|------|---------|
| `0b13e95` | fix(bench): change shadow configs G/H from 6-bit to 2-bit |
| `9621a95` | chore: update submodule to include frequency-aware eviction |
| `a3f7ec1` | feat(mobile): frequency-aware eviction + PP-as-profiling hotset |
| `27e1f5f` | feat(mobile): cache clearing between inferences in bench_quality |
| `5d73d1e` | feat(mobile): add streaming cache limit and LRU eviction |
| `4758641` | bench(mobile): add config R — mobile streaming benchmark |
| `c0592e5` | bench: PP shadow bypass fix verified — G@2048 = standard = 85% |

---

## Appendix: Why PP Correctness is Non-Negotiable

MoE expert miss during prefill is catastrophic. Even 1% PP miss rate → 0% quality.

**Root cause:** PP writes KV cache. A wrong expert output (from sentinel remap) injects
incorrect values into the KV cache. All subsequent TG tokens attend to this corrupted
state, causing cascading errors.

Experimental proof:
- pool=32, zero_out miss policy → 0/20 MATH-500 (all wrong)
- pool=128, zero_out miss policy → 0/20 (still all wrong, despite 92% hit rate)
- pool=32 + full shadow → 17/20 = 85% (= standard)

**This finding drove both tracks:** Desktop uses shadow bypass for PP; Mobile uses
NVMe streaming for PP. Both guarantee 100% PP expert coverage.
