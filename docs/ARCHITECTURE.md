# FlashMLX: Complete System Architecture & Data Flow

> **Last Updated**: 2026-04-02 | **Version**: 4.0 (3PIR — Three-Phase Interleaved Reconstruction)

---

## 1. System Overview

FlashMLX is a **memory-policy runtime** for Apple Silicon LLM inference, implementing 5 optimization routes that work as a coordinated system:

| Route | Name | Target | Core Technique |
|-------|------|--------|----------------|
| **Route 0** | Density Router | Compression control | Discrete compression levels + model card modes |
| **Route 1** | Expert Offloading | Parameter memory | Three-tier GPU/CPU/SSD expert pool |
| **Route 3** | Scored P2 + Flat Buffer | KV cache memory | AM scoring + pluggable quantization |
| **Route 4** | Chunked Prefill | PP peak memory | Streaming eviction + interleaved scheduling |
| **Route 5** | Context Recall (KV-Direct) | Recall after compression | h^(0) checkpointing + reconstruction |
| **3PIR** | Three-Phase Interleaved Reconstruction | Non-blocking recall | PP/TG/RC three-phase scheduling |

Routes are not alternatives — they compose:

```
Route 0 (control plane)
    ├─ Route 3 (KV compression substrate)
    │   ├─ Route 4 (chunked prefill during PP)
    │   └─ Route 5 (h^(0) backup for recall)
    └─ Route 1 (expert offloading, orthogonal)

3PIR (cross-cutting):
    Route 5 reconstruction → non-blocking chunks
    Scheduled as third phase in PP/TG/RC pipeline
```

---

## 2. Repository Structure

### 2.1 FlashMLX SDK (`/src/flashmlx/`)

| File | LOC | Purpose |
|------|-----|---------|
| `config.py` | 362 | `CacheConfig`, `OffloadConfig`, `FlashMLXConfig`, `DensityLevel` enum, `snap_to_nearest()` |
| `model_cards.py` | 267 | `ModelCard`, `ModeConfig` — per-model JSON configs as single source of truth |
| `reconstruction.py` | 605 | `ReconstructionController` API — programmatic h^(0) → K/V reconstruction + async API |
| `rc_engine.py` | 430 | `RCEngine` — chunk-level non-blocking reconstruction engine (3PIR) |
| `capabilities.py` | ~80 | `detect_capabilities()`, `recommend_config()`, `ModelCapabilities` |
| `__init__.py` | 107 | Public API surface (re-exports from mlx-lm + SDK classes) |
| `integration/thunderomlx.py` | 159 | `setup_flashmlx()` entry point + `ThunderOMLXAdapter` |
| `integration/protocol.py` | — | `FlashMLXProvider` protocol definition |
| `profiler/*` | 8 files | Instrumentation & latency profiling toolkit |

### 2.2 mlx-lm Core Engine (`/mlx-lm-source/mlx_lm/models/`)

| File | LOC | Purpose |
|------|-----|---------|
| `cache.py` | 1810 | Base `KVCache`, `RotatingKVCache`; attention mask helpers |
| **`cache_factory.py`** | 497 | Smart factory: auto-selects strategy, detects hybrid models, creates caches |
| **`triple_layer_cache.py`** | 2034 | **Route 3 Core**: Recent(L0) + Warm(L1/Q4) + Cold(L2/AM). Flat buffer. Scored mode. |
| **`kv_direct_cache.py`** | 950 | **Route 5 Core**: `H0Store`, `reconstruct_prefix_kv()`, `reconstruct_prefix_kv_stateful()`, `BatchedH0View`, monkey-patch |
| `quantization_strategies.py` | 1088 | `Q4_0Quantizer`, `PolarQuantizer`, `TurboQuantizer`, base `QuantizationStrategy` |
| `expert_offload.py` | 3084 | **Route 1**: Three-tier expert management (GPU→CPU→SSD) |
| `h0_probe.py` | 119 | Attention importance probe (runs first N layers for scoring) |
| `hybrid_cache.py` | 414 | SSM+Attention hybrid model handling |
| `double_layer_cache.py` | 674 | Older two-layer variant; AM calibration registry |

### 2.3 Model Cards (`/model_cards/`)

| File | Model | Optimal Strategy |
|------|-------|-----------------|
| `qwen3-1.7b-mlx-4bit.json` | Qwen3-1.7B | scored_pq + q8_0 |
| `qwen3-8b-mlx-4bit.json` | Qwen3-8B | scored_pq + q8_0 |
| `qwen3.5-35b-a3b.json` | Qwen3.5-35B | scored_pq + expert offload |

### 2.4 Benchmarks (`/benchmarks/`, ~130 files)

| Category | Key Files | Purpose |
|----------|-----------|---------|
| Main | `bench_card.py` | Universal model card benchmark |
| Route 0 | `bench_density_modes.py`, `bench_recall_d.py` | Compression levels + recall quality |
| Route 1 | `bench_expert_offload.py`, `bench_pool_*.py` | Expert offloading efficiency |
| Route 5 | `bench_h0_persistence.py`, `bench_route5_overhead.py` | h^(0) reconstruction costs |
| 3PIR | `bench_3pir.py` | Chunk correctness, RCEngine, async API, scheduler |
| Throughput | `bench_modes_throughput.py` | PP/TG/memory across all modes |

---

## 3. End-to-End Data Flow

### 3.1 Complete Pipeline

```
User Prompt (text)
    │
    ▼
tokenizer.encode() → token_ids (mx.array)
    │
    ▼
╔══════════════════════════════════════════════════════════════════╗
║  PREFILL PHASE (multi-token)                                    ║
║                                                                  ║
║  embed_tokens(token_ids) → h^(0)  ← [Route 5: append to H0Store]║
║      │                                                           ║
║      ▼                                                           ║
║  Layer 0..N-1 forward (with TripleLayerKVCache)                  ║
║      │                                                           ║
║      ├─ Each layer: Q,K,V = proj(h)                              ║
║      ├─ Cache: K,V → TripleLayerKVCache.update_and_fetch()       ║
║      │   ├─ Recent (L0, bf16, 512 tokens max)                   ║
║      │   ├─ Overflow → Warm (L1, Q4_0) ← [Route 4: streaming]   ║
║      │   └─ Overflow → Cold (L2, AM scored) ← [Route 3]         ║
║      │                                                           ║
║      └─ Attention(Q, K_all, V_all) → h_next                     ║
║                                                                  ║
║  [Route 4: chunk=512 tokens, evict between chunks]               ║
╚══════════════════════════════════════════════════════════════════╝
    │
    ▼
╔══════════════════════════════════════════════════════════════════╗
║  TRANSITION: FIRST TG TOKEN                                     ║
║                                                                  ║
║  Scored Fast Promotion (if scored_mode):                         ║
║      AM scoring on clean bf16 → keep hot tokens only             ║
║      Allocate flat buffer (Q8_0/Q4_0/bf16)                       ║
║      Copy hot tokens → flat buffer (quantize-on-write)           ║
║      Free L0/L1/L2 staging caches                                ║
║                                                                  ║
║  [Optional Route 5: auto_reconstruct → inject h^(0) K/V]        ║
╚══════════════════════════════════════════════════════════════════╝
    │
    ▼
╔══════════════════════════════════════════════════════════════════╗
║  TOKEN GENERATION (each step)                                    ║
║                                                                  ║
║  Flat Fast Path:                                                 ║
║      Write new K,V to flat[offset]  ← O(1) slice assignment     ║
║      Read flat[0:offset+1]          ← dequant on read           ║
║      Attention(Q_new, K_flat, V_flat) → logits                  ║
║      Sample → next token                                         ║
║      offset++                                                    ║
╚══════════════════════════════════════════════════════════════════╝
    │
    ▼ (optional, on demand)
╔══════════════════════════════════════════════════════════════════╗
║  RECONSTRUCTION (cold path, triggered by ReconstructionController)║
║                                                                  ║
║  h0_store.get_range(0, N) → h^(0) tokens                        ║
║      │                                                           ║
║      ▼ Chunked replay (64 tok/chunk, ~30ms/chunk)                ║
║  for chunk in [0:64, 64:128, ...]:                               ║
║      for layer in model.layers:                                  ║
║          h = layer(h_chunk, mask, temp_cache)                    ║
║      mx.eval(h)  ← yield GPU between chunks                     ║
║      │                                                           ║
║      ▼                                                           ║
║  Exact K/V → inject_reconstruction() (with dedup)                ║
╚══════════════════════════════════════════════════════════════════╝
```

### 3.2 Function Call Chain

```
generate_step()                         # mlx_lm/generate.py
  └─ model(tokens, cache=cache_list)    # Model.__call__
      └─ embed_tokens(tokens) → h       # (Route 5: h0_store.append here)
      └─ for layer in self.layers:
          └─ layer(h, mask, cache=cache[i])
              └─ self_attn(h, mask, cache)
                  └─ cache.update_and_fetch(keys, values)
                      ├─ [prefill] _update_slow_path()
                      │   ├─ append to Recent (L0)
                      │   ├─ _manage_aging() → Warm (L1, Q4)
                      │   └─ _manage_aging() → Cold (L2, AM)
                      │
                      └─ [decode] _update_flat_path()
                          ├─ _write_flat(k, v)  ← quantize on write
                          └─ _fetch_flat()      ← dequant on read
```

---

## 4. Route 3: Scored P2 + Flat Buffer (KV Cache Compression)

### 4.1 Triple Layer Cache Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ TripleLayerKVCache (per attention layer)                     │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │ Recent (L0)  │→ │  Warm (L1)   │→ │    Cold (L2)      │  │
│  │ bf16, exact  │  │ Q4_0 packed  │  │ AM compressed     │  │
│  │ 512 tokens   │  │ 2048 tokens  │  │ variable          │  │
│  └──────────────┘  └──────────────┘  └───────────────────┘  │
│                                                              │
│  Transition at first TG token:                               │
│  ┌──────────────────────────────────────────────────────┐    │
│  │ Flat Buffer (replaces L0/L1/L2)                      │    │
│  │ Q8_0 / Q4_0 / TurboQuant / bf16                     │    │
│  │ Pre-allocated, O(1) write, fast dequant on read      │    │
│  └──────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Flat Buffer Quantization Options

| `flat_quant` | Storage | Dequant Cost | Compression | Recommended |
|-------------|---------|-------------|-------------|-------------|
| `None` (bf16) | (B,H,S,D) bf16 | 0 | 0% | Max speed |
| **`q8_0`** | int8 + scales | Low | ~50% | **Default** |
| `q4_0` | uint8 nibble-packed + scales | Moderate | ~75% | Memory-constrained |
| `turboquant` | PolarQuant packed | Moderate | ~75% | Experimental |

### 4.3 AM Scoring (Attention Matching)

During PP→TG transition (scored_mode=True):

1. Compute attention weights from last query: `W = softmax(Q @ K^T / sqrt(d))`
2. Score each token by aggregated attention weight
3. Keep top-K hot tokens (based on compression_ratio)
4. Apply pinned protection (first N tokens never evicted)
5. Quantize hot tokens into flat buffer

### 4.4 Key Instance Attributes (TripleLayerKVCache)

```python
# Configuration
scored_mode: bool           # P2 Scored mode (skip Warm layer)
enable_cold_am: bool        # AM compression on Cold layer
compression_ratio: float    # AM keep ratio (0 = adaptive)
pinned_tokens: int          # System prompt protection
flat_quant: str             # Quantization strategy for flat buffer

# L0 (Recent)
recent_keys, recent_values: mx.array     # (B, H, S, D) bf16

# L1 (Warm) — skipped in scored_mode
warm_keys, warm_values: mx.array         # (B, H, S, D//2) uint8 (Q4_0)

# L2 (Cold) — AM compressed
cold_compressed_keys, cold_compressed_values: mx.array

# Flat Mode (after first TG token)
_flat_mode: bool
_flat_keys, _flat_values: mx.array       # Pre-allocated buffer
_flat_offset: int                         # Write pointer
_flat_keys_scales, _flat_values_scales: mx.array  # Quant metadata

# Route 5 integration
_h0_store: H0Store                       # Shared h^(0) archive
_recon_keys, _recon_values: mx.array     # Injected reconstructed K/V
_flat_prefix_token_count: int            # Dedup threshold

# Controller reference
_reconstruction_controller: ReconstructionController
```

---

## 5. Route 5: Context Recall (KV-Direct)

### 5.1 Core Principle

From paper [KV-Direct (arXiv: 2603.19664)](https://arxiv.org/abs/2603.19664):

> All layers' K/V are derived from h^(0) = embed_tokens(x). Save h^(0), reconstruct any layer's K/V on demand.

| Storage | Bytes/Token (Qwen3-8B) | Precision |
|---------|:---:|:---:|
| Full K/V (36 layers) | 147,456 B | 100% |
| **h^(0) only (bf16)** | **8,192 B** | **100% reconstructible** |
| Compression ratio | **18x** | — |

### 5.2 H0Store

```python
class H0Store:
    _quant: str | None    # None='bf16' | 'q8' | 'q4'
    _h0: mx.array         # (B, T, d_hidden) or quantized equivalent
    _scales: mx.array     # Quantization scales (q8/q4 only)
    _count: int           # Tokens stored

    def append(h0: mx.array)              # Add tokens (called by monkey-patched embed_tokens)
    def get_range(start, end) → mx.array  # Retrieve + dequantize
    def get_evicted(n) → mx.array         # Get oldest N tokens
```

### 5.3 Monkey-Patch Architecture

```python
# In kv_direct_cache.py: apply_h0_capture_only()

original_class = model.model.__class__

class PatchedModel(original_class):
    def __call__(self, inputs, cache=None):
        h = self.embed_tokens(inputs)     # Original computation
        if self._h0_store is not None:
            self._h0_store.append(h)      # Zero-cost capture
        # ... continue forward pass as normal

model.model.__class__ = PatchedModel

# Safety: _KV_DIRECT_PATCHED sentinel prevents double-patch
# Safety: unpatch_model() restores original __class__
# Safety: batch_size > 1 raises (h^(0) is sequence-level)
```

### 5.4 Reconstruction

```python
def reconstruct_prefix_kv(inner_model, h0_store, start, end, chunk_size=64):
    """Replay h^(0) through all layers to get exact K/V."""
    assert start == 0  # Only prefix reconstruction supported (causal constraint)

    kv_list = [None] * n_layers
    for chunk_start in range(0, end, chunk_size):
        h = h0_store.get_range(chunk_start, chunk_end)
        temp_cache = make_fresh_cache()
        for i, layer in enumerate(inner_model.layers):
            h = layer(h, mask, cache=temp_cache[i])
            kv_list[i] = (temp_cache[i].keys, temp_cache[i].values)
        mx.eval(h)  # Force eval, yield GPU

    return kv_list  # [(k0,v0), (k1,v1), ..., (kN,vN)]
```

### 5.5 ReconstructionController SDK

```python
from flashmlx import ReconstructionController

# Factory: auto-discovers h0_store, inner_model, probe from cache
recon = ReconstructionController.from_cache(cache_list, model)

# Query
if recon.available:
    stats = recon.stats         # h0_tokens, h0_bytes, probe_available, ...
    cost = recon.estimate_cost(n_tokens=4096)  # time_ms_est, memory_mb_est

# Reconstruct (thread-safe, non-blocking lock)
result = recon.reconstruct()                                    # full
result = recon.reconstruct(strategy="partial", max_tokens=4096) # partial
result = recon.reconstruct(strategy="targeted", coverage=0.95)  # probe-guided

# Result
if result.success:
    print(f"{result.tokens_reconstructed} tokens, "
          f"{result.layers_injected} layers, {result.time_ms:.0f}ms")

# Cleanup
recon.clear()  # Free reconstructed K/V memory
```

**Key Design Decisions:**

| Decision | Why |
|----------|-----|
| **Null Object pattern** | `from_cache()` returns `NullReconstructionController` when h^(0) unavailable — caller never checks None |
| **Non-blocking lock** | `lock.acquire(blocking=False)` — won't stall scheduler thread during 20s reconstruction |
| **Frozen dataclasses** | `ReconStats`, `ReconCostEstimate` are thread-safe, passable across threads |
| **Doesn't change make_prompt_cache** | `from_cache()` factory discovers controller — zero API breakage |

---

## 6. Three-Phase Interleaved Reconstruction (3PIR)

> **Detailed design document**: [`docs/3PIR-ARCHITECTURE.md`](./3PIR-ARCHITECTURE.md)

### 6.1 Problem

Route 5's `reconstruct_prefix_kv()` is **synchronous blocking** — 8K tokens takes ~20s, freezing the entire scheduler. This is unacceptable for multi-Agent serving (ThunderOMLX).

### 6.2 Core Innovation

Split reconstruction into a **third schedulable phase** that interleaves with PP (Prefill) and TG (Token Generation):

```
Scheduler.step()
    ├── _schedule_waiting()       # PP phase (compute-bound)
    ├── batch_generator.next()    # TG phase (bandwidth-bound)
    ├── rc_scheduler.try_rc_step()  # ★ RC phase (compute-bound, NEW)
    └── _process_batch_responses()
```

**TG/RC complementarity on Apple Silicon UMA**:
- TG is memory-bandwidth-bound (KV reads)
- RC is compute-bound (layer replay)
- Natural complement — RC uses TG's idle compute cycles

### 6.3 Architecture

```
FlashMLX (Library)              ThunderOMLX (Server)
┌─────────────────────┐         ┌─────────────────────┐
│ rc_engine.py        │         │ rc_scheduler.py     │
│  RCEngine           │◀────────│  RCScheduler        │
│  RCSequenceState    │         │  RCBudget           │
│  RCChunkResult      │         │  RCRequest          │
│  process_chunk()    │         │  try_rc_step()      │
│  inject_completed() │         │  round-robin + budget│
└─────────────────────┘         └─────────────────────┘
         │                                │
         ▼                                ▼
┌─────────────────────┐         ┌─────────────────────┐
│ kv_direct_cache.py  │         │ scheduler.py        │
│  reconstruct_       │         │  step() += RC phase │
│  prefix_kv_stateful │         │  enqueue_rc_if_     │
│  BatchedH0View      │         │  needed()           │
└─────────────────────┘         └─────────────────────┘
```

### 6.4 Budget-Based Scheduling

| GPU Load | Max Chunks | Time Budget | Rationale |
|----------|:---:|:---:|:---|
| Idle (0 decode, 0 prefill) | 8 | 32ms | Aggressive — GPU free |
| PP only (0 decode) | 4 | 16ms | Moderate — PP uses compute too |
| Light TG (1-2 decode) | 2 | 6ms | Conservative — TG has priority |
| Heavy TG (3-4 decode) | 1 | 3ms | Minimal — tiny gaps only |
| Saturated (5+ decode) | 0 | 0ms | Paused — no budget |

### 6.5 Verified Results (Qwen3-1.7B, M4 Max)

| Test | Result |
|------|--------|
| **Bit-exact correctness** | PASS (max diff = 0.00000000) — chunked == blocking |
| **RCEngine chunks** | 4 chunks × 28 layers, 964ms total |
| **Async API round-trip** | success=True via start/step/complete |
| **Scheduler simulation** | 4 steps, 1 injection, budget respected |

### 6.6 Key API

```python
# FlashMLX: Chunk-level engine
from flashmlx import RCEngine, RCSequenceState, RCChunkResult

engine = RCEngine(chunk_size=512)
state = engine.register_sequence(seq_id, h0_store, inner_model, cache_list)
while not state.is_complete:
    result = engine.process_chunk(state)  # ~1.3ms per chunk
engine.inject_completed(state)  # Atomic injection

# FlashMLX: Async API on ReconstructionController
state = recon.reconstruct_async_start()
while not state.is_complete:
    result = recon.reconstruct_async_step(state)
final = recon.reconstruct_async_complete(state)

# ThunderOMLX: Scheduler integration
from omlx.rc_scheduler import RCScheduler, RCBudget, RCRequest
sched = RCScheduler(rc_engine=engine)
sched.add_rc_request(RCRequest(request_id="req-1", h0_store=..., ...))
results = sched.try_rc_step(num_decoding=2, num_prefilling=0)
```

---

## 7. Route 1: Expert Offloading

### 7.1 Three-Tier Architecture

```
┌──────────────────────────┐
│ GPU Hot Pool (Tier 0)    │ ← Active experts, zero-latency access
│ pool_size=128-192        │   gather_qmm directly on GPU
├──────────────────────────┤
│ CPU Warm Cache (Tier 1)  │ ← numpy arrays in UMA system RAM
│ ~8-12 GB                 │   mx.array(numpy_arr) = fast on Apple Silicon
├──────────────────────────┤
│ SSD Cold Store (Tier 2)  │ ← safetensors via pread() + OS page cache
│ model files on disk      │   Slowest path, avoided at runtime
└──────────────────────────┘
```

### 7.2 Two-Phase Compact Pool

```
PP Phase (Discovery):
    full pool (all 256 experts loaded)
    → collect activation histogram
    → zero overhead (just counting)

PP→TG Transition:
    compact(pool_size=128)
    → sort by activation count
    → keep top-128 hot experts on GPU
    → evict rest to CPU warm cache
    → remap table for O(1) lookup

TG Phase (Generation):
    use compact pool
    → 99%+ hit rate (hot experts stay hot)
    → miss → promote from CPU (fast on UMA)
```

### 7.3 Key Innovation: Zero-Sync Hot Path

```python
# No .item() or .tolist() in hot path (kills lazy eval)
# Pre-clamped remap: mx.minimum(expert_idx, pool_size-1)
# Speculative execution: assume hit, verify on cold path only
```

### 7.4 Results (Qwen3.5-35B-A3B)

| Config | TG tok/s | Memory | Savings |
|--------|:---:|:---:|:---:|
| No offload | 90.0 | 18.21 GB | — |
| Compact pool=192 | 90.9 | 13.99 GB | -23% |
| **Compact pool=128** | **92.8** | **9.77 GB** | **-46%** |

### 7.5 Expert Affinity Scheduling

Three-tier storage solves *where* experts live; affinity scheduling
solves *which requests* run together.

#### Problem

8 concurrent decode → tokens scatter to 30+ distinct experts →
each expert processes 1 token → gather_qmm does matvec instead of
matmul → GPU parallelism wasted.

#### Solution

FlashMLX captures per-step expert routing (zero-cost shape op) and
exports it via ThunderOMLXBridge. ThunderOMLX's scheduler checks
Jaccard similarity between a candidate request's expert signature and
the running batch's expert union. Low-overlap requests are deferred
(max 3 times, anti-starvation guaranteed).

#### Data Flow

```
FlashMoeSwitchGLU._pool_call()
  → _last_decode_indices = indices[..., 0, :]   # [B,K], <100ns
       ↓
ThunderOMLXBridge.get_batch_routing_by_uid()
  → {uid: frozenset(expert_ids)}                 # ~30μs
       ↓
Scheduler._update_expert_signatures()
  → Request._expert_signature = frozenset(...)
       ↓
Scheduler._schedule_waiting()
  → Jaccard(candidate, batch) < 0.3 → defer (max 3×)
       ↓
Higher per-expert group_size → GPU throughput ↑
```

#### Results (Qwen3.5-35B-A3B / 8 concurrent / M4 Pro 48GB)

| Metric | OFF | ON | Change |
|---|---:|---:|---:|
| Aggregate TPS | 67.3 | **92.9** | **+38%** |
| Wave-2 TPS | 53.9 | **85.1** | **+58%** |
| TTFT | 12,687 ms | **7,016 ms** | **-45%** |

Full design: [`expert-affinity-scheduling.md`](expert-affinity-scheduling.md)

---

## 8. Route 0: Density Router

### 8.1 Discrete Compression Levels

```python
class DensityLevel(Enum):
    keep_80 = (0.80, 1.25)    # Light compression
    keep_50 = (0.50, 2.0)     # Default (balanced)
    keep_33 = (0.33, 3.0)     # Moderate
    keep_20 = (0.20, 5.0)     # Aggressive
    keep_10 = (0.10, 10.0)    # Ultra-aggressive
```

### 8.2 Product Modes (Model Cards)

| Mode | density_scale | Strategy | h^(0) | Use Case |
|------|:---:|:---|:---:|:---|
| `balanced` | 0.0 | scored_pq | No | Daily use, best TG speed |
| `ultra_long` | +1.5 | scored_pq | No | 32K+ contexts, lowest memory |
| `recall_first` | +2.5 | scored_kv_direct | Yes + auto | Detail retrieval, maximum compression + backup |

### 8.3 Mode Selection Flow

```python
card = load_card_or_detect(model, model_path)
cache_kwargs = card.to_cache_kwargs(mode="recall_first")
# → { "kv_cache": "scored_kv_direct", "kv_flat_quant": "q8_0",
#     "density_scale": 2.5, "probe_layers": 3, "auto_reconstruct": True }

cache = make_prompt_cache(model, **cache_kwargs)
```

---

## 9. Cache Factory: Strategy Selection

### 9.1 `make_optimized_cache()` Decision Tree

```python
# cache_factory.py: make_optimized_cache()

if strategy == "standard":
    return [KVCache(...) for each layer]  # No compression

elif strategy == "triple":
    return [TripleLayerKVCache(...) for each layer]  # L0/L1/L2

elif strategy == "triple_am":
    return [TripleLayerKVCache(..., enable_cold_am=True)]  # + AM on L2

elif strategy == "triple_pq":
    return [TripleLayerKVCache(..., warm_quantizer=PolarQuantizer())]

elif strategy == "scored_pq":
    # Route 3: Scored mode (skip Warm, AM on bf16 Recent → flat)
    caches = [TripleLayerKVCache(..., scored_mode=True)]
    return caches

elif strategy in ("scored_kv_direct", "kv_direct"):
    # Route 3 + Route 5: scored compression + h^(0) capture
    caches = [TripleLayerKVCache(..., scored_mode=True)]
    h0_store = H0Store(quant=h0_quant)
    apply_h0_capture_only(model, h0_store)  # Monkey-patch

    # Install probe if requested
    if probe_layers > 0:
        probe = H0Probe(inner_model, n_probe_layers=probe_layers)

    # Create ReconstructionController
    recon_ctrl = ReconstructionController(
        inner_model=inner, cache_list=caches,
        h0_store=h0_store, probe=probe
    )
    for c in caches:
        c._reconstruction_controller = recon_ctrl

    return caches
```

### 9.2 Hybrid Model Detection

```python
# For SSM+Attention models (e.g., Qwen3.5-35B with 30 SSM + 10 Attn layers)
if model has both SSM and Attention layers:
    attn_indices = detect_attention_layers(model)
    for i, layer in enumerate(model.layers):
        if i in attn_indices:
            caches[i] = TripleLayerKVCache(...)  # Compress attention layers
        else:
            caches[i] = KVCache(...)  # Standard for SSM layers
```

---

## 10. Model Cards System

### 10.1 Structure

```json
{
  "model_id": "qwen3-8b-mlx-4bit",
  "architecture": {
    "type": "pure_transformer",
    "num_layers": 36, "attention_layers": 36,
    "head_dim": 128, "num_kv_heads": 8
  },
  "optimal": {
    "strategy": "scored_pq",
    "flat_quant": "q8_0",
    "calibration_file": "calibrations/am_calibration_qwen3-8b_2.0x_onpolicy.pkl"
  },
  "benchmarks": {
    "8k": { "pp_toks": 430.7, "tg_toks": 27.9, "tg_mem_mb": 240 },
    "32k": { "pp_toks": 409.5, "tg_toks": 21.6, "tg_mem_mb": 529 }
  },
  "modes": {
    "balanced": { "density_scale": 0.0, "strategy": "scored_pq" },
    "ultra_long": { "density_scale": 1.5 },
    "recall_first": { "density_scale": 2.5, "strategy": "scored_kv_direct", "probe_layers": 3 }
  }
}
```

### 10.2 Loading Chain

```python
# 1. Try model_cards/*.json matching model_id
card = load_card(model_path)

# 2. Fallback: auto-detect from model architecture
if not card:
    card = load_card_or_detect(model, model_path)

# 3. Convert to cache kwargs
kwargs = card.to_cache_kwargs(mode="balanced")
cache = make_prompt_cache(model, **kwargs)
```

---

## 11. Quantization Strategies

### 11.1 Class Hierarchy

```
QuantizationStrategy (ABC)
├── Q4_0Quantizer        # Symmetric 4-bit, group_size=32, nibble-packed
├── PolarQuantizer       # 2-4 bit, random rotation + Lloyd-Max (data-oblivious)
└── TurboQuantizer       # PolarQuant + damped QJL residual correction
```

### 11.2 Q4_0 Layout (default Warm layer quantization)

```
Input:  (B, H, S, D) bf16
Group:  reshape → (B, H, S, D//32, 32)
Scale:  max(|group|) / 7.0 per group → (B, H, S, D//32) bf16
Quant:  round(x / scale), clip to [-7, 7], offset to [1, 15]
Pack:   pair nibbles → uint8 → (B, H, S, D//2)
Output: packed (B, H, S, D//2) uint8 + scales (B, H, S, D//32) bf16
```

### 11.3 Q8_0 Flat Buffer (default scored_pq flat quantization)

```
Input:  (B, H, S, D) bf16
Scale:  max(|token|) / 127.0 per token → (B, H, S, 1) bf16
Quant:  round(x / scale), clip to [-127, 127]
Store:  int8 (B, H, S, D) + scales (B, H, S, 1) bf16
Dequant: multiply on read (very fast, single mul)
```

---

## 12. Integration Boundaries

### 12.1 Three-Layer Architecture

```
┌──────────────────────────────────────────────────────┐
│ ThunderOMLX (Scheduler / Orchestrator)                │
│                                                       │
│  batched.py: _apply_flashmlx() → FlashMLXConfig      │
│  scheduler.py: _make_flashmlx_cache() → cache_list   │
│  scheduler.py: _recover_from_cache_error()            │
│  settings_v2.py: FlashMLXSettingsV2                   │
│                                                       │
│  ReconstructionController.from_cache(cache, model)     │
│       → recon.reconstruct(strategy="targeted")        │
│  NEW: RCScheduler → try_rc_step() in step() loop      │
│       → non-blocking 3PIR reconstruction              │
└───────────────────────┬──────────────────────────────┘
                        │ imports
                        ▼
┌──────────────────────────────────────────────────────┐
│ FlashMLX SDK (/src/flashmlx/)                         │
│                                                       │
│  config.py       → CacheConfig, FlashMLXConfig       │
│  model_cards.py  → load_card(), to_cache_kwargs()    │
│  reconstruction.py → ReconstructionController        │
│  rc_engine.py    → RCEngine (3PIR chunk engine)      │
│  capabilities.py → recommend_config()                 │
│  integration/    → setup_flashmlx() entry point      │
└───────────────────────┬──────────────────────────────┘
                        │ imports
                        ▼
┌──────────────────────────────────────────────────────┐
│ mlx-lm-source (/mlx_lm/models/)                      │
│                                                       │
│  cache_factory.py    → make_optimized_cache()        │
│  triple_layer_cache.py → TripleLayerKVCache          │
│  kv_direct_cache.py  → H0Store, reconstruct_*       │
│  quantization_strategies.py → Q4/Q8/Polar/Turbo     │
│  expert_offload.py   → ExpertOffloadManager          │
└──────────────────────────────────────────────────────┘
```

### 12.2 ThunderOMLX Integration Points

```python
# In ThunderOMLX scheduler.py (existing):
def _make_flashmlx_cache(self, model, model_path):
    from flashmlx import load_card_or_detect, make_prompt_cache
    card = load_card_or_detect(model, model_path)
    kwargs = card.to_cache_kwargs(mode=self.settings.flashmlx_mode)
    return make_prompt_cache(model, **kwargs)

# NEW: Programmatic reconstruction control
def _handle_quality_issue(self, cache_list, model):
    from flashmlx import ReconstructionController
    recon = ReconstructionController.from_cache(cache_list, model)
    if recon.available:
        cost = recon.estimate_cost()
        if cost.time_ms_est < 5000:  # Budget: max 5 seconds
            result = recon.reconstruct(strategy="targeted", coverage=0.95)
            return result.success
    return False
```

---

## 13. Performance Baselines

### 13.1 Qwen3-8B (M4 Pro 48GB) — 32K Context

| Metric | Standard | scored_pq + Q8 | Change |
|--------|:---:|:---:|:---:|
| PP tok/s | 269.5 | **409.5** | **+52%** |
| TG tok/s | 16.1 | **21.6** | **+34%** |
| TTFT | 121.6s | **80.0s** | **-34%** |
| PP Peak | 4,840 MB | **526 MB** | **-89%** |
| TG Memory | 4,647 MB | **529 MB** | **-89%** |
| Quality | PASS | **PASS** | Lossless |

### 13.2 Route 0 Mode Comparison (8K Context, M4 Pro 48GB)

| Mode | PP tok/s | TG tok/s | TTFT | TG Mem | Recall (6 needles) |
|------|:---:|:---:|:---:|:---:|:---:|
| standard | 395 | 23.9 | — | 1,193 MB | 6/6 |
| scored_pq (balanced) | 417 (+6%) | 24.6 (+3%) | — | 242 MB (-80%) | 6/6 |
| ultra_long (10x) | 409 (+4%) | 26.0 (+9%) | — | 152 MB (-87%) | 4/6 |
| recall_first (10x+h0) | 338 (-14%) | 26.1 (+9%) | — | 217 MB (-82%) | 4/6 |
| recall_first+RECON | — | — | — | — | **6/6** |
| recall_first+TARGETED | — | — | — | — | **6/6** |
| recall_first+AUTO | 172 (-56%) | 17.7 (-26%) | — | 1,411 MB (+18%) | **6/6** |

Key insight: `recall_first+RECON/TARGETED` restores recall to 6/6 while `ultra_long` alone drops to 4/6. This validates the "aggressive compress + backend rescue" thesis.

### 13.3 Qwen3.5-35B-A3B-4bit (M4 Pro 48GB) — 16K Batch=4

**Non-interleaved** (fair TG measurement):

| Metric | Community mlx-lm | FlashMLX v2.0 | Change |
|--------|:---:|:---:|:---:|
| TG tok/s | ~120 | **~120** | **≈0% (maintained)** |
| GPU Peak | 28.05 GB | **21.10 GB** | **-25%** |
| Model Memory | 18.16 GB | **11.37 GB** | **-37%** |
| Quality | 4/4 PASS | **4/4 PASS** | Lossless |

**Interleaved** (production mode):

| Metric | Community mlx-lm | FlashMLX v2.0 | Change |
|--------|:---:|:---:|:---:|
| TTFT | 82.0s | **21.1s** | **-74%** |
| GPU Peak | 28.05 GB | **13.78 GB** | **-51%** |
| Model Memory | 18.16 GB | **11.42 GB** | **-37%** |
| Quality | 4/4 PASS | **4/4 PASS** | Lossless |

> Note: TG throughput varies with thermal state on M4 Pro Mac Mini (~120 tok/s cold, ~45 tok/s sustained).
> Expert offloading maintains TG parity with community mlx-lm while reducing model memory by 37%.

---

## 14. Key Architectural Decisions

| Decision | Choice | Why | Alternative Rejected |
|----------|--------|-----|---------------------|
| Flat buffer at TG transition | One-shot promotion | Avoids PP double-buffer; O(1) TG write | Pipeline L0→L1→L2 (PP memory 2x) |
| Q8_0 default flat quant | Balance speed/memory | Q4 nibble-unpack costs > bandwidth saved | Q4_0 (TG -45%) |
| Scored mode on bf16 | AM scoring quality | Scoring quantized tokens = noisy scores | AM on Q4 (accuracy loss) |
| Monkey-patch for h^(0) | Zero model changes | Works with any model architecture | Modify model code (maintenance hell) |
| Only prefix reconstruction | Causal correctness | RoPE + causal mask requires start=0 | Sparse [50:100] (position corruption) |
| Non-blocking reconstruction lock | Scheduler freedom | 20s reconstruction won't block other agents | Blocking lock (stalls scheduler) |
| Null Object for NullController | API simplicity | ThunderOMLX never checks None | Optional returns (null checks everywhere) |
| 3PIR chunk-level RC | Non-blocking recall | 8K recon from 20s blocking → ~200ms interleaved | Thread-based async (GIL contention) |
| Round-robin + budget scheduler | Fair + safe | TG priority preserved, RC uses idle time only | Fixed allocation (TG latency degradation) |
| Stateful temp_caches across chunks | Bit-exact correctness | Persistent KVCache accumulates across chunks | Re-create each chunk (causal mask corruption) |

---

## 15. Cemetery: Killed Approaches

| Approach | Why Killed | Lesson |
|----------|-----------|--------|
| Pipeline L0→L1→L2 cache | PP memory 2x, no real benefit | Simple models beat elegant models |
| AM compress-and-reconstruct | AM scoring good; AM as codec = lossy | Right tool for right job |
| Unbounded β solver | Long sequences → β diverges to [-171, +221] | Industrial constraints must be explicit |
| Off-policy calibration | Layers 19-36 error blowup | Must calibrate with compressed distribution |
| `.item()` in hot path | GPU→CPU sync kills lazy eval | Move work to cold path |
| AM on hybrid (SSM+Attn) | SSM layers amplify attention compression error | Structure incompatibility ≠ tuning problem |
| Q4_0 flat buffer default | TG -45% — nibble unpack too expensive | Lower bit ≠ better system |
| Sparse [50:100] reconstruction | RoPE position corruption + causal mask break | Physics says no |
| Per-step auto-reconstruct | Reconstruction is cold-path; TG is hot-path | Never mix hot and cold |
| Thread-based async RC | GIL contention on Apple Silicon; MLX has its own dispatch | Match the runtime model |
| Re-create caches per chunk | Causal mask breaks: each chunk sees wrong sequence length | Persistent state across chunks |

---

*This document is the single source of truth for FlashMLX system architecture. It covers the state as of v4.0-3pir. See [3PIR-ARCHITECTURE.md](./3PIR-ARCHITECTURE.md) for detailed 3PIR design.*
