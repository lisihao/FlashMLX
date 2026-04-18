# Expert Affinity Scheduling — MoE Batch Composition Optimization

> **TL;DR**: When 8 users chat concurrently with a 256-expert MoE model, FCFS
> scheduling scatters tokens so that each expert processes only 1 token per step.
> `gather_qmm` does matvec instead of matmul and GPU parallelism is wasted.
> Expert Affinity Scheduling groups requests with overlapping expert routing into
> the same batch, raising per-expert `group_size` from 1 to 4–6 and delivering
> **+38% aggregate TPS, +58% wave-2 TPS, -45% TTFT** with zero model or kernel
> changes.

---

## 1. Problem Statement

Qwen3.5-35B-A3B is a triple-hybrid architecture: 30 SSM layers + 10 attention
layers + 40 MoE layers, each with 256 experts and `top_k = 8`.

When 8 concurrent decode requests run through FCFS scheduling:

```
8 tokens × 8 experts/token = 64 expert activations per step
spread across 30+ distinct experts
→ per-expert group_size ≈ 64 / 30 ≈ 2, often 1
→ gather_qmm does matvec (not matmul)
→ GPU ALU utilization < 15%
```

Worse, **batch scaling is flat**: going from batch=1 to batch=8 yields roughly
the same per-token throughput. The root cause is not memory bandwidth — it is
that FCFS scheduling is blind to expert routing patterns.

### The Insight

Different users tend to activate overlapping subsets of experts (especially
within the same domain or system prompt). If we group requests that share
experts, each expert receives 4–6 tokens per step instead of 1, turning
scattered matvec into useful matmul.

---

## 2. Architecture Overview

End-to-end data flow from expert activation to scheduling decision:

```
FlashMLX (inference engine)                    ThunderOMLX (serving scheduler)
─────────────────────────────                  ──────────────────────────────

FlashMoeSwitchGLU._pool_call()
  │
  │ indices shape: [B, seq, top_k]
  │ decode guard: seq_len == 1 AND ndim >= 3
  ▼
_last_decode_indices = indices[..., 0, :]      ← zero-cost shape op, <100ns
  │   shape: [B, top_k]
  │
  ▼
ThunderOMLXBridge.get_batch_routing()
  │ sample 4 evenly-spaced MoE layers
  │ union expert IDs per batch position
  │ overhead: ~30μs
  ▼
ThunderOMLXBridge.get_batch_routing_by_uid()
  │ map batch positions → request UIDs
  │ return: {uid: frozenset(expert_ids)}
  ▼
                                               Scheduler._update_expert_signatures()
                                                 │ req._expert_signature = frozenset(...)
                                                 ▼
                                               Scheduler._schedule_waiting()
                                                 │ for each candidate in waiting queue:
                                                 │   Jaccard(candidate, batch) < 0.3?
                                                 │     YES → defer (max 3×, anti-starvation)
                                                 │     NO  → admit to batch
                                                 ▼
                                               Higher per-expert group_size
                                               → GPU throughput ↑
```

Five files, ~100 lines of core logic. No model weights changed. No custom
kernels added. Pure scheduling optimization.

---

## 3. Three-Tier Expert Storage (Foundation)

Expert Affinity Scheduling builds on top of the three-tier expert storage
system that solves *where* experts live. This section documents the foundation.

### 3.1 Storage Tiers

```
┌───────────────────────────────────────────────────────────┐
│                    Tier 0: GPU Hot Pool                     │
│  mx.take() lookup  •  <1μs  •  pool_size configurable     │
│  In-memory mx.array with remap table                       │
│  Identity optimization: skip mx.take when pool == full     │
└────────────────────────────┬──────────────────────────────┘
                             │ miss
                             ▼
┌───────────────────────────────────────────────────────────┐
│                  Tier 1: CPU Warm Cache                     │
│  numpy arrays on UMA  •  ~6μs promotion  •  273 GB/s      │
│  Demoted experts from pool compaction stay here             │
└────────────────────────────┬──────────────────────────────┘
                             │ miss
                             ▼
┌───────────────────────────────────────────────────────────┐
│                   Tier 2: SSD Cold Store                    │
│  pread() + OS page cache  •  ~240μs  •  zero SSD writes   │
│  ThreadPoolExecutor for parallel loading                   │
└───────────────────────────────────────────────────────────┘
```

**Key constraint**: pread-only access to SSD. Zero writes during inference.
OS page cache provides transparent caching for recently-loaded experts.

### 3.2 Three Phases: Discovery → Compact → Generation

```
Phase 1: Discovery (Prefill)
  Full pool prebuilt (all frequently-used experts resident)
  PP indices buffered — no GPU sync in hot path
  ExpertTelemetry records every activation

Phase 2: Compact (PP → TG transition)
  Triggered when seq_len drops from >1 to 1
  np.bincount() on aggregated PP indices (single GPU→CPU sync)
  Top-K experts kept; non-hot demoted to CPU warm cache
  Coverage gate: auto-expand if 95% coverage needs more slots

Phase 3: Generation (Decode)
  Compact pool serves 99%+ of expert lookups
  Misses promote from CPU (fast on UMA) or SSD (240μs)
  Pool maintenance: every 8 tokens, in-place slot swap
  Dynamic re-compaction uses TG telemetry for long sessions
```

### 3.3 ExpertTelemetry

Activation tracking with frequency + recency scoring:

```python
class ExpertTelemetry:
    _freq: np.ndarray       # [num_layers, num_experts] lifetime counts
    _recency: np.ndarray    # [num_layers, num_experts] last token position
    _window_freq: np.ndarray  # rolling window for trend detection

    def predict_hot_experts(self, layer_idx, top_k=16):
        # Score = 0.6 * freq_norm + 0.4 * recency_norm
        # Returns top-K expert IDs sorted by hotness

    def get_cold_experts(self, layer_idx, pool_expert_ids, keep_min=8):
        # Never evicts below keep_min experts
        # Returns expert IDs sorted by coldness (coldest first)
```

**Drift detection**: When the rolling window diverges significantly from
lifetime frequency, the telemetry signals that expert usage patterns have
shifted — triggering pool re-compaction.

### 3.4 Zero-Sync Hot Path

The critical invariant for decode throughput:

```python
# In the hot path (_pool_call), we NEVER:
#   - Call .item() or .tolist() (kills MLX lazy evaluation)
#   - Trigger GPU→CPU synchronization
#   - Allocate new mx.arrays beyond the pre-allocated pool

# We DO:
#   - Use pre-clamped remap: mx.minimum(expert_idx, pool_size-1)
#   - Speculative execution: assume pool hit, verify on cold path only
#   - In-place slot swap for pool maintenance (every 8 tokens)
```

### 3.5 Pool Maintenance

Every 8 decode tokens, the pool swaps cold experts for hot candidates:

```
Maintenance step (every 8 tokens):
  1. ExpertTelemetry.get_cold_experts() → cold list
  2. ExpertTelemetry.predict_hot_experts() → hot candidates not in pool
  3. For each (cold, hot) pair:
     a. Demote cold expert to CPU warm cache
     b. Promote hot expert from CPU/SSD into freed slot
     c. Update remap table in-place
  4. No GPU sync — remap update takes effect on next decode step
```

---

## 4. Expert Routing Capture (Zero-Cost)

The first piece of the affinity pipeline: capturing which experts each token
activates during decode.

### 4.1 Implementation

```python
# expert_offload.py, FlashMoeSwitchGLU._pool_call(), lines 1056-1060
#
# indices contract: [batch, seq, top_k] (see __call__ docstring).
# Use explicit slice instead of squeeze to fail loud on shape violations.

if seq_len == 1 and indices.ndim >= 3:
    self._last_decode_indices = indices[..., 0, :]  # [B,1,K] → [B,K]
```

This same capture is also present in the fallback path (line 1095-1096) for
pool-miss scenarios.

### 4.2 Design Decisions

| Decision | Choice | Why |
|---|---|---|
| Capture only during decode | `seq_len == 1` guard | Prefill routing is noisy (many tokens); decode is what matters for batch composition |
| Explicit `[..., 0, :]` over `squeeze` | Shape contract clarity | `squeeze` silently succeeds on wrong shapes; explicit slice fails loud |
| `ndim >= 3` guard | Defensive | Skips capture on malformed indices rather than crashing |
| Store on `self` (instance attr) | Zero allocation | No dict lookup, no heap allocation — just pointer assignment |

### 4.3 Performance

- **Hot path overhead**: < 100ns (shape manipulation, zero-copy)
- **Memory overhead**: One `[B, K]` tensor reference per MoE layer (negligible)
- **GPU sync**: None — this is a lazy graph operation

---

## 5. ThunderOMLXBridge: Routing Export

The bridge translates per-layer routing into per-request expert signatures
that the scheduler can compare.

### 5.1 Layer Sampling Strategy

```python
# expert_offload.py, ThunderOMLXBridge, line 1912
_AFFINITY_SAMPLE_LAYERS = 4

# 4 layers × 8 experts/token → 32 expert IDs per position
# — enough for Jaccard similarity while keeping overhead ~30μs.
```

Layers are sampled at even spacing across all MoE layers:

```python
layer_indices = sorted(self._flash_layers.keys())  # e.g. [0, 2, 4, ..., 78]
n = len(layer_indices)
step = max(1, n // _AFFINITY_SAMPLE_LAYERS)
sampled = [layer_indices[i] for i in range(0, n, step)][:_AFFINITY_SAMPLE_LAYERS]
# → e.g. [0, 20, 40, 60] for 40 MoE layers
```

### 5.2 `get_batch_routing()`

Returns `{batch_position: [expert_ids]}` — the union of activated experts
across sampled layers for each position in the batch.

```python
def get_batch_routing(self) -> Dict[int, List[int]]:
    result: Dict[int, set] = {}
    for layer_idx in sampled:
        switch = self._flash_layers[layer_idx]
        indices = switch._last_decode_indices  # [B, K]
        if indices is None or indices.ndim != 2:
            continue
        arr = np.array(indices, copy=False)     # cheap memcpy on UMA
        for pos in range(arr.shape[0]):
            result.setdefault(pos, set()).update(arr[pos].tolist())
    return {pos: sorted(ids) for pos, ids in result.items()}
```

**Zero-copy optimization**: `np.array(copy=False)` leverages UMA — the
`indices` tensor is already materialized by the prior `y.tolist()` call in
the decode loop, so this is a pointer cast, not a GPU sync.

### 5.3 `get_batch_routing_by_uid()`

Maps batch positions to request UIDs so the scheduler can associate expert
signatures with specific requests:

```python
def get_batch_routing_by_uid(
    self, uid_to_batch_pos: Dict[int, int]
) -> Dict[int, frozenset]:
    batch_routing = self.get_batch_routing()
    if not batch_routing:
        return {}
    result = {}
    for uid, pos in uid_to_batch_pos.items():
        if pos in batch_routing:
            result[uid] = frozenset(batch_routing[pos])
    return result
```

**Step overhead**: ~30μs total (4 layers × `np.array` + set union).

---

## 6. Affinity-Aware Batch Scheduling

The scheduler uses Jaccard similarity to decide whether a waiting request
should join the current batch or be deferred.

### 6.1 Jaccard Similarity

```
Jaccard(A, B) = |A ∩ B| / |A ∪ B|

Where:
  A = candidate request's expert signature (frozenset of expert IDs)
  B = running batch's expert union (frozenset of all active expert IDs)
```

- **High Jaccard (≥ 0.3)**: Candidate shares many experts with the batch →
  admit. Per-expert `group_size` increases → `gather_qmm` does real matmul.
- **Low Jaccard (< 0.3)**: Candidate would scatter tokens to new experts →
  defer (unless anti-starvation kicks in).

### 6.2 Agent Bonus

Requests from the same `agent_id` (same system prompt) tend to activate
similar experts. The scheduler adds a configurable bonus:

```python
if candidate.agent_id is not None:
    if any(req.agent_id == candidate.agent_id for req in self.running.values()):
        score += config.expert_affinity_agent_bonus  # default: +0.2
        score = min(1.0, score)  # clamp
```

### 6.3 Cold Start Handling

New requests have no routing history (`_expert_signature = None`). The
scheduler returns a neutral score of **0.5** — above the default threshold
of 0.3. This means cold-start requests are always admitted, which is the
correct behavior: you need to observe their routing before you can judge
affinity.

### 6.4 Insertion Point

The affinity check is inserted into `Scheduler._schedule_waiting()` after
existing cache-homogeneity checks but before final batch assembly:

```python
# scheduler.py, step(), lines 4181-4200

if (self.config.enable_expert_affinity
        and self._flashmlx_ctx
        and self._flashmlx_ctx.get("offload_ctx") is not None
        and num_decoding > 0):

    batch_sig = self._compute_batch_expert_signature()
    if batch_sig:
        affinity = self._expert_affinity_score(request, batch_sig)
        if (affinity < self.config.expert_affinity_threshold
                and request._expert_affinity_deferrals
                    < self.config.expert_affinity_max_deferrals):
            request._expert_affinity_deferrals += 1
            self.waiting.append(request)  # tail, not front
            continue  # skip, try next candidate
```

### 6.5 Guard Conditions

The affinity check is only active when ALL conditions hold:

| Condition | Why |
|---|---|
| `enable_expert_affinity` is True | Feature flag — off by default |
| `offload_ctx` is not None | Only relevant for MoE models with expert offloading |
| `num_decoding > 0` | Need at least one running decode to compute batch signature |
| `batch_sig` is not None | At least one running request has a signature |

---

## 7. Anti-Starvation Mechanism

A request deferred too many times must eventually be admitted, regardless of
affinity score.

### 7.1 Implementation

```python
# Per-request state (request.py, lines 191-193):
_expert_signature: Optional[frozenset] = None
_expert_affinity_deferrals: int = 0

# Admission logic:
if (affinity < threshold
        and request._expert_affinity_deferrals < max_deferrals):  # default: 3
    request._expert_affinity_deferrals += 1
    self.waiting.append(request)  # tail re-queue
    continue  # skip THIS request, try next in queue
```

### 7.2 Design Details

| Detail | Choice | Why |
|---|---|---|
| `continue` (not `break`) | Skip only the current request | Other waiting requests may have better affinity |
| Tail re-queue (`append`) | Deferred request goes to end of queue | Gives fresher requests a chance; prevents livelock |
| `max_deferrals = 3` | Configurable upper bound | Worst case: 3 × queue_depth × step_time |

### 7.3 Worst-Case Delay Analysis

```
Assumptions:
  queue_depth = 8 requests
  step_time ≈ 20ms (decode step at full batch)
  max_deferrals = 3

Worst case per request:
  3 deferrals × 8 positions × 20ms = 480ms additional latency

In practice:
  Most requests are deferred 0-1 times (natural overlap exists)
  Average additional latency: < 60ms
```

---

## 8. Configuration Reference

| Parameter | Default | Env Var | Description | Tuning |
|---|---|---|---|---|
| `enable_expert_affinity` | `False` | `OMLX_EXPERT_AFFINITY` | Master switch | Enable for MoE models with ≥ 4 concurrent users |
| `expert_affinity_threshold` | `0.3` | — | Min Jaccard to admit | Lower (0.2) = less deferral, less grouping. Higher (0.5) = more grouping, more latency risk |
| `expert_affinity_max_deferrals` | `3` | — | Anti-starvation bound | Higher = better grouping at cost of tail latency |
| `expert_affinity_agent_bonus` | `0.2` | — | Same-agent Jaccard bonus | Increase for homogeneous agent workloads |

### Recommended Configurations

```yaml
# High-throughput serving (8+ concurrent users, same domain)
enable_expert_affinity: true
expert_affinity_threshold: 0.25
expert_affinity_max_deferrals: 4
expert_affinity_agent_bonus: 0.3

# Mixed workload (diverse users, different domains)
enable_expert_affinity: true
expert_affinity_threshold: 0.3     # default
expert_affinity_max_deferrals: 3   # default
expert_affinity_agent_bonus: 0.2   # default

# Latency-sensitive (minimize tail latency)
enable_expert_affinity: true
expert_affinity_threshold: 0.2
expert_affinity_max_deferrals: 2
expert_affinity_agent_bonus: 0.1
```

---

## 9. Benchmark Results

### 9.1 Setup

- **Model**: Qwen3.5-35B-A3B (256 experts, K=8)
- **Hardware**: M4 Pro 48GB
- **Workload**: Staggered 8 concurrent chat requests
- **Protocol**: A/B comparison, affinity OFF vs ON, 3 rounds each

### 9.2 Aggregate Results

| Metric | Affinity OFF | Affinity ON | Delta |
|---|---:|---:|---:|
| Aggregate Gen TPS | 67.3 tok/s | **92.9 tok/s** | **+38.0%** |
| Wave-2 Gen TPS | 53.9 tok/s | **85.1 tok/s** | **+57.8%** |
| TTFT | 12,687 ms | **7,016 ms** | **-44.7%** |

### 9.3 Stability

```
Affinity OFF:
  Round 1: 72.1 tok/s
  Round 2: 65.8 tok/s
  Round 3: 64.0 tok/s  ← degrades round-over-round

Affinity ON:
  Round 1: 91.2 tok/s
  Round 2: 93.5 tok/s
  Round 3: 94.0 tok/s  ← stable across rounds
```

Without affinity, the pool maintenance system fights against scattered routing
patterns — expert churn increases round-over-round. With affinity, grouped
requests share experts consistently, reducing pool churn to near-zero.

### 9.4 Per-Expert group_size Distribution

```
Affinity OFF:
  group_size=1: 68%    ← matvec (wasted GPU)
  group_size=2: 22%
  group_size=3+: 10%

Affinity ON:
  group_size=1: 15%
  group_size=2: 25%
  group_size=3: 30%
  group_size=4+: 30%   ← real matmul (full GPU utilization)
```

---

## 10. Implementation Map

### 10.1 File Summary

| File | Component | LOC Changed | Role |
|---|---|---|---|
| `FlashMLX/mlx-lm-source/mlx_lm/models/expert_offload.py` | Routing capture + Bridge | ~30 | Capture `_last_decode_indices`; export via Bridge |
| `FlashMLX/src/flashmlx/integration/protocol.py` | Protocol definition | ~25 | `get_batch_routing()` / `get_batch_routing_by_uid()` interface |
| `FlashMLX/src/flashmlx/integration/thunderomlx.py` | Protocol implementation | ~20 | Delegates to `offload_ctx.bridge` |
| `ThunderOMLX/src/omlx/scheduler.py` | Scheduling logic | ~50 | Signature update, Jaccard scoring, admission control |
| `ThunderOMLX/src/omlx/request.py` | Request state | ~5 | `_expert_signature`, `_expert_affinity_deferrals` |

**Total**: ~130 lines of core logic across 5 files.

### 10.2 Per-File Details

**`expert_offload.py`** (FlashMLX):
- Lines 1056-1060: `_last_decode_indices` capture in `_pool_call()`
- Lines 1095-1096: Same capture in fallback path
- Lines 1909-1912: `_AFFINITY_SAMPLE_LAYERS = 4` constant + rationale
- Lines 1914-1953: `get_batch_routing()` implementation
- Lines 1955-1979: `get_batch_routing_by_uid()` implementation

**`protocol.py`** (FlashMLX):
- Lines 153-177: Protocol methods for expert affinity feedback

**`thunderomlx.py`** (FlashMLX):
- Lines 255-273: Delegation to `ThunderOMLXBridge`

**`scheduler.py`** (ThunderOMLX):
- Lines 1102-1105: Configuration parameters
- Lines 1999-2030: `_update_expert_signatures()`
- Lines 2032-2047: `_compute_batch_expert_signature()` (cached)
- Lines 2049-2084: `_expert_affinity_score()` (Jaccard + agent bonus)
- Lines 4181-4200: Admission control in `step()`

**`request.py`** (ThunderOMLX):
- Lines 191-193: `_expert_signature` and `_expert_affinity_deferrals`

---

## 11. Edge Cases & Limitations

| Edge Case | Handling | Notes |
|---|---|---|
| Single concurrent user | Affinity check skipped (`num_decoding > 0` but no batch signature to compare) | No overhead for single-user workload |
| All users same domain | High Jaccard everywhere → all admitted → behaves like FCFS | No harm, marginal benefit from natural overlap |
| All users different domains | Low Jaccard → frequent deferrals → anti-starvation kicks in | Worst case: 480ms extra latency per request |
| Non-MoE model | `offload_ctx is None` → check skipped entirely | Zero overhead for dense models |
| Prefill phase | `_last_decode_indices` not captured (`seq_len > 1`) | Cold start handling (score=0.5) ensures admission |
| Model with < 4 MoE layers | Fewer than 4 layers sampled | Still works — less signal but Jaccard is still valid |
| Expert signature drift | Re-computed every decode step | Signatures track live routing, not stale history |
| `agent_id` is None | Agent bonus skipped | No crash — just no bonus |
| Queue empty | No candidates to defer | Loop exits immediately |

### Known Limitations

1. **Layer sampling is uniform**: The 4 sampled layers are evenly spaced.
   If expert routing is highly non-uniform across layers (e.g., early layers
   use broad routing, late layers are focused), the uniform sampling may miss
   the most discriminative layers. An adaptive sampling strategy is a possible
   future extension.

2. **Jaccard is set-based, not frequency-weighted**: Two requests that both
   activate expert #42 are treated the same regardless of whether expert #42
   handles 50% or 5% of their computation. A weighted Jaccard variant could
   improve grouping quality at the cost of more complex signature tracking.

3. **Single-step signatures**: The expert signature is based on the most
   recent decode step only. Autoregressive routing can shift rapidly — a
   multi-step exponential moving average could smooth this, at the cost of
   delayed adaptation.

---

## Appendix A: Why Not Sort by Expert Count?

An alternative to Jaccard-based scheduling is to sort requests by the number
of unique experts they activate and batch "narrow" requests (few experts)
together. We tried this and found it inferior because:

1. Expert *identity* matters more than expert *count*. Two requests using 20
   experts each can have 100% overlap (Jaccard=1.0) or 0% overlap
   (Jaccard=0.0). Count-based sorting cannot distinguish these cases.

2. Sorting imposes a global ordering that conflicts with FIFO fairness.
   Jaccard-based deferral preserves FIFO as a baseline and only perturbs it
   within the anti-starvation bound.

## Appendix B: Integration with ThunderOMLX

Expert Affinity Scheduling is a ThunderOMLX-side optimization that consumes
data from FlashMLX. The two systems are cleanly separated:

```
FlashMLX responsibility:
  - Capture routing (zero-cost)
  - Export routing via ThunderOMLXBridge
  - No scheduling logic, no deferral decisions

ThunderOMLX responsibility:
  - Consume routing via bridge protocol
  - Compute Jaccard similarity
  - Make admission/deferral decisions
  - Enforce anti-starvation
```

This separation means FlashMLX can be used standalone (without ThunderOMLX)
with zero overhead — the `_last_decode_indices` capture is a shape operation
that costs < 100ns and has no side effects when nobody reads it.
