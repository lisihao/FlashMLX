# FlashMLX

**O(1) Memory KV Cache Compression for Apple Silicon**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![MLX](https://img.shields.io/badge/MLX-0.31+-green.svg)](https://github.com/ml-explore/mlx)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

FlashMLX 在 MLX-LM 推理中实现了 **O(1) 内存复杂度的 Prefill** 和 **97% KV Cache 压缩**，无质量损失。

> 32K 上下文：KV 内存从 4,572 MB 降到 147 MB，TG 速度反而快 54%。

## Performance

Qwen3-8B-MLX (Q8) on Apple M4 Pro 24GB，所有数据来自独立子进程串行测试：

### 32K Context — scored_q8 (recommended) vs standard

| Metric | Standard | FlashMLX | Change |
|--------|----------|----------|--------|
| **PP Speed** | 213.6 tok/s | 372.8 tok/s | **+74.5%** |
| **TG Speed** | 16.0 tok/s | 24.7 tok/s | **+54.4%** |
| **TTOF** | 151.7s | 86.9s | **-42.7%** |
| **KV Memory (TG)** | 4,572 MB | 147 MB | **-96.8%** |
| **Quality** | PASS | PASS | lossless |

### 16K Context

| Metric | Standard | FlashMLX | Change |
|--------|----------|----------|--------|
| **PP Speed** | 275.2 tok/s | 361.8 tok/s | **+31.5%** |
| **TG Speed** | 18.9 tok/s | 24.7 tok/s | **+30.7%** |
| **TTOF** | 58.4s | 44.3s | **-24.1%** |
| **KV Memory (TG)** | 2,268 MB | 129 MB | **-94.3%** |
| **Quality** | PASS | PASS | lossless |

### All Configurations

| Config | Ctx | PP tok/s | TG tok/s | TTOF | KV TG | Quality |
|--------|-----|----------|----------|------|-------|---------|
| standard | 16K | 275.2 | 18.9 | 58.4s | 2,268 MB | PASS |
| scored_bf16 | 16K | 362.4 | 26.4 | 44.2s | 252 MB | PASS |
| **scored_q8** | **16K** | **361.8** | **24.7** | **44.3s** | **129 MB** | **PASS** |
| scored_q4 | 16K | 378.9 | 19.4 | 42.3s | 72 MB | PASS |
| standard | 32K | 213.6 | 16.0 | 151.7s | 4,572 MB | PASS |
| scored_bf16 | 32K | 369.5 | 26.2 | 87.7s | 288 MB | PASS |
| **scored_q8** | **32K** | **372.8** | **24.7** | **86.9s** | **147 MB** | **PASS** |
| scored_q4 | 32K | 376.9 | 16.1 | 86.0s | 81 MB | PASS |

## Quick Start

```python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Qwen3-8B-Instruct")

# One line — auto-calibration on first use (~26s), cached afterwards (<1ms)
result = generate(model, tokenizer, prompt="Your long prompt here...",
                  kv_cache="scored_pq", kv_flat_quant="q8_0")
```

No manual calibration, no config files, no code changes needed.

### Advanced Options

```python
result = generate(
    model, tokenizer, prompt=prompt,
    kv_cache="scored_pq",        # AM-scored chunked prefill + streaming eviction
    kv_flat_quant="q8_0",        # Flat buffer quantization: None (bf16), "q8_0", "q4_0"
    kv_scored_max_cache=2048,    # Max tokens retained after eviction (default: 2048)
    kv_calibration="/path/to/custom.pkl",  # Optional: custom calibration file
)
```

## How It Works

```
                    FlashMLX KV Cache Architecture (v0.9.2)
    ┌────────────────────────────────────────────────────────────────────┐
    │                                                                    │
    │   Prefill (Chunked)                     Decode (Streaming)         │
    │   ┌──────────────┐                      ┌──────────────────┐      │
    │   │ chunk=512     │                      │ Flat Buffer (Q8) │      │
    │   │ → model()     │                      │ max_cache=2048   │      │
    │   │ → eval()      │                      │ int8 + bf16 scale│      │
    │   │ → if >2048:   │                      │ O(1) per step    │      │
    │   │   AM evict    │── promote ──────────→│                  │      │
    │   └──────────────┘                      └──────────────────┘      │
    │        │                                        │                  │
    │        │ PP Peak: ~773 MB (O(1))                │ TG: 24.7 tok/s  │
    │        │ PP Speed: 373 tok/s                    │ TG KV: 147 MB   │
    │                                                                    │
    │   ┌────────────────────────────────────────────────────────────┐   │
    │   │  Auto-Calibration                                          │   │
    │   │  First use: ~26s (diverse corpus prefill → AM scoring)     │   │
    │   │  Cached: <1ms (~/.cache/flashmlx/calibrations/)            │   │
    │   └────────────────────────────────────────────────────────────┘   │
    └────────────────────────────────────────────────────────────────────┘
```

### Key Innovations

1. **Chunked Prefill + Streaming Eviction** — Process input in 512-token chunks, evict low-scoring tokens when cache exceeds threshold. Turns O(N^2) prefill memory into O(1).

2. **AM-Scored Token Selection** — Attention Matching importance scoring determines which tokens survive eviction. Calibrated offline with diverse corpus, cached per model architecture.

3. **Q8_0 Flat Buffer** — Surviving tokens stored as per-token int8 + bf16 scale. 50% memory reduction with <7% speed cost. The sweet spot between bf16 (fast, big) and Q4_0 (small, slow).

4. **Auto-Calibration** — New model? First `scored_pq` call triggers automatic calibration (~26s). Saved to `~/.cache/flashmlx/calibrations/`, reused across sessions.

### Why is it FASTER?

Standard attention is O(N^2). FlashMLX bounds the cache at 2048 tokens, so attention becomes O(chunk * 2048) = O(1) per chunk. At 32K, standard PP drops to 213 tok/s while FlashMLX maintains 373 tok/s.

## Configuration Guide

| Config | KV Memory | TG Speed | Use Case |
|--------|-----------|----------|----------|
| `scored_pq` (bf16) | -89% @ 16K | +40% | Maximum speed |
| `scored_pq` + `q8_0` | -94% @ 16K | +31% | **Recommended default** |
| `scored_pq` + `q4_0` | -97% @ 16K | +3% | Maximum compression |

**Recommendation**: `kv_flat_quant="q8_0"` is the sweet spot. Q4_0's additional memory savings are marginal (147 MB → 81 MB at 32K) but the 30% TG speed penalty is steep.

## Core Files

```
FlashMLX/mlx-lm-source/mlx_lm/
├── generate.py                      # Entry point — kv_cache, kv_flat_quant params
├── models/
│   ├── cache.py                     # make_prompt_cache() routing
│   ├── cache_factory.py             # Strategy factory + adaptive params
│   ├── triple_layer_cache.py        # Scored P2 + flat buffer + Q8/Q4 quantization
│   ├── am_calibrator.py             # Auto-calibration system
│   └── quantization_strategies.py   # Pluggable quantizers (Q4_0, Q8_0, PolarQuant)
```

## Development Journey

This project started as a paper reproduction of Attention Matching (AM) and evolved into something beyond any single paper. The full story:

- **Day 1-2**: AM works single-layer, 36-layer = gibberish. Found rank-deficient beta matrices.
- **Day 3**: Almost gave up. Found two critical bugs (unbounded beta, biased query sampling).
- **Day 3-4**: On-Policy calibration — the key insight no paper discusses.
- **Day 5-6**: Triple-Layer Cache architecture born from accumulated failures.
- **Day 6-7**: Chunked Prefill + Streaming Eviction = O(1) PP memory.
- **Day 7+**: Auto-calibration + Q8_0 flat buffer = production-ready.

Full article: [From Paper to Production (Chinese)](/.solar/ARTICLE-week-in-review.md)

## Research References

- **Attention Matching**: [Fast KV Cache Compaction](https://arxiv.org/abs/2602.16284) — Core scoring algorithm
- **H2O**: [Heavy-Hitter Oracle](https://arxiv.org/abs/2306.14048) — Token eviction concept
- **StreamingLLM**: [Efficient Streaming LLMs](https://arxiv.org/abs/2309.17453) — Attention sink preservation
- **MLX**: https://github.com/ml-explore/mlx — Apple's ML framework
- **MLX-LM**: https://github.com/ml-explore/mlx-lm — LLM inference

## License

MIT License — see [LICENSE](LICENSE)

---

*FlashMLX v0.9.2 — Built for Apple Silicon*
