# FlashMLX

> **A memory-policy runtime for local LLM inference on Apple Silicon.**
> Not a KV-cache trick. Not a port of a CUDA serving stack. A system-level rewrite
> of how parameter memory, prefill activation, KV residency, batching and quality
> protection work together on M-series Macs.

[![Platform](https://img.shields.io/badge/platform-Apple%20Silicon-black)](https://developer.apple.com/documentation/apple-silicon)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![Status](https://img.shields.io/badge/status-research%2Falpha-orange)]()

---

## What is FlashMLX?

FlashMLX is a research runtime built on top of [MLX](https://github.com/ml-explore/mlx)
and a forked copy of [`mlx-lm`](https://github.com/ml-explore/mlx-examples). It
answers a narrow question: **what does local LLM inference on Apple Silicon look
like when you stop pretending it is CUDA and start building for Unified Memory
Architecture (UMA)?**

We ship five composable optimization "routes" that work at the `mlx-lm` cache /
scheduler layer — no CUDA, no `vllm`, no server. Everything runs on one Mac.

**Concretely, FlashMLX ships today:**

| Route | Name | Targets | Maturity |
|:-----:|------|---------|----------|
| **0** | Density Router | Compression control plane (Model Card + discrete density modes) | Stable |
| **1** | Expert Offloading | MoE parameter residency (GPU / CPU / SSD tiers) | Stable |
| **3** | Scored P2 + Flat Buffer | KV-cache memory & TG speed (AM scoring + Q8 flat buffer) | Stable |
| **4** | Chunked Prefill | PP peak memory (streaming eviction + interleaved schedule) | Stable |
| **5** | Context Recall (KV-Direct) | Lossless reconstruction of evicted tokens via h⁽⁰⁾ replay | Beta |

Plus experimental VLM bridge for Qwen2-VL / Gemma 4.

---

## Why bother?

Running a 35B MoE or a 32K-context 8B on an M-series Mac usually hits one of
these walls:

- **Parameter residency** — 256-expert MoE keeps every expert live in UMA.
- **Prefill peak** — activations for a 32K prompt push peak memory well past the
  live working set.
- **TG memory bandwidth** — decode reads the entire KV cache every step, so
  throughput falls off with context length.
- **Time-to-first-token in batch** — naive prefill-then-decode stalls every
  downstream request behind the slowest prefill.

FlashMLX treats all four as a single memory-policy problem. No component is
CUDA-translated. Everything is designed around the fact that CPU and GPU share
the same physical memory.

---

## Performance snapshot

### Qwen3-8B-MLX-4bit / 32K context / M4 Max 64GB
| Metric | Standard | FlashMLX (`scored_pq` + Q8) | Change |
|---|---:|---:|---:|
| PP throughput | 269.5 tok/s | **409.5 tok/s** | **+51.9%** |
| TG throughput | 16.1 tok/s | **21.6 tok/s** | **+34.2%** |
| TTFT | 121.6 s | **80.0 s** | **-34.2%** |
| PP peak memory | 4,840 MB | **526 MB** | **-89.1%** |
| TG KV memory | 4,647 MB | **529 MB** | **-88.6%** |
| Quality | pass | **pass** | no degradation |

### Qwen3.5-35B-A3B / 16K / batch=4 / M4 Pro 48GB
| Metric | Community `mlx-lm` | FlashMLX v2.0 | Change |
|---|---:|---:|---:|
| TG throughput | 115.8 tok/s | **196.9 tok/s** | **+70.0%** |
| TTFT | 82.0 s | **21.1 s** | **-74.3%** |
| GPU peak memory | 28.01 GB | **13.78 GB** | **-50.8%** |
| Model residency | 18.21 GB | **11.42 GB** | **-37.3%** |
| Quality | 4/4 pass | **4/4 pass** | no degradation |

Numbers above are reproducible — see `benchmarks/` and `examples/` for the exact
scripts and model cards. Raw logs from the latest Gemma 4 run live in
`docs/gemma31b_longctx_benchmark_2026-04-10.log` (gitignored by default).

---

## Architecture at a glance

```
Route 0 (control plane — picks the strategy)
    ├─ Route 3 (KV compression: Recent/Warm/Cold → Flat buffer)
    │     ├─ Route 4 (chunked prefill + streaming eviction)
    │     └─ Route 5 (h⁽⁰⁾ archive → on-demand prefix reconstruction)
    └─ Route 1 (expert offloading for MoE — orthogonal)
```

A single line of user code picks a mode:

```python
from flashmlx import load_card_or_detect, make_prompt_cache

card  = load_card_or_detect(model, "/path/to/model")
cache = make_prompt_cache(model, **card.to_cache_kwargs(mode="balanced"))
# mode ∈ {"balanced", "ultra_long", "recall_first"}
```

`balanced` is the recommended daily driver. `ultra_long` trades a bit of recall
for much lower KV footprint at 32K+. `recall_first` compresses hardest but keeps
an h⁽⁰⁾ archive so evicted tokens can be reconstructed exactly when an upstream
scheduler asks.

For a full walkthrough with data flow diagrams, tombstones ("what we killed and
why"), and design decisions, see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Repository layout

```
FlashMLX/
├── src/flashmlx/             # User-facing SDK (configs, model cards, reconstruction)
│   ├── config.py             #   CacheConfig, FlashMLXConfig, DensityLevel
│   ├── model_cards.py        #   ModelCard, ModeConfig — single source of truth per model
│   ├── reconstruction.py     #   ReconstructionController — programmatic h⁽⁰⁾ replay
│   ├── capabilities.py       #   Hardware probing and preset recommendation
│   ├── integration/          #   ThunderOMLX adapter and friends
│   └── vlm_bridge.py         #   Experimental VLM hook for mlx-vlm fork
│
├── mlx-lm-source/mlx_lm/models/   # Vendored mlx-lm fork (the engine)
│   ├── cache_factory.py           #   make_optimized_cache() + hybrid architecture detection
│   ├── triple_layer_cache.py      #   Route 3 core: Recent/Warm/Cold + Flat buffer
│   ├── kv_direct_cache.py         #   Route 5 core: H0Store, reconstruct_prefix_kv()
│   ├── quantization_strategies.py #   Q4_0, Q8_0, PolarQuant, TurboQuant backends
│   └── expert_offload.py          #   Route 1 core: three-tier expert pool
│
├── mlx-vlm-source/              # Vendored mlx-vlm fork (VLM experiments)
├── mlx-source/                  # Vendored MLX core fork (rarely modified)
├── model_cards/                 # Per-model JSON configs (optimal + modes + benchmarks)
├── examples/                    # Runnable demos and benchmarks
├── benchmarks/                  # Formal benchmark suite
├── docs/                        # Architecture, API reference, per-route docs
├── tests/                       # Unit and integration tests
└── scripts/                     # Calibration generators and utility scripts
```

The repo vendors three upstream forks (`mlx-source`, `mlx-lm-source`,
`mlx-vlm-source`) because most of the runtime work lives inside the cache factory
and generate loop — patches that are hard to maintain as external monkey-patches.

---

## Getting started

### Requirements

- **Hardware**: Apple Silicon Mac (M1 / M2 / M3 / M4, any variant). 32GB+ unified
  memory recommended for anything beyond 7B.
- **OS**: macOS 14 Sonoma or newer. Metal 3 required.
- **Python**: 3.10 or newer.
- **Disk**: Keep your model weights on **internal SSD**, not an external USB
  drive. MLX uses `mmap` for weights; cold page faults from a USB drive will
  silently halve TG throughput.

### Setup

```bash
git clone https://github.com/lisihao/FlashMLX.git
cd FlashMLX

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt    # or: pip install -e .
```

The vendored `mlx-lm-source` and `mlx-vlm-source` are on the `sys.path` via the
examples — no separate install step. If you want to use FlashMLX as a library
from outside the repo, install it editable:

```bash
pip install -e .
```

### Download a model

We test against:

- `mlx-community/Qwen3-8B-MLX-4bit` — default performance target
- `mlx-community/Qwen3-1.7B-MLX-4bit` — fast iteration for CI
- `mlx-community/Llama-3.2-3B-Instruct-4bit` — triple_pq reference
- `mlx-community/Qwen3.5-35B-A3B-MLX` — MoE / expert offloading target

```bash
python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('mlx-community/Qwen3-8B-MLX-4bit', local_dir='./models/qwen3-8b')"
```

### Run your first benchmark

```bash
# Quick smoke test — compares standard vs scored_pq at 4K
python examples/bench_card.py --model ./models/qwen3-8b --context 4096

# Full Route 0 discrete benchmark (balanced / ultra_long / recall_first)
python benchmarks/bench_route0_discrete.py --model ./models/qwen3-8b
```

If the cache factory logs `[CacheFactory] scored_pq: ...` and final numbers land
within 10% of the table above, your setup is good.

---

## Development workflow

### Branch model

- `main` — released checkpoints. Protected.
- `phase1-mlx-lm-upgrade` — current active development branch (this is where
  most recent commits live).
- Feature branches: `feat/<short-slug>`, `fix/<short-slug>`, `docs/<slug>`.

Please branch off the most recent active branch, not off `main`, until we cut the
next stable release.

### Running tests

```bash
# Unit tests for the FlashMLX SDK
pytest tests/ -v

# Regression suite for the mlx-lm cache factory
pytest tests/cache/ -v

# Route 0 density router
pytest tests/test_route0_density.py -v
```

There are about 2900 tests across the full regression suite. A single-laptop run
is ~5 minutes on an M4 Pro.

### Running benchmarks

Each benchmark is a standalone script under `examples/` or `benchmarks/`. They
print a clearly labelled table and exit non-zero if quality checks fail.

```bash
# KV compression strategies on a short / long context
python examples/bench_card.py              --model ./models/qwen3-8b
python benchmarks/bench_route0_discrete.py --model ./models/qwen3-8b
python benchmarks/bench_density_modes.py   --model ./models/qwen3-8b

# Gemma 4 hybrid-cache long-context sweep
GEMMA4_MODEL_PATH=./models/gemma-4-31B python examples/bench_gemma4_long_context.py

# Expert offloading on an MoE model
python examples/mac_benchmark.py --model ./models/qwen3.5-35b-a3b
```

When you add a new benchmark, also record results in the relevant
`model_cards/*.json` under the `benchmarks` key so future runs have a reference
baseline.

### Updating a Model Card

```bash
python examples/bench_card.py --model ./models/qwen3-8b --update-card
```

This writes the new throughput / memory numbers back into
`model_cards/qwen3-8b-mlx-4bit.json` so the next run of `load_card_or_detect`
picks them up automatically.

---

## Contributing

FlashMLX is early-stage research code. Contributions are genuinely wanted — not
as a slogan, but because there are concrete places where we need help from
people outside the core team.

### Good places to start

1. **Model Cards for new models**. Drop a JSON into `model_cards/`, run
   `bench_card.py --update-card`, open a PR. Small, self-contained, immediately
   useful.
2. **Quantization backends**. `quantization_strategies.py` has a clean ABC.
   Adding a new `QuantizationStrategy` subclass (e.g. a new TurboAngle preset)
   is a great first PR.
3. **Benchmarks on other M-series chips**. We currently have M4 Pro 48GB and
   M4 Max 64GB numbers. M1 Max, M2 Ultra, M3 Pro and base M4 data are all
   missing and would be valuable.
4. **VLM support for additional models**. The vlm_bridge currently handles
   Qwen2-VL and Gemma 4. LLaVA, InternVL, and Phi-Vision are open.
5. **Documentation**. `docs/ARCHITECTURE.md` is in Mandarin for historical
   reasons. English ports, diagrams, and diataxis-style how-to guides are very
   welcome.

### Open research questions

These are the harder problems we're tracking. If you want to take one, open an
issue first so we can agree on scope.

- **Weight quantization beyond uniform 4-bit.** Mixed-bit / sensitivity-aware
  PTQ targeting `down_proj`, `o_proj`, and FFN layers. Ceiling on a 31B 4-bit
  model is roughly 1.7× TG if you drop the weight read from ~20GB to ~12GB.
- **Speculative decoding on Apple Silicon.** Narrow-beam self-speculation,
  Apple-style recurrent drafting, or EAGLE-3. No MLX implementation exists
  upstream.
- **Block verification (ICLR 2025).** A ~5–8% wall-clock win if integrated on
  top of speculative decoding.
- **Cross-session prefix caching.** Route 5 already captures h⁽⁰⁾; promoting it
  to a persistent, content-addressed prefix cache would unlock the "system
  prompt shared across 100 agent calls" scenario for ~5× throughput on agentic
  workloads.
- **Perplexity-based density signal.** The current key-norm surprise signal
  cannot detect "unique information made of common words." A perplexity- or
  attention-entropy-based alternative is an open problem — see
  [`docs/route0-density-router-design.md`](docs/route0-density-router-design.md)
  Phase 5.

### Code style

- Python: `black` with default settings (line length 88). `ruff` for linting.
- Imports: sorted by `isort` (profile=`black`).
- Type hints required for new public APIs in `src/flashmlx/`. Internal
  mlx-lm patches can match the surrounding style.
- No emojis in committed code or comments. (User-facing READMEs and docs are
  fine.)

### Pull request guidelines

- One logical change per PR. Separate refactors from features.
- Include a benchmark or test result in the PR description if the change
  touches a hot path (cache factory, triple layer cache, decode loop).
- Reference the relevant paper (arXiv or conference) in the commit message if
  the change implements a published technique.
- CI is currently a laptop-based regression run; please confirm `pytest tests/`
  passes locally and paste the last 20 lines into the PR description.

### Non-goals

- **CUDA support.** This is not that project.
- **Serving multiple users over HTTP.** ThunderOMLX is a separate project for
  that. FlashMLX provides the scheduling primitives; it is not itself a server.
- **Training / fine-tuning.** FlashMLX is inference-only.

---

## Related work and references

FlashMLX stands on a lot of recent papers. The ones we actually implemented,
benchmarked, or explicitly rejected:

- [KV-Direct (arXiv:2603.19664)](https://arxiv.org/abs/2603.19664) — the h⁽⁰⁾
  reconstruction idea at the core of Route 5.
- **PolarQuant / TurboQuant** — data-oblivious KV codecs used by `triple_pq`
  and `scored_pq`.
- **Density-Aware Semi-Dynamic Compression (arXiv:2603.25926)** — basis for
  Route 0's five discrete density levels.
- **MemBoost + MemCollab + Meta-Harness trilogy (2603.23234 / 26557 / 28052)** —
  informs the ThunderOMLX integration and the AME semantic cache layer.
- **H2O / StreamingLLM / SnapKV** — eviction-only approaches we benchmarked
  against and ultimately treated as one tool among many rather than the answer.

Each implemented route has a design note in `docs/`, and each killed approach
has a tombstone entry explaining why it did not make it into the main path.

---

## License

FlashMLX is released under the **MIT License** — the same license used by the
upstream `mlx`, `mlx-lm`, and `mlx-vlm` projects it builds on. See
[`LICENSE`](LICENSE) for the full text.

The repository vendors copies of `mlx`, `mlx-lm`, and `mlx-vlm`, each with
their own upstream MIT License (© 2023 Apple Inc.) preserved in the
respective `mlx-source/`, `mlx-lm-source/`, and `mlx-vlm-source/` directories.
When you redistribute FlashMLX (source or binary), please keep both the root
LICENSE and the vendored upstream LICENSE files intact.

Contributors retain copyright on their changes; contributions are accepted
under the same MIT terms as the rest of the project.

---

## Getting in touch

- **Bugs and feature requests**: open an issue at
  https://github.com/lisihao/FlashMLX/issues
- **Discussions**: GitHub Discussions is enabled on the repo.
- **Security reports**: please open a private security advisory via GitHub
  rather than a public issue.

If you're onboarding to contribute, introduce yourself in the Discussions board
first — we're happy to pair new contributors with a mentor for their first PR.

---

**FlashMLX is a runtime built for Apple Silicon, not a tech demo.**
If that resonates, file an issue, run a benchmark, or open a PR.
