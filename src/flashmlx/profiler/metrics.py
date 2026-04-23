"""
FlashMLX 8-Metric Profiling Framework — Metric Collectors

8 fixed metrics for undeniable profiling evidence:
  1. Kernel launches per token (mx.eval count)
  2. MoE path time ratio
  3. Per-expert token histogram
  4. Grouped GEMM group count/size distribution
  5. Dequant time ratio
  6. GPU command buffer gaps (xctrace)
  7. Batch scaling curves
  8. Context length prefill/decode split
"""

import time
import statistics
from collections import defaultdict
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod

import mlx.core as mx


class MetricCollector(ABC):
    """Base class for all metric collectors."""

    @abstractmethod
    def reset(self):
        """Reset collected data."""

    @abstractmethod
    def to_dict(self) -> dict:
        """Return collected metrics as a dictionary."""

    def _stats(self, values: List[float]) -> dict:
        """Compute summary statistics for a list of values."""
        if not values:
            return {"count": 0, "mean": 0, "min": 0, "max": 0, "p50": 0, "p95": 0}
        sorted_v = sorted(values)
        n = len(sorted_v)
        return {
            "count": n,
            "mean": statistics.mean(sorted_v),
            "min": sorted_v[0],
            "max": sorted_v[-1],
            "p50": sorted_v[n // 2],
            "p95": sorted_v[min(int(n * 0.95), n - 1)],
        }


# ---------------------------------------------------------------------------
# Metric 1: Kernel launches per token
# ---------------------------------------------------------------------------

class KernelCounter(MetricCollector):
    """Count mx.eval / mx.async_eval invocations per decode token.

    Install before generation, call mark_token() between tokens,
    uninstall after generation.
    """

    def __init__(self):
        self._orig_eval = None
        self._orig_async_eval = None
        self._current_count = 0
        self._token_counts: List[int] = []
        self._installed = False

    def reset(self):
        self._current_count = 0
        self._token_counts = []

    def install(self):
        if self._installed:
            return
        self._orig_eval = mx.eval
        self._orig_async_eval = mx.async_eval

        counter = self

        _real_eval = self._orig_eval
        _real_async = self._orig_async_eval

        def counting_eval(*args, **kwargs):
            counter._current_count += 1
            return _real_eval(*args, **kwargs)

        def counting_async(*args, **kwargs):
            counter._current_count += 1
            return _real_async(*args, **kwargs)

        mx.eval = counting_eval
        mx.async_eval = counting_async
        self._installed = True

    def uninstall(self):
        if not self._installed:
            return
        mx.eval = self._orig_eval
        mx.async_eval = self._orig_async_eval
        self._installed = False

    def mark_token(self):
        """Call between tokens to record the count for the previous token."""
        self._token_counts.append(self._current_count)
        self._current_count = 0

    def to_dict(self) -> dict:
        if self._current_count > 0:
            self._token_counts.append(self._current_count)
            self._current_count = 0
        return {
            "metric": "kernel_launches_per_token",
            "per_token": self._token_counts,
            "stats": self._stats([float(c) for c in self._token_counts]),
        }


# ---------------------------------------------------------------------------
# Metrics 2, 3, 4: MoE profiling (time ratio, expert histogram, GEMM stats)
# ---------------------------------------------------------------------------

class MoEProfiler(MetricCollector):
    """Profile MoE layers: time ratio, expert distribution, GEMM group sizes.

    Wraps SwitchGLU/SwitchMLP __call__ and captures:
      - Wall-clock time per invocation (metric 2)
      - Expert indices per invocation (metric 3)
      - Sorted index group sizes (metric 4)
    """

    def __init__(self):
        self._moe_times: List[float] = []
        self._forward_times: List[float] = []
        self._expert_counts: Dict[int, int] = defaultdict(int)
        self._group_sizes: List[int] = []
        self._pending_indices: List[Any] = []  # deferred tolist()
        self._patched_modules: List[tuple] = []  # (cls, attr_name, original_fn)
        self._installed = False

    def reset(self):
        self._moe_times = []
        self._forward_times = []
        self._expert_counts = defaultdict(int)
        self._group_sizes = []
        self._pending_indices = []

    def install(self, model):
        """Find and wrap all SwitchGLU/SwitchMLP modules in the model."""
        if self._installed:
            return

        from mlx_lm.models.switch_layers import SwitchGLU, SwitchMLP

        profiler = self

        # Use named_modules() for robust traversal of any model architecture
        moe_modules = []
        if hasattr(model, 'named_modules'):
            for name, mod in model.named_modules():
                if isinstance(mod, (SwitchGLU, SwitchMLP)):
                    moe_modules.append((mod, name))

        # Patch each class only once (all instances share the same __call__)
        patched_classes = set()
        for moe_module, path in moe_modules:
            cls = moe_module.__class__
            if cls in patched_classes:
                continue
            patched_classes.add(cls)

            original_call = cls.__call__

            def make_wrapper(orig_fn):
                def wrapped_call(self_mod, x, indices, *args, **kwargs):
                    # Capture expert indices (metric 3) — defer tolist to avoid sync
                    profiler._pending_indices.append(indices)

                    # Time the MoE dispatch (metric 2) — no mx.eval to avoid overhead
                    start = time.perf_counter()
                    result = orig_fn(self_mod, x, indices, *args, **kwargs)
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    profiler._moe_times.append(elapsed_ms)

                    return result
                return wrapped_call

            cls.__call__ = make_wrapper(original_call)
            self._patched_modules.append((cls, '__call__', original_call))

        self._installed = True
        self._num_moe_layers = len(moe_modules)

    def uninstall(self):
        if not self._installed:
            return
        for cls, attr_name, original_fn in self._patched_modules:
            setattr(cls, attr_name, original_fn)
        self._patched_modules = []
        self._installed = False

    def record_forward_time(self, time_ms: float):
        """Record total forward pass time for ratio calculation."""
        self._forward_times.append(time_ms)

    @property
    def num_moe_layers(self) -> int:
        return getattr(self, '_num_moe_layers', 0)

    def _process_pending_indices(self):
        """Process deferred expert indices into histogram + group sizes."""
        for indices in self._pending_indices:
            try:
                idx_flat = indices.flatten().tolist()
            except Exception:
                continue

            # Expert histogram (metric 3)
            for eid in idx_flat:
                self._expert_counts[eid] += 1

            # Group sizes from sorted indices (metric 4)
            sorted_idx = sorted(idx_flat)
            if sorted_idx:
                current_expert = sorted_idx[0]
                group_size = 1
                for i in range(1, len(sorted_idx)):
                    if sorted_idx[i] == current_expert:
                        group_size += 1
                    else:
                        self._group_sizes.append(group_size)
                        current_expert = sorted_idx[i]
                        group_size = 1
                self._group_sizes.append(group_size)

        self._pending_indices = []

    def to_dict(self) -> dict:
        self._process_pending_indices()

        total_moe_ms = sum(self._moe_times) if self._moe_times else 0.0
        total_forward_ms = sum(self._forward_times) if self._forward_times else 0.0
        moe_ratio = total_moe_ms / total_forward_ms if total_forward_ms > 0 else 0.0

        return {
            "metric": "moe_profiling",
            "num_moe_layers": self.num_moe_layers,
            "moe_time_ratio": {
                "total_moe_ms": total_moe_ms,
                "total_forward_ms": total_forward_ms,
                "ratio": moe_ratio,
                "per_call_stats": self._stats(self._moe_times),
            },
            "expert_histogram": dict(sorted(self._expert_counts.items())),
            "total_expert_activations": sum(self._expert_counts.values()),
            "gemm_groups": {
                "stats": self._stats([float(g) for g in self._group_sizes]),
                "total_groups": len(self._group_sizes),
            },
        }


# ---------------------------------------------------------------------------
# Metric 5: Dequant time ratio
# ---------------------------------------------------------------------------

class DequantProfiler(MetricCollector):
    """Measure KV cache dequantization time as a ratio of total forward time.

    Wraps QuantizationStrategy.dequantize() on all cache layers.
    """

    def __init__(self):
        self._dequant_times: List[float] = []
        self._forward_times: List[float] = []
        self._patched: List[tuple] = []
        self._installed = False

    def reset(self):
        self._dequant_times = []
        self._forward_times = []

    def install(self, prompt_cache):
        """Wrap dequantize() on each cache layer's quantizer."""
        if self._installed:
            return

        profiler = self

        for i, cache in enumerate(prompt_cache):
            if cache is None:
                continue
            quantizer = getattr(cache, '_quantizer', None) or getattr(cache, 'quantizer', None)
            if quantizer is None:
                continue
            if not hasattr(quantizer, 'dequantize'):
                continue

            original_dequant = quantizer.dequantize

            def make_wrapper(orig_fn):
                def timed_dequant(*args, **kwargs):
                    start = time.perf_counter()
                    result = orig_fn(*args, **kwargs)
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    profiler._dequant_times.append(elapsed_ms)
                    return result
                return timed_dequant

            quantizer.dequantize = make_wrapper(original_dequant)
            self._patched.append((quantizer, 'dequantize', original_dequant))

        self._installed = True

    def uninstall(self):
        if not self._installed:
            return
        for obj, attr, original in self._patched:
            setattr(obj, attr, original)
        self._patched = []
        self._installed = False

    def record_forward_time(self, time_ms: float):
        """Record total forward pass time for ratio calculation."""
        self._forward_times.append(time_ms)

    def to_dict(self) -> dict:
        total_dequant_ms = sum(self._dequant_times) if self._dequant_times else 0.0
        total_forward_ms = sum(self._forward_times) if self._forward_times else 0.0
        ratio = total_dequant_ms / total_forward_ms if total_forward_ms > 0 else 0.0

        return {
            "metric": "dequant_time_ratio",
            "total_dequant_ms": total_dequant_ms,
            "total_forward_ms": total_forward_ms,
            "ratio": ratio,
            "num_dequant_calls": len(self._dequant_times),
            "per_call_stats": self._stats(self._dequant_times),
        }


# ---------------------------------------------------------------------------
# Metric 6: GPU command buffer gaps
# ---------------------------------------------------------------------------

class GPUGapAnalyzer(MetricCollector):
    """Analyze GPU idle gaps from Metal System Trace.

    This metric requires a separate xctrace capture run.
    Use parse_metal_trace.py to generate the data, then feed it here.
    """

    def __init__(self):
        self._gap_data: Optional[dict] = None

    def reset(self):
        self._gap_data = None

    def set_data(self, gap_data: dict):
        """Set pre-parsed gap data from parse_metal_trace.py."""
        self._gap_data = gap_data

    def to_dict(self) -> dict:
        if self._gap_data is None:
            return {
                "metric": "gpu_command_buffer_gaps",
                "status": "not_collected",
                "note": "Run with --metrics 6 to capture Metal trace",
            }
        return {
            "metric": "gpu_command_buffer_gaps",
            "status": "collected",
            **self._gap_data,
        }


# ---------------------------------------------------------------------------
# Metric 7: Batch scaling curves
# ---------------------------------------------------------------------------

class BatchScaler(MetricCollector):
    """Measure throughput at different batch sizes.

    Runs generation sweep externally — not a runtime hook.
    """

    def __init__(self):
        self._results: Dict[int, dict] = {}

    def reset(self):
        self._results = {}

    def run_sweep(
        self,
        model,
        tokenizer,
        prompt: str,
        batch_sizes: List[int],
        max_tokens: int = 50,
        warmup_tokens: int = 5,
    ):
        """Run batch size sweep and record tok/s for each size."""
        from mlx_lm import generate as mlx_generate

        for bs in batch_sizes:
            print(f"  Batch size {bs}...")

            # Prepare batched prompt
            messages = [prompt] * bs
            encoded = [tokenizer.encode(m) for m in messages]

            # Warmup
            for msg in messages[:1]:
                _ = mlx_generate(model, tokenizer, prompt=msg, max_tokens=warmup_tokens, verbose=False)

            # Timed run — generate for each prompt in batch sequentially
            # (MLX doesn't natively batch decode, so we time per-item)
            token_counts = []
            start = time.perf_counter()
            for msg in messages:
                result = mlx_generate(model, tokenizer, prompt=msg, max_tokens=max_tokens, verbose=False)
                token_counts.append(max_tokens)
            total_time = time.perf_counter() - start

            total_tokens = sum(token_counts)
            tok_per_sec = total_tokens / total_time if total_time > 0 else 0

            self._results[bs] = {
                "batch_size": bs,
                "total_tokens": total_tokens,
                "total_time_s": total_time,
                "tok_per_sec": tok_per_sec,
                "ms_per_token": (total_time / total_tokens * 1000) if total_tokens > 0 else 0,
            }

    def to_dict(self) -> dict:
        return {
            "metric": "batch_scaling_curves",
            "results": self._results,
            "scaling_efficiency": self._compute_efficiency(),
        }

    def _compute_efficiency(self) -> dict:
        """Compute scaling efficiency relative to batch=1."""
        if not self._results or 1 not in self._results:
            return {}
        base_tps = self._results[1]["tok_per_sec"]
        if base_tps == 0:
            return {}
        return {
            bs: {
                "speedup": data["tok_per_sec"] / base_tps,
                "efficiency": (data["tok_per_sec"] / base_tps) / bs,
            }
            for bs, data in sorted(self._results.items())
        }


# ---------------------------------------------------------------------------
# Metric 8: Context length prefill/decode split
# ---------------------------------------------------------------------------

class ContextLengthProfiler(MetricCollector):
    """Measure prefill and decode time at different context lengths.

    Runs generation sweep externally — not a runtime hook.
    """

    def __init__(self):
        self._results: Dict[int, dict] = {}

    def reset(self):
        self._results = {}

    def run_sweep(
        self,
        model,
        tokenizer,
        context_lengths: List[int],
        decode_tokens: int = 50,
        warmup_tokens: int = 5,
    ):
        """Run context length sweep, measuring prefill TTFT and decode tok/s."""
        from mlx_lm import stream_generate

        # Build a long base text for padding
        base_text = "The quick brown fox jumps over the lazy dog. " * 200

        for ctx_len in context_lengths:
            print(f"  Context length {ctx_len}...")

            # Truncate base text to approximate target context length
            tokens = tokenizer.encode(base_text)
            if len(tokens) < ctx_len:
                # Repeat to reach target
                repeats = (ctx_len // len(tokens)) + 1
                tokens = (tokens * repeats)[:ctx_len]
            else:
                tokens = tokens[:ctx_len]

            prompt_text = tokenizer.decode(tokens)

            # Warmup
            count = 0
            for _ in stream_generate(model, tokenizer, prompt=prompt_text[:200], max_tokens=warmup_tokens):
                count += 1
                if count >= warmup_tokens:
                    break

            # Timed run
            token_times = []
            gen_start = time.perf_counter()

            for resp in stream_generate(
                model, tokenizer, prompt=prompt_text, max_tokens=decode_tokens
            ):
                mx.eval(resp)
                token_times.append(time.perf_counter())

            if not token_times:
                self._results[ctx_len] = {
                    "context_length": ctx_len,
                    "status": "failed",
                }
                continue

            # First token time = prefill
            prefill_ms = (token_times[0] - gen_start) * 1000

            # Decode times = inter-token latencies
            decode_times_ms = []
            for i in range(1, len(token_times)):
                decode_times_ms.append((token_times[i] - token_times[i - 1]) * 1000)

            decode_mean_ms = statistics.mean(decode_times_ms) if decode_times_ms else 0
            decode_tps = 1000.0 / decode_mean_ms if decode_mean_ms > 0 else 0

            # Memory snapshot
            metal_mb = 0.0
            try:
                metal_mb = mx.get_active_memory() / (1024 * 1024)
            except Exception:
                pass

            self._results[ctx_len] = {
                "context_length": ctx_len,
                "actual_tokens": len(tokens),
                "prefill_ms": prefill_ms,
                "decode_tokens": len(token_times) - 1,
                "decode_mean_ms_per_token": decode_mean_ms,
                "decode_tok_per_sec": decode_tps,
                "total_time_ms": (token_times[-1] - gen_start) * 1000,
                "metal_memory_mb": metal_mb,
            }

    def to_dict(self) -> dict:
        return {
            "metric": "context_length_split",
            "results": self._results,
        }
