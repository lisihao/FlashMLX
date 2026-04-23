#!/usr/bin/env python3
"""
FlashMLX 8-Metric Profiling Framework

8 fixed metrics for undeniable performance evidence:
  1. Kernel launches per token
  2. MoE path time ratio
  3. Per-expert token histogram
  4. Grouped GEMM group count/size distribution
  5. Dequant time ratio
  6. GPU command buffer gaps (xctrace Metal trace)
  7. Batch scaling curves
  8. Context length prefill/decode split

Usage:
  # All online metrics (1-5) on a model
  python benchmarks/profile_8metrics.py --model ~/models/qwen3.5-35b-a3b \\
    --metrics 1,2,3,4,5 --tokens 50

  # Batch scaling sweep (metric 7)
  python benchmarks/profile_8metrics.py --model ~/models/qwen3.5-35b-a3b \\
    --metrics 7 --batch-sizes 1,2,4,8

  # Context length sweep (metric 8)
  python benchmarks/profile_8metrics.py --model ~/models/qwen3.5-35b-a3b \\
    --metrics 8 --context-lengths 2048,4096,8192

  # GPU trace (metric 6)
  python benchmarks/profile_8metrics.py --model ~/models/qwen3.5-35b-a3b \\
    --metrics 6 --tokens 30

  # All metrics
  python benchmarks/profile_8metrics.py --model ~/models/qwen3.5-35b-a3b \\
    --metrics all --tokens 50 --output profiling_data/baseline.json

  # Compare with previous run
  python benchmarks/profile_8metrics.py --model ~/models/qwen3.5-35b-a3b \\
    --metrics all --tokens 50 --compare profiling_data/baseline.json
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add project roots to path
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))
sys.path.insert(0, str(_project_root / "mlx-lm-source"))

import mlx.core as mx
from mlx_lm import load, stream_generate

from flashmlx.profiler.metrics import (
    KernelCounter,
    MoEProfiler,
    DequantProfiler,
    GPUGapAnalyzer,
    BatchScaler,
    ContextLengthProfiler,
)
from flashmlx.profiler.report import ProfileReport


DEFAULT_PROMPT = (
    "Explain the key differences between transformer and recurrent neural "
    "network architectures, focusing on their attention mechanisms, training "
    "efficiency, and ability to capture long-range dependencies in sequences."
)


def parse_metrics_arg(metrics_str: str) -> set:
    """Parse --metrics argument into a set of metric numbers."""
    if metrics_str.strip().lower() == "all":
        return {1, 2, 3, 4, 5, 6, 7, 8}
    return {int(m.strip()) for m in metrics_str.split(",")}


# ---------------------------------------------------------------------------
# Online metrics (1-5): collected during a single generation pass
# ---------------------------------------------------------------------------

def run_online_metrics(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int,
    metrics_to_run: set,
    kv_cache: str = "",
    kv_bits: int = 3,
) -> dict:
    """Run metrics 1-5 during a single generation pass.

    Returns dict of metric name -> MetricCollector.
    """
    collectors = {}

    # Metric 1: kernel counter
    kernel_counter = None
    if 1 in metrics_to_run:
        kernel_counter = KernelCounter()
        kernel_counter.install()
        collectors["kernel_launches"] = kernel_counter

    # Metric 2,3,4: MoE profiler
    moe_profiler = None
    if metrics_to_run & {2, 3, 4}:
        moe_profiler = MoEProfiler()
        moe_profiler.install(model)
        if moe_profiler.num_moe_layers == 0:
            print("  (No MoE layers found — metrics 2,3,4 will be empty)")
        collectors["moe"] = moe_profiler

    # Metric 5: dequant profiler — needs prompt_cache, installed after first token

    print(f"  Generating {max_tokens} tokens...")

    gen_kwargs = {}
    if kv_cache:
        gen_kwargs["kv_cache"] = kv_cache
        gen_kwargs["kv_bits"] = kv_bits

    token_count = 0
    gen_start = time.perf_counter()

    for resp in stream_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, **gen_kwargs):
        token_count += 1

        # Mark token boundary for kernel counter
        if kernel_counter is not None:
            kernel_counter.mark_token()

        # Record per-token forward time for MoE/dequant ratio
        now = time.perf_counter()
        if token_count == 1:
            prefill_ms = (now - gen_start) * 1000
            token_start = now
        else:
            token_ms = (now - token_start) * 1000
            if moe_profiler is not None:
                moe_profiler.record_forward_time(token_ms)
            token_start = now

    total_time = time.perf_counter() - gen_start

    # Uninstall hooks
    if kernel_counter is not None:
        kernel_counter.uninstall()
    if moe_profiler is not None:
        moe_profiler.uninstall()

    print(f"  Generated {token_count} tokens in {total_time:.2f}s "
          f"({token_count / total_time:.1f} tok/s)")

    return collectors


# ---------------------------------------------------------------------------
# Metric 6: GPU command buffer gaps
# ---------------------------------------------------------------------------

def run_gpu_trace(
    model_path: str,
    prompt: str,
    max_tokens: int,
) -> GPUGapAnalyzer:
    """Capture Metal System Trace and parse GPU gaps."""
    from parse_metal_trace import capture_trace, analyze_from_trace

    analyzer = GPUGapAnalyzer()

    trace_dir = _project_root / "profiling_data"
    trace_dir.mkdir(exist_ok=True)
    trace_path = str(trace_dir / "profile_8m_trace.gputrace")

    # Build the command that xctrace will launch
    venv_python = str(_project_root / "venv" / "bin" / "python")
    if not Path(venv_python).exists():
        venv_python = sys.executable

    inline_script = f'''
import mlx.core as mx
from mlx_lm import load, stream_generate

model, tokenizer = load("{model_path}")
count = 0
for resp in stream_generate(model, tokenizer, prompt="""{prompt[:200]}""", max_tokens={max_tokens}):
    count += 1
print(f"Generated {{count}} tokens")
'''

    trace_file = capture_trace(
        cmd=[venv_python, "-c", inline_script],
        output_path=trace_path,
        duration_ms=30000,
    )

    if trace_file:
        gap_data = analyze_from_trace(trace_file)
        analyzer.set_data(gap_data)
    else:
        analyzer.set_data({"status": "capture_failed"})

    return analyzer


# ---------------------------------------------------------------------------
# Metric 7: Batch scaling
# ---------------------------------------------------------------------------

def run_batch_sweep(
    model,
    tokenizer,
    prompt: str,
    batch_sizes: list,
    max_tokens: int,
) -> BatchScaler:
    """Run batch size scaling sweep."""
    scaler = BatchScaler()
    scaler.run_sweep(model, tokenizer, prompt, batch_sizes, max_tokens)
    return scaler


# ---------------------------------------------------------------------------
# Metric 8: Context length sweep
# ---------------------------------------------------------------------------

def run_context_sweep(
    model,
    tokenizer,
    context_lengths: list,
    decode_tokens: int,
) -> ContextLengthProfiler:
    """Run context length prefill/decode split sweep."""
    profiler = ContextLengthProfiler()
    profiler.run_sweep(model, tokenizer, context_lengths, decode_tokens)
    return profiler


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="FlashMLX 8-Metric Profiling Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="Model path or HF name")
    parser.add_argument("--metrics", default="all",
                        help="Comma-separated metric numbers (1-8) or 'all'")
    parser.add_argument("--tokens", type=int, default=50,
                        help="Max tokens to generate (default: 50)")
    parser.add_argument("--prompt", default=None,
                        help="Custom prompt (default: built-in)")
    parser.add_argument("--batch-sizes", default="1,2,4,8",
                        help="Batch sizes for metric 7 (default: 1,2,4,8)")
    parser.add_argument("--context-lengths", default="2048,4096,8192",
                        help="Context lengths for metric 8")
    parser.add_argument("--kv-cache", default="",
                        help="KV cache strategy (e.g., 'polar', 'scored_pq')")
    parser.add_argument("--kv-bits", type=int, default=3,
                        help="KV cache quantization bits")
    parser.add_argument("--output", default=None,
                        help="Save JSON report to this path")
    parser.add_argument("--compare", default=None,
                        help="Compare with a previous JSON report")

    args = parser.parse_args()
    metrics_to_run = parse_metrics_arg(args.metrics)
    prompt = args.prompt or DEFAULT_PROMPT

    print("=" * 64)
    print(" FlashMLX 8-Metric Profiler")
    print("=" * 64)
    print(f"  Model:   {args.model}")
    print(f"  Metrics: {sorted(metrics_to_run)}")
    print(f"  Tokens:  {args.tokens}")
    if args.kv_cache:
        print(f"  KV:      {args.kv_cache} ({args.kv_bits}-bit)")
    print()

    # Load model
    print("Loading model...")
    model, tokenizer = load(args.model)
    print(f"  Metal memory after load: {mx.get_active_memory() / 1e9:.2f} GB")
    print()

    # Initialize report
    model_name = Path(args.model).name
    report = ProfileReport(f"8metrics_{model_name}", model_name)

    # --- Online metrics 1-5 ---
    online_metrics = metrics_to_run & {1, 2, 3, 4, 5}
    if online_metrics:
        print(f"Running online metrics {sorted(online_metrics)}...")
        collectors = run_online_metrics(
            model, tokenizer, prompt, args.tokens,
            online_metrics, args.kv_cache, args.kv_bits,
        )
        for name, collector in collectors.items():
            report.add_metric(name, collector)
        print()

    # --- Metric 6: GPU trace ---
    if 6 in metrics_to_run:
        print("Running metric 6 (GPU command buffer gaps)...")
        try:
            gap_analyzer = run_gpu_trace(args.model, prompt, min(args.tokens, 30))
            report.add_metric("gpu_gaps", gap_analyzer)
        except Exception as e:
            print(f"  Metric 6 failed: {e}")
            report.add_raw("gpu_gaps", {
                "metric": "gpu_command_buffer_gaps",
                "status": "error",
                "error": str(e),
            })
        print()

    # --- Metric 7: Batch scaling ---
    if 7 in metrics_to_run:
        batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
        print(f"Running metric 7 (batch scaling: {batch_sizes})...")
        scaler = run_batch_sweep(model, tokenizer, prompt, batch_sizes, args.tokens)
        report.add_metric("batch_scaling", scaler)
        print()

    # --- Metric 8: Context length ---
    if 8 in metrics_to_run:
        ctx_lengths = [int(c) for c in args.context_lengths.split(",")]
        print(f"Running metric 8 (context lengths: {ctx_lengths})...")
        ctx_profiler = run_context_sweep(model, tokenizer, ctx_lengths, args.tokens)
        report.add_metric("context_split", ctx_profiler)
        print()

    # --- Output ---
    report.print_console()

    if args.output:
        report.save_json(args.output)

    # --- Comparison ---
    if args.compare:
        print("Loading comparison baseline...")
        baseline = ProfileReport.load_json(args.compare)
        comp = ProfileReport.compare(baseline, report)
        ProfileReport.print_comparison(comp)


if __name__ == "__main__":
    main()
