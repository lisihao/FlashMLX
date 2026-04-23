#!/usr/bin/env python3
"""
TEP Phase A — Routing Consistency Diagnostics (SRP / SCH)

Implements two metrics from "Not All Models Suit Expert Offloading" (arXiv 2505.16056):

  SRP (Segment Routing Predictability):
    For each segment of S consecutive tokens, find the fixed expert set that
    maximizes F1 score (precision × recall vs actual per-token routing).
    High SRP → experts are stable within segments → good for caching/offloading.

  SCH (Segment Cache Hit Rate):
    Simulate an oracle LRU cache with capacity = rho × top_k experts per layer.
    Count cache hits across all tokens. High SCH → offloading is effective.

Usage:
    python3 benchmarks/bench_routing_consistency.py [--model PATH] [--tokens N]
"""

import argparse
import gc
import json
import os
import sys
import time

import numpy as np

MLX_LM_SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mlx-lm-source")
sys.path.insert(0, MLX_LM_SOURCE)

import mlx.core as mx
from mlx_lm import load


# ============================================================================
# Routing trace collection
# ============================================================================

def collect_routing_trace(model, tokenizer, prompt_text: str,
                          max_tokens: int = 256) -> dict:
    """Run inference and capture per-layer per-token expert routing decisions.

    Returns dict with:
        traces: list of (layer_idx, token_pos, [expert_ids]) tuples
        num_layers: int
        num_experts: int
        top_k: int
        total_tokens: int
    """
    from mlx_lm.generate import stream_generate

    # Find MoE layers and monkey-patch to capture routing
    inner = model
    for attr in ("model", "model", "language_model", "model"):
        if hasattr(inner, attr):
            inner = getattr(inner, attr)

    moe_layers = []
    for i, layer in enumerate(inner.layers):
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate"):
            moe_layers.append((i, layer.mlp))

    if not moe_layers:
        raise RuntimeError("No MoE layers found in model")

    # Storage for routing decisions
    traces = []
    token_counter = [0]
    num_experts = moe_layers[0][1].gate.weight.shape[0]
    top_k = getattr(moe_layers[0][1], "top_k", 8)

    original_calls = {}

    def make_hook(layer_idx, moe_block):
        original = moe_block.__class__.__call__

        def hooked_call(self_inner, x):
            # Capture gate logits before dispatching
            gates = self_inner.gate(x)
            gates_soft = mx.softmax(gates, axis=-1, precise=True)
            k = getattr(self_inner, "top_k", 8)
            inds = mx.argpartition(gates_soft, kth=-k, axis=-1)[..., -k:]

            # Record indices (materialize to CPU)
            inds_np = np.array(inds.reshape(-1, k).tolist())
            for t in range(inds_np.shape[0]):
                traces.append((
                    layer_idx,
                    token_counter[0] + t,
                    sorted(inds_np[t].tolist()),
                ))

            # Call original
            return original(self_inner, x)

        return hooked_call, original

    # Install hooks
    for layer_idx, moe_block in moe_layers:
        hooked, orig = make_hook(layer_idx, moe_block)
        original_calls[layer_idx] = (moe_block, orig)
        moe_block.__class__.__call__ = hooked

    # Run generation
    messages = [{"role": "user", "content": prompt_text}]
    formatted = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    for response in stream_generate(model, tokenizer, formatted,
                                     max_tokens=max_tokens):
        token_counter[0] += 1

    # Restore original __call__
    for layer_idx, (moe_block, orig) in original_calls.items():
        moe_block.__class__.__call__ = orig

    return {
        "traces": traces,
        "num_layers": len(moe_layers),
        "layer_indices": [idx for idx, _ in moe_layers],
        "num_experts": num_experts,
        "top_k": top_k,
        "total_tokens": token_counter[0],
    }


# ============================================================================
# SRP: Segment Routing Predictability
# ============================================================================

def compute_srp(traces: list, layer_indices: list, num_experts: int,
                top_k: int, total_tokens: int,
                segment_length: int = 16) -> dict:
    """Compute SRP for each layer.

    For each segment of `segment_length` tokens, find the fixed expert set
    (of size top_k) that maximizes mean F1 against actual per-token routing.
    """
    # Organize traces by (layer, token)
    routing = {}  # (layer_idx, token_pos) -> set(expert_ids)
    for layer_idx, token_pos, expert_ids in traces:
        routing[(layer_idx, token_pos)] = set(expert_ids)

    results = {}
    for layer_idx in layer_indices:
        segment_f1s = []
        num_segments = total_tokens // segment_length

        for seg in range(num_segments):
            start = seg * segment_length
            end = start + segment_length

            # Count expert frequency in this segment
            freq = np.zeros(num_experts, dtype=np.int32)
            token_sets = []
            for t in range(start, end):
                key = (layer_idx, t)
                if key in routing:
                    s = routing[key]
                    token_sets.append(s)
                    for eid in s:
                        freq[eid] += 1

            if not token_sets:
                continue

            # Best fixed set = top-k most frequent experts in segment
            best_set = set(np.argsort(freq)[-top_k:].tolist())

            # Compute F1 for each token in segment
            f1s = []
            for s in token_sets:
                tp = len(s & best_set)
                fp = len(best_set - s)
                fn = len(s - best_set)
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                f1s.append(f1)

            segment_f1s.append(np.mean(f1s))

        srp = float(np.mean(segment_f1s)) if segment_f1s else 0.0
        results[layer_idx] = {
            "srp": srp,
            "num_segments": len(segment_f1s),
        }

    overall_srp = float(np.mean([r["srp"] for r in results.values()])) if results else 0.0
    return {
        "overall_srp": overall_srp,
        "segment_length": segment_length,
        "per_layer": results,
        "tier": _srp_tier(overall_srp),
    }


def _srp_tier(srp: float) -> str:
    if srp > 0.50:
        return "Excellent"
    elif srp > 0.45:
        return "Good"
    elif srp > 0.35:
        return "Moderate"
    else:
        return "Poor"


# ============================================================================
# SCH: Segment Cache Hit Rate
# ============================================================================

def compute_sch(traces: list, layer_indices: list, num_experts: int,
                top_k: int, total_tokens: int,
                cache_ratios: list = None) -> dict:
    """Compute SCH for each layer at different cache sizes.

    Simulates an oracle LRU cache with capacity = rho × top_k experts.
    """
    if cache_ratios is None:
        cache_ratios = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]

    # Organize traces by (layer, token) ordered by token
    layer_sequences = {}  # layer_idx -> [(token_pos, [expert_ids])]
    for layer_idx, token_pos, expert_ids in traces:
        if layer_idx not in layer_sequences:
            layer_sequences[layer_idx] = []
        layer_sequences[layer_idx].append((token_pos, expert_ids))

    for layer_idx in layer_sequences:
        layer_sequences[layer_idx].sort(key=lambda x: x[0])

    results = {}
    for rho in cache_ratios:
        cache_size = max(1, int(rho * top_k))
        layer_hits = {}

        for layer_idx in layer_indices:
            seq = layer_sequences.get(layer_idx, [])
            if not seq:
                continue

            # Simulate LRU cache
            cache = []  # ordered: most recent at end
            hits = 0
            total = 0

            for _, expert_ids in seq:
                for eid in expert_ids:
                    total += 1
                    if eid in cache:
                        hits += 1
                        cache.remove(eid)
                        cache.append(eid)
                    else:
                        cache.append(eid)
                        if len(cache) > cache_size:
                            cache.pop(0)  # evict LRU

            layer_hits[layer_idx] = hits / total if total > 0 else 1.0

        overall = float(np.mean(list(layer_hits.values()))) if layer_hits else 0.0
        results[rho] = {
            "cache_size": cache_size,
            "overall_sch": overall,
            "per_layer": layer_hits,
        }

    return {
        "cache_ratios": {
            rho: r["overall_sch"] for rho, r in results.items()
        },
        "top_k": top_k,
        "detail": results,
        "optimal_rho": _find_inflection(results),
    }


def _find_inflection(results: dict) -> float:
    """Find the cache ratio where diminishing returns kick in."""
    ratios = sorted(results.keys())
    if len(ratios) < 2:
        return ratios[0] if ratios else 2.0

    schs = [results[r]["overall_sch"] for r in ratios]
    # Find where marginal gain drops below 2%
    for i in range(1, len(ratios)):
        if schs[i] - schs[i - 1] < 0.02:
            return ratios[i - 1]
    return ratios[-1]


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="TEP Phase A — Routing Consistency (SRP/SCH)")
    parser.add_argument("--model", default="/Volumes/toshiba/models/qwen3.5-35b-mlx",
                        help="Path to MoE model")
    parser.add_argument("--tokens", type=int, default=128,
                        help="Max tokens to generate for trace collection")
    parser.add_argument("--segment", type=int, default=16,
                        help="Segment length for SRP computation")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: stdout)")
    args = parser.parse_args()

    print("=" * 70)
    print("  TEP Phase A — Routing Consistency Diagnostics (SRP / SCH)")
    print(f"  Model: {args.model}")
    print(f"  Tokens: {args.tokens} | Segment: {args.segment}")
    print("=" * 70)

    # Load model
    print("\n  Loading model...")
    t0 = time.perf_counter()
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())
    load_time = time.perf_counter() - t0
    mem = mx.metal.get_active_memory() / 1024 / 1024 / 1024
    print(f"  Loaded in {load_time:.1f}s, {mem:.2f} GB")

    # Collect routing trace
    prompt = (
        "Write a detailed explanation of how transformer attention mechanisms "
        "work, including multi-head attention, scaled dot-product attention, "
        "and their computational complexity. Then compare with SSM architectures."
    )
    print(f"\n  Collecting routing trace ({args.tokens} tokens)...")
    t0 = time.perf_counter()
    trace_data = collect_routing_trace(model, tokenizer, prompt,
                                        max_tokens=args.tokens)
    trace_time = time.perf_counter() - t0
    print(f"  Collected {len(trace_data['traces'])} routing decisions "
          f"across {trace_data['num_layers']} MoE layers "
          f"in {trace_time:.1f}s")
    print(f"  Experts: {trace_data['num_experts']}, "
          f"Top-k: {trace_data['top_k']}, "
          f"Tokens: {trace_data['total_tokens']}")

    # Compute SRP
    print(f"\n  Computing SRP (segment={args.segment})...")
    srp_result = compute_srp(
        trace_data["traces"],
        trace_data["layer_indices"],
        trace_data["num_experts"],
        trace_data["top_k"],
        trace_data["total_tokens"],
        segment_length=args.segment,
    )
    print(f"  Overall SRP: {srp_result['overall_srp']:.4f} → {srp_result['tier']}")

    # Compute SCH
    print(f"\n  Computing SCH (LRU cache simulation)...")
    sch_result = compute_sch(
        trace_data["traces"],
        trace_data["layer_indices"],
        trace_data["num_experts"],
        trace_data["top_k"],
        trace_data["total_tokens"],
    )
    print(f"  SCH by cache ratio:")
    for rho, sch in sorted(sch_result["cache_ratios"].items()):
        marker = " ← optimal" if rho == sch_result["optimal_rho"] else ""
        print(f"    rho={rho:.1f}x ({int(rho * trace_data['top_k'])} experts): "
              f"hit rate = {sch:.2%}{marker}")

    # Summary
    report = {
        "model": args.model,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "tokens": args.tokens,
            "segment_length": args.segment,
        },
        "trace": {
            "num_layers": trace_data["num_layers"],
            "num_experts": trace_data["num_experts"],
            "top_k": trace_data["top_k"],
            "total_tokens": trace_data["total_tokens"],
            "total_routing_decisions": len(trace_data["traces"]),
        },
        "srp": srp_result,
        "sch": sch_result,
    }

    print(f"\n{'=' * 70}")
    print(f"  VERDICT: {srp_result['tier']} routing consistency")
    print(f"  SRP = {srp_result['overall_srp']:.4f}")
    print(f"  SCH@2x = {sch_result['cache_ratios'].get(2.0, 0):.2%}")
    print(f"  Optimal cache = {sch_result['optimal_rho']:.1f}x top_k")
    if srp_result["tier"] in ("Excellent", "Good"):
        print(f"  → This model is a GOOD candidate for expert offloading + predictive prefetch")
    else:
        print(f"  → This model may NOT benefit much from expert offloading")
    print(f"{'=' * 70}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n  Report saved to {args.output}")
    else:
        # Also save to .solar/
        solar_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".solar")
        os.makedirs(solar_dir, exist_ok=True)
        out_path = os.path.join(solar_dir, "tep-routing-consistency.json")
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n  Report saved to {out_path}")

    del model, tokenizer
    gc.collect()

    return report


if __name__ == "__main__":
    main()
