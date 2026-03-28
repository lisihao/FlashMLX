#!/usr/bin/env python3
"""
Benchmark Triple-Layer KV Cache vs Double-Layer KV Cache

Compare:
    - 2-layer: [Old Prefix (AM)] + [Recent (exact)]
    - 3-layer: [Cold (AM)] + [Warm (quant)] + [Recent (exact)]

Test metrics:
    - Memory usage
    - Quality (repetition, coherence, relevance)
    - Speed (tokens/sec)
"""

import sys
import time
import json
from pathlib import Path
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.double_layer_cache import DoubleLayerKVCache
sys.path.insert(0, str(Path(__file__).parent / "mlx-lm-source"))
from mlx_lm.models.triple_layer_cache import TripleLayerKVCache


# Agent debugging session prompt
AGENT_DEBUG_PROMPT = """You are a senior production support engineer investigating a critical issue. The application has been experiencing intermittent 500 errors over the past 24 hours. The error occurs randomly, affecting about 15% of requests. Initial investigation shows:

1. Database connections are stable
2. Memory usage is normal
3. CPU usage shows occasional spikes
4. Error logs show: "Connection pool exhausted" every few minutes
5. The issue started after deploying version 2.1.3
6. Version 2.1.3 added a new background worker for data processing

Based on the evidence, what is the root cause of the 500 errors?"""


def compute_quality_metrics(text: str) -> dict:
    """Compute quality metrics"""
    lines = text.strip().split('\n')
    if not lines:
        return {"repetition": 0.0, "coherence": 0.0, "relevance": 0.0}

    # Repetition: ratio of repeated lines
    unique_lines = set(lines)
    repetition_ratio = 1.0 - (len(unique_lines) / len(lines))

    # Coherence: ratio of lines with meaningful content (>10 chars)
    meaningful_lines = [l for l in lines if len(l.strip()) > 10]
    coherence_score = len(meaningful_lines) / len(lines) if lines else 0.0

    # Relevance: mentions problem-related keywords
    keywords = ["connection", "pool", "worker", "database", "error", "cause", "issue"]
    relevant_lines = [l for l in lines if any(k in l.lower() for k in keywords)]
    relevance_score = len(relevant_lines) / len(lines) if lines else 0.0

    return {
        "repetition": repetition_ratio,
        "coherence": coherence_score,
        "relevance": relevance_score
    }


def benchmark_double_layer(
    model,
    tokenizer,
    num_layers: int = 36,
    num_generate: int = 100
) -> dict:
    """
    Benchmark 2-layer system (baseline)
    """
    print(f"\n{'='*80}")
    print(f"Testing: 2-LAYER SYSTEM (Baseline)")
    print(f"{'='*80}")

    # Tokenize prompt
    input_ids = mx.array(tokenizer.encode(AGENT_DEBUG_PROMPT))
    prompt_len = input_ids.shape[0]
    print(f"Prompt length: {prompt_len} tokens")

    # Create 2-layer caches
    caches = []
    for layer_idx in range(num_layers):
        cache = DoubleLayerKVCache(
            memory_budget_mb=2.0,
            recent_window_size=512,
            compression_ratio=1.5,
            calibration_dir="/tmp/am_calibrations_ultra_dense",
            layer_idx=layer_idx,
            enable_compression=True,
            selection_strategy="nearest"
        )
        caches.append(cache)

    print(f"Created {len(caches)} 2-layer caches")

    # Tokenize
    tokens = tokenizer.encode(AGENT_DEBUG_PROMPT)
    y = mx.array([tokens])

    # Prefill phase: run prompt through model to populate caches
    print("\nPrefilling caches...")
    logits = model(y[:, :-1], cache=caches)
    mx.eval(logits)

    # Generate phase
    print(f"Generating {num_generate} tokens...")
    y = mx.array([[tokens[-1]]])
    start_time = time.time()

    generated_tokens = []
    for i in range(num_generate):
        logits = model(y, cache=caches)
        y = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        mx.eval(y)

        token_id = y[0, 0].item()
        generated_tokens.append(token_id)

        if token_id == tokenizer.eos_token_id:
            break

    gen_time = time.time() - start_time

    # Decode generated text
    generated_text = tokenizer.decode(generated_tokens)

    # Compute memory usage
    total_memory_mb = 0
    old_prefix_mb = 0
    recent_mb = 0

    for cache in caches:
        if hasattr(cache, 'get_memory_usage'):
            mem = cache.get_memory_usage()
            total_memory_mb += mem['total_mb']
            old_prefix_mb += mem.get('old_prefix_mb', 0)
            recent_mb += mem.get('recent_mb', 0)

    # Compute quality
    quality = compute_quality_metrics(generated_text)

    # Tokens per second
    tps = num_generate / gen_time if gen_time > 0 else 0

    results = {
        "system": "2-layer",
        "prompt_len": prompt_len,
        "generated_tokens": num_generate,
        "generation_time_sec": gen_time,
        "tokens_per_second": tps,
        "memory": {
            "total_mb": total_memory_mb,
            "old_prefix_mb": old_prefix_mb,
            "recent_mb": recent_mb
        },
        "quality": quality,
        "output_preview": generated_text[:200]
    }

    # Print results
    print(f"\n2-LAYER Results:")
    print(f"  Memory: {total_memory_mb:.1f} MB (Old: {old_prefix_mb:.1f} MB, Recent: {recent_mb:.1f} MB)")
    print(f"  Speed: {tps:.2f} tok/s")
    print(f"  Quality:")
    print(f"    Repetition: {quality['repetition']*100:.2f}%")
    print(f"    Coherence: {quality['coherence']*100:.2f}%")
    print(f"    Relevance: {quality['relevance']*100:.2f}%")
    print(f"\n  Output (first 200 chars):")
    print(f"  {generated_text[:200]}")

    # Save full output
    output_file = "/tmp/triple_2layer_output.txt"
    with open(output_file, "w") as f:
        f.write(generated_text)
    print(f"\n  Full output saved to: {output_file}")

    return results


def benchmark_triple_layer(
    model,
    tokenizer,
    num_layers: int = 36,
    num_generate: int = 100
) -> dict:
    """
    Benchmark 3-layer system

    L0 (Recent): 0-512 tokens, exact
    L1 (Warm): 512-2048 tokens, KV quant (~2x)
    L2 (Cold): 2048+ tokens, AM (~1.5x)
    """
    print(f"\n{'='*80}")
    print(f"Testing: 3-LAYER SYSTEM")
    print(f"{'='*80}")

    # Tokenize prompt
    input_ids = mx.array(tokenizer.encode(AGENT_DEBUG_PROMPT))
    prompt_len = input_ids.shape[0]
    print(f"Prompt length: {prompt_len} tokens")

    # Create 3-layer caches
    caches = []
    for layer_idx in range(num_layers):
        cache = TripleLayerKVCache(
            recent_size=512,       # L0: 0-512
            warm_size=1536,        # L1: 512-2048 (2048 - 512 = 1536)
            calibration_dir="/tmp/am_calibrations_ultra_dense",
            layer_idx=layer_idx,
            compression_ratio=1.5,
            selection_strategy="nearest",
            quant_bits=4,
            enable_warm_quant=True,
            enable_cold_am=True
        )
        caches.append(cache)

    print(f"Created {len(caches)} 3-layer caches")
    print(f"  L0 (Recent): 0-512 tokens (exact)")
    print(f"  L1 (Warm): 512-2048 tokens (Q4 quant, ~2x compression)")
    print(f"  L2 (Cold): 2048+ tokens (AM R1.5, ~1.5x compression)")

    # Tokenize
    tokens = tokenizer.encode(AGENT_DEBUG_PROMPT)
    y = mx.array([tokens])

    # Prefill phase: run prompt through model to populate caches
    print("\nPrefilling caches...")
    logits = model(y[:, :-1], cache=caches)
    mx.eval(logits)

    # Generate phase
    print(f"Generating {num_generate} tokens...")
    y = mx.array([[tokens[-1]]])
    start_time = time.time()

    generated_tokens = []
    for i in range(num_generate):
        logits = model(y, cache=caches)
        y = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        mx.eval(y)

        token_id = y[0, 0].item()
        generated_tokens.append(token_id)

        if token_id == tokenizer.eos_token_id:
            break

    gen_time = time.time() - start_time

    # Decode generated text
    generated_text = tokenizer.decode(generated_tokens)

    # Compute memory usage
    total_memory_mb = 0
    cold_mb = 0
    warm_mb = 0
    recent_mb = 0

    for cache in caches:
        if hasattr(cache, 'get_memory_usage'):
            mem = cache.get_memory_usage()
            total_memory_mb += mem['total_mb']
            cold_mb += mem.get('cold_mb', 0)
            warm_mb += mem.get('warm_mb', 0)
            recent_mb += mem.get('recent_mb', 0)

    # Compute quality
    quality = compute_quality_metrics(generated_text)

    # Tokens per second
    tps = num_generate / gen_time if gen_time > 0 else 0

    results = {
        "system": "3-layer",
        "prompt_len": prompt_len,
        "generated_tokens": num_generate,
        "generation_time_sec": gen_time,
        "tokens_per_second": tps,
        "memory": {
            "total_mb": total_memory_mb,
            "cold_mb": cold_mb,
            "warm_mb": warm_mb,
            "recent_mb": recent_mb
        },
        "quality": quality,
        "output_preview": generated_text[:200]
    }

    # Print results
    print(f"\n3-LAYER Results:")
    print(f"  Memory: {total_memory_mb:.1f} MB (Cold: {cold_mb:.1f} MB, Warm: {warm_mb:.1f} MB, Recent: {recent_mb:.1f} MB)")
    print(f"  Speed: {tps:.2f} tok/s")
    print(f"  Quality:")
    print(f"    Repetition: {quality['repetition']*100:.2f}%")
    print(f"    Coherence: {quality['coherence']*100:.2f}%")
    print(f"    Relevance: {quality['relevance']*100:.2f}%")
    print(f"\n  Output (first 200 chars):")
    print(f"  {generated_text[:200]}")

    # Save full output
    output_file = "/tmp/triple_3layer_output.txt"
    with open(output_file, "w") as f:
        f.write(generated_text)
    print(f"\n  Full output saved to: {output_file}")

    return results


def main():
    print("Triple-Layer KV Cache Benchmark")
    print("=" * 80)

    # Load model
    print("\nLoading Qwen3-8B model...")
    model_path = "/Volumes/toshiba/models/qwen3-8b-mlx"
    model, tokenizer = load(model_path)
    num_layers = len(model.model.layers)
    print(f"Model loaded: {num_layers} layers")

    # Test both systems
    all_results = []

    # Test 2-layer (baseline)
    result_2layer = benchmark_double_layer(
        model=model,
        tokenizer=tokenizer,
        num_layers=num_layers,
        num_generate=100
    )
    all_results.append(result_2layer)

    # Cool down
    print("\nCooling down for 5 seconds...")
    time.sleep(5)

    # Test 3-layer
    result_3layer = benchmark_triple_layer(
        model=model,
        tokenizer=tokenizer,
        num_layers=num_layers,
        num_generate=100
    )
    all_results.append(result_3layer)

    # Summary comparison
    print(f"\n{'='*80}")
    print("SUMMARY COMPARISON")
    print(f"{'='*80}")

    baseline = result_2layer

    print(f"\n{'System':<15} {'Memory (MB)':<15} {'Savings':<12} {'Quality':<30} {'Status':<10}")
    print(f"{'-'*80}")

    for result in all_results:
        mem = result['memory']['total_mb']
        savings = (baseline['memory']['total_mb'] - mem) / baseline['memory']['total_mb'] * 100 if baseline['memory']['total_mb'] > 0 else 0

        quality = result['quality']
        rep = quality['repetition'] * 100
        coh = quality['coherence'] * 100
        rel = quality['relevance'] * 100

        # Quality status
        if rep > 20 or coh < 50 or rel < 30:
            status = "❌ BAD"
        elif rep > 10 or coh < 60 or rel < 40:
            status = "⚠️  FAIR"
        else:
            status = "✅ GOOD"

        print(f"{result['system']:<15} {mem:<15.1f} {f'+{savings:.1f}%':<12} R:{rep:.0f}% C:{coh:.0f}% Rel:{rel:.0f}%{'':<5} {status:<10}")

    # Memory breakdown
    print(f"\n{'='*80}")
    print("MEMORY BREAKDOWN")
    print(f"{'='*80}")

    print("\n2-Layer:")
    print(f"  Old Prefix: {result_2layer['memory']['old_prefix_mb']:.1f} MB")
    print(f"  Recent:     {result_2layer['memory']['recent_mb']:.1f} MB")
    print(f"  Total:      {result_2layer['memory']['total_mb']:.1f} MB")

    print("\n3-Layer:")
    print(f"  Cold:   {result_3layer['memory']['cold_mb']:.1f} MB")
    print(f"  Warm:   {result_3layer['memory']['warm_mb']:.1f} MB")
    print(f"  Recent: {result_3layer['memory']['recent_mb']:.1f} MB")
    print(f"  Total:  {result_3layer['memory']['total_mb']:.1f} MB")

    # Save results
    results_file = "/tmp/triple_layer_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDetailed results saved to: {results_file}")

    print("\n" + "="*80)
    print("Benchmark complete!")
    print("="*80)

    # Final verdict
    if baseline['memory']['total_mb'] > 0:
        mem_savings = (baseline['memory']['total_mb'] - result_3layer['memory']['total_mb']) / baseline['memory']['total_mb'] * 100
    else:
        mem_savings = 0.0

    quality_3layer = result_3layer['quality']
    quality_2layer = result_2layer['quality']

    print(f"\n🎯 VERDICT:")
    print(f"  Memory savings: +{mem_savings:.1f}%")
    print(f"  Quality delta:")
    print(f"    Repetition: {(quality_3layer['repetition'] - quality_2layer['repetition'])*100:+.1f} pp")
    print(f"    Coherence: {(quality_3layer['coherence'] - quality_2layer['coherence'])*100:+.1f} pp")
    print(f"    Relevance: {(quality_3layer['relevance'] - quality_2layer['relevance'])*100:+.1f} pp")

    if mem_savings > 10 and quality_3layer['coherence'] >= quality_2layer['coherence'] * 0.95:
        print(f"\n  ✅ 3-layer system SUCCESS: significant memory savings with acceptable quality")
    elif mem_savings > 0 and quality_3layer['coherence'] >= quality_2layer['coherence']:
        print(f"\n  ✅ 3-layer system PASS: memory savings with maintained quality")
    else:
        print(f"\n  ⚠️  3-layer system needs tuning")


if __name__ == "__main__":
    main()
