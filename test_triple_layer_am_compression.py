#!/usr/bin/env python3
"""
Three-Layer KV Cache with AM Compression Test

Architecture:
    L0 (Recent):  0-512 tokens, exact storage
    L1 (Warm):    512-2048 tokens, no quantization (quantization disabled due to quality issues)
    L2 (Cold):    2048+ tokens, AM compression (R2.0 or R3.0)

Test Configurations:
    1. Baseline: TripleLayer without Cold compression (enable_cold_am=False)
    2. R2.0: TripleLayer with Cold R2.0 compression
    3. R3.0: TripleLayer with Cold R3.0 compression

Metrics:
    - Memory usage (MB) per layer (Recent, Warm, Cold)
    - Total memory vs Baseline
    - Output quality (repetition, coherence, relevance)
    - Speed (prefill, generate tokens/sec)
"""

import sys
sys.path.insert(0, '/Users/lisihao/FlashMLX/mlx-lm-source')

import mlx.core as mx
from mlx_lm import load
import argparse
from datetime import datetime
import time
import json

from mlx_lm.models.triple_layer_cache import TripleLayerKVCache

def log(msg, end='\n'):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True, end=end)

# Long test prompt (repeated scenario to reach ~3000+ tokens)
BASE_SCENARIO = """You are investigating a critical production issue. The application has intermittent 500 errors affecting 15% of requests since version 2.1.3 deployment.

Investigation findings:
- Error: "Connection pool exhausted"
- Database pool: 10 connections + 5 overflow
- Background worker added in 2.1.3
- Worker processes 50-100 batches/minute
- Each batch creates a connection

Code analysis shows:
```python
def sync_to_external_api(record):
    response = requests.post(api_url, json=record.to_dict())
    if response.status_code != 200:
        db.session.execute(log_query)  # Creates new connection!
```

The system has been experiencing these issues for several days now. The database connection pool is configured with 10 base connections and 5 overflow connections. The error occurs when all 15 connections are in use.

Looking at the logs, we can see:
- Peak request rate: 200 req/sec
- Average request duration: 50ms
- Background worker batch processing: 50-100 batches/minute
- Each batch processes 10-20 records
- Error spike correlates with worker activity

Additional context:
- The worker was added to handle data synchronization with an external API
- The synchronization happens in real-time for each record update
- The external API has variable response times (100ms-2000ms)
- Failed API calls trigger retry logic with exponential backoff
- Error logging uses a separate database session

What is the root cause and how should we fix it?"""

# Repeat 12 times to get ~3000+ tokens
LONG_PROMPT = "\n\n".join([
    f"--- Investigation Report #{i+1} ---\n{BASE_SCENARIO}"
    for i in range(12)
]) + "\n\nProvide comprehensive analysis of all investigation reports. Identify the common root cause and propose a solution."


def compute_quality(text: str) -> dict:
    """Compute quality metrics for generated text."""
    lines = [l for l in text.strip().split('\n') if l.strip()]
    if not lines:
        return {"repetition": 0.0, "coherence": 0.0, "relevance": 0.0}

    # Repetition: ratio of duplicate lines
    unique = set(lines)
    rep = 1.0 - (len(unique) / len(lines))

    # Coherence: ratio of meaningful lines (>10 chars)
    meaningful = [l for l in lines if len(l.strip()) > 10]
    coh = len(meaningful) / len(lines) if lines else 0.0

    # Relevance: keyword matching
    keywords = ["connection", "pool", "worker", "error", "leak", "database", "scenario"]
    relevant = [l for l in lines if any(k in l.lower() for k in keywords)]
    rel = len(relevant) / len(lines) if lines else 0.0

    return {"repetition": rep, "coherence": coh, "relevance": rel}


def benchmark_configuration(
    name: str,
    model,
    tokenizer,
    cache_config: dict,
    num_generate: int = 100
):
    """
    Benchmark a specific triple-layer cache configuration.

    Returns
    -------
    dict : Performance metrics
    """
    log(f"\n{'='*80}")
    log(f"Benchmarking: {name}")
    log(f"{'='*80}")

    # Tokenize
    tokens = tokenizer.encode(LONG_PROMPT)
    prompt_len = len(tokens)
    log(f"Prompt length: {prompt_len} tokens")

    # Create caches
    num_layers = len(model.model.layers)
    cache_list = [
        TripleLayerKVCache(layer_idx=i, **cache_config)
        for i in range(num_layers)
    ]

    # Prefill
    log("Step 1: Prefill...")
    y = mx.array([tokens])
    mx.eval(y)
    mx.clear_cache()

    prefill_start = time.time()
    logits = model(y[:, :-1], cache=cache_list)
    mx.eval(logits)
    prefill_time = time.time() - prefill_start

    prefill_tps = prompt_len / prefill_time
    log(f"  Prefill: {prefill_tps:.2f} tokens/sec ({prefill_time:.3f}s)")

    # Generate
    log(f"Step 2: Generate {num_generate} tokens...")
    y = mx.array([[tokens[-1]]])

    generate_start = time.time()
    generated_tokens = []

    for i in range(num_generate):
        logits = model(y, cache=cache_list)
        y = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        mx.eval(y)

        token_id = y[0, 0].item()
        generated_tokens.append(token_id)

        if token_id == tokenizer.eos_token_id:
            break

    generate_time = time.time() - generate_start
    generate_tps = len(generated_tokens) / generate_time

    log(f"  Generated {len(generated_tokens)} tokens")
    log(f"  TG speed: {generate_tps:.2f} tokens/sec ({generate_time:.3f}s)")

    # Memory usage
    total_memory_mb = 0
    layer_memory = {}

    for cache in cache_list:
        if hasattr(cache, 'get_memory_usage'):
            mem = cache.get_memory_usage()
            total_memory_mb += mem.get('total_mb', 0)
            for k, v in mem.items():
                if k.endswith('_mb') or k.endswith('_tokens'):
                    layer_memory[k] = layer_memory.get(k, 0) + v

    log(f"  Memory: {total_memory_mb:.1f} MB")
    if 'cold_mb' in layer_memory:
        log(f"    Cold:   {layer_memory.get('cold_mb', 0):.1f} MB ({layer_memory.get('cold_tokens', 0)} tokens)")
        log(f"    Warm:   {layer_memory.get('warm_mb', 0):.1f} MB ({layer_memory.get('warm_tokens', 0)} tokens)")
        log(f"    Recent: {layer_memory.get('recent_mb', 0):.1f} MB ({layer_memory.get('recent_tokens', 0)} tokens)")

    # Output text
    output_text = tokenizer.decode(generated_tokens)
    log(f"  Output preview: {output_text[:100]}...")

    # Quality metrics
    quality = compute_quality(output_text)
    log(f"  Quality: R:{quality['repetition']*100:.0f}% C:{quality['coherence']*100:.0f}% Rel:{quality['relevance']*100:.0f}%")

    # Quality status
    if quality['repetition'] > 0.2:
        status = "❌ HIGH REPETITION"
    elif quality['coherence'] < 0.5:
        status = "❌ LOW COHERENCE"
    elif quality['relevance'] < 0.3:
        status = "⚠️  LOW RELEVANCE"
    else:
        status = "✅ GOOD"
    log(f"  Status: {status}")

    return {
        'name': name,
        'prompt_tokens': prompt_len,
        'generated_tokens': len(generated_tokens),
        'prefill_tps': prefill_tps,
        'generate_tps': generate_tps,
        'total_memory_mb': total_memory_mb,
        'layer_memory': layer_memory,
        'output_text': output_text,
        'quality': quality,
        'status': status
    }


def main():
    parser = argparse.ArgumentParser(description='Three-Layer KV Cache AM Compression Test')
    parser.add_argument('--model-path', default='/Volumes/toshiba/models/qwen3-8b-mlx',
                        help='Path to model')
    parser.add_argument('--calibration-dir', default='/tmp/am_calibrations_ultra_dense',
                        help='Calibration directory')
    parser.add_argument('--compression-ratio', type=float, default=2.0,
                        help='AM compression ratio (2.0 or 3.0)')
    parser.add_argument('--num-generate', type=int, default=100,
                        help='Number of tokens to generate')
    parser.add_argument('--recent-size', type=int, default=512,
                        help='Recent layer size')
    parser.add_argument('--warm-size', type=int, default=1536,
                        help='Warm layer size')
    args = parser.parse_args()

    log("=" * 80)
    log("🔬 Three-Layer KV Cache with AM Compression Test")
    log("=" * 80)
    log(f"Model: {args.model_path}")
    log(f"Calibration: {args.calibration_dir}")
    log(f"Compression ratio: R{args.compression_ratio}")
    log(f"Recent size: {args.recent_size}, Warm size: {args.warm_size}")

    # Load model
    log("\nLoading model...")
    model, tokenizer = load(args.model_path)
    num_layers = len(model.model.layers)
    log(f"✓ Model loaded: {num_layers} layers")

    # Results storage
    results = []

    # ========================================================================
    # Test 1: Baseline (No Cold Compression)
    # ========================================================================
    result_baseline = benchmark_configuration(
        name="Baseline (No Cold Compression)",
        model=model,
        tokenizer=tokenizer,
        cache_config={
            'recent_size': args.recent_size,
            'warm_size': args.warm_size,
            'calibration_dir': args.calibration_dir,
            'compression_ratio': args.compression_ratio,
            'selection_strategy': 'nearest',
            'enable_warm_quant': False,  # Disabled due to quality issues
            'enable_cold_am': False       # Baseline: no compression
        },
        num_generate=args.num_generate
    )
    results.append(result_baseline)

    time.sleep(3)  # Cool down

    # ========================================================================
    # Test 2: With Cold AM Compression (R2.0 or R3.0)
    # ========================================================================
    result_compressed = benchmark_configuration(
        name=f"With Cold AM Compression (R{args.compression_ratio})",
        model=model,
        tokenizer=tokenizer,
        cache_config={
            'recent_size': args.recent_size,
            'warm_size': args.warm_size,
            'calibration_dir': args.calibration_dir,
            'compression_ratio': args.compression_ratio,
            'selection_strategy': 'nearest',
            'enable_warm_quant': False,  # Disabled due to quality issues
            'enable_cold_am': True        # Enable AM compression
        },
        num_generate=args.num_generate
    )
    results.append(result_compressed)

    # ========================================================================
    # Summary
    # ========================================================================
    log("\n\n" + "=" * 80)
    log(f"📊 Summary: R{args.compression_ratio} AM Compression Impact")
    log("=" * 80)

    baseline = results[0]
    compressed = results[1]

    # Memory comparison
    log(f"\n{'Configuration':<40} {'Total Memory':<20} {'Cold Memory':<20}")
    log("-" * 80)
    log(f"{baseline['name']:<40} {baseline['total_memory_mb']:>10.1f} MB      {baseline['layer_memory'].get('cold_mb', 0):>10.1f} MB")
    log(f"{compressed['name']:<40} {compressed['total_memory_mb']:>10.1f} MB      {compressed['layer_memory'].get('cold_mb', 0):>10.1f} MB")

    # Memory savings
    if baseline['total_memory_mb'] > 0:
        memory_reduction = (baseline['total_memory_mb'] - compressed['total_memory_mb']) / baseline['total_memory_mb'] * 100
        log(f"\nMemory savings: {memory_reduction:+.1f}%")

        cold_baseline = baseline['layer_memory'].get('cold_mb', 0)
        cold_compressed = compressed['layer_memory'].get('cold_mb', 0)
        if cold_baseline > 0:
            cold_reduction = (cold_baseline - cold_compressed) / cold_baseline * 100
            log(f"Cold layer reduction: {cold_reduction:+.1f}%")

    # Quality comparison
    log("\n" + "=" * 80)
    log("Quality Comparison")
    log("=" * 80)
    log(f"\n{'Configuration':<40} {'Repetition':<15} {'Coherence':<15} {'Relevance':<15} {'Status':<15}")
    log("-" * 100)
    for result in results:
        q = result['quality']
        log(f"{result['name']:<40} {q['repetition']*100:>10.0f}%     {q['coherence']*100:>10.0f}%     {q['relevance']*100:>10.0f}%     {result['status']:<15}")

    # Speed comparison
    log("\n" + "=" * 80)
    log("Speed Comparison")
    log("=" * 80)
    log(f"\n{'Configuration':<40} {'Prefill TPS':<20} {'Generate TPS':<20}")
    log("-" * 80)
    for result in results:
        log(f"{result['name']:<40} {result['prefill_tps']:>15.2f}      {result['generate_tps']:>15.2f}")

    # Output samples
    log("\n" + "=" * 80)
    log("Output Samples (First 150 chars)")
    log("=" * 80)
    for result in results:
        log(f"\n{result['name']}:")
        log(f"  {result['output_text'][:150]}...")

    # Save results
    output_file = f"/tmp/triple_layer_r{args.compression_ratio}_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    log(f"\n✓ Results saved to: {output_file}")

    log("\n" + "=" * 80)


if __name__ == '__main__':
    main()
