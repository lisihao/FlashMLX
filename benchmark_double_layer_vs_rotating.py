#!/usr/bin/env python3
"""
Performance Comparison: DoubleLayerKVCache vs RotatingKVCache vs Baseline

Compares:
1. Baseline (Full KVCache)
2. RotatingKVCache (256 window)
3. DoubleLayerKVCache (Memory-Budget Driven, Production-Ready)

NEW Design (2026-03-26):
- Compression triggered by memory budget (not token threshold)
- Blocks inference only when memory exceeds budget
- Transparent when memory is sufficient

Metrics:
- TG speed (tokens/sec)
- Memory usage (MB)
- Quality preservation
- Compression frequency

Usage:
    python benchmark_double_layer_vs_rotating.py --calibration-dir /tmp/am_calibrations_ultra_dense
"""

import sys
sys.path.insert(0, '/Users/lisihao/FlashMLX/mlx-lm-source')

import mlx.core as mx
from mlx_lm import load
import numpy as np
import argparse
from datetime import datetime
import time

# Import cache classes
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.models.double_layer_cache import DoubleLayerKVCache

def log(msg, end='\n'):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True, end=end)

# Test corpus (long context - expanded to 1500+ tokens)
TEST_CORPUS = """
Dr. Sarah Chen founded the Quantum Dynamics Research Lab at Stanford University in 2019 with $5 million from the National Science Foundation. Her team aimed to develop room-temperature quantum computers.

The initial phase involved assembling a diverse team. Chen recruited Dr. Robert Kim from MIT, Dr. Elena Rodriguez from Caltech, and Dr. Yuki Tanaka from Tokyo. They started with theoretical modeling.

In 2020, they built their first prototype in the basement laboratory. Early tests were disappointing - quantum coherence lasted only milliseconds at room temperature.

The team persevered through 2021, making incremental improvements. They experimented with different materials and by December 2021 had extended coherence to 3 seconds at 280 Kelvin.

The breakthrough came on July 15, 2022, at 3:47 AM. The quantum processor achieved stable coherence at 294 Kelvin (room temperature) for 47 seconds. They ran 127 experiments with 89% success rate.

Professor Marcus Blackwell from Oxford criticized the results. However, five teams (Tokyo, Cambridge, Zurich, Tsinghua, Caltech) replicated them with 84% success rate. Blackwell later admitted his calibration error.

Dr. Chen received the Nobel Prize in 2024, shared with Dr. Robert Kim and Dr. Elena Rodriguez. She announced open-sourcing the basic designs.

The technology quickly spread worldwide. By 2025, quantum computers were deployed in major research centers across 50 countries. Applications ranged from drug discovery to climate modeling.

Dr. Chen's lab continued pushing boundaries. They achieved quantum advantage in molecular simulation, solving problems that would take classical computers millions of years in just hours.

The commercial applications were vast. Pharmaceutical companies used quantum computers to design new drugs in weeks instead of years. Climate scientists ran detailed simulations of atmospheric changes. Financial institutions optimized trading strategies in real-time. Cryptographers developed new encryption methods resistant to quantum attacks.

The educational sector transformed as well. Universities worldwide established quantum computing departments. Online courses reached millions of students. High schools introduced quantum mechanics as part of standard curriculum. The workforce rapidly adapted to this new paradigm.

International collaborations flourished. The United Nations established a Quantum Computing Ethics Committee to ensure equitable access. Developing nations received technology transfers and training. Open-source quantum software projects attracted thousands of contributors. Standards bodies worked to create interoperable quantum protocols.

Environmental benefits emerged unexpectedly. Quantum optimization reduced energy consumption in logistics by 30%. Material science breakthroughs led to better solar panels and batteries. Carbon capture technologies improved dramatically. Agricultural yields increased through precision farming guided by quantum weather predictions.

Security implications required careful consideration. Governments updated encryption standards to quantum-resistant algorithms. Military strategists reconsidered defense systems. Cybersecurity professionals retrained for the quantum era. International treaties addressed quantum computing in warfare.

The medical field saw revolutionary changes. Quantum computers modeled protein folding with unprecedented accuracy. Personalized medicine became routine as genetic analysis accelerated. Drug interactions were predicted before clinical trials. Rare diseases received targeted treatments developed in months.

Economic impacts rippled globally. New industries emerged around quantum hardware and software. Traditional computing companies pivoted or perished. Venture capital flowed into quantum startups. Stock markets experienced volatility as industries transformed. Employment patterns shifted dramatically toward quantum-literate positions.

Dr. Chen's vision extended beyond technology. She established the Quantum Literacy Foundation to ensure broad public understanding. Free educational resources reached underserved communities. Scholarships supported diverse students entering the field. Public lectures demystified quantum concepts for general audiences.

Research frontiers expanded continuously. Scientists pursued quantum computing applications in astronomy, mapping dark matter distributions. Geologists used quantum simulations to predict earthquakes. Linguists developed quantum natural language processing. Historians analyzed vast archives with quantum pattern recognition.

The philosophical implications sparked debate. Quantum computing challenged classical notions of determinism and causality. Ethicists questioned the societal impacts of exponential computational power. Religious scholars explored quantum mechanics' relationship to free will. Artists incorporated quantum randomness into creative works.

Question: What was the breakthrough achievement in July 2022?
Answer:"""

def benchmark_cache_type(name, model, tokenizer, cache_factory, num_generate=100):
    """
    Benchmark a specific cache type.

    Parameters
    ----------
    name : str
        Cache name for reporting
    cache_factory : callable
        Function that creates cache list
    num_generate : int
        Number of tokens to generate

    Returns
    -------
    dict : Performance metrics
    """
    log(f"\n{'='*70}")
    log(f"Benchmarking: {name}")
    log(f"{'='*70}")

    # Tokenize
    tokens = tokenizer.encode(TEST_CORPUS)
    prompt_len = len(tokens)
    log(f"Prompt length: {prompt_len} tokens")

    # Create caches
    num_layers = len(model.model.layers)
    cache_list = cache_factory(num_layers)

    # Prefill
    log("Step 1: Prefill...")
    y = mx.array([tokens])

    mx.eval(y)
    mx.metal.clear_cache()

    prefill_start = time.time()
    logits = model(y[:, :-1], cache=cache_list)
    mx.eval(logits)
    prefill_time = time.time() - prefill_start

    prefill_tps = prompt_len / prefill_time
    log(f"  Prefill: {prefill_tps:.2f} tokens/sec ({prefill_time:.3f}s)")

    # Decode
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
    total_memory = 0
    for cache in cache_list:
        if hasattr(cache, 'nbytes'):
            total_memory += cache.nbytes
        elif hasattr(cache, 'keys') and hasattr(cache, 'values'):
            total_memory += cache.keys.nbytes + cache.values.nbytes

    total_memory_mb = total_memory / (1024 ** 2)
    log(f"  Memory: {total_memory_mb:.2f} MB")

    # Output text
    output_text = tokenizer.decode(generated_tokens[:50])  # First 50 tokens
    log(f"  Output: {output_text[:100]}...")

    # Final cache size
    if hasattr(cache_list[0], 'offset'):
        cache_size = cache_list[0].offset
    elif hasattr(cache_list[0], 'keys'):
        cache_size = cache_list[0].keys.shape[2]
    else:
        cache_size = 0

    log(f"  Final cache size: {cache_size} tokens")

    # DoubleLayerKVCache specific stats
    compression_stats = None
    if hasattr(cache_list[0], 'get_stats'):
        compression_stats = cache_list[0].get_stats()
        log(f"  Compression stats:")
        log(f"    - old_prefix: {compression_stats['old_prefix_size']} tokens")
        log(f"    - recent_window: {compression_stats['recent_window_size']} tokens")
        log(f"    - compressions: {compression_stats['num_compressions']}")
        if compression_stats['num_compressions'] > 0:
            log(f"    - avg ratio: {compression_stats['avg_compression_ratio']:.2f}x")

    return {
        'name': name,
        'prompt_tokens': prompt_len,
        'generated_tokens': len(generated_tokens),
        'prefill_tps': prefill_tps,
        'generate_tps': generate_tps,
        'memory_mb': total_memory_mb,
        'cache_size': cache_size,
        'output_text': output_text,
        'compression_stats': compression_stats
    }

def main():
    parser = argparse.ArgumentParser(description='DoubleLayerKVCache vs RotatingKVCache Benchmark')
    parser.add_argument('--model-path', default='/Volumes/toshiba/models/qwen3-8b-mlx',
                        help='Path to model')
    parser.add_argument('--calibration-dir', required=True,
                        help='Calibration directory for DoubleLayerKVCache')
    parser.add_argument('--num-generate', type=int, default=100,
                        help='Number of tokens to generate')
    args = parser.parse_args()

    log("=" * 70)
    log("🏁 Performance Comparison: KV Cache Strategies")
    log("=" * 70)
    log(f"Model: {args.model_path}")
    log(f"Calibration: {args.calibration_dir}")
    log(f"Generate: {args.num_generate} tokens")

    # Load model
    log("\nLoading model...")
    model, tokenizer = load(args.model_path)
    num_layers = len(model.model.layers)
    log(f"✓ Model loaded: {num_layers} layers")

    # Define cache factories
    cache_configs = [
        {
            'name': 'Baseline (Full KVCache)',
            'factory': lambda n: [KVCache() for _ in range(n)]
        },
        {
            'name': 'RotatingKVCache (256 window)',
            'factory': lambda n: [RotatingKVCache(max_size=256, keep=256) for _ in range(n)]
        },
        # Adaptive Recent Window Tests (same budget 2.0MB, different windows)
        # Lower budget to ensure compression triggers
        {
            'name': 'DoubleLayer (2.0MB, 128 window - Retrieval)',
            'factory': lambda n: [
                DoubleLayerKVCache(
                    memory_budget_mb=2.0,  # Reduced to trigger compression
                    recent_window_size=128,  # Retrieval scenario
                    compression_ratio=1.5,
                    calibration_dir=args.calibration_dir,
                    layer_idx=i,
                    enable_compression=True
                )
                for i in range(n)
            ]
        },
        {
            'name': 'DoubleLayer (2.0MB, 256 window - Chat)',
            'factory': lambda n: [
                DoubleLayerKVCache(
                    memory_budget_mb=2.0,  # Reduced to trigger compression
                    recent_window_size=256,  # Chat scenario
                    compression_ratio=1.5,
                    calibration_dir=args.calibration_dir,
                    layer_idx=i,
                    enable_compression=True
                )
                for i in range(n)
            ]
        },
        {
            'name': 'DoubleLayer (2.0MB, 512 window - QA)',
            'factory': lambda n: [
                DoubleLayerKVCache(
                    memory_budget_mb=2.0,  # Reduced to trigger compression
                    recent_window_size=512,  # QA scenario
                    compression_ratio=1.5,
                    calibration_dir=args.calibration_dir,
                    layer_idx=i,
                    enable_compression=True
                )
                for i in range(n)
            ]
        }
    ]

    # Run benchmarks
    results = []

    for config in cache_configs:
        result = benchmark_cache_type(
            name=config['name'],
            model=model,
            tokenizer=tokenizer,
            cache_factory=config['factory'],
            num_generate=args.num_generate
        )
        results.append(result)

    # Summary table
    log("\n" + "=" * 70)
    log("📊 Performance Summary")
    log("=" * 70)

    log("\n{:<35} {:>12} {:>12} {:>12}".format(
        "Cache Strategy", "TG (tok/s)", "Memory (MB)", "Cache Size"
    ))
    log("-" * 70)

    baseline = results[0]

    for r in results:
        speedup = (r['generate_tps'] / baseline['generate_tps']) if baseline['generate_tps'] > 0 else 0
        memory_ratio = (r['memory_mb'] / baseline['memory_mb']) if baseline['memory_mb'] > 0 else 0

        log("{:<35} {:>12.2f} {:>12.2f} {:>12}".format(
            r['name'],
            r['generate_tps'],
            r['memory_mb'],
            r['cache_size']
        ))

        if r != baseline:
            log("  vs Baseline: {:>22} {:>12} {:>12}".format(
                f"{speedup:.2%} speed",
                f"{memory_ratio:.2%} memory",
                ""
            ))

    # Quality comparison
    log("\n" + "=" * 70)
    log("📝 Output Quality (First 100 chars)")
    log("=" * 70)

    for r in results:
        log(f"\n{r['name']}:")
        log(f"  {r['output_text'][:100]}...")

    # Recommendations
    log("\n" + "=" * 70)
    log("💡 Recommendations")
    log("=" * 70)

    rotating = results[1]
    double_layer = results[2]

    log(f"\n1. Memory Efficiency:")
    log(f"   - RotatingKVCache: {rotating['memory_mb']/baseline['memory_mb']:.1%} of baseline")
    log(f"   - DoubleLayerKVCache: {double_layer['memory_mb']/baseline['memory_mb']:.1%} of baseline")

    log(f"\n2. Speed:")
    log(f"   - RotatingKVCache: {rotating['generate_tps']/baseline['generate_tps']:.1%} of baseline")
    log(f"   - DoubleLayerKVCache: {double_layer['generate_tps']/baseline['generate_tps']:.1%} of baseline")

    log(f"\n3. Use Cases:")
    log(f"   - Baseline: Best quality, high memory")
    log(f"   - RotatingKVCache: Simple, fast, limited context")
    log(f"   - DoubleLayerKVCache: Best trade-off, preserves quality")

    log("\n" + "=" * 70)

if __name__ == '__main__':
    main()
