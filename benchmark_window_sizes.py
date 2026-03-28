#!/usr/bin/env python3
"""
Dynamic Recent Window Size Quality Benchmark

Test different recent_window_size values and measure:
1. Memory savings
2. Output quality (重点检查)
3. Speed

Window sizes to test:
- 512 (baseline)
- 256 (recommended)
- 128 (aggressive)
"""

import sys
sys.path.insert(0, '/Users/lisihao/FlashMLX/mlx-lm-source')

import mlx.core as mx
from mlx_lm import load
import time
from datetime import datetime

from mlx_lm.models.cache import KVCache
from mlx_lm.models.double_layer_cache import DoubleLayerKVCache
from benchmark_adaptive_window import WORKLOAD_AGENT as TEST_PROMPT

def log(msg, end='\n'):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True, end=end)

def benchmark_window_size(
    window_size: int,
    model,
    tokenizer,
    num_generate: int = 100
):
    """Test specific window size"""
    log(f"\n{'='*70}")
    log(f"Testing Window Size: {window_size}")
    log(f"{'='*70}")

    # Tokenize
    tokens = tokenizer.encode(TEST_PROMPT)
    prompt_len = len(tokens)
    log(f"Prompt length: {prompt_len} tokens")

    # Create caches
    num_layers = len(model.model.layers)
    cache_list = []

    for layer_idx in range(num_layers):
        cache = DoubleLayerKVCache(
            memory_budget_mb=2.0,
            recent_window_size=window_size,  # 动态调整
            compression_ratio=1.5,  # Uniform R1.5
            calibration_dir="/tmp/am_calibrations_ultra_dense",
            layer_idx=layer_idx,
            enable_compression=True,
            selection_strategy="nearest"
        )
        cache_list.append(cache)

    # Prefill
    log("Step 1: Prefill...")
    y = mx.array([tokens])
    mx.eval(y)
    mx.clear_cache()

    prefill_start = time.time()
    logits = model(y[:, :-1], cache=cache_list)
    mx.eval(logits)
    prefill_time = time.time() - prefill_start

    log(f"  Prefill: {prompt_len / prefill_time:.2f} tokens/sec ({prefill_time:.3f}s)")

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
    tg_speed = len(generated_tokens) / generate_time

    log(f"  Generated {len(generated_tokens)} tokens")
    log(f"  TG speed: {tg_speed:.2f} tokens/sec ({generate_time:.3f}s)")

    # Memory usage
    total_memory = sum(cache.nbytes for cache in cache_list) / 1024**2
    log(f"  Memory: {total_memory:.2f} MB")

    # Decode output
    output = tokenizer.decode(generated_tokens)

    # Quality checks
    log(f"\n[Quality Checks]")

    # 1. Length check
    if len(generated_tokens) < num_generate * 0.8:
        log(f"  ⚠️  Short output: {len(generated_tokens)}/{num_generate} tokens")
    else:
        log(f"  ✓ Output length: {len(generated_tokens)} tokens")

    # 2. Repetition detection
    lines = output.split('\n')
    unique_lines = len(set(lines))
    total_lines = len(lines) if len(lines) > 0 else 1
    repetition_ratio = 1.0 - (unique_lines / total_lines)
    log(f"  Repetition: {repetition_ratio:.2%}")
    if repetition_ratio > 0.3:
        log(f"  ⚠️  High repetition detected")

    # 3. Coherence check
    common_words = ["the", "and", "is", "to", "of", "in", "that", "for"]
    word_count = sum(1 for word in common_words if word in output.lower())
    coherence_score = word_count / len(common_words)
    log(f"  Coherence: {coherence_score:.2%} ({word_count}/{len(common_words)} common words)")
    if coherence_score < 0.5:
        log(f"  ⚠️  Low coherence (possible gibberish)")

    # 4. Relevance check
    relevant_keywords = ["connection", "database", "fix", "solution", "code", "bug", "error"]
    relevant_count = sum(1 for kw in relevant_keywords if kw in output.lower())
    relevance_score = relevant_count / len(relevant_keywords)
    log(f"  Relevance: {relevance_score:.2%} ({relevant_count}/{len(relevant_keywords)} keywords)")
    if relevance_score < 0.3:
        log(f"  ⚠️  Low relevance to context")

    # Output preview
    log(f"\n[Output Preview] First 200 chars:")
    log(f"  {output[:200]}")

    return {
        "window_size": window_size,
        "memory_mb": total_memory,
        "tg_speed": tg_speed,
        "output_tokens": len(generated_tokens),
        "repetition": repetition_ratio,
        "coherence": coherence_score,
        "relevance": relevance_score,
        "output": output
    }

def main():
    model_path = "/Volumes/toshiba/models/qwen3-8b-mlx"

    log("="*70)
    log("🔬 Dynamic Recent Window Quality Test")
    log("="*70)
    log(f"Model: {model_path}")
    log(f"Workload: {len(TEST_PROMPT)} chars agent debugging")
    log(f"Compression: Uniform R1.5")
    log("")

    # Load model
    log("Loading model...")
    model, tokenizer = load(model_path)
    log(f"✓ Model loaded: {len(model.model.layers)} layers")

    # Test window sizes: 512 (baseline), 256 (recommended), 128 (aggressive)
    window_sizes = [512, 256, 128]
    results = []

    for window_size in window_sizes:
        try:
            result = benchmark_window_size(window_size, model, tokenizer, num_generate=100)
            results.append(result)

            # Save output
            output_file = f"/tmp/window_{window_size}_output.txt"
            with open(output_file, 'w') as f:
                f.write(result['output'])
            log(f"  Saved output to: {output_file}")

        except Exception as e:
            log(f"\n❌ Error testing window={window_size}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    log("\n" + "="*70)
    log("📊 Summary: Window Size Quality Comparison")
    log("="*70)
    log("")

    # Header
    print(f"{'Window':<10} {'Memory (MB)':<15} {'Savings':<12} {'TG Speed':<12} {'Repetition':<12} {'Coherence':<12} {'Relevance':<12}")
    print("-"*90)

    baseline = results[0] if results else None

    for result in results:
        window = result["window_size"]
        memory = result["memory_mb"]
        speed = result["tg_speed"]
        repetition = result["repetition"]
        coherence = result["coherence"]
        relevance = result["relevance"]

        if baseline and window != baseline["window_size"]:
            savings = (baseline["memory_mb"] - memory) / baseline["memory_mb"] * 100
            savings_str = f"+{savings:.1f}%"
        else:
            savings_str = "baseline"

        print(f"{window:<10} {memory:<15.1f} {savings_str:<12} {speed:<12.2f} {repetition:<12.2%} {coherence:<12.2%} {relevance:<12.2%}")

    # Quality verdict
    log("\n" + "="*70)
    log("🎯 Quality Verdict")
    log("="*70)

    if len(results) >= 2:
        baseline = results[0]
        optimized = results[1]  # window=256
        aggressive = results[2] if len(results) > 2 else None

        log(f"\nWindow 256 vs 512 (baseline):")
        log(f"  Memory savings: +{(baseline['memory_mb'] - optimized['memory_mb']) / baseline['memory_mb'] * 100:.1f}%")
        log(f"  Repetition delta: {optimized['repetition'] - baseline['repetition']:+.2%}")
        log(f"  Coherence delta: {optimized['coherence'] - baseline['coherence']:+.2%}")
        log(f"  Relevance delta: {optimized['relevance'] - baseline['relevance']:+.2%}")

        quality_ok = (
            abs(optimized['repetition'] - baseline['repetition']) < 0.1 and
            abs(optimized['coherence'] - baseline['coherence']) < 0.2 and
            abs(optimized['relevance'] - baseline['relevance']) < 0.2
        )

        if quality_ok:
            log(f"\n✅ Window=256: Quality maintained! Safe to use.")
        else:
            log(f"\n⚠️  Window=256: Quality issues detected.")

        if aggressive:
            log(f"\nWindow 128 vs 512 (baseline):")
            log(f"  Memory savings: +{(baseline['memory_mb'] - aggressive['memory_mb']) / baseline['memory_mb'] * 100:.1f}%")
            log(f"  Repetition delta: {aggressive['repetition'] - baseline['repetition']:+.2%}")
            log(f"  Coherence delta: {aggressive['coherence'] - baseline['coherence']:+.2%}")
            log(f"  Relevance delta: {aggressive['relevance'] - baseline['relevance']:+.2%}")

            quality_ok = (
                abs(aggressive['repetition'] - baseline['repetition']) < 0.1 and
                abs(aggressive['coherence'] - baseline['coherence']) < 0.2 and
                abs(aggressive['relevance'] - baseline['relevance']) < 0.2
            )

            if quality_ok:
                log(f"\n✅ Window=128: Quality maintained! Safe to use.")
            else:
                log(f"\n⚠️  Window=128: Quality issues detected.")

    log("\n✅ Test complete!")

if __name__ == "__main__":
    main()
