#!/usr/bin/env python3
"""Quick PCA E2E test - 3B model only."""

import sys
sys.path.insert(0, '.')

import os
import json
import time
from datetime import datetime
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step, stream_generate
from mlx_lm.models.cache import KVTCPromptCache, make_prompt_cache
from mlx_lm.models.kvtc_codec import KVTCCodecConfig
from mlx_lm.models.kvtc_pca_codec import fit_pca_calibration


def _is_kvtc_supported(layer_cache):
    state = layer_cache.state
    if not isinstance(state, tuple) or len(state) != 2:
        return False
    keys, values = state
    return getattr(keys, "ndim", 0) == 4 and getattr(values, "ndim", 0) == 4


def _prefill_prompt(model, tokenizer, prompt_text):
    if tokenizer.has_chat_template:
        messages = [{"role": "user", "content": prompt_text}]
        prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            continue_final_message=True,
        )
    else:
        prompt = tokenizer.encode(prompt_text)

    cache = make_prompt_cache(model, None)
    y = mx.array(prompt)
    for _ in generate_step(y, model, max_tokens=0, prompt_cache=cache):
        pass

    return cache, prompt


def _layer_calibration(layer_cache, calibrations):
    if calibrations is None:
        return None
    keys, values = layer_cache.state
    return calibrations[(keys.shape[-1], values.shape[-1])]


def _apply_kvtc_compression(cache, calibrations):
    if calibrations is None:
        return cache

    compressed_cache = []
    for c in cache:
        if _is_kvtc_supported(c):
            calibration = _layer_calibration(c, calibrations)
            compressed_cache.append(
                KVTCPromptCache.from_cache(c, calibration=calibration)
            )
        else:
            compressed_cache.append(c)

    return compressed_cache


def test_pca_e2e():
    """Test PCA compression on 3B model."""
    print("=" * 70)
    print("KVTC PCA E2E Test - 3B Model")
    print("=" * 70)
    print()

    model_path = "mlx-community/Qwen2.5-3B-Instruct-4bit"

    # Load model
    print("📦 Loading model...")
    t0 = time.time()
    model, tokenizer = load(model_path)
    load_time = time.time() - t0
    print(f"   ✅ Loaded in {load_time:.2f}s")
    print()

    # Test configurations
    configs = [
        ("PCA-8 (rank=8)", KVTCCodecConfig(rank=8, bits=4, group_size=16)),
        ("PCA-16 (rank=16)", KVTCCodecConfig(rank=16, bits=4, group_size=16)),
        ("No-Compression", None),
    ]

    test_prompts = [
        "The future of artificial intelligence is",
        "In the year 2050, technology will",
        "Explain quantum computing in simple terms:",
    ]

    results = []

    for config_name, config in configs:
        print("-" * 70)
        print(f"Testing: {config_name}")
        print("-" * 70)

        # Calibration
        calibrations = None
        if config is not None:
            print("   校准中...")
            calib_prompt = test_prompts[0]
            cache, _ = _prefill_prompt(model, tokenizer, calib_prompt)

            # Fit calibrations
            groups = {}
            for layer_cache in cache:
                if layer_cache.empty() or not _is_kvtc_supported(layer_cache):
                    continue
                keys, values = layer_cache.state
                group_key = (keys.shape[-1], values.shape[-1])
                groups.setdefault(group_key, {"keys": [], "values": []})
                groups[group_key]["keys"].append(keys.reshape(-1, keys.shape[-1]))
                groups[group_key]["values"].append(values.reshape(-1, values.shape[-1]))

            calibrations = {}
            for group_key, group in groups.items():
                calibrations[group_key] = fit_pca_calibration(
                    [group["keys"][0]], [group["values"][0]], config
                )

            print("   ✅ 校准完成")

        # Test generation
        gen_results = []
        for prompt in test_prompts:
            cache, prompt_tokens = _prefill_prompt(model, tokenizer, prompt)

            if calibrations is not None:
                cache = _apply_kvtc_compression(cache, calibrations)

            # Generate
            t0 = time.time()
            response_text = ""
            token_count = 0

            for gen_response in stream_generate(
                model, tokenizer, prompt_tokens,
                max_tokens=100, prompt_cache=cache
            ):
                response_text += gen_response.text
                token_count += 1

            gen_time = time.time() - t0
            tok_per_s = token_count / gen_time if gen_time > 0 else 0

            gen_results.append({
                'tokens': token_count,
                'tok_per_s': tok_per_s,
                'response': response_text[:100]
            })

            print(f"   ✓ {prompt[:40]}... → {token_count} tokens, {tok_per_s:.1f} tok/s")

        avg_tok_per_s = sum(r['tok_per_s'] for r in gen_results) / len(gen_results)
        results.append((config_name, avg_tok_per_s, gen_results))

        print(f"   📊 Average: {avg_tok_per_s:.2f} tok/s")
        print()

        mx.metal.clear_cache()

    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()

    print(f"{'Configuration':20s}  {'Avg Speed':>12s}  {'Quality':>10s}")
    print(f"{'-'*20}  {'-'*12}  {'-'*10}")

    for config_name, avg_speed, gen_results in results:
        # Check quality by response length
        avg_tokens = sum(r['tokens'] for r in gen_results) / len(gen_results)
        quality = "✅ Good" if avg_tokens > 50 else "❌ Poor"
        print(f"{config_name:20s}  {avg_speed:11.2f}  {quality:>10s}")

    print()

    # Show sample responses for all prompts
    prompts_short = [
        "AI future",
        "2050 tech",
        "Quantum",
    ]

    for i, prompt_label in enumerate(prompts_short):
        print("=" * 70)
        print(f"Sample Responses - {prompt_label}")
        print("=" * 70)
        print()

        for config_name, _, gen_results in results:
            print(f"{config_name}:")
            print(f"  {gen_results[i]['response'][:200]}")
            print()

    return 0


def main():
    print()
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 18 + "KVTC PCA E2E Test - 3B" + " " * 27 + "║")
    print("╚" + "=" * 68 + "╝")
    print()

    try:
        return test_pca_e2e()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
