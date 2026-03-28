#!/usr/bin/env python3
"""
Test 3-layer system WITHOUT Warm quantization on LONG prompt

Configuration:
  - Recent: 512 tokens, exact
  - Warm: 1536 tokens, exact (no quantization)
  - Cold: 2048+ tokens, AM compression

Long prompt strategy: Repeat agent debugging scenario to reach ~3400 tokens
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


# Base scenario (977 tokens)
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

What is the root cause?"""

# Repeat 4 times to get ~3900 tokens
LONG_PROMPT = "\n\n--- Scenario 1 ---\n" + BASE_SCENARIO + \
              "\n\n--- Scenario 2 ---\n" + BASE_SCENARIO + \
              "\n\n--- Scenario 3 ---\n" + BASE_SCENARIO + \
              "\n\n--- Scenario 4 ---\n" + BASE_SCENARIO + \
              "\n\nProvide comprehensive analysis of all scenarios."


def compute_quality(text: str) -> dict:
    lines = [l for l in text.strip().split('\n') if l.strip()]
    if not lines:
        return {"repetition": 0.0, "coherence": 0.0, "relevance": 0.0}

    unique = set(lines)
    rep = 1.0 - (len(unique) / len(lines))

    meaningful = [l for l in lines if len(l.strip()) > 10]
    coh = len(meaningful) / len(lines) if lines else 0.0

    keywords = ["connection", "pool", "worker", "error", "leak"]
    relevant = [l for l in lines if any(k in l.lower() for k in keywords)]
    rel = len(relevant) / len(lines) if lines else 0.0

    return {"repetition": rep, "coherence": coh, "relevance": rel}


def test_system(name: str, cache_class, cache_kwargs: dict, model, tokenizer, num_layers=36):
    print(f"\n{'='*80}")
    print(f"{name}")
    print(f"{'='*80}")

    # Tokenize
    tokens = tokenizer.encode(LONG_PROMPT)
    prompt_len = len(tokens)
    print(f"Prompt: {prompt_len} tokens")
    y = mx.array([tokens])

    # Create caches
    caches = []
    for i in range(num_layers):
        cache = cache_class(layer_idx=i, **cache_kwargs)
        caches.append(cache)

    # Prefill
    print("Prefilling...")
    t0 = time.time()
    logits = model(y[:, :-1], cache=caches)
    mx.eval(logits)
    print(f"  Prefill: {prompt_len / (time.time() - t0):.1f} tok/s")

    # Generate
    print("Generating 100 tokens...")
    y = mx.array([[tokens[-1]]])
    t0 = time.time()

    generated = []
    for i in range(100):
        logits = model(y, cache=caches)
        y = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        mx.eval(y)
        generated.append(y[0, 0].item())
        if y[0, 0].item() == tokenizer.eos_token_id:
            break

    gen_time = time.time() - t0
    output = tokenizer.decode(generated)

    # Memory
    total_mb = 0
    layer_mem = {}
    for cache in caches:
        if hasattr(cache, 'get_memory_usage'):
            mem = cache.get_memory_usage()
            total_mb += mem.get('total_mb', 0)
            for k, v in mem.items():
                if k.endswith('_mb') or k.endswith('_tokens'):
                    layer_mem[k] = layer_mem.get(k, 0) + v

    quality = compute_quality(output)
    tps = len(generated) / gen_time if gen_time > 0 else 0

    # Print
    print(f"\nResults:")
    print(f"  Memory: {total_mb:.1f} MB")
    if 'cold_mb' in layer_mem:
        print(f"    Cold:   {layer_mem.get('cold_mb', 0):.1f} MB ({layer_mem.get('cold_tokens', 0)} tok)")
        print(f"    Warm:   {layer_mem.get('warm_mb', 0):.1f} MB ({layer_mem.get('warm_tokens', 0)} tok)")
        print(f"    Recent: {layer_mem.get('recent_mb', 0):.1f} MB ({layer_mem.get('recent_tokens', 0)} tok)")
    elif 'old_prefix_mb' in layer_mem:
        print(f"    Old Prefix: {layer_mem.get('old_prefix_mb', 0):.1f} MB")
        print(f"    Recent:     {layer_mem.get('recent_mb', 0):.1f} MB")

    print(f"  Speed: {tps:.1f} tok/s")
    print(f"  Quality: R:{quality['repetition']*100:.0f}% C:{quality['coherence']*100:.0f}% Rel:{quality['relevance']*100:.0f}%")
    print(f"  Output: {output[:120]}...")

    # Check quality
    if quality['repetition'] > 0.2:
        status = "❌ HIGH REPETITION"
    elif quality['coherence'] < 0.5:
        status = "❌ LOW COHERENCE"
    else:
        status = "✅ GOOD"
    print(f"  Status: {status}")

    return {
        "name": name,
        "prompt_len": prompt_len,
        "memory_mb": total_mb,
        "layer_memory": layer_mem,
        "quality": quality,
        "tps": tps,
        "output": output
    }


def main():
    print("3-Layer (No Warm Quant) vs 2-Layer on LONG Prompt")
    print("=" * 80)

    model_path = "/Volumes/toshiba/models/qwen3-8b-mlx"
    model, tokenizer = load(model_path)
    num_layers = len(model.model.layers)

    results = []

    # 2-layer baseline
    r2 = test_system(
        "2-Layer (Baseline)",
        DoubleLayerKVCache,
        {
            "memory_budget_mb": 2.0,
            "recent_window_size": 512,
            "compression_ratio": 1.5,
            "calibration_dir": "/tmp/am_calibrations_ultra_dense",
            "enable_compression": True,
            "selection_strategy": "nearest"
        },
        model, tokenizer, num_layers
    )
    results.append(r2)

    time.sleep(3)

    # 3-layer (NO Warm quant)
    r3 = test_system(
        "3-Layer (No Warm Quant)",
        TripleLayerKVCache,
        {
            "recent_size": 512,
            "warm_size": 1536,
            "calibration_dir": "/tmp/am_calibrations_ultra_dense",
            "compression_ratio": 1.5,
            "selection_strategy": "nearest",
            "enable_warm_quant": False,  # NO quantization
            "enable_cold_am": True
        },
        model, tokenizer, num_layers
    )
    results.append(r3)

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    baseline_mem = r2['memory_mb']
    for r in results:
        mem_delta = ((baseline_mem - r['memory_mb']) / baseline_mem * 100) if baseline_mem > 0 else 0
        q = r['quality']
        print(f"\n{r['name']}:")
        print(f"  Memory: {r['memory_mb']:.1f} MB (savings: {mem_delta:+.1f}%)")
        print(f"  Quality: R:{q['repetition']*100:.0f}% C:{q['coherence']*100:.0f}% Rel:{q['relevance']*100:.0f}%")

    # Save
    with open("/tmp/3layer_noquant_long_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to: /tmp/3layer_noquant_long_results.json")


if __name__ == "__main__":
    main()
