#!/usr/bin/env python3
"""
Diagnose Warm layer quantization impact

Test configurations:
1. 3-layer with Warm quant (baseline, has quality issue)
2. 3-layer without Warm quant (diagnostic)

If quality improves → quantization is the problem
If quality stays bad → layering logic has issues
"""

import sys
import time
import json
from pathlib import Path
import mlx.core as mx
from mlx_lm import load
sys.path.insert(0, str(Path(__file__).parent / "mlx-lm-source"))
from mlx_lm.models.triple_layer_cache import TripleLayerKVCache


# Agent debugging prompt (1429 tokens)
AGENT_PROMPT = """You are a senior production support engineer investigating a critical issue. The application has been experiencing intermittent 500 errors over the past 24 hours. The error occurs randomly, affecting about 15% of requests. Initial investigation shows:

1. Database connections are stable
2. Memory usage is normal
3. CPU usage shows occasional spikes
4. Error logs show: "Connection pool exhausted" every few minutes
5. The issue started after deploying version 2.1.3
6. Version 2.1.3 added a new background worker for data processing

Additional context from the development team:

The new background worker was introduced to handle asynchronous data synchronization with an external API. The worker runs every 30 seconds and processes batches of records that need to be synced. Each batch can contain up to 100 records. The worker creates a new database connection for each batch processing cycle.

The database connection pool is configured with the following settings:
- Pool size: 10 connections
- Max overflow: 5 connections
- Pool recycle: 3600 seconds
- Pool timeout: 30 seconds

The application architecture consists of:
- 4 web servers running the main application
- 2 worker servers running background jobs
- 1 scheduler server running periodic tasks
- PostgreSQL database (version 13.5)
- Redis for session storage and caching

Recent deployment history:
- Version 2.1.0: Released 3 weeks ago, stable
- Version 2.1.1: Hotfix for UI bug, released 2 weeks ago
- Version 2.1.2: Performance improvements, released 1 week ago
- Version 2.1.3: Background worker addition, released yesterday

The 500 errors are distributed across all web servers roughly evenly. Error rate increased from <0.1% to 15% immediately after the 2.1.3 deployment. No errors are logged on the worker or scheduler servers.

Database monitoring shows:
- Average active connections: 8-12
- Peak active connections: 45 (exceeds pool + overflow)
- Connection wait time: Occasional spikes to 30+ seconds
- Database CPU: Normal (20-30%)
- Database memory: Normal (40% utilized)

Application monitoring shows:
- Request latency: Normal for successful requests (50-200ms)
- Request latency: Timeout (30s) for failed requests
- Memory usage: Stable across all servers
- CPU usage: Occasional spikes to 80% during error periods

The error stacktrace shows:
```
sqlalchemy.exc.TimeoutError: QueuePool limit of size 10 overflow 5 reached, connection timed out, timeout 30.00
  File "/app/api/handlers.py", line 45, in handle_request
    result = db.session.execute(query)
  File "/app/models/base.py", line 120, in execute
    return self.connection.execute(*args, **kwargs)
```

The team has tried several debugging approaches:
1. Increased database connection pool size to 15 - no improvement
2. Added connection pooling to Redis - no effect on errors
3. Reviewed recent code changes - no obvious connection leaks found
4. Checked for long-running queries - none found
5. Monitored network connectivity - stable

Additional investigation revealed:
- The background worker processes 50-100 batches per minute
- Each batch processing takes 200-500ms
- The worker creates a connection at the start of batch processing
- The worker should close the connection after batch completion
- Code review shows proper try/finally blocks for connection cleanup

However, when examining the actual connection count during error periods:
- Database reports 45 active connections
- Application pool reports only 15 connections (pool + overflow)
- This 30-connection discrepancy is suspicious

Further analysis of the worker code shows:
```python
def process_batch(batch):
    conn = db.engine.connect()
    try:
        for record in batch:
            # Process record
            sync_to_external_api(record)
            # Update database
            conn.execute(update_query)
    finally:
        conn.close()

def sync_to_external_api(record):
    # Makes HTTP request to external API
    response = requests.post(api_url, json=record.to_dict(), timeout=30)
    if response.status_code != 200:
        # Log error and retry
        db.session.execute(log_query)  # Creates new connection!
```

Based on the evidence, what is the root cause of the 500 errors?"""


def compute_quality_metrics(text: str) -> dict:
    """Compute quality metrics"""
    lines = text.strip().split('\n')
    if not lines:
        return {"repetition": 0.0, "coherence": 0.0, "relevance": 0.0}

    unique_lines = set(lines)
    repetition_ratio = 1.0 - (len(unique_lines) / len(lines))

    meaningful_lines = [l for l in lines if len(l.strip()) > 10]
    coherence_score = len(meaningful_lines) / len(lines) if lines else 0.0

    keywords = ["connection", "pool", "worker", "database", "error", "cause", "leak"]
    relevant_lines = [l for l in lines if any(k in l.lower() for k in keywords)]
    relevance_score = len(relevant_lines) / len(lines) if lines else 0.0

    return {
        "repetition": repetition_ratio,
        "coherence": coherence_score,
        "relevance": relevance_score
    }


def test_config(config_name: str, enable_quant: bool, model, tokenizer, num_layers=36):
    """Test a specific configuration"""
    print(f"\n{'='*80}")
    print(f"Testing: {config_name}")
    print(f"  Warm quantization: {'ENABLED' if enable_quant else 'DISABLED'}")
    print(f"{'='*80}")

    # Tokenize
    tokens = tokenizer.encode(AGENT_PROMPT)
    prompt_len = len(tokens)
    print(f"Prompt length: {prompt_len} tokens")
    y = mx.array([tokens])

    # Create caches
    caches = []
    for layer_idx in range(num_layers):
        cache = TripleLayerKVCache(
            recent_size=512,
            warm_size=1536,
            calibration_dir="/tmp/am_calibrations_ultra_dense",
            layer_idx=layer_idx,
            compression_ratio=1.5,
            selection_strategy="nearest",
            quant_bits=4,
            enable_warm_quant=enable_quant,  # KEY: toggle quantization
            enable_cold_am=True
        )
        caches.append(cache)

    # Prefill
    print("Prefilling...")
    prefill_start = time.time()
    logits = model(y[:, :-1], cache=caches)
    mx.eval(logits)
    prefill_time = time.time() - prefill_start
    print(f"  Prefill: {prompt_len / prefill_time:.2f} tok/s")

    # Generate
    print("Generating 100 tokens...")
    y = mx.array([[tokens[-1]]])
    gen_start = time.time()

    generated_tokens = []
    for i in range(100):
        logits = model(y, cache=caches)
        y = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        mx.eval(y)

        token_id = y[0, 0].item()
        generated_tokens.append(token_id)

        if token_id == tokenizer.eos_token_id:
            break

    gen_time = time.time() - gen_start
    generated_text = tokenizer.decode(generated_tokens)

    # Memory
    total_mb = 0
    cold_mb = warm_mb = recent_mb = 0
    cold_tokens = warm_tokens = recent_tokens = 0

    for cache in caches:
        if hasattr(cache, 'get_memory_usage'):
            mem = cache.get_memory_usage()
            total_mb += mem.get('total_mb', 0)
            cold_mb += mem.get('cold_mb', 0)
            warm_mb += mem.get('warm_mb', 0)
            recent_mb += mem.get('recent_mb', 0)
            cold_tokens += mem.get('cold_tokens', 0)
            warm_tokens += mem.get('warm_tokens', 0)
            recent_tokens += mem.get('recent_tokens', 0)

    # Quality
    quality = compute_quality_metrics(generated_text)
    tps = len(generated_tokens) / gen_time if gen_time > 0 else 0

    # Results
    print(f"\n{config_name} Results:")
    print(f"  Memory: {total_mb:.1f} MB")
    print(f"    Cold:   {cold_mb:.1f} MB ({cold_tokens} tokens)")
    print(f"    Warm:   {warm_mb:.1f} MB ({warm_tokens} tokens)")
    print(f"    Recent: {recent_mb:.1f} MB ({recent_tokens} tokens)")
    print(f"  Speed: {tps:.2f} tok/s")
    print(f"  Quality:")
    print(f"    Repetition: {quality['repetition']*100:.2f}%")
    print(f"    Coherence: {quality['coherence']*100:.2f}%")
    print(f"    Relevance: {quality['relevance']*100:.2f}%")
    print(f"  Output preview:")
    print(f"    {generated_text[:150]}...")

    # Save
    output_file = f"/tmp/warm_quant_{config_name.replace(' ', '_').lower()}_output.txt"
    with open(output_file, "w") as f:
        f.write(generated_text)

    return {
        "config": config_name,
        "enable_quant": enable_quant,
        "prompt_len": prompt_len,
        "generated_tokens": len(generated_tokens),
        "tps": tps,
        "memory": {
            "total_mb": total_mb,
            "cold_mb": cold_mb,
            "warm_mb": warm_mb,
            "recent_mb": recent_mb,
            "cold_tokens": cold_tokens,
            "warm_tokens": warm_tokens,
            "recent_tokens": recent_tokens
        },
        "quality": quality,
        "output_preview": generated_text[:200]
    }


def main():
    print("Warm Layer Quantization Diagnostic Test")
    print("=" * 80)

    # Load model
    print("\nLoading model...")
    model_path = "/Volumes/toshiba/models/qwen3-8b-mlx"
    model, tokenizer = load(model_path)
    num_layers = len(model.model.layers)
    print(f"Model: Qwen3-8B ({num_layers} layers)")

    results = []

    # Test 1: WITH quantization (baseline, has quality issue)
    result_with_quant = test_config(
        config_name="WITH_Quant",
        enable_quant=True,
        model=model,
        tokenizer=tokenizer,
        num_layers=num_layers
    )
    results.append(result_with_quant)

    # Cool down
    print("\nCooling down...")
    time.sleep(3)

    # Test 2: WITHOUT quantization (diagnostic)
    result_no_quant = test_config(
        config_name="NO_Quant",
        enable_quant=False,
        model=model,
        tokenizer=tokenizer,
        num_layers=num_layers
    )
    results.append(result_no_quant)

    # Compare
    print(f"\n{'='*80}")
    print("DIAGNOSTIC COMPARISON")
    print(f"{'='*80}")

    baseline = result_with_quant
    test = result_no_quant

    print(f"\n{'Config':<15} {'Memory':<15} {'Quality':<35} {'Verdict':<15}")
    print(f"{'-'*80}")

    for r in results:
        q = r['quality']
        mem = r['memory']['total_mb']

        # Quality verdict
        if q['repetition'] < 0.1 and q['coherence'] > 0.6:
            verdict = "✅ GOOD"
        elif q['repetition'] < 0.2 and q['coherence'] > 0.5:
            verdict = "⚠️  FAIR"
        else:
            verdict = "❌ BAD"

        print(f"{r['config']:<15} {mem:<15.1f} R:{q['repetition']*100:.0f}% C:{q['coherence']*100:.0f}% Rel:{q['relevance']*100:.0f}%{'':<10} {verdict:<15}")

    # Analysis
    print(f"\n{'='*80}")
    print("ANALYSIS")
    print(f"{'='*80}")

    q_with = baseline['quality']
    q_no = test['quality']

    rep_delta = (q_no['repetition'] - q_with['repetition']) * 100
    coh_delta = (q_no['coherence'] - q_with['coherence']) * 100
    rel_delta = (q_no['relevance'] - q_with['relevance']) * 100

    mem_with = baseline['memory']['total_mb']
    mem_no = test['memory']['total_mb']
    mem_increase = ((mem_no - mem_with) / mem_with * 100) if mem_with > 0 else 0

    print(f"\nQuality Delta (NO_Quant - WITH_Quant):")
    print(f"  Repetition: {rep_delta:+.1f} pp {'✅ Better' if rep_delta < 0 else '❌ Worse'}")
    print(f"  Coherence:  {coh_delta:+.1f} pp {'✅ Better' if coh_delta > 0 else '❌ Worse'}")
    print(f"  Relevance:  {rel_delta:+.1f} pp {'✅ Better' if rel_delta > 0 else '❌ Worse'}")

    print(f"\nMemory Delta:")
    print(f"  WITH_Quant: {mem_with:.1f} MB (Warm: {baseline['memory']['warm_mb']:.1f} MB)")
    print(f"  NO_Quant:   {mem_no:.1f} MB (Warm: {test['memory']['warm_mb']:.1f} MB)")
    print(f"  Increase:   {mem_increase:+.1f}% (expected ~2x for Warm layer)")

    # Conclusion
    print(f"\n{'='*80}")
    print("CONCLUSION")
    print(f"{'='*80}")

    if coh_delta > 10 and rep_delta < -10:
        print("\n✅ DIAGNOSIS: Quantization IS the problem!")
        print("   - Disabling quantization significantly improves quality")
        print("   - Trade-off: Memory increases (Warm layer uncompressed)")
        print("\n   Next step: Optimize quantization (方案 C)")
    elif coh_delta > 5 or rep_delta < -5:
        print("\n⚠️  DIAGNOSIS: Quantization contributes to problem")
        print("   - Modest quality improvement without quantization")
        print("   - But not the only issue")
        print("\n   Next step: Optimize quantization AND check layering logic")
    else:
        print("\n❌ DIAGNOSIS: Quantization is NOT the main problem")
        print("   - Quality similar with or without quantization")
        print("   - Issue likely in layering logic or other factors")
        print("\n   Next step: Debug layering logic, check AM compression")

    # Save
    with open("/tmp/warm_quant_diagnostic.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: /tmp/warm_quant_diagnostic.json")


if __name__ == "__main__":
    main()
