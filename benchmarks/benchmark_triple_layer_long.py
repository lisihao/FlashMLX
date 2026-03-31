#!/usr/bin/env python3
"""
Benchmark Triple-Layer KV Cache with LONG prompt to trigger layering

Prompt strategy: Use 3384 tokens (same as window test) to exceed Recent=512,
triggering Warm and Cold layers.

Expected behavior:
    - Tokens 0-2872 → should move to Cold (AM compressed)
    - Tokens 2872-3384 → should be in Warm or Recent
    - New generated tokens → Recent

This will properly test the 3-layer system!
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


# Long agent debugging session (3384 tokens)
LONG_AGENT_PROMPT = """You are a senior production support engineer investigating a critical issue. The application has been experiencing intermittent 500 errors over the past 24 hours. The error occurs randomly, affecting about 15% of requests. Initial investigation shows:

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

The scheduler server runs these periodic tasks:
- Data cleanup job (every hour)
- Report generation (every 6 hours)
- Cache warming (every 30 minutes)
- Health check pings (every 5 minutes)

Each scheduled task also creates database connections:
```python
@scheduler.task(interval=3600)
def cleanup_old_data():
    conn = db.engine.connect()
    try:
        cleanup_records(conn)
    finally:
        conn.close()

def cleanup_records(conn):
    # Delete old records
    conn.execute(delete_query)
    # Update statistics
    with db.engine.connect() as stats_conn:  # Another connection!
        stats_conn.execute(update_stats_query)
```

The web application request handler also has this pattern:
```python
@app.route('/api/data')
def get_data():
    results = db.session.query(Data).filter(...).all()

    # Enrich with external data
    for result in results:
        external_info = fetch_external_info(result.id)
        result.external = external_info

    return jsonify(results)

def fetch_external_info(record_id):
    # Look up additional info
    with db.engine.connect() as conn:  # New connection outside pool!
        return conn.execute(lookup_query).fetchone()
```

System resource analysis shows:
- File descriptors: Normal usage (2000/10000)
- TCP connections: High (8000/10000)
- Process count: Normal (150/500)
- Disk I/O: Normal
- Network bandwidth: Normal

The error pattern shows:
- Errors cluster in 2-3 minute bursts
- Followed by 5-10 minutes of normal operation
- Pattern repeats throughout the day
- No correlation with specific times or traffic levels

Database query analysis:
- No slow queries detected
- Query plan analysis shows proper index usage
- No table locks or blocking queries
- Transaction rollback rate: <0.01%

The team's current hypothesis:
"The background worker is somehow leaking connections, but we can't find the leak in the code. The connection pool exhaustion happens periodically, suggesting some kind of accumulation and release pattern."

What is the actual root cause of the 500 errors? Provide a detailed analysis with evidence."""


def compute_quality_metrics(text: str) -> dict:
    """Compute quality metrics"""
    lines = text.strip().split('\n')
    if not lines:
        return {"repetition": 0.0, "coherence": 0.0, "relevance": 0.0}

    # Repetition
    unique_lines = set(lines)
    repetition_ratio = 1.0 - (len(unique_lines) / len(lines))

    # Coherence
    meaningful_lines = [l for l in lines if len(l.strip()) > 10]
    coherence_score = len(meaningful_lines) / len(lines) if lines else 0.0

    # Relevance
    keywords = ["connection", "pool", "worker", "database", "error", "cause", "leak"]
    relevant_lines = [l for l in lines if any(k in l.lower() for k in keywords)]
    relevance_score = len(relevant_lines) / len(lines) if lines else 0.0

    return {
        "repetition": repetition_ratio,
        "coherence": coherence_score,
        "relevance": relevance_score
    }


def benchmark_system(
    system_name: str,
    cache_class,
    cache_kwargs: dict,
    model,
    tokenizer,
    num_layers: int = 36,
    num_generate: int = 100
) -> dict:
    """Generic benchmark function"""

    print(f"\n{'='*80}")
    print(f"Testing: {system_name}")
    print(f"{'='*80}")

    # Tokenize
    tokens = tokenizer.encode(LONG_AGENT_PROMPT)
    prompt_len = len(tokens)
    print(f"Prompt length: {prompt_len} tokens")
    y = mx.array([tokens])

    # Create caches
    caches = []
    for layer_idx in range(num_layers):
        cache = cache_class(layer_idx=layer_idx, **cache_kwargs)
        caches.append(cache)

    print(f"Created {len(caches)} {system_name} caches")

    # Prefill
    print("\nPrefilling caches...")
    prefill_start = time.time()
    logits = model(y[:, :-1], cache=caches)
    mx.eval(logits)
    prefill_time = time.time() - prefill_start
    print(f"  Prefill: {prompt_len / prefill_time:.2f} tokens/sec ({prefill_time:.3f}s)")

    # Generate
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
    generated_text = tokenizer.decode(generated_tokens)

    # Memory usage
    total_memory_mb = 0
    layer_memory = {}

    for cache in caches:
        if hasattr(cache, 'get_memory_usage'):
            mem = cache.get_memory_usage()
            total_memory_mb += mem.get('total_mb', 0)

            # Collect layer-specific memory
            for key, value in mem.items():
                if key.endswith('_mb') and key != 'total_mb':
                    layer_memory[key] = layer_memory.get(key, 0) + value
                elif key.endswith('_tokens'):
                    layer_memory[key] = layer_memory.get(key, 0) + value

    # Quality
    quality = compute_quality_metrics(generated_text)
    tps = len(generated_tokens) / gen_time if gen_time > 0 else 0

    results = {
        "system": system_name,
        "prompt_len": prompt_len,
        "generated_tokens": len(generated_tokens),
        "generation_time_sec": gen_time,
        "tokens_per_second": tps,
        "memory": {"total_mb": total_memory_mb, **layer_memory},
        "quality": quality,
        "output_preview": generated_text[:200]
    }

    # Print results
    print(f"\n{system_name} Results:")
    print(f"  Total memory: {total_memory_mb:.1f} MB")

    if 'cold_mb' in layer_memory:
        print(f"    Cold:   {layer_memory.get('cold_mb', 0):.1f} MB ({layer_memory.get('cold_tokens', 0)} tokens)")
        print(f"    Warm:   {layer_memory.get('warm_mb', 0):.1f} MB ({layer_memory.get('warm_tokens', 0)} tokens)")
        print(f"    Recent: {layer_memory.get('recent_mb', 0):.1f} MB ({layer_memory.get('recent_tokens', 0)} tokens)")
    elif 'old_prefix_mb' in layer_memory:
        print(f"    Old Prefix: {layer_memory.get('old_prefix_mb', 0):.1f} MB")
        print(f"    Recent:     {layer_memory.get('recent_mb', 0):.1f} MB")

    print(f"  Speed: {tps:.2f} tok/s")
    print(f"  Quality: R:{quality['repetition']*100:.0f}% C:{quality['coherence']*100:.0f}% Rel:{quality['relevance']*100:.0f}%")
    print(f"  Output preview: {generated_text[:150]}...")

    # Save output
    output_file = f"/tmp/triple_{system_name.replace('-', '_').replace(' ', '_').lower()}_output.txt"
    with open(output_file, "w") as f:
        f.write(generated_text)

    return results


def main():
    print("Triple-Layer KV Cache Benchmark (LONG PROMPT)")
    print("=" * 80)

    # Load model
    print("\nLoading Qwen3-8B model...")
    model_path = "/Volumes/toshiba/models/qwen3-8b-mlx"
    model, tokenizer = load(model_path)
    num_layers = len(model.model.layers)
    print(f"Model loaded: {num_layers} layers")

    all_results = []

    # Test 2-layer (baseline)
    result_2layer = benchmark_system(
        system_name="2-layer",
        cache_class=DoubleLayerKVCache,
        cache_kwargs={
            "memory_budget_mb": 2.0,
            "recent_window_size": 512,
            "compression_ratio": 1.5,
            "calibration_dir": "/tmp/am_calibrations_ultra_dense",
            "enable_compression": True,
            "selection_strategy": "nearest"
        },
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
    result_3layer = benchmark_system(
        system_name="3-layer",
        cache_class=TripleLayerKVCache,
        cache_kwargs={
            "recent_size": 512,
            "warm_size": 1536,
            "calibration_dir": "/tmp/am_calibrations_ultra_dense",
            "compression_ratio": 1.5,
            "selection_strategy": "nearest",
            "quant_bits": 4,
            "enable_warm_quant": True,
            "enable_cold_am": True
        },
        model=model,
        tokenizer=tokenizer,
        num_layers=num_layers,
        num_generate=100
    )
    all_results.append(result_3layer)

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    baseline = result_2layer
    baseline_mem = baseline['memory']['total_mb']

    for result in all_results:
        mem = result['memory']['total_mb']
        savings = (baseline_mem - mem) / baseline_mem * 100 if baseline_mem > 0 else 0

        q = result['quality']
        status = "✅" if q['coherence'] >= 0.6 and q['repetition'] < 0.2 else "❌"

        print(f"\n{result['system']}:")
        print(f"  Memory: {mem:.1f} MB (savings: {savings:+.1f}%)")
        print(f"  Quality: {status} R:{q['repetition']*100:.0f}% C:{q['coherence']*100:.0f}% Rel:{q['relevance']*100:.0f}%")

    # Save
    with open("/tmp/triple_layer_long_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*80}")
    print("Complete!")


if __name__ == "__main__":
    main()
