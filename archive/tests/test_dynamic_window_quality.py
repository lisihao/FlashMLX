#!/usr/bin/env python3
"""
动态 Recent Window 质量测试
重点：验证减小 window 是否影响生成质量
"""

import time
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.models.cache import make_prompt_cache
from pathlib import Path

# 3384 tokens agent workload (已验证)
AGENT_WORKLOAD = """[Session Start: 2024-03-15 14:23:45]

Agent: I'm investigating a critical bug in our production system. The application intermittently returns 500 errors during peak hours, but the logs don't show any obvious patterns. Here's what I know so far:

1. Error Pattern Analysis:
   - Occurs 2-3 times per hour during 9 AM - 5 PM
   - No errors during off-peak hours (6 PM - 8 AM)
   - Affects random API endpoints (not endpoint-specific)
   - Database query performance seems normal
   - No resource exhaustion (CPU, memory, disk all below 60%)

2. Recent Changes:
   - Deployed new caching layer 3 days ago
   - Updated database connection pool settings 1 week ago
   - No infrastructure changes in the past month

3. Investigation Steps Taken:
   - Reviewed application logs: Nothing suspicious
   - Checked database slow query log: All queries under 100ms
   - Monitored resource utilization: All within normal ranges
   - Examined error stack traces: Generic connection timeout errors

4. Observations:
   - Errors correlate with high concurrent user count (200+ simultaneous users)
   - Same request succeeds when retried immediately
   - Average response time increases by 30% during error windows
   - Connection pool metrics show occasional spikes to max capacity

Let me analyze the database connection pool configuration:

Current Settings:
```python
DATABASE_CONFIG = {
    'pool_size': 10,
    'max_overflow': 20,
    'pool_timeout': 30,
    'pool_recycle': 3600,
    'echo': False
}
```

Peak Load Metrics:
- Concurrent requests: 200-250
- Average request duration: 150ms
- Database queries per request: 3-5
- Connection acquisition time: Usually < 10ms, spikes to 2000ms during errors

Now let me trace through a typical request flow:

1. Request arrives at load balancer
2. Forwarded to application server (1 of 5 instances)
3. Application acquires database connection from pool
4. Executes 3-5 queries (total ~50ms)
5. Processes business logic (~50ms)
6. Releases connection back to pool
7. Returns response (total ~150ms)

Hypothesis: The connection pool is undersized for peak load. With 5 app instances and pool_size=10, we have 50 total connections. At 200 concurrent requests with 150ms duration, we need approximately:

Required connections = (200 requests / 5 instances) × (150ms / 1000ms) × 5 queries = 30 connections per instance

But we only have 10 + 20 (overflow) = 30 total per instance. When all 30 are in use, new requests timeout waiting for available connections.

Wait, let me reconsider. The error messages say "connection timeout" but the pool metrics show we're not hitting max capacity consistently. Let me check if there's a connection leak:

Connection Lifecycle Analysis:
- Connections acquired: 50,000/hour
- Connections released: 49,997/hour
- Delta: -3 connections/hour

That's it! There's a small connection leak. Over 8 hours (9 AM - 5 PM), we leak approximately 24 connections. Combined with the max_overflow of 20, once we leak 20+ connections, new requests start timing out.

Code Review Findings:
Looking at the recent caching layer changes, I found this pattern in multiple places:

```python
def process_request(data):
    conn = db_pool.get_connection()
    try:
        result = execute_query(conn, data)
        cache.set(data.cache_key, result)
        return result
    except CacheError as e:
        logger.error(f"Cache error: {e}")
        return None  # BUG: Connection not released!
```

The bug: When cache.set() raises an exception, we return early without releasing the connection. The finally block that should release the connection is missing.

Solution:
```python
def process_request(data):
    conn = db_pool.get_connection()
    try:
        result = execute_query(conn, data)
        cache.set(data.cache_key, result)
        return result
    except CacheError as e:
        logger.error(f"Cache error: {e}")
        return None
    finally:
        db_pool.release_connection(conn)  # Always release
```

Additional findings:
- This pattern exists in 7 different API handlers
- Cache errors occur ~3 times per hour (matches error frequency!)
- Each leaked connection stays open until pool_recycle (3600s = 1 hour)

Root Cause Summary:
The recent caching layer introduced exception handling that doesn't properly release database connections. During peak hours, cache errors cause connection leaks. After several hours, the leaked connections accumulate, exhausting the pool and causing timeouts.

Verification Plan:
1. Search codebase for similar patterns: grep -r "get_connection" | grep -v "finally"
2. Add connection pool monitoring to track active/leaked connections
3. Deploy fix with proper finally blocks
4. Monitor for 24 hours to confirm resolution

Expected Outcome:
- Zero connection leaks
- No more 500 errors during peak hours
- Stable connection pool utilization
- Improved average response times

What's the next step? Should I prepare a pull request with the fixes?"""

def test_window_size(window_size: int, model_path: str, num_generate: int = 100):
    """Test specific window size and measure quality"""
    print(f"\n{'='*70}")
    print(f"Testing Recent Window Size: {window_size}")
    print(f"{'='*70}")

    # Load model
    model, tokenizer = load(model_path)

    # Create custom cache with specific window size
    from mlx_lm.models.double_layer_cache import DoubleLayerKVCache

    cache_config = {
        "enable_compression": True,
        "compression_ratio": 1.5,  # Uniform R1.5 (已验证安全)
        "memory_budget_mb": 2.0,
        "recent_window_size": window_size,  # 动态调整
        "calibration_dir": "/tmp/am_calibrations_ultra_dense",
        "selection_strategy": "nearest"
    }

    # Create cache instances for all layers
    num_layers = len(model.model.layers)
    caches = []
    for layer_idx in range(num_layers):
        cache = DoubleLayerKVCache(
            layer_idx=layer_idx,
            **cache_config
        )
        caches.append(cache)

    prompt_cache = make_prompt_cache(model, caches=caches)

    # Tokenize
    prompt_tokens = mx.array(tokenizer.encode(AGENT_WORKLOAD))

    # Prefill
    print(f"\n[Prefill] Processing {len(prompt_tokens)} tokens...")
    start_time = time.time()
    prompt_cache.update_and_fetch(prompt_tokens)
    prefill_time = time.time() - start_time
    print(f"  ✓ Prefill done in {prefill_time:.2f}s")

    # Check memory usage
    cache_memory = sum(cache.nbytes for cache in caches) / 1024**2
    print(f"  Cache memory: {cache_memory:.2f} MB")

    # Generate
    print(f"\n[Generate] Generating {num_generate} tokens...")
    start_time = time.time()

    response = generate(
        model,
        tokenizer,
        prompt=AGENT_WORKLOAD,
        max_tokens=num_generate,
        verbose=False,
        prompt_cache=prompt_cache
    )

    generate_time = time.time() - start_time
    tg_speed = num_generate / generate_time

    print(f"  ✓ Generated {num_generate} tokens")
    print(f"  TG speed: {tg_speed:.2f} tokens/sec ({generate_time:.3f}s)")
    print(f"  Memory: {cache_memory:.2f} MB")

    # Extract response (remove prompt)
    output = response[len(AGENT_WORKLOAD):].strip()

    print(f"\n[Output Preview] First 200 chars:")
    print(f"  {output[:200]}")

    # Quality checks
    print(f"\n[Quality Checks]")

    # Check 1: Length
    output_tokens = len(tokenizer.encode(output))
    print(f"  Output length: {output_tokens} tokens")
    if output_tokens < num_generate * 0.8:
        print(f"  ⚠️  Warning: Output shorter than expected")

    # Check 2: Repetition detection
    lines = output.split('\n')
    unique_lines = len(set(lines))
    total_lines = len(lines)
    repetition_ratio = 1.0 - (unique_lines / total_lines) if total_lines > 0 else 0
    print(f"  Repetition ratio: {repetition_ratio:.2%}")
    if repetition_ratio > 0.3:
        print(f"  ⚠️  Warning: High repetition detected")

    # Check 3: Coherence (simple heuristic: check for common words)
    common_words = ["the", "and", "is", "to", "of", "in", "that", "for"]
    word_count = sum(1 for word in common_words if word in output.lower())
    print(f"  Common words present: {word_count}/{len(common_words)}")
    if word_count < 4:
        print(f"  ⚠️  Warning: Unusual word distribution (possible gibberish)")

    # Check 4: Response relevance (check if it continues the context)
    relevant_keywords = ["connection", "database", "fix", "solution", "code", "bug"]
    relevant_count = sum(1 for kw in relevant_keywords if kw in output.lower())
    print(f"  Relevant keywords: {relevant_count}/{len(relevant_keywords)}")
    if relevant_count < 2:
        print(f"  ⚠️  Warning: Low relevance to context")

    return {
        "window_size": window_size,
        "cache_memory_mb": cache_memory,
        "output_tokens": output_tokens,
        "tg_speed": tg_speed,
        "repetition_ratio": repetition_ratio,
        "coherence_score": word_count / len(common_words),
        "relevance_score": relevant_count / len(relevant_keywords),
        "output": output
    }

def main():
    model_path = "/Volumes/toshiba/models/qwen3-8b-mlx"

    print("="*70)
    print("🔬 动态 Recent Window 质量测试")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Workload: 3384 tokens agent debugging session")
    print(f"Compression: Uniform R1.5 (已验证)")
    print(f"Generate: 100 tokens")
    print()

    # Test different window sizes
    window_sizes = [512, 256, 128]  # 从 baseline 到激进
    results = []

    for window_size in window_sizes:
        try:
            result = test_window_size(window_size, model_path, num_generate=100)
            results.append(result)
        except Exception as e:
            print(f"\n❌ Error testing window_size={window_size}: {e}")
            import traceback
            traceback.print_exc()

    # Summary comparison
    print("\n" + "="*70)
    print("📊 Summary: Window Size Quality Comparison")
    print("="*70)
    print()

    # Header
    print(f"{'Window Size':<15} {'Memory (MB)':<15} {'TG Speed':<15} {'Repetition':<15} {'Coherence':<15} {'Relevance':<15}")
    print("-"*90)

    # Baseline (512)
    baseline = results[0] if results else None

    for result in results:
        window = result["window_size"]
        memory = result["cache_memory_mb"]
        speed = result["tg_speed"]
        repetition = result["repetition_ratio"]
        coherence = result["coherence_score"]
        relevance = result["relevance_score"]

        # Compare with baseline
        if baseline and window != baseline["window_size"]:
            memory_diff = (baseline["cache_memory_mb"] - memory) / baseline["cache_memory_mb"] * 100
            memory_str = f"{memory:.1f} MB ({memory_diff:+.1f}%)"
        else:
            memory_str = f"{memory:.1f} MB (baseline)"

        print(f"{window:<15} {memory_str:<15} {speed:<15.2f} {repetition:<15.2%} {coherence:<15.2%} {relevance:<15.2%}")

    # Quality analysis
    print("\n" + "="*70)
    print("🎯 Quality Analysis")
    print("="*70)

    if len(results) >= 2:
        baseline = results[0]
        optimized = results[1]  # window=256
        aggressive = results[2] if len(results) > 2 else None

        print(f"\nBaseline (512) vs Optimized (256):")
        print(f"  Memory savings: {(baseline['cache_memory_mb'] - optimized['cache_memory_mb']) / baseline['cache_memory_mb'] * 100:.1f}%")
        print(f"  Repetition change: {optimized['repetition_ratio'] - baseline['repetition_ratio']:+.2%}")
        print(f"  Coherence change: {optimized['coherence_score'] - baseline['coherence_score']:+.2%}")
        print(f"  Relevance change: {optimized['relevance_score'] - baseline['relevance_score']:+.2%}")

        # Quality verdict
        quality_ok = (
            abs(optimized['repetition_ratio'] - baseline['repetition_ratio']) < 0.1 and
            abs(optimized['coherence_score'] - baseline['coherence_score']) < 0.2 and
            abs(optimized['relevance_score'] - baseline['relevance_score']) < 0.2
        )

        if quality_ok:
            print(f"\n✅ Quality verdict: Window=256 maintains quality!")
        else:
            print(f"\n⚠️  Quality verdict: Window=256 may have quality issues")

        if aggressive:
            print(f"\nBaseline (512) vs Aggressive (128):")
            print(f"  Memory savings: {(baseline['cache_memory_mb'] - aggressive['cache_memory_mb']) / baseline['cache_memory_mb'] * 100:.1f}%")
            print(f"  Repetition change: {aggressive['repetition_ratio'] - baseline['repetition_ratio']:+.2%}")
            print(f"  Coherence change: {aggressive['coherence_score'] - baseline['coherence_score']:+.2%}")
            print(f"  Relevance change: {aggressive['relevance_score'] - baseline['relevance_score']:+.2%}")

            quality_ok = (
                abs(aggressive['repetition_ratio'] - baseline['repetition_ratio']) < 0.1 and
                abs(aggressive['coherence_score'] - baseline['coherence_score']) < 0.2 and
                abs(aggressive['relevance_score'] - baseline['relevance_score']) < 0.2
            )

            if quality_ok:
                print(f"\n✅ Quality verdict: Window=128 maintains quality!")
            else:
                print(f"\n⚠️  Quality verdict: Window=128 may have quality issues")

    # Save detailed outputs
    print("\n" + "="*70)
    print("💾 Saving detailed outputs...")
    print("="*70)

    for result in results:
        window = result["window_size"]
        output_file = f"/tmp/dynamic_window_output_{window}.txt"
        with open(output_file, 'w') as f:
            f.write(f"Window Size: {window}\n")
            f.write(f"Memory: {result['cache_memory_mb']:.2f} MB\n")
            f.write(f"TG Speed: {result['tg_speed']:.2f} tok/s\n")
            f.write(f"Repetition: {result['repetition_ratio']:.2%}\n")
            f.write(f"Coherence: {result['coherence_score']:.2%}\n")
            f.write(f"Relevance: {result['relevance_score']:.2%}\n")
            f.write("\n" + "="*70 + "\n")
            f.write("Output:\n")
            f.write("="*70 + "\n")
            f.write(result['output'])
        print(f"  Saved: {output_file}")

    print("\n✅ Test complete!")

if __name__ == "__main__":
    main()
