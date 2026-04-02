"""
3PIR Benchmark: Three-Phase Interleaved Reconstruction validation and performance.

Tests:
  1. Bit-exact correctness: chunked stateful vs blocking reconstruction
  2. RCEngine chunk-by-chunk performance
  3. Async API round-trip
  4. Simulated scheduler interleaving latency

Usage:
    python3 benchmarks/bench_3pir.py /path/to/model
    python3 benchmarks/bench_3pir.py /path/to/model --prompt-tokens 4096
    python3 benchmarks/bench_3pir.py /path/to/model --prompt-tokens 8192 --chunk-size 1024
"""

import argparse
import gc
import sys
import time

sys.path.insert(0, "/Users/lisihao/FlashMLX/mlx-lm-source")
sys.path.insert(0, "/Users/lisihao/FlashMLX/src")
sys.path.insert(0, "/Users/lisihao/ThunderOMLX/src")

import mlx.core as mx
import numpy as np

from flashmlx import ReconstructionController
from flashmlx.model_cards import load_card_or_detect
from flashmlx.rc_engine import RCEngine, RCSequenceState, RCChunkResult
from mlx_lm import load
from mlx_lm.models.cache import KVCache, make_prompt_cache
from mlx_lm.models.cache_factory import get_cache_info
from mlx_lm.models.kv_direct_cache import (
    BatchedH0View,
    _find_inner_model,
    extract_kv_from_temp_caches,
    reconstruct_prefix_kv,
    reconstruct_prefix_kv_stateful,
    unpatch_model,
)


def get_mem_mb():
    try:
        return mx.get_active_memory() / (1024 * 1024)
    except AttributeError:
        return mx.metal.get_active_memory() / (1024 * 1024)


FILLER_PARA = (
    "The development of artificial intelligence has progressed rapidly in recent years. "
    "Machine learning algorithms continue to improve across various benchmarks. Research "
    "teams around the world are exploring new architectures for language understanding. "
    "The computational requirements for training large models have grown exponentially. "
    "Transfer learning has enabled smaller teams to build on pre-trained foundations. "
    "Ethical considerations remain central to AI development discussions globally. "
)


def build_filler_prompt(tokenizer, target_tokens):
    filler_tokens = len(tokenizer.encode(FILLER_PARA))
    n_paras = (target_tokens // filler_tokens) + 5
    prefix = "Read the following document carefully.\n\n"
    text = prefix + FILLER_PARA * n_paras
    tokens = tokenizer.encode(text)
    if len(tokens) > target_tokens:
        tokens = tokens[:target_tokens]
        text = tokenizer.decode(tokens)
    return text, len(tokens)


def prefill_and_get_h0(model, tokenizer, prompt, cache_kwargs):
    """Run prefill to fill h0_store, return (cache, h0_store, inner_model)."""
    unpatch_model(model)
    cache = make_prompt_cache(model, **cache_kwargs)

    prompt_tokens = mx.array(tokenizer.encode(prompt))
    out = model(prompt_tokens.reshape(1, -1), cache=cache)
    mx.eval(out)

    info = get_cache_info(cache)
    h0_tokens = info.get("h0_count", 0) or info.get("h0_tokens", 0)

    # Find h0_store
    h0_store = None
    for c in cache:
        h0_store = getattr(c, "_h0_store", None)
        if h0_store is not None:
            break

    inner_model = _find_inner_model(model)
    return cache, h0_store, inner_model, h0_tokens


# ---------------------------------------------------------------------------
# Test 1: Bit-exact correctness
# ---------------------------------------------------------------------------

def test_bit_exact(model, tokenizer, prompt, cache_kwargs, chunk_size=512):
    """Verify that chunked stateful reconstruction produces identical K/V."""
    print("\n=== Test 1: Bit-exact correctness ===")
    print(f"  chunk_size={chunk_size}")

    gc.collect(); mx.clear_cache()
    cache, h0_store, inner_model, h0_tokens = prefill_and_get_h0(
        model, tokenizer, prompt, cache_kwargs
    )

    if h0_tokens == 0:
        print("  SKIP: no h0 tokens available")
        return False

    print(f"  h0_tokens={h0_tokens}")

    # Method A: Blocking reconstruction (ground truth)
    t0 = time.perf_counter()
    kv_blocking = reconstruct_prefix_kv(inner_model, h0_store, 0, h0_tokens, chunk_size=0)
    mx.eval(*[k for pair in kv_blocking for k in pair])
    blocking_ms = (time.perf_counter() - t0) * 1000

    # Method B: Stateful chunked reconstruction
    num_layers = len(inner_model.layers)
    temp_caches = [KVCache() for _ in range(num_layers)]
    h0_full = h0_store.get_range(0, h0_tokens)
    mx.eval(h0_full)

    t0 = time.perf_counter()
    for offset in range(0, h0_tokens, chunk_size):
        end = min(offset + chunk_size, h0_tokens)
        chunk = h0_full[:, offset:end, :]
        reconstruct_prefix_kv_stateful(inner_model, chunk, temp_caches)
        mx.eval(temp_caches[-1].keys)
    chunked_ms = (time.perf_counter() - t0) * 1000

    kv_chunked = extract_kv_from_temp_caches(temp_caches)

    # Compare K/V for every layer
    all_match = True
    max_diff_k = 0.0
    max_diff_v = 0.0

    for i in range(num_layers):
        k_block, v_block = kv_blocking[i]
        k_chunk, v_chunk = kv_chunked[i]

        # Check shapes match
        if k_block.shape != k_chunk.shape or v_block.shape != v_chunk.shape:
            print(f"  Layer {i}: SHAPE MISMATCH "
                  f"block={k_block.shape} vs chunk={k_chunk.shape}")
            all_match = False
            continue

        # Check values
        diff_k = mx.abs(k_block - k_chunk).max().item()
        diff_v = mx.abs(v_block - v_chunk).max().item()
        max_diff_k = max(max_diff_k, diff_k)
        max_diff_v = max(max_diff_v, diff_v)

        if diff_k > 1e-5 or diff_v > 1e-5:
            print(f"  Layer {i}: MISMATCH k_diff={diff_k:.6f} v_diff={diff_v:.6f}")
            all_match = False

    status = "PASS" if all_match else "FAIL"
    print(f"\n  Result: {status}")
    print(f"  Max diff: keys={max_diff_k:.8f}, values={max_diff_v:.8f}")
    print(f"  Blocking: {blocking_ms:.0f}ms | Chunked: {chunked_ms:.0f}ms "
          f"({chunked_ms/blocking_ms:.2f}x)")

    return all_match


# ---------------------------------------------------------------------------
# Test 2: RCEngine chunk-by-chunk performance
# ---------------------------------------------------------------------------

def test_rc_engine(model, tokenizer, prompt, cache_kwargs, chunk_size=512):
    """Benchmark RCEngine chunk-by-chunk reconstruction."""
    print("\n=== Test 2: RCEngine chunk-by-chunk performance ===")
    print(f"  chunk_size={chunk_size}")

    gc.collect(); mx.clear_cache()
    cache, h0_store, inner_model, h0_tokens = prefill_and_get_h0(
        model, tokenizer, prompt, cache_kwargs
    )

    if h0_tokens == 0:
        print("  SKIP: no h0 tokens available")
        return

    # Create engine and register
    engine = RCEngine(chunk_size=chunk_size)
    mem_before = get_mem_mb()

    t0 = time.perf_counter()
    state = engine.register_sequence(
        seq_id="bench_agent",
        h0_store=h0_store,
        inner_model=inner_model,
        target_cache_list=[c for c in cache if hasattr(c, "inject_reconstruction")],
    )
    reg_ms = (time.perf_counter() - t0) * 1000

    print(f"  Registration: {reg_ms:.0f}ms (h0 dequant + temp_cache creation)")
    print(f"  Target: {state.total_tokens} tokens, "
          f"{(state.total_tokens + chunk_size - 1) // chunk_size} chunks")

    # Process chunks
    chunk_times = []
    total_t0 = time.perf_counter()

    while not state.is_complete:
        result = engine.process_chunk(state)
        chunk_times.append(result.time_ms)

    total_ms = (time.perf_counter() - total_t0) * 1000

    # Inject
    t_inject = time.perf_counter()
    layers, mem_mb = engine.inject_completed(state)
    inject_ms = (time.perf_counter() - t_inject) * 1000

    mem_after = get_mem_mb()

    print(f"\n  Chunks processed: {len(chunk_times)}")
    print(f"  Per-chunk: min={min(chunk_times):.1f}ms, "
          f"max={max(chunk_times):.1f}ms, "
          f"avg={sum(chunk_times)/len(chunk_times):.1f}ms, "
          f"p50={sorted(chunk_times)[len(chunk_times)//2]:.1f}ms")
    print(f"  Total reconstruction: {total_ms:.0f}ms")
    print(f"  Injection: {inject_ms:.0f}ms ({layers} layers, {mem_mb:.1f}MB)")
    print(f"  Memory delta: {mem_after - mem_before:.0f}MB")

    # Simulated 3PIR interleaving latency
    tg_step_ms = 12.7  # Typical M4 Max decode step
    n_chunks = len(chunk_times)
    avg_chunk_ms = sum(chunk_times) / n_chunks
    interleaved_ms = n_chunks * tg_step_ms  # RC hidden behind TG steps
    blocking_ms = total_ms

    print(f"\n  --- 3PIR Benefit ---")
    print(f"  Blocking RC latency: {blocking_ms:.0f}ms (stops all TG)")
    print(f"  3PIR perceived latency: {interleaved_ms:.0f}ms "
          f"(hidden behind {n_chunks} TG steps @ {tg_step_ms:.1f}ms/step)")
    print(f"  Speedup: {blocking_ms/interleaved_ms:.1f}x perception, "
          f"TG unblocked")
    print(f"  RC overhead per TG step: +{avg_chunk_ms:.1f}ms "
          f"({avg_chunk_ms/tg_step_ms*100:.0f}% of TG step)")


# ---------------------------------------------------------------------------
# Test 3: Async API round-trip
# ---------------------------------------------------------------------------

def test_async_api(model, tokenizer, prompt, cache_kwargs, chunk_size=512):
    """Test ReconstructionController async API end-to-end."""
    print("\n=== Test 3: Async API round-trip ===")

    gc.collect(); mx.clear_cache()
    cache, h0_store, inner_model, h0_tokens = prefill_and_get_h0(
        model, tokenizer, prompt, cache_kwargs
    )

    if h0_tokens == 0:
        print("  SKIP: no h0 tokens available")
        return

    recon = ReconstructionController.from_cache(cache, model)
    if not recon.available:
        print("  SKIP: reconstruction not available")
        return

    print(f"  h0_tokens={h0_tokens}, probe={recon.stats.probe_available}")

    # Start async reconstruction
    t0 = time.perf_counter()
    state = recon.reconstruct_async_start(
        strategy="full",
        chunk_size=chunk_size,
        seq_id="async_bench",
    )

    if state is None:
        print("  FAIL: reconstruct_async_start returned None")
        return

    # Process chunks
    n_chunks = 0
    while not state.is_complete:
        result = recon.reconstruct_async_step(state)
        n_chunks += 1

    # Complete
    final_result = recon.reconstruct_async_complete(state)
    total_ms = (time.perf_counter() - t0) * 1000

    print(f"  Chunks: {n_chunks}")
    print(f"  Result: success={final_result.success}, "
          f"tokens={final_result.tokens_reconstructed}, "
          f"layers={final_result.layers_injected}")
    print(f"  Total: {total_ms:.0f}ms (API) / {final_result.time_ms:.0f}ms (engine)")
    print(f"  Status: {'PASS' if final_result.success else 'FAIL'}")


# ---------------------------------------------------------------------------
# Test 4: RCScheduler simulation
# ---------------------------------------------------------------------------

def test_scheduler_sim(model, tokenizer, prompt, cache_kwargs, chunk_size=512):
    """Simulate scheduler interleaving with RC."""
    print("\n=== Test 4: Scheduler interleaving simulation ===")

    gc.collect(); mx.clear_cache()
    cache, h0_store, inner_model, h0_tokens = prefill_and_get_h0(
        model, tokenizer, prompt, cache_kwargs
    )

    if h0_tokens == 0:
        print("  SKIP: no h0 tokens available")
        return

    from omlx.rc_scheduler import RCScheduler, RCBudget, RCRequest

    engine = RCEngine(chunk_size=chunk_size)
    budget = RCBudget()
    sched = RCScheduler(rc_engine=engine, budget=budget)

    # Enqueue request
    target_caches = [c for c in cache if hasattr(c, "inject_reconstruction")]
    req = RCRequest(
        request_id="sim_agent_a",
        h0_store=h0_store,
        cache_list=target_caches,
        inner_model=inner_model,
    )
    sched.add_rc_request(req)

    # Simulate scheduler steps with varying decode load
    scenarios = [
        ("idle (0 decode)", 0, 0),
        ("light (2 decode)", 2, 0),
        ("heavy (4 decode)", 4, 0),
    ]

    for label, n_dec, n_pp in scenarios:
        max_chunks, time_ms = budget.compute(n_dec, n_pp)
        print(f"  Budget [{label}]: max_chunks={max_chunks}, time={time_ms}ms")

    # Run with 2 decode agents (light load)
    print(f"\n  Running with 2 decode agents (light load)...")
    step_count = 0
    t0 = time.perf_counter()

    while sched.has_work():
        results = sched.try_rc_step(num_decoding=2, num_prefilling=0)
        step_count += 1
        if step_count > 500:  # safety limit
            break

    total_ms = (time.perf_counter() - t0) * 1000
    stats = sched.stats()

    print(f"  Steps: {step_count}")
    print(f"  Total RC time: {total_ms:.0f}ms")
    print(f"  Chunks processed: {stats['total_chunks']}")
    print(f"  Injections: {stats['total_injections']}")
    print(f"  Avg RC per step: {total_ms/max(step_count,1):.1f}ms")

    # Simulated TG impact
    tg_step_ms = 12.7
    overhead_pct = (total_ms / step_count / tg_step_ms) * 100 if step_count > 0 else 0
    print(f"  TG overhead: {overhead_pct:.1f}% per step (at {tg_step_ms}ms/step)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="3PIR Benchmark")
    parser.add_argument("model_path", help="Path to model directory")
    parser.add_argument("--prompt-tokens", type=int, default=4096)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--skip-bitexact", action="store_true",
                        help="Skip bit-exact test (slow for large prompts)")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    model, tokenizer = load(args.model_path)

    card = load_card_or_detect(model, args.model_path)
    print(f"Card: {card.model_name}")

    # Use recall_first mode if available, else fallback to scored_kv_direct
    if card.modes and "recall_first" in card.modes:
        cache_kwargs = card.to_cache_kwargs(mode="recall_first")
        cache_kwargs = {k: v for k, v in cache_kwargs.items() if k != "auto_reconstruct"}
    else:
        # Fallback: manually create scored_kv_direct config (enables h^(0))
        cache_kwargs = {
            "kv_cache": "scored_kv_direct",
            "kv_compression_ratio": 0.1,  # 10x compression, aggressive eviction
        }
    print(f"Cache kwargs: {cache_kwargs}")

    # Build prompt
    print(f"\nBuilding {args.prompt_tokens:,}-token filler prompt...")
    prompt, actual_tokens = build_filler_prompt(tokenizer, args.prompt_tokens)
    print(f"  Actual: {actual_tokens:,} tokens")

    # Warmup
    print("Warming up...")
    warm = mx.array(tokenizer.encode("Hello")).reshape(1, -1)
    model(warm)
    mx.eval(model.parameters())

    print(f"\n{'='*80}")
    print(f"  3PIR Benchmark | {card.model_name} | {actual_tokens:,} tokens | chunk={args.chunk_size}")
    print(f"{'='*80}")

    # Run tests
    if not args.skip_bitexact:
        test_bit_exact(model, tokenizer, prompt, cache_kwargs, args.chunk_size)

    test_rc_engine(model, tokenizer, prompt, cache_kwargs, args.chunk_size)
    test_async_api(model, tokenizer, prompt, cache_kwargs, args.chunk_size)
    test_scheduler_sim(model, tokenizer, prompt, cache_kwargs, args.chunk_size)

    print(f"\n{'='*80}")
    print("  Done!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
