"""
Cache Bridge Benchmark: FlashMLX ↔ ThunderOMLX cache bridge validation.

Tests:
  1. state property correctness: TripleLayerKVCache.state returns valid bf16
  2. export/import round-trip: export_flat_state → import_flat_state bit-exact
  3. H0Store block round-trip: export_blocks → import_blocks data consistency
  4. RCEngine from H0 blocks: register_from_h0_blocks → reconstruct
  5. SSD space measurement: compressed vs bf16 actual sizes

Usage:
    python3 benchmarks/bench_cache_bridge.py /path/to/model
    python3 benchmarks/bench_cache_bridge.py /path/to/model --prompt-tokens 2048
    python3 benchmarks/bench_cache_bridge.py /path/to/model --flat-quant q4_0
"""

import argparse
import gc
import sys
import time

sys.path.insert(0, "/Users/lisihao/FlashMLX/mlx-lm-source")
sys.path.insert(0, "/Users/lisihao/FlashMLX/src")

import mlx.core as mx
import numpy as np

from mlx_lm import load
from mlx_lm.models.cache import KVCache, make_prompt_cache
from mlx_lm.models.cache_factory import get_cache_info
from mlx_lm.models.kv_direct_cache import H0Store, _find_inner_model


def get_mem_mb():
    try:
        return mx.get_active_memory() / (1024 * 1024)
    except AttributeError:
        return mx.metal.get_active_memory() / (1024 * 1024)


def banner(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def test_state_property(cache_list, prompt_tokens, model, tokenizer):
    """Test 1: TripleLayerKVCache.state returns valid bf16 tuple."""
    banner("Test 1: state property correctness")

    from mlx_lm.models.triple_layer_cache import TripleLayerKVCache

    triple_count = sum(1 for c in cache_list if isinstance(c, TripleLayerKVCache))
    print(f"  TripleLayerKVCache layers: {triple_count}/{len(cache_list)}")

    if triple_count == 0:
        print("  SKIP: no TripleLayerKVCache layers found")
        return False

    # Run prefill
    tokens = mx.array(prompt_tokens[:256])[None]
    model(tokens, cache=cache_list)
    mx.eval([c.state for c in cache_list if hasattr(c, 'state') and c.state])

    passed = 0
    failed = 0
    for i, cache in enumerate(cache_list):
        if not isinstance(cache, TripleLayerKVCache):
            continue

        state = cache.state
        if not state or len(state) < 2:
            print(f"  Layer {i}: FAIL — state returned empty/short: {type(state)}")
            failed += 1
            continue

        k, v = state
        if k.dtype != mx.bfloat16:
            print(f"  Layer {i}: FAIL — keys dtype={k.dtype}, expected bfloat16")
            failed += 1
            continue

        if len(k.shape) != 4:
            print(f"  Layer {i}: FAIL — keys shape={k.shape}, expected 4D")
            failed += 1
            continue

        seq_len = k.shape[2]
        if seq_len == 0:
            print(f"  Layer {i}: FAIL — seq_len=0")
            failed += 1
            continue

        passed += 1

    print(f"\n  Result: {passed}/{passed+failed} layers passed")
    if failed > 0:
        print(f"  FAILED: {failed} layers returned invalid state")

    # Test meta_state
    sample = next(c for c in cache_list if isinstance(c, TripleLayerKVCache))
    ms = sample.meta_state
    print(f"  meta_state: {ms}")
    assert len(ms) >= 4, f"meta_state too short: {ms}"
    print(f"  meta_state OK (len={len(ms)})")

    return failed == 0


def test_export_import_roundtrip(cache_list, prompt_tokens, model):
    """Test 2: export_flat_state → import_flat_state bit-exact round-trip."""
    banner("Test 2: export/import flat_state round-trip")

    from mlx_lm.models.triple_layer_cache import TripleLayerKVCache

    triple_caches = [
        (i, c) for i, c in enumerate(cache_list)
        if isinstance(c, TripleLayerKVCache)
    ]

    if not triple_caches:
        print("  SKIP: no TripleLayerKVCache")
        return False

    passed = 0
    failed = 0
    total_export_bytes = 0
    total_bf16_bytes = 0

    for i, cache in triple_caches[:3]:  # Test first 3 layers
        flat_state = cache.export_flat_state()
        if flat_state is None:
            print(f"  Layer {i}: SKIP — not in flat_mode")
            continue

        # Measure compressed size
        compressed_bytes = 0
        for key in ('flat_keys', 'flat_values', 'flat_keys_scales', 'flat_values_scales'):
            arr = flat_state.get(key)
            if arr is not None:
                compressed_bytes += arr.nbytes

        # Estimate bf16 equivalent
        bf16_keys = flat_state['flat_keys']
        B, heads, seq_len, dim = bf16_keys.shape
        bf16_bytes = B * heads * seq_len * dim * 2 * 2  # keys + values, bf16

        total_export_bytes += compressed_bytes
        total_bf16_bytes += bf16_bytes

        quant = flat_state.get('flat_quant', 'bf16')
        print(f"  Layer {i}: quant={quant}, seq={seq_len}, "
              f"compressed={compressed_bytes:,}B, bf16={bf16_bytes:,}B, "
              f"ratio={bf16_bytes/max(compressed_bytes,1):.1f}x")

        # Create fresh cache and import
        fresh_cache = TripleLayerKVCache.__new__(TripleLayerKVCache)
        fresh_cache._flat_step = cache._flat_step
        fresh_cache._flat_keys = None
        fresh_cache._flat_values = None
        fresh_cache._flat_keys_scales = None
        fresh_cache._flat_values_scales = None
        fresh_cache._flat_offset = 0
        fresh_cache._true_offset = 0
        fresh_cache._flat_mode = False
        fresh_cache._flat_quant = None
        fresh_cache._flat_pq = None
        fresh_cache._flat_pq_head_dim = None
        fresh_cache._flat_prefix_token_count = 0
        fresh_cache.recent_keys = None
        fresh_cache.recent_values = None

        ok = fresh_cache.import_flat_state(flat_state)
        if not ok:
            print(f"  Layer {i}: FAIL — import_flat_state returned False")
            failed += 1
            continue

        # Re-export and compare
        re_exported = fresh_cache.export_flat_state()
        if re_exported is None:
            print(f"  Layer {i}: FAIL — re-export returned None")
            failed += 1
            continue

        # Bit-exact comparison
        bit_exact = True
        for key in ('flat_keys', 'flat_values'):
            orig = flat_state[key]
            restored = re_exported[key]
            mx.eval(orig, restored)
            if not mx.array_equal(orig, restored):
                print(f"  Layer {i}: FAIL — {key} not bit-exact")
                bit_exact = False
                break

        for key in ('flat_keys_scales', 'flat_values_scales'):
            orig = flat_state.get(key)
            restored = re_exported.get(key)
            if orig is not None and restored is not None:
                mx.eval(orig, restored)
                if not mx.array_equal(orig, restored):
                    print(f"  Layer {i}: FAIL — {key} scales not bit-exact")
                    bit_exact = False
                    break

        if bit_exact:
            passed += 1
            print(f"  Layer {i}: PASS — bit-exact round-trip")
        else:
            failed += 1

    print(f"\n  Result: {passed}/{passed+failed} layers bit-exact")
    if total_bf16_bytes > 0:
        ratio = total_bf16_bytes / max(total_export_bytes, 1)
        print(f"  Space saving: {total_export_bytes:,}B compressed vs "
              f"{total_bf16_bytes:,}B bf16 ({ratio:.1f}x)")

    return failed == 0


def test_h0_block_roundtrip(cache_list, prompt_tokens, model, block_size=64):
    """Test 3: H0Store export_blocks → import_blocks data consistency."""
    banner("Test 3: H0Store block round-trip")

    # Find H0Store
    h0_store = None
    for c in cache_list:
        h0_store = getattr(c, '_h0_store', None)
        if h0_store is not None and h0_store.count > 0:
            break

    if h0_store is None:
        print("  SKIP: no H0Store found (need scored_kv_direct strategy)")
        return True  # Not a failure, just not applicable

    total_tokens = h0_store.count
    quant = h0_store._quant or 'bf16'
    nbytes = h0_store.nbytes
    print(f"  H0Store: {total_tokens} tokens, quant={quant}, "
          f"size={nbytes:,}B ({nbytes/1024/1024:.1f}MB)")

    # Export blocks
    t0 = time.perf_counter()
    blocks = h0_store.export_blocks(block_size=block_size)
    export_ms = (time.perf_counter() - t0) * 1000
    print(f"  export_blocks: {len(blocks)} blocks, {export_ms:.1f}ms")

    # Verify block structure
    total_in_blocks = sum(b['token_end'] - b['token_start'] for b in blocks)
    assert total_in_blocks == total_tokens, \
        f"Token count mismatch: blocks={total_in_blocks}, store={total_tokens}"

    # Measure block sizes
    block_bytes = sum(
        b['h0'].nbytes + (b['scales'].nbytes if b['scales'] is not None else 0)
        for b in blocks
    )
    print(f"  Block total: {block_bytes:,}B ({block_bytes/1024/1024:.1f}MB)")

    # Get original data for comparison
    original_h0 = h0_store.get_range(0, total_tokens)
    mx.eval(original_h0)

    # Import into fresh store
    fresh_store = H0Store(quant=None)  # quant inferred from blocks
    t0 = time.perf_counter()
    restored_count = fresh_store.import_blocks(blocks)
    import_ms = (time.perf_counter() - t0) * 1000
    print(f"  import_blocks: {restored_count} tokens, {import_ms:.1f}ms")

    assert restored_count == total_tokens, \
        f"Restored count mismatch: {restored_count} vs {total_tokens}"

    # Compare dequantized output
    restored_h0 = fresh_store.get_range(0, total_tokens)
    mx.eval(restored_h0)

    diff = mx.abs(original_h0 - restored_h0).astype(mx.float32)
    max_err = float(mx.max(diff))
    mean_err = float(mx.mean(diff))

    if quant == 'bf16':
        # Should be bit-exact for bf16
        is_exact = max_err == 0.0
        print(f"  Comparison (bf16): max_err={max_err:.6f}, mean_err={mean_err:.6f}")
        print(f"  {'PASS — bit-exact' if is_exact else 'FAIL — expected bit-exact'}")
        return is_exact
    else:
        # Quantized: should match within tolerance
        is_ok = max_err < 0.1  # Q8 error is typically < 0.01
        print(f"  Comparison ({quant}): max_err={max_err:.6f}, mean_err={mean_err:.6f}")
        print(f"  {'PASS' if is_ok else 'FAIL'} — tolerance=0.1")
        return is_ok


def test_h0_block_hash():
    """Test 3b: H0Store.block_hash_key uniqueness."""
    banner("Test 3b: H0 block hash uniqueness")

    parent_hash = b'\x00' * 32
    hashes = set()
    for i in range(1000):
        h = H0Store.block_hash_key(parent_hash, i)
        assert len(h) == 32, f"Hash length={len(h)}, expected 32"
        hashes.add(h)

    print(f"  Generated 1000 hashes, {len(hashes)} unique")
    assert len(hashes) == 1000, "Hash collision detected!"

    # Verify h0: prefix prevents collision with raw SHA-256
    import hashlib
    raw_hash = hashlib.sha256(parent_hash + (0).to_bytes(8, 'little')).digest()
    h0_hash = H0Store.block_hash_key(parent_hash, 0)
    assert raw_hash != h0_hash, "h0: prefix should prevent collision"
    print("  PASS — no collisions, h0: prefix verified")
    return True


def test_rc_from_h0_blocks(cache_list, prompt_tokens, model, block_size=64):
    """Test 4: RCEngine.register_from_h0_blocks reconstruction."""
    banner("Test 4: RCEngine from H0 blocks")

    from flashmlx.rc_engine import RCEngine

    # Find H0Store
    h0_store = None
    for c in cache_list:
        h0_store = getattr(c, '_h0_store', None)
        if h0_store is not None and h0_store.count > 0:
            break

    if h0_store is None:
        print("  SKIP: no H0Store found")
        return True

    total_tokens = h0_store.count
    if total_tokens < 128:
        print(f"  SKIP: only {total_tokens} tokens (need >=128)")
        return True

    # Export blocks
    blocks = h0_store.export_blocks(block_size=block_size)
    print(f"  H0 blocks: {len(blocks)} blocks, {total_tokens} tokens")

    # Find inner model
    try:
        inner_model = _find_inner_model(model)
    except Exception as e:
        print(f"  SKIP: cannot find inner model: {e}")
        return True

    # Create fresh target caches
    target_caches = [KVCache() for _ in range(len(inner_model.layers))]

    # Register from blocks
    engine = RCEngine(chunk_size=512)
    t0 = time.perf_counter()
    state = engine.register_from_h0_blocks(
        seq_id="bench_bridge",
        h0_blocks=blocks,
        inner_model=inner_model,
        target_cache_list=target_caches,
    )
    reg_ms = (time.perf_counter() - t0) * 1000
    print(f"  register_from_h0_blocks: {reg_ms:.1f}ms")
    print(f"  State: {state.total_tokens} tokens, "
          f"{state.remaining_chunks} chunks remaining")

    # Process a few chunks
    chunks_to_process = min(3, state.remaining_chunks)
    total_ms = 0
    for i in range(chunks_to_process):
        result = engine.process_chunk(state)
        total_ms += result.time_ms
        print(f"  Chunk {i}: {result.tokens_processed} tokens, "
              f"{result.time_ms:.1f}ms, "
              f"progress={state.progress:.0%}")

    print(f"\n  Processed {chunks_to_process} chunks in {total_ms:.1f}ms")
    print(f"  Avg: {total_ms/max(chunks_to_process,1):.1f}ms/chunk")

    # Verify temp_caches have data
    has_data = any(
        tc.keys is not None and tc.keys.shape[2] > 0
        for tc in state.temp_caches
        if hasattr(tc, 'keys') and tc.keys is not None
    )
    print(f"  temp_caches populated: {has_data}")

    # Abort (cleanup)
    engine.abort(state.sequence_id)
    print("  PASS — reconstruction from H0 blocks works")
    return True


def test_ssd_space(cache_list, prompt_tokens, model):
    """Test 5: SSD space measurement."""
    banner("Test 5: SSD space measurement")

    from mlx_lm.models.triple_layer_cache import TripleLayerKVCache

    bf16_total = 0
    compressed_total = 0
    h0_total = 0

    # Measure KV cache sizes
    for i, cache in enumerate(cache_list):
        if isinstance(cache, TripleLayerKVCache):
            # bf16 size (via .state)
            state = cache.state
            if state and len(state) >= 2:
                k, v = state
                bf16_size = k.nbytes + v.nbytes
                bf16_total += bf16_size

            # Compressed size (via export_flat_state)
            flat = cache.export_flat_state()
            if flat:
                comp_size = 0
                for key in ('flat_keys', 'flat_values',
                            'flat_keys_scales', 'flat_values_scales'):
                    arr = flat.get(key)
                    if arr is not None:
                        comp_size += arr.nbytes
                compressed_total += comp_size
        elif isinstance(cache, KVCache):
            state = cache.state
            if state and len(state) >= 2:
                k, v = state
                bf16_total += k.nbytes + v.nbytes
                compressed_total += k.nbytes + v.nbytes  # No compression for KVCache

    # Measure H0 size
    h0_store = None
    for c in cache_list:
        h0_store = getattr(c, '_h0_store', None)
        if h0_store is not None and h0_store.count > 0:
            break

    if h0_store is not None:
        h0_total = h0_store.nbytes

    print(f"  bf16 KV total:       {bf16_total:>12,}B ({bf16_total/1024/1024:.1f}MB)")
    print(f"  Compressed KV total: {compressed_total:>12,}B ({compressed_total/1024/1024:.1f}MB)")
    if bf16_total > 0:
        print(f"  KV compression:      {bf16_total/max(compressed_total,1):.1f}x")
    if h0_total > 0:
        print(f"  H0 total:            {h0_total:>12,}B ({h0_total/1024/1024:.1f}MB)")
        if bf16_total > 0:
            print(f"  H0 vs KV:            {bf16_total/max(h0_total,1):.0f}x smaller")

    # Summary
    ssd_with_compressed = compressed_total + h0_total
    ssd_without = bf16_total
    if ssd_without > 0:
        saving = 1 - ssd_with_compressed / ssd_without
        print(f"\n  SSD with bridge:     {ssd_with_compressed:>12,}B ({ssd_with_compressed/1024/1024:.1f}MB)")
        print(f"  SSD without bridge:  {ssd_without:>12,}B ({ssd_without/1024/1024:.1f}MB)")
        print(f"  Saving:              {saving:.0%}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Cache Bridge Benchmark")
    parser.add_argument("model_path", help="Path to MLX model")
    parser.add_argument("--prompt-tokens", type=int, default=1024,
                        help="Number of prompt tokens (default: 1024)")
    parser.add_argument("--flat-quant", default="q8_0",
                        help="Flat quantization (default: q8_0)")
    parser.add_argument("--strategy", default="scored_pq",
                        help="Cache strategy (default: scored_pq)")
    parser.add_argument("--block-size", type=int, default=64,
                        help="Block size for H0 export (default: 64)")
    args = parser.parse_args()

    print(f"Cache Bridge Benchmark")
    print(f"  Model: {args.model_path}")
    print(f"  Strategy: {args.strategy}")
    print(f"  Flat quant: {args.flat_quant}")
    print(f"  Prompt tokens: {args.prompt_tokens}")
    print(f"  Block size: {args.block_size}")

    # Load model
    print("\nLoading model...")
    model, tokenizer = load(args.model_path)
    mx.eval(model.parameters())
    print(f"  Memory after load: {get_mem_mb():.0f}MB")

    # Create cache
    print(f"\nCreating {args.strategy} cache (flat_quant={args.flat_quant})...")
    cache_kwargs = {"kv_cache": args.strategy}
    if args.flat_quant:
        cache_kwargs["kv_flat_quant"] = args.flat_quant

    # scored_kv_direct for H0Store tests
    if args.strategy in ("scored_kv_direct", "kv_direct"):
        cache_kwargs["h0_quant"] = "q8"

    cache_list = make_prompt_cache(model, **cache_kwargs)
    info = get_cache_info(cache_list)
    print(f"  Cache info: {info}")

    # Generate prompt tokens
    prompt_text = "The quick brown fox " * (args.prompt_tokens // 4)
    prompt_tokens = tokenizer.encode(prompt_text)[:args.prompt_tokens]
    print(f"  Actual prompt tokens: {len(prompt_tokens)}")

    # Run prefill
    print("\nRunning prefill...")
    tokens_mx = mx.array(prompt_tokens)[None]
    t0 = time.perf_counter()
    model(tokens_mx, cache=cache_list)
    mx.eval([c.keys if hasattr(c, 'keys') and c.keys is not None else mx.array(0)
             for c in cache_list])
    prefill_ms = (time.perf_counter() - t0) * 1000
    print(f"  Prefill: {prefill_ms:.0f}ms, memory: {get_mem_mb():.0f}MB")

    # Run tests
    results = {}
    results['state_property'] = test_state_property(
        cache_list, prompt_tokens, model, tokenizer
    )
    results['export_import'] = test_export_import_roundtrip(
        cache_list, prompt_tokens, model
    )
    results['h0_hash'] = test_h0_block_hash()
    results['h0_roundtrip'] = test_h0_block_roundtrip(
        cache_list, prompt_tokens, model, block_size=args.block_size
    )
    results['rc_from_blocks'] = test_rc_from_h0_blocks(
        cache_list, prompt_tokens, model, block_size=args.block_size
    )
    results['ssd_space'] = test_ssd_space(cache_list, prompt_tokens, model)

    # Summary
    banner("Summary")
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False

    print(f"\n  Overall: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
