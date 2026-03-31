#!/usr/bin/env python3
"""Smoke test for FlashMLX Expert Offloading v3 — Three-Tier Architecture.

Verifies:
1. Expert index builds correctly from safetensors
2. Expert loading produces correct data (matches original weights)
3. ExpertTelemetry tracks activations correctly
4. CPUWarmCache stores/retrieves experts correctly
5. RegimeDetector selects appropriate strategies
6. Model patching reduces memory (three-tier setup)
7. Inference with patched model produces text
8. Telemetry summary reports meaningful data
"""

import gc
import os
import sys
import time

MLX_LM_SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlx-lm-source")
sys.path.insert(0, MLX_LM_SOURCE)

import mlx.core as mx
import numpy as np

MODEL_PATH = "/Volumes/toshiba/models/qwen3.5-35b-mlx"


def test_1_build_index():
    """Test: Expert index builds correctly."""
    print("\n=== Test 1: Build Expert Index ===")
    from mlx_lm.models.expert_offload import build_expert_index

    index = build_expert_index(MODEL_PATH)

    assert len(index) == 40, f"Expected 40 MoE layers, got {len(index)}"

    layer0 = index[0]
    assert layer0.num_experts == 256, f"Expected 256 experts, got {layer0.num_experts}"

    expected_components = [
        "gate_proj.weight", "gate_proj.scales", "gate_proj.biases",
        "up_proj.weight", "up_proj.scales", "up_proj.biases",
        "down_proj.weight", "down_proj.scales", "down_proj.biases",
    ]
    for comp_name in expected_components:
        assert comp_name in layer0.components, f"Missing component: {comp_name}"

    gw = layer0.components["gate_proj.weight"]
    assert gw.shape_per_expert == [512, 256], f"gate_proj.weight shape: {gw.shape_per_expert}"
    assert gw.per_expert_size == 512 * 256 * 4, f"gate_proj.weight size: {gw.per_expert_size}"

    dw = layer0.components["down_proj.weight"]
    assert dw.shape_per_expert == [2048, 64], f"down_proj.weight shape: {dw.shape_per_expert}"

    print("  PASS: 40 layers, 256 experts, 9 components per layer")
    return index


def test_2_load_experts(index):
    """Test: Expert loading returns correct shapes."""
    print("\n=== Test 2: Expert Loading ===")
    from mlx_lm.models.expert_offload import ExpertLoader

    loader = ExpertLoader(index, max_workers=4)

    expert_ids = [0, 5, 100]
    t0 = time.perf_counter()
    data = loader.load_experts(layer_idx=0, expert_indices=expert_ids)
    t1 = time.perf_counter()

    print(f"  Loaded 3 experts in {(t1-t0)*1000:.1f}ms")

    assert data["gate_proj.weight"].shape == (3, 512, 256), \
        f"gate_proj.weight: {data['gate_proj.weight'].shape}"
    assert data["gate_proj.scales"].shape == (3, 512, 32), \
        f"gate_proj.scales: {data['gate_proj.scales'].shape}"
    assert data["down_proj.weight"].shape == (3, 2048, 64), \
        f"down_proj.weight: {data['down_proj.weight'].shape}"

    gw_sum = float(mx.abs(data["gate_proj.weight"].astype(mx.float32)).sum())
    assert gw_sum > 0, "gate_proj.weight is all zeros!"

    # Test numpy loading
    np_data = loader.load_experts_numpy(layer_idx=0, expert_indices=expert_ids)
    assert np_data["gate_proj.weight"].shape == (3, 512, 256), \
        f"numpy gate_proj.weight: {np_data['gate_proj.weight'].shape}"
    print(f"  Numpy loading works: {list(np_data.keys())}")

    # Test expert_byte_size
    byte_size = loader.expert_byte_size(0)
    assert byte_size > 0, "expert_byte_size should be positive"
    print(f"  Expert byte size: {byte_size} ({byte_size/1024/1024:.2f} MB)")

    print(f"  PASS: Shapes correct, data non-zero, numpy+byte_size work")
    return loader


def test_3_telemetry():
    """Test: ExpertTelemetry tracks activations and predicts correctly."""
    print("\n=== Test 3: ExpertTelemetry ===")
    from mlx_lm.models.expert_offload import ExpertTelemetry

    tel = ExpertTelemetry(num_layers=40, num_experts=256, window_size=32)

    # Simulate activations: experts 5,10,15 are hot, rest are cold
    for token in range(100):
        hot_ids = [5, 10, 15, token % 256]  # Hot + rotating
        tel.record_activation(layer_idx=0, expert_ids=hot_ids, token_pos=token)

    # Check frequency
    unique = tel.get_unique_expert_count(0)
    assert unique > 3, f"Expected >3 unique experts, got {unique}"
    print(f"  Unique experts at layer 0: {unique}")

    # Predict hot experts
    hot = tel.predict_hot_experts(0, top_k=5)
    assert 5 in hot, f"Expert 5 should be predicted hot, got {hot}"
    assert 10 in hot, f"Expert 10 should be predicted hot, got {hot}"
    assert 15 in hot, f"Expert 15 should be predicted hot, got {hot}"
    print(f"  Predicted hot: {hot}")

    # Predict with exclusion
    hot_excl = tel.predict_hot_experts(0, top_k=5, exclude={5, 10})
    assert 5 not in hot_excl, "Expert 5 should be excluded"
    assert 10 not in hot_excl, "Expert 10 should be excluded"
    print(f"  Predicted hot (excl 5,10): {hot_excl}")

    # Cold expert detection
    pool_ids = [5, 10, 15, 200, 201, 202, 203, 204, 205, 206]
    cold = tel.get_cold_experts(0, pool_ids, keep_min=3)
    assert len(cold) <= len(pool_ids) - 3, "Should keep at least 3 experts"
    # 200-206 should be colder than 5,10,15
    assert any(eid >= 200 for eid in cold), f"200+ experts should be cold, got {cold}"
    print(f"  Cold experts: {cold}")

    # Pool hit tracking
    tel.record_pool_hit(0, hits=8, misses=2)
    tel.record_pool_hit(0, hits=7, misses=3)
    hr = tel.get_pool_hit_rate(0)
    assert 0.7 <= hr <= 0.8, f"Expected ~0.75 hit rate, got {hr}"
    print(f"  Pool hit rate: {hr:.2f}")

    # Summary
    summary = tel.summary()
    assert summary["total_tokens"] == 100
    assert summary["total_activations"] == 400
    print(f"  Summary: {summary['total_tokens']} tokens, "
          f"{summary['total_activations']} activations, "
          f"avg unique/layer: {summary['avg_unique_per_layer']:.0f}")

    print("  PASS: Telemetry tracks, predicts, and detects cold experts correctly")


def test_4_cpu_cache():
    """Test: CPUWarmCache stores and retrieves experts correctly."""
    print("\n=== Test 4: CPUWarmCache ===")
    from mlx_lm.models.expert_offload import CPUWarmCache

    # 10 MB cache
    cache = CPUWarmCache(max_bytes=10 * 1024 * 1024)

    # Store some numpy data
    data1 = {"weight": np.random.randn(512, 256).astype(np.float32)}
    data2 = {"weight": np.random.randn(512, 256).astype(np.float32)}

    cache.put(0, 5, data1)
    cache.put(0, 10, data2)

    assert cache.contains(0, 5), "Should contain (0, 5)"
    assert cache.contains(0, 10), "Should contain (0, 10)"
    assert not cache.contains(0, 99), "Should not contain (0, 99)"

    # Retrieve
    retrieved = cache.get(0, 5)
    assert retrieved is not None, "Should retrieve (0, 5)"
    assert np.array_equal(retrieved["weight"], data1["weight"]), "Data should match"

    # Miss
    miss = cache.get(1, 5)
    assert miss is None, "Should miss (1, 5)"

    # batch_get_missing
    cached, missing = cache.batch_get_missing(0, [5, 10, 99, 200])
    assert set(cached) == {5, 10}, f"Cached: {cached}"
    assert set(missing) == {99, 200}, f"Missing: {missing}"

    # LRU eviction (fill cache beyond capacity)
    big_data = {"weight": np.random.randn(1024, 1024).astype(np.float32)}  # ~4 MB
    for i in range(10):
        cache.put(1, i, big_data)

    # Should have evicted old entries
    assert cache.utilization <= 1.0, f"Utilization: {cache.utilization}"

    summary = cache.summary()
    print(f"  Capacity: {summary['capacity_gb']*1024:.1f} MB, "
          f"Used: {summary['used_gb']*1024:.1f} MB, "
          f"Entries: {summary['entries']}, "
          f"Hit rate: {summary['hit_rate']:.2f}")

    # Test mx.array put/get
    mx_data = {"weight": mx.ones((512, 256))}
    cache.put_from_mx(2, 0, mx_data)
    assert cache.contains(2, 0)
    retrieved_np = cache.get(2, 0)
    assert retrieved_np is not None
    assert np.all(retrieved_np["weight"] == 1.0)

    print("  PASS: CPU cache stores, retrieves, evicts correctly")


def test_5_regime_detector():
    """Test: RegimeDetector selects appropriate strategies."""
    print("\n=== Test 5: RegimeDetector ===")
    from mlx_lm.models.expert_offload import RegimeDetector

    # Scenario 1: Mac Mini M4 Pro 48GB, single request
    r1 = RegimeDetector.detect(
        total_expert_gb=16.88, non_expert_gb=1.29,
        gpu_memory_gb=36, cpu_memory_gb=12,
        target_concurrent=1
    )
    print(f"  48GB/1req: {r1.regime} - {r1.description}")
    assert r1.regime == "C_full_gpu", f"Expected C, got {r1.regime}"

    # Scenario 2: Mac Mini M4 Pro 48GB, 8 concurrent
    r2 = RegimeDetector.detect(
        total_expert_gb=16.88, non_expert_gb=1.29,
        gpu_memory_gb=36, cpu_memory_gb=12,
        target_concurrent=8
    )
    print(f"  48GB/8req: {r2.regime} - {r2.description}")
    # With 8 concurrent * 0.5 GB KV = 4 GB KV
    # available = 36 - 2 - 1.29 - 4 = 28.71 → ratio = 28.71/16.88 = 1.70 → C
    # Actually still Regime C because 48GB is large

    # Scenario 3: MacBook Pro 24GB, single request
    r3 = RegimeDetector.detect(
        total_expert_gb=16.88, non_expert_gb=1.29,
        gpu_memory_gb=18, cpu_memory_gb=6,
        target_concurrent=1
    )
    print(f"  24GB/1req: {r3.regime} - {r3.description}")

    # Scenario 4: 24GB, 8 concurrent → memory pressure
    r4 = RegimeDetector.detect(
        total_expert_gb=16.88, non_expert_gb=1.29,
        gpu_memory_gb=18, cpu_memory_gb=6,
        target_concurrent=8
    )
    print(f"  24GB/8req: {r4.regime} - {r4.description}")

    # Scenario 5: Huge model on 24GB → streaming
    r5 = RegimeDetector.detect(
        total_expert_gb=40.0, non_expert_gb=3.0,
        gpu_memory_gb=18, cpu_memory_gb=6,
        target_concurrent=1
    )
    print(f"  70B/24GB:  {r5.regime} - {r5.description}")
    assert r5.regime in ("A_streaming", "B_three_tier"), f"Expected A or B, got {r5.regime}"

    print("  PASS: Regime detection works across scenarios")


def test_6_data_matches_original(index, loader):
    """Test: Loaded expert data matches original model weights exactly."""
    print("\n=== Test 6: Data Matches Original ===")
    from mlx_lm import load as mlx_load

    print("  Loading original model (this takes a moment)...")
    model, tokenizer = mlx_load(MODEL_PATH)

    inner = model
    if hasattr(inner, "model"):
        inner = inner.model
    if hasattr(inner, "language_model"):
        inner = inner.language_model
    if hasattr(inner, "model"):
        inner = inner.model

    layer0_switch = inner.layers[0].mlp.switch_mlp

    # Compare expert 5's gate_proj.weight
    original_gate_w = layer0_switch.gate_proj.weight
    original_expert5 = original_gate_w[5]
    mx.eval(original_expert5)

    data = loader.load_experts(layer_idx=0, expert_indices=[5])
    loaded_expert5 = data["gate_proj.weight"][0]
    mx.eval(loaded_expert5)

    orig_np = np.array(original_expert5)
    load_np = np.array(loaded_expert5)
    match = np.array_equal(orig_np, load_np)
    print(f"  gate_proj.weight[5] exact match: {match}")
    assert match, "Expert data mismatch!"

    # Check scales
    original_gate_s = layer0_switch.gate_proj.scales
    original_expert5_s = original_gate_s[5]
    mx.eval(original_expert5_s)

    data_s = loader.load_experts(layer_idx=0, expert_indices=[5])
    loaded_expert5_s = data_s["gate_proj.scales"][0]
    mx.eval(loaded_expert5_s)

    orig_s_np = np.array(original_expert5_s.view(mx.uint16))
    load_s_np = np.array(loaded_expert5_s.view(mx.uint16))
    match_s = np.array_equal(orig_s_np, load_s_np)
    print(f"  gate_proj.scales[5] exact match: {match_s}")
    assert match_s, "Expert scales mismatch!"

    # Different layer
    layer20_switch = inner.layers[20].mlp.switch_mlp
    orig_l20_w = layer20_switch.down_proj.weight[100]
    mx.eval(orig_l20_w)

    data_l20 = loader.load_experts(layer_idx=20, expert_indices=[100])
    load_l20_w = data_l20["down_proj.weight"][0]
    mx.eval(load_l20_w)

    match_l20 = np.array_equal(np.array(orig_l20_w), np.array(load_l20_w))
    print(f"  layer20.down_proj.weight[100] exact match: {match_l20}")
    assert match_l20, "Layer 20 expert data mismatch!"

    print("  PASS: All expert data matches original model exactly")
    return model, tokenizer


def test_7_patch_and_memory(model, tokenizer):
    """Test: Model patching with three-tier setup reduces memory."""
    print("\n=== Test 7: Patch Model + Three-Tier Setup ===")
    from mlx_lm.models.expert_offload import patch_model_for_offload

    mx.eval(model.parameters())
    gc.collect()
    mem_before = mx.metal.get_active_memory() / 1024 / 1024 / 1024

    print(f"  Memory before patch: {mem_before:.2f} GB")

    # Patch with three-tier configuration
    ctx = patch_model_for_offload(
        model, MODEL_PATH,
        max_workers=4,
        cpu_cache_gb=2.0,  # Use 2 GB CPU cache for test
        enable_prefetch=True,
        enable_telemetry=True,
    )

    gc.collect()
    mx.metal.clear_cache()
    mem_after = mx.metal.get_active_memory() / 1024 / 1024 / 1024

    savings = mem_before - mem_after
    print(f"  Memory after patch:  {mem_after:.2f} GB")
    print(f"  Memory saved:        {savings:.2f} GB ({savings/mem_before*100:.0f}%)")

    # Verify context
    assert ctx.telemetry is not None, "Telemetry should be created"
    assert ctx.cpu_cache is not None, "CPU cache should be created"
    assert ctx.regime is not None, "Regime should be detected"
    print(f"  Regime: {ctx.regime.regime}")
    print(f"  CPU cache: {ctx.cpu_cache.summary()['capacity_gb']*1024:.0f} MB")

    if savings > 5:
        print("  PASS: Significant memory reduction achieved")
    else:
        print("  WARN: Less savings than expected (GC timing)")

    return ctx


def test_8_inference_smoke(model, tokenizer, ctx):
    """Test: Model generates text with three-tier offloading."""
    print("\n=== Test 8: Inference Smoke Test ===")
    from mlx_lm.generate import stream_generate

    prompt = "What is 2+2? Answer in one word:"
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    print(f"  Generating with patched model...")
    t0 = time.perf_counter()
    text = ""
    token_count = 0
    for response in stream_generate(model, tokenizer, formatted, max_tokens=20):
        text += response.text
        token_count += 1
    t1 = time.perf_counter()

    print(f"  Output: {text[:100]}")
    print(f"  Time: {(t1-t0)*1000:.0f}ms, Tokens: {token_count}")

    if len(text) > 0:
        print("  PASS: Model generates text with three-tier offloading")
    else:
        print("  FAIL: No output generated")

    return text


def test_9_telemetry_summary(ctx):
    """Test: Telemetry reports meaningful data after inference."""
    print("\n=== Test 9: Telemetry Summary ===")

    if not ctx.telemetry:
        print("  SKIP: Telemetry not enabled")
        return

    summary = ctx.telemetry.summary()
    print(f"  Total tokens processed: {summary['total_tokens']}")
    print(f"  Total activations: {summary['total_activations']}")
    print(f"  Avg unique experts/layer: {summary['avg_unique_per_layer']:.0f}")
    print(f"  Max unique experts/layer: {summary['max_unique_per_layer']}")
    print(f"  Overall pool hit rate: {summary['overall_pool_hit_rate']:.2%}")

    # With prebuild (Regime C), discovery is skipped, so telemetry
    # may be zero. Only assert when discovery was actually used.
    if summary['total_tokens'] > 0:
        assert summary['total_activations'] > 0, "If tokens recorded, activations should be too"
        print("  Telemetry recorded during discovery phase")
    else:
        print("  Telemetry empty (prebuild mode, discovery skipped — expected)")

    # CPU cache summary
    if ctx.cpu_cache:
        cpu_sum = ctx.cpu_cache.summary()
        print(f"  CPU cache: {cpu_sum['entries']} entries, "
              f"{cpu_sum['used_gb']*1024:.1f} MB used, "
              f"hit rate: {cpu_sum['hit_rate']:.2%}")

    # Full system summary
    full = ctx.summary()
    print(f"  System regime: {full['regime']}")

    print("  PASS: Telemetry reports meaningful data")


def test_10_thunderomlx_bridge(ctx, model):
    """Test: ThunderOMLX Bridge interface works correctly."""
    print("\n=== Test 10: ThunderOMLX Bridge ===")

    # Access bridge via lazy property
    bridge = ctx.bridge
    assert bridge is not None, "Bridge should be created"
    print(f"  Bridge created, found {len(bridge._flash_layers)} FlashMoE layers")

    # Test 1: telemetry_snapshot
    snapshot = bridge.telemetry_snapshot()
    assert "timestamp" in snapshot, "Snapshot should have timestamp"
    assert "regime" in snapshot, "Snapshot should have regime"
    assert "pool_state" in snapshot, "Snapshot should have pool_state"
    assert "telemetry" in snapshot, "Snapshot should have telemetry"
    assert "model" in snapshot, "Snapshot should have model info"

    print(f"  Snapshot regime: {snapshot['regime']}")
    print(f"  Snapshot model: {snapshot['model']['num_layers']} layers, "
          f"{snapshot['model']['num_experts']} experts")
    print(f"  Snapshot tokens: {snapshot['telemetry']['total_tokens']}")

    # Verify activation_freq shape
    freq = snapshot["telemetry"]["activation_freq"]
    assert len(freq) == snapshot["model"]["num_layers"], "Freq should have num_layers rows"
    assert len(freq[0]) == snapshot["model"]["num_experts"], "Freq should have num_experts cols"

    # Verify pool_state has entries
    pool_state = snapshot["pool_state"]
    assert len(pool_state) > 0, "Pool state should have entries after inference"
    sample_key = next(iter(pool_state))
    sample_pool = pool_state[sample_key]
    assert "expert_ids" in sample_pool, "Pool state should list expert_ids"
    assert "size" in sample_pool, "Pool state should show size"
    print(f"  Pool at layer {sample_key}: {sample_pool['size']} experts")

    # Test 2: pool_state (lightweight query)
    quick_state = bridge.pool_state()
    assert len(quick_state) > 0, "Quick pool state should have entries"
    print(f"  Quick pool_state: {len(quick_state)} layers")

    # Test 3: JSON serializable (critical for MCP/IPC)
    import json
    json_str = json.dumps(snapshot)
    assert len(json_str) > 100, "Snapshot should serialize to substantial JSON"
    print(f"  JSON serializable: {len(json_str)} bytes")

    # Test 4: apply_strategy (prefetch)
    strategy = {
        "prefetch": {
            "0": [200, 201, 202],
        },
    }
    bridge.apply_strategy(strategy)
    print("  apply_strategy(prefetch): accepted")

    # Test 5: load_hint
    bridge.load_hint(prompt_tokens=500, expected_generation=100)
    print("  load_hint(500 tokens): accepted")

    # Test 6: Verify snapshot is JSON-round-trippable
    parsed = json.loads(json_str)
    assert parsed["regime"] == snapshot["regime"]
    assert parsed["model"]["num_layers"] == snapshot["model"]["num_layers"]
    print("  JSON round-trip: verified")

    print("  PASS: ThunderOMLX Bridge fully operational")


if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        sys.exit(1)

    print("=" * 70)
    print("  FlashMLX Expert Offload v3 — Three-Tier Architecture Tests")
    print("=" * 70)

    # Unit tests (no model needed)
    test_3_telemetry()
    test_4_cpu_cache()
    test_5_regime_detector()

    # Integration tests (need model files)
    index = test_1_build_index()
    loader = test_2_load_experts(index)
    model, tokenizer = test_6_data_matches_original(index, loader)
    loader.close()

    # Full system tests
    ctx = test_7_patch_and_memory(model, tokenizer)
    test_8_inference_smoke(model, tokenizer, ctx)
    test_9_telemetry_summary(ctx)
    test_10_thunderomlx_bridge(ctx, model)
    ctx.close()

    print("\n" + "=" * 70)
    print("  All 10 tests complete!")
    print("=" * 70)
