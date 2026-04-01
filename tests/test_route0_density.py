"""
Route 0 Density Router — unit tests for discretization logic.

Tests:
    1. DensityLevel enum values (keep_ratio, compression_ratio, log2_ratio)
    2. snap_to_nearest correctness at boundaries and with scale
    3. CacheConfig density fields (mode, scale, effective_density_scale)
    4. Density signal extraction from importance masks
    5. Parameter threading (config → cache_kwargs → factory_kwargs)
"""

import math
import sys

sys.path.insert(0, "/Users/lisihao/FlashMLX/mlx-lm-source")
sys.path.insert(0, "/Users/lisihao/FlashMLX/src")

import numpy as np

from flashmlx.config import CacheConfig, DensityLevel, snap_to_nearest


# ---------------------------------------------------------------------------
# 1. DensityLevel enum
# ---------------------------------------------------------------------------

def test_density_level_values():
    """Each level has correct keep_ratio, compression_ratio, log2_ratio."""
    expected = [
        ("keep_80", 0.80, 1.25),
        ("keep_50", 0.50, 2.0),
        ("keep_33", 0.33, 3.0),
        ("keep_20", 0.20, 5.0),
        ("keep_10", 0.10, 10.0),
    ]
    for name, keep, ratio in expected:
        lvl = DensityLevel[name]
        assert lvl.keep_ratio == keep, f"{name}: keep_ratio {lvl.keep_ratio} != {keep}"
        assert lvl.compression_ratio == ratio, f"{name}: compression_ratio mismatch"
        assert abs(lvl.log2_ratio - math.log2(ratio)) < 1e-6, f"{name}: log2_ratio mismatch"


def test_density_levels_ordered():
    """Levels are ordered by increasing compression (log2_ratio)."""
    levels = list(DensityLevel)
    log2s = [lvl.log2_ratio for lvl in levels]
    assert log2s == sorted(log2s), f"Levels not ordered: {log2s}"


# ---------------------------------------------------------------------------
# 2. snap_to_nearest
# ---------------------------------------------------------------------------

def test_snap_exact_matches():
    """log2 values matching a level exactly → that level."""
    for lvl in DensityLevel:
        result = snap_to_nearest(lvl.log2_ratio)
        assert result == lvl, f"log2={lvl.log2_ratio} → {result.name}, expected {lvl.name}"


def test_snap_midpoint_keep80_keep50():
    """Midpoint between keep_80 and keep_50 goes to the closer one."""
    mid = (DensityLevel.keep_80.log2_ratio + DensityLevel.keep_50.log2_ratio) / 2
    result = snap_to_nearest(mid)
    # Should be one of the two
    assert result in (DensityLevel.keep_80, DensityLevel.keep_50)


def test_snap_scale_positive():
    """scale=+1 shifts target by +1 in log2 space (more compression)."""
    # Start at keep_50 (log2=1.0), scale=+1 → adjusted=2.0
    # Closest to log2(5.0)=2.32 → keep_20
    result = snap_to_nearest(1.0, scale=1.0)
    assert result == DensityLevel.keep_20, f"Expected keep_20, got {result.name}"


def test_snap_scale_negative():
    """scale=-1 shifts target by -1 in log2 space (less compression)."""
    # Start at keep_50 (log2=1.0), scale=-1 → adjusted=0.0
    # Closest to log2(1.25)=0.32 → keep_80
    result = snap_to_nearest(1.0, scale=-1.0)
    assert result == DensityLevel.keep_80, f"Expected keep_80, got {result.name}"


def test_snap_extreme_high():
    """Very high log2 target → keep_10 (maximum compression)."""
    result = snap_to_nearest(10.0)
    assert result == DensityLevel.keep_10


def test_snap_extreme_low():
    """Very low (negative) log2 target → keep_80 (minimum compression)."""
    result = snap_to_nearest(-5.0)
    assert result == DensityLevel.keep_80


def test_snap_scale_sweep():
    """Sweeping scale from -2 to +3 produces monotonically increasing compression."""
    base_log2 = 1.0  # keep_50 baseline
    prev_ratio = 0.0
    for scale in [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]:
        lvl = snap_to_nearest(base_log2, scale=scale)
        assert lvl.compression_ratio >= prev_ratio, \
            f"scale={scale}: {lvl.compression_ratio} < prev {prev_ratio}"
        prev_ratio = lvl.compression_ratio


# ---------------------------------------------------------------------------
# 3. CacheConfig density fields
# ---------------------------------------------------------------------------

def test_config_density_defaults():
    """Default density_mode is 'off', density_scale is 0.0."""
    cfg = CacheConfig()
    assert cfg.density_mode == "off"
    assert cfg.density_scale == 0.0


def test_config_density_mode_validation():
    """Invalid density_mode raises ValueError."""
    import pytest
    try:
        CacheConfig(density_mode="invalid")
        assert False, "Should have raised ValueError"
    except Exception as e:
        assert "Unknown density_mode" in str(e) or "validation error" in str(e).lower()


def test_config_effective_density_scale():
    """effective_density_scale resolves mode presets correctly."""
    cases = [
        ("off", 0.0, 0.0),
        ("balanced", 0.0, 0.0),
        ("ultra_long", 0.0, 1.5),
        ("recall_first", 0.0, 2.5),
    ]
    for mode, manual_scale, expected in cases:
        cfg = CacheConfig(density_mode=mode, density_scale=manual_scale)
        assert cfg.effective_density_scale() == expected, \
            f"mode={mode}, scale={manual_scale}: got {cfg.effective_density_scale()}, expected {expected}"


def test_config_effective_density_scale_manual_override():
    """Manual density_scale overrides mode preset."""
    cfg = CacheConfig(density_mode="ultra_long", density_scale=3.0)
    assert cfg.effective_density_scale() == 3.0  # manual overrides 1.5


def test_config_to_cache_kwargs_density():
    """to_cache_kwargs includes density fields when mode != off."""
    cfg = CacheConfig(strategy="scored_pq", density_mode="balanced")
    kwargs = cfg.to_cache_kwargs()
    assert "density_mode" in kwargs
    assert kwargs["density_mode"] == "balanced"
    assert "density_scale" in kwargs


def test_config_to_cache_kwargs_no_density():
    """to_cache_kwargs omits density fields when mode == off."""
    cfg = CacheConfig(strategy="scored_pq")
    kwargs = cfg.to_cache_kwargs()
    assert "density_mode" not in kwargs
    assert "density_scale" not in kwargs


# ---------------------------------------------------------------------------
# 4. Density signal extraction
# ---------------------------------------------------------------------------

def test_density_signal_basic():
    """_extract_density_signal computes correct metrics."""
    from mlx_lm.models.triple_layer_cache import TripleLayerKVCache
    cache = TripleLayerKVCache(layer_idx=0, scored_mode=True)

    masks = [
        np.array([True, False, True, True, False] * 2),  # 60% keep
        np.array([True, True, False, False, True] * 2),  # 60% keep
    ]
    sig = cache._extract_density_signal(masks, 1024, 512, 256, 256)

    assert sig["keep_ratio"] == 0.5
    assert sig["n_chunks"] == 2
    assert abs(sig["log2_ratio"] - 1.0) < 0.1  # log2(2.0) ≈ 1.0
    assert sig["concentration"] >= 0.0
    assert len(sig["chunk_concentrations"]) == 2


def test_density_signal_empty():
    """Empty masks → default values."""
    from mlx_lm.models.triple_layer_cache import TripleLayerKVCache
    cache = TripleLayerKVCache(layer_idx=0, scored_mode=True)

    sig = cache._extract_density_signal([], 100, 0, 0, 0)
    assert sig["keep_ratio"] == 1.0
    assert sig["log2_ratio"] == 0.0
    assert sig["n_chunks"] == 0


# ---------------------------------------------------------------------------
# 5. Parameter threading
# ---------------------------------------------------------------------------

def test_triple_layer_cache_accepts_density():
    """TripleLayerKVCache stores density_mode and density_scale."""
    from mlx_lm.models.triple_layer_cache import TripleLayerKVCache
    cache = TripleLayerKVCache(
        layer_idx=0, scored_mode=True,
        density_mode="recall_first", density_scale=2.5,
    )
    assert cache._density_mode == "recall_first"
    assert cache._density_scale == 2.5


def test_factory_kwargs_include_density():
    """to_factory_kwargs outputs density fields when active."""
    cfg = CacheConfig(strategy="scored_pq", density_mode="ultra_long", density_scale=1.5)
    fkw = cfg.to_factory_kwargs()
    assert fkw["density_mode"] == "ultra_long"
    assert fkw["density_scale"] == 1.5


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            print(f"  PASS  {test_fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {test_fn.__name__}: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"  Route 0 Density Router: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    if failed:
        sys.exit(1)
