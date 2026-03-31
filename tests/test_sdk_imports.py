"""
Test that all public FlashMLX SDK imports work correctly.

Run: python -m pytest tests/test_sdk_imports.py -v
"""

import pytest


def test_version():
    import flashmlx
    assert flashmlx.__version__ == "1.0.0"


def test_top_level_imports():
    """All top-level exports should be importable."""
    from flashmlx import (
        load,
        generate,
        stream_generate,
        FlashMLXConfig,
        CacheConfig,
        OffloadConfig,
        detect_capabilities,
        recommend_config,
        ModelCapabilities,
        make_prompt_cache,
        make_optimized_cache,
        VALID_STRATEGIES,
        get_quantizer,
        QuantizationStrategy,
    )
    assert callable(load)
    assert callable(generate)
    assert callable(stream_generate)
    assert callable(detect_capabilities)
    assert callable(recommend_config)
    assert callable(make_prompt_cache)
    assert callable(make_optimized_cache)
    assert callable(get_quantizer)
    assert len(VALID_STRATEGIES) > 0


def test_config_creation():
    """Config should create with defaults and validate."""
    from flashmlx import FlashMLXConfig, CacheConfig, OffloadConfig

    config = FlashMLXConfig()
    assert config.cache.strategy == "standard"
    assert config.cache.flat_quant is None
    assert config.offload.enabled is False

    config2 = FlashMLXConfig(
        cache=CacheConfig(strategy="scored_pq", flat_quant="q8_0"),
        offload=OffloadConfig(enabled=True, pool_size=8),
    )
    assert config2.cache.strategy == "scored_pq"
    assert config2.cache.flat_quant == "q8_0"
    assert config2.offload.pool_size == 8


def test_config_validation():
    """Config should reject invalid values."""
    from flashmlx import CacheConfig

    with pytest.raises(Exception):
        CacheConfig(strategy="invalid_strategy")

    with pytest.raises(Exception):
        CacheConfig(flat_quant="invalid_quant")

    with pytest.raises(Exception):
        CacheConfig(warm_bits=7)


def test_config_serialization():
    """Config should round-trip through JSON."""
    from flashmlx import FlashMLXConfig, CacheConfig

    config = FlashMLXConfig(
        cache=CacheConfig(
            strategy="scored_pq",
            flat_quant="q8_0",
            compression_ratio=3.0,
            calibration_file="/path/to/cal.pkl",
        ),
    )

    json_str = config.model_dump_json()
    config2 = FlashMLXConfig.model_validate_json(json_str)

    assert config2.cache.strategy == "scored_pq"
    assert config2.cache.flat_quant == "q8_0"
    assert config2.cache.compression_ratio == 3.0
    assert config2.cache.calibration_file == "/path/to/cal.pkl"


def test_config_to_cache_kwargs():
    """to_cache_kwargs should produce correct dict for make_prompt_cache."""
    from flashmlx import CacheConfig

    # Default config — no kwargs needed
    cfg = CacheConfig()
    assert cfg.to_cache_kwargs() == {}

    # scored_pq + q8_0
    cfg = CacheConfig(
        strategy="scored_pq",
        flat_quant="q8_0",
        calibration_file="/cal.pkl",
    )
    kwargs = cfg.to_cache_kwargs()
    assert kwargs["kv_cache"] == "scored_pq"
    assert kwargs["kv_flat_quant"] == "q8_0"
    assert kwargs["kv_calibration"] == "/cal.pkl"
    assert "kv_compression_ratio" not in kwargs  # default, not included


def test_config_to_factory_kwargs():
    """to_factory_kwargs should produce correct dict for make_optimized_cache."""
    from flashmlx import CacheConfig

    cfg = CacheConfig(strategy="scored_pq", flat_quant="q8_0")
    kwargs = cfg.to_factory_kwargs()
    assert kwargs["strategy"] == "scored_pq"
    assert kwargs["flat_quant"] == "q8_0"
    assert kwargs["recent_size"] == 512
    assert kwargs["warm_size"] == 2048


def test_cache_submodule_imports():
    """Cache submodule should re-export all key types."""
    from flashmlx.cache import (
        KVCache,
        RotatingKVCache,
        TripleLayerKVCache,
        make_prompt_cache,
        make_optimized_cache,
        VALID_STRATEGIES,
        get_cache_info,
        QuantizationStrategy,
        get_quantizer,
        Q4_0Quantizer,
        Q8_0Quantizer,
        PolarQuantizer,
        TurboQuantizer,
        NoOpQuantizer,
        auto_calibrate,
    )
    assert callable(make_prompt_cache)
    assert callable(auto_calibrate)


def test_offload_submodule_imports():
    """Offload submodule should re-export key classes."""
    from flashmlx.offload import (
        patch_model_for_offload,
        OffloadContext,
        FlashBatchGenerator,
        FlashMoeSwitchGLU,
        ThunderOMLXBridge,
    )
    assert callable(patch_model_for_offload)


def test_integration_imports():
    """Integration layer should be importable."""
    from flashmlx.integration import (
        FlashMLXProvider,
        setup_flashmlx,
        flashmlx_settings_schema,
    )
    assert callable(setup_flashmlx)
    assert callable(flashmlx_settings_schema)


def test_protocol_checkable():
    """FlashMLXProvider should be a runtime-checkable protocol."""
    from flashmlx.integration import FlashMLXProvider
    from flashmlx.integration.thunderomlx import ThunderOMLXAdapter

    adapter = ThunderOMLXAdapter()
    assert isinstance(adapter, FlashMLXProvider)


def test_flashmlx_settings_schema():
    """Schema should be a valid JSON Schema dict."""
    from flashmlx.integration import flashmlx_settings_schema

    schema = flashmlx_settings_schema()
    assert isinstance(schema, dict)
    assert "properties" in schema
    assert "cache" in schema["properties"]
    assert "offload" in schema["properties"]


def test_valid_strategies_content():
    """VALID_STRATEGIES should contain expected strategies."""
    from flashmlx import VALID_STRATEGIES

    assert "standard" in VALID_STRATEGIES
    assert "scored_pq" in VALID_STRATEGIES
    assert "triple" in VALID_STRATEGIES
    assert "auto" in VALID_STRATEGIES


def test_model_capabilities_dataclass():
    """ModelCapabilities should be a proper dataclass."""
    from flashmlx import ModelCapabilities

    caps = ModelCapabilities()
    assert caps.model_type == "transformer"
    assert caps.supports_scored_pq is True
    assert caps.supports_turboquant is True
    assert caps.warnings == []
