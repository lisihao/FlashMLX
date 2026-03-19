"""
Tests for FlashMLX core functionality
"""

import pytest
import mlx.core as mx
from flashmlx.core import flash_attention, FlashMLXEngine


def test_flash_attention_shape():
    """Test Flash Attention output shape"""
    batch_size = 2
    seq_len = 128
    num_heads = 8
    head_dim = 64

    q = mx.random.normal((batch_size, seq_len, num_heads, head_dim))
    k = mx.random.normal((batch_size, seq_len, num_heads, head_dim))
    v = mx.random.normal((batch_size, seq_len, num_heads, head_dim))

    output, _ = flash_attention(q, k, v)

    assert output.shape == (batch_size, seq_len, num_heads, head_dim)


def test_flash_attention_scale():
    """Test Flash Attention with custom scale"""
    seq_len = 64
    head_dim = 32

    q = mx.random.normal((1, seq_len, 1, head_dim))
    k = mx.random.normal((1, seq_len, 1, head_dim))
    v = mx.random.normal((1, seq_len, 1, head_dim))

    scale = 1.0 / (head_dim ** 0.5)
    output, _ = flash_attention(q, k, v, scale=scale)

    assert output.shape == (1, seq_len, 1, head_dim)


def test_engine_initialization():
    """Test FlashMLX engine initialization"""
    engine = FlashMLXEngine(model_path="/tmp/test_model")
    assert engine.model_path == "/tmp/test_model"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
