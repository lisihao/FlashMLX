#!/usr/bin/env python3
"""简单测试：文本续写"""

import sys
sys.path.insert(0, '/Users/lisihao/FlashMLX/mlx-lm-source')

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.hybrid_cache import HybridKVCache

def test_simple():
    model_path = "/Volumes/toshiba/models/qwen3-8b-mlx"
    calibration_file = "calibrations/am_calibration_qwen3-8b_2.0x_onpolicy.pkl"

    prompt = "The capital of France is"

    # ===== Test 1: Uncompressed =====
    print("Test 1: Uncompressed")
    model, tokenizer = load(model_path)
    tokens = mx.array(tokenizer.encode(prompt))

    cache = [None] * len(model.model.layers)
    logits = model(tokens[None, :], cache=cache)

    response = []
    for i in range(10):
        token = mx.argmax(logits[0, -1]).item()
        response.append(token)
        logits = model(mx.array([[token]]), cache=cache)

    answer_uncompressed = tokenizer.decode(response)
    print(f"✓ Uncompressed: {answer_uncompressed}\n")

    # ===== Test 2: Compressed =====
    print("Test 2: Compressed")
    model, tokenizer = load(model_path)
    tokens = mx.array(tokenizer.encode(prompt))

    cache = [
        HybridKVCache(compression_ratio=2.0, calibration_file=calibration_file, layer_idx=i)
        for i in range(len(model.model.layers))
    ]

    logits = model(tokens[None, :], cache=cache)

    # Compress
    for c in cache:
        c.compress()

    response = []
    for i in range(10):
        token = mx.argmax(logits[0, -1]).item()
        response.append(token)
        logits = model(mx.array([[token]]), cache=cache)

    answer_compressed = tokenizer.decode(response)
    print(f"✓ Compressed: {answer_compressed}\n")

    # Compare
    print(f"Match: {answer_uncompressed == answer_compressed}")

if __name__ == "__main__":
    test_simple()
