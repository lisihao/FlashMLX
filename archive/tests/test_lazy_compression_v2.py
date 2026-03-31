#!/usr/bin/env python3
"""Test lazy AM compression - 正确的测试方式"""

import sys
sys.path.insert(0, '/Users/lisihao/FlashMLX/mlx-lm-source')

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.hybrid_cache import HybridKVCache
import time

def test_compression_quality():
    """
    正确的测试方式：
    - Test 1: Uncompressed 完整流程
    - Test 2: Compressed 完整流程（prefill → compress → generate）
    """
    print("=" * 70)
    print("🧪 Lazy AM Compression Quality Test (v2)")
    print("=" * 70)

    model_path = "/Volumes/toshiba/models/qwen3-8b-mlx"
    calibration_file = "calibrations/am_calibration_qwen3-8b_2.0x_onpolicy.pkl"

    # Prompt
    prompt = """The Quantum Dynamics Research Lab was founded in 2018 by Dr. Sarah Chen,
who received $5 million in initial funding from the National Science Foundation.

The lab's primary research goal was to achieve stable quantum coherence for 100 microseconds,
significantly longer than the industry standard at the time.

Dr. Chen recruited a team of five researchers:
- Dr. Robert Kim, a theoretical physicist from MIT
- Dr. Elena Rodriguez, an experimental physicist from Caltech
- Dr. James Wilson, a quantum computing specialist
- Dr. Lisa Zhang, a materials scientist
- Dr. Michael Brown, a cryogenics engineer

In their first year, the team conducted 150 experiments. Of these, 134 successfully demonstrated
quantum coherence. The early tests showed coherence lasting 15 microseconds on average.

Question: What was the success rate of the experiments?"""

    # =====================================
    # Test 1: Uncompressed
    # =====================================
    print("\n" + "=" * 70)
    print("Test 1: Uncompressed (Baseline)")
    print("=" * 70)

    model, tokenizer = load(model_path)

    # Tokenize
    tokens = mx.array(tokenizer.encode(prompt))
    print(f"Prompt tokens: {tokens.shape[0]}")

    # Generate (uncompressed)
    print("\nGenerating (uncompressed)...")
    response_uncompressed = []

    # Prefill
    cache = [None] * len(model.model.layers)
    logits = model(tokens[None, :], cache=cache)

    # Generate 20 tokens
    for i in range(20):
        token = mx.argmax(logits[0, -1]).item()
        response_uncompressed.append(token)
        logits = model(mx.array([[token]]), cache=cache)

    answer_uncompressed = tokenizer.decode(response_uncompressed)
    print(f"✓ Answer (uncompressed): {answer_uncompressed}")

    # =====================================
    # Test 2: Compressed
    # =====================================
    print("\n" + "=" * 70)
    print("Test 2: Compressed (AM)")
    print("=" * 70)

    # Reload model (fresh start)
    model, tokenizer = load(model_path)

    # Create hybrid cache with calibration
    print(f"Loading calibration from: {calibration_file}")
    cache = [
        HybridKVCache(
            compression_ratio=2.0,
            calibration_file=calibration_file,
            layer_idx=i
        )
        for i in range(len(model.model.layers))
    ]

    # Prefill
    print("Prefilling...")
    logits = model(tokens[None, :], cache=cache)

    # Compress
    print("Compressing cache...")
    for i, c in enumerate(cache):
        before, after = c.compress()
    print(f"✓ Compressed: {before} → {after} tokens")

    # Generate 20 tokens (after compression)
    print("\nGenerating (compressed)...")
    response_compressed = []

    for i in range(20):
        token = mx.argmax(logits[0, -1]).item()
        response_compressed.append(token)
        logits = model(mx.array([[token]]), cache=cache)

    answer_compressed = tokenizer.decode(response_compressed)
    print(f"✓ Answer (compressed): {answer_compressed}")

    # =====================================
    # Compare
    # =====================================
    print("\n" + "=" * 70)
    print("📊 COMPARISON")
    print("=" * 70)
    print(f"Uncompressed: {answer_uncompressed}")
    print(f"Compressed:   {answer_compressed}")

    if answer_uncompressed.strip()[:20] == answer_compressed.strip()[:20]:
        print("\n✅ PASS: Answers match (first 20 chars)")
    else:
        print("\n❌ FAIL: Answers differ")
        print(f"   Uncompressed starts: '{answer_uncompressed[:50]}'")
        print(f"   Compressed starts:   '{answer_compressed[:50]}'")

if __name__ == "__main__":
    test_compression_quality()
