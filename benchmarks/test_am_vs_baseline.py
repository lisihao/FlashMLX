#!/usr/bin/env python3
"""
AM vs Baseline 对比测试

快速验证 AM 压缩后的输出质量是否正常
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "mlx-lm-source"))

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import KVCache, ArraysCache
from mlx_lm.models.compacted_cache import CompactedKVCache


def test_generation(model, tokenizer, prompt, cache_name, cache):
    """测试生成质量"""
    print(f"\n{'='*70}")
    print(f"Testing {cache_name}")
    print(f"{'='*70}")

    tokens = mx.array([tokenizer.encode(prompt)])

    # PP phase
    logits = model(tokens, cache=cache)
    mx.eval(logits)

    # TG phase
    generated_tokens = []
    next_token = mx.argmax(logits[:, -1, :], axis=-1).item()

    max_gen = 100

    for _ in range(max_gen):
        if next_token == tokenizer.eos_token_id:
            break

        generated_tokens.append(next_token)
        tokens = mx.array([[next_token]])
        logits = model(tokens, cache=cache)
        mx.eval(logits)
        next_token = mx.argmax(logits[:, -1, :], axis=-1).item()

    output = tokenizer.decode(generated_tokens)

    print(f"\nGenerated {len(generated_tokens)} tokens:")
    print(f"{'-'*70}")
    print(output)
    print(f"{'-'*70}")

    return output


def main():
    print("="*70)
    print("AM vs Baseline 对比测试")
    print("="*70)

    model_path = "/Volumes/toshiba/models/qwen3-8b-mlx"

    print("\nLoading model...")
    model, tokenizer = load(model_path)

    num_layers = len(model.layers)

    # 简短 prompt 避免压缩
    prompt = "What is machine learning?"
    print(f"\nPrompt: {prompt}")
    print(f"Prompt tokens: {len(tokenizer.encode(prompt))}")

    # Test 1: Baseline (无压缩)
    baseline_cache = ArraysCache(size=num_layers)
    for i in range(num_layers):
        baseline_cache[i] = KVCache()

    baseline_output = test_generation(model, tokenizer, prompt, "Baseline (无压缩)", baseline_cache)

    # Test 2: AM 压缩 (使用 Fast Path)
    am_cache = ArraysCache(size=num_layers)
    for i in range(num_layers):
        am_cache[i] = CompactedKVCache(
            max_size=512,
            compression_ratio=2.0,
            use_quality_path=False  # Fast Path
        )

    am_output = test_generation(model, tokenizer, prompt, "AM (Fast Path)", am_cache)

    # Test 3: AM 压缩 (使用 Quality Path)
    am_quality_cache = ArraysCache(size=num_layers)
    for i in range(num_layers):
        am_quality_cache[i] = CompactedKVCache(
            max_size=512,
            compression_ratio=2.0,
            use_quality_path=True  # Quality Path
        )

    am_quality_output = test_generation(model, tokenizer, prompt, "AM (Quality Path)", am_quality_cache)

    # 对比
    print(f"\n{'='*70}")
    print("对比分析")
    print(f"{'='*70}")

    print(f"\n1. Baseline 输出长度: {len(baseline_output.split())}")
    print(f"2. AM (Fast Path) 输出长度: {len(am_output.split())}")
    print(f"3. AM (Quality Path) 输出长度: {len(am_quality_output.split())}")

    print(f"\n4. 输出是否相同:")
    print(f"   Baseline vs AM (Fast): {'✅ 相同' if baseline_output == am_output else '❌ 不同'}")
    print(f"   Baseline vs AM (Quality): {'✅ 相同' if baseline_output == am_quality_output else '❌ 不同'}")

    print(f"\n5. 输出质量评估:")
    print(f"   Baseline: {'✅ 正常' if len(baseline_output.split()) > 20 else '❌ 异常'}")
    print(f"   AM (Fast): {'✅ 正常' if len(am_output.split()) > 20 else '❌ 异常'}")
    print(f"   AM (Quality): {'✅ 正常' if len(am_quality_output.split()) > 20 else '❌ 异常'}")


if __name__ == "__main__":
    main()
