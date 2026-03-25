#!/usr/bin/env python3
"""
AM 压缩质量验证 - 触发压缩后的输出对比
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "mlx-lm-source"))

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import KVCache, ArraysCache
from mlx_lm.models.compacted_cache import CompactedKVCache


def generate_long_prompt(target_tokens=600):
    """生成指定长度的 prompt"""
    base = "Machine learning is a powerful technique. "
    prompt = ""
    while len(prompt.split()) < target_tokens:
        prompt += base
    return prompt


def test_with_compression_check(model, tokenizer, prompt, cache_name, cache):
    """测试并检查是否真的触发了压缩"""
    print(f"\n{'='*70}")
    print(f"Testing {cache_name}")
    print(f"{'='*70}")

    tokens = mx.array([tokenizer.encode(prompt)])
    prompt_tokens = tokens.shape[1]
    print(f"Prompt tokens: {prompt_tokens}")

    # PP phase
    logits = model(tokens, cache=cache)
    mx.eval(logits)

    # 检查压缩统计
    if hasattr(cache[0], 'get_stats'):
        stats = cache[0].get_stats()
        print(f"Layer 0 stats:")
        print(f"  Compressions: {stats['num_compressions']}")
        print(f"  Current size: {stats['current_size']}")
        if stats['num_compressions'] > 0:
            print(f"  ✅ 压缩已触发")
        else:
            print(f"  ⚠️  压缩未触发")

    # TG phase - 只生成少量 tokens 以便对比
    generated_tokens = []
    next_token = mx.argmax(logits[:, -1, :], axis=-1).item()

    max_gen = 50

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
    print(output[:200])
    if len(output) > 200:
        print("...")
    print(f"{'-'*70}")

    return output


def main():
    print("="*70)
    print("AM 压缩质量验证")
    print("="*70)

    model_path = "/Volumes/toshiba/models/qwen3-8b-mlx"

    print("\nLoading model...")
    model, tokenizer = load(model_path)

    num_layers = len(model.layers)

    # 生成长 prompt（确保触发压缩）
    prompt = generate_long_prompt(target_tokens=600)
    actual_tokens = len(tokenizer.encode(prompt))
    print(f"\nGenerated prompt with {actual_tokens} tokens")

    # Test 1: Baseline (无压缩，大 max_size)
    baseline_cache = ArraysCache(size=num_layers)
    for i in range(num_layers):
        baseline_cache[i] = CompactedKVCache(
            max_size=10000,  # 很大，不会触发压缩
            enable_compression=False
        )

    baseline_output = test_with_compression_check(
        model, tokenizer, prompt, "Baseline (无压缩)", baseline_cache
    )

    # Test 2: AM 压缩 (小 max_size，必定触发)
    am_cache = ArraysCache(size=num_layers)
    for i in range(num_layers):
        am_cache[i] = CompactedKVCache(
            max_size=256,  # 很小，必定触发压缩
            compression_ratio=2.0,
            use_quality_path=False  # Fast Path
        )

    am_output = test_with_compression_check(
        model, tokenizer, prompt, "AM (Fast Path, 压缩)", am_cache
    )

    # Test 3: AM 压缩 (Quality Path)
    am_quality_cache = ArraysCache(size=num_layers)
    for i in range(num_layers):
        am_quality_cache[i] = CompactedKVCache(
            max_size=256,
            compression_ratio=2.0,
            use_quality_path=True  # Quality Path
        )

    am_quality_output = test_with_compression_check(
        model, tokenizer, prompt, "AM (Quality Path, 压缩)", am_quality_cache
    )

    # 对比
    print(f"\n{'='*70}")
    print("对比分析")
    print(f"{'='*70}")

    # 简单质量评估：检查是否有重复模式
    def has_repetition(text, min_repeat=3):
        """检查是否有重复短语（重复3次以上）"""
        words = text.split()
        for i in range(len(words) - 10):
            phrase = " ".join(words[i:i+3])
            if text.count(phrase) >= min_repeat:
                return True, phrase
        return False, None

    baseline_rep, baseline_phrase = has_repetition(baseline_output)
    am_rep, am_phrase = has_repetition(am_output)
    am_quality_rep, am_quality_phrase = has_repetition(am_quality_output)

    print(f"\n1. 输出长度:")
    print(f"   Baseline: {len(baseline_output.split())} words")
    print(f"   AM (Fast): {len(am_output.split())} words")
    print(f"   AM (Quality): {len(am_quality_output.split())} words")

    print(f"\n2. 重复模式检测:")
    print(f"   Baseline: {'❌ 有重复' if baseline_rep else '✅ 无重复'}")
    if baseline_rep:
        print(f"      重复短语: \"{baseline_phrase}\"")

    print(f"   AM (Fast): {'❌ 有重复' if am_rep else '✅ 无重复'}")
    if am_rep:
        print(f"      重复短语: \"{am_phrase}\"")

    print(f"   AM (Quality): {'❌ 有重复' if am_quality_rep else '✅ 无重复'}")
    if am_quality_rep:
        print(f"      重复短语: \"{am_quality_phrase}\"")

    print(f"\n3. 输出是否相同:")
    print(f"   Baseline vs AM (Fast): {'✅ 相同' if baseline_output == am_output else '❌ 不同'}")
    print(f"   Baseline vs AM (Quality): {'✅ 相同' if baseline_output == am_quality_output else '❌ 不同'}")
    print(f"   AM (Fast) vs AM (Quality): {'✅ 相同' if am_output == am_quality_output else '❌ 不同'}")

    print(f"\n4. 质量结论:")
    if not am_rep and not baseline_rep:
        print(f"   ✅ AM 压缩后输出质量正常")
    elif am_rep and not baseline_rep:
        print(f"   ❌ AM 压缩导致输出质量下降（出现重复）")
    elif baseline_rep:
        print(f"   ⚠️  Baseline 本身就有重复（模型问题？）")


if __name__ == "__main__":
    main()
