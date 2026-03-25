#!/usr/bin/env python3
"""
测试 AM 压缩时机 - 验证 TG 阶段是否会多次压缩
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "mlx-lm-source"))

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import ArraysCache
from mlx_lm.models.compacted_cache import CompactedKVCache


def main():
    print("="*70)
    print("AM 压缩时机测试")
    print("="*70)

    model_path = "/Volumes/toshiba/models/qwen3-8b-mlx"

    print("\nLoading model...")
    model, tokenizer = load(model_path)
    num_layers = len(model.layers)

    # 使用一个非常小的 max_size 来强制触发多次压缩
    max_size = 50
    compression_ratio = 2.0

    print(f"\n配置: max_size={max_size}, ratio={compression_ratio}")

    # 创建 AM cache
    cache = ArraysCache(size=num_layers)
    for i in range(num_layers):
        cache[i] = CompactedKVCache(
            max_size=max_size,
            compression_ratio=compression_ratio
        )

    # 使用中等长度 prompt（确保超过 max_size）
    prompt = "Machine learning is a branch of artificial intelligence. " * 10
    tokens = mx.array([tokenizer.encode(prompt)])
    prompt_tokens = tokens.shape[1]

    print(f"Prompt tokens: {prompt_tokens}")
    print(f"Expected to trigger compression: {prompt_tokens > max_size}")

    # PP phase
    print(f"\n{'='*70}")
    print("PP Phase")
    print(f"{'='*70}")

    logits = model(tokens, cache=cache)
    mx.eval(logits)

    # 检查第 0 层的压缩统计
    layer0_stats = cache[0].get_stats()
    print(f"Layer 0 after PP:")
    print(f"  Compressions: {layer0_stats['num_compressions']}")
    print(f"  Current size: {layer0_stats['current_size']}")
    print(f"  Tokens before: {layer0_stats['total_tokens_before']}")
    print(f"  Tokens after: {layer0_stats['total_tokens_after']}")

    # TG phase - 生成多个 tokens
    print(f"\n{'='*70}")
    print("TG Phase - 逐个 token 观察压缩")
    print(f"{'='*70}")

    next_token = mx.argmax(logits[:, -1, :], axis=-1).item()

    for i in range(20):  # 生成 20 个 tokens
        # 生成前的状态
        stats_before = cache[0].get_stats()

        # 生成 token
        tokens = mx.array([[next_token]])
        logits = model(tokens, cache=cache)
        mx.eval(logits)
        next_token = mx.argmax(logits[:, -1, :], axis=-1).item()

        # 生成后的状态
        stats_after = cache[0].get_stats()

        # 检查是否触发了新的压缩
        if stats_after['num_compressions'] > stats_before['num_compressions']:
            print(f"Token {i+1}: ✅ 触发压缩！")
            print(f"  Before: {stats_before['current_size']} tokens")
            print(f"  After:  {stats_after['current_size']} tokens")
            print(f"  Total compressions: {stats_after['num_compressions']}")
        else:
            # 只在关键点打印
            if i in [0, 4, 9, 14, 19]:
                print(f"Token {i+1}: ⚠️  未压缩, cache size: {stats_after['current_size']}")

    # 最终统计
    final_stats = cache[0].get_stats()
    print(f"\n{'='*70}")
    print("最终统计 (Layer 0)")
    print(f"{'='*70}")
    print(f"Total compressions: {final_stats['num_compressions']}")
    print(f"Current cache size: {final_stats['current_size']}")
    print(f"Total tokens before all compressions: {final_stats['total_tokens_before']}")
    print(f"Total tokens after all compressions: {final_stats['total_tokens_after']}")

    if final_stats['num_compressions'] > 0:
        avg_ratio = final_stats['total_tokens_before'] / final_stats['total_tokens_after']
        print(f"Average compression ratio: {avg_ratio:.2f}x")

    # 结论
    print(f"\n{'='*70}")
    print("结论")
    print(f"{'='*70}")

    if final_stats['num_compressions'] == 1:
        print("✅ AM 只在 PP 阶段压缩 1 次")
        print("   TG 阶段不会再次压缩（即使超过 max_size）")
    elif final_stats['num_compressions'] > 1:
        print(f"⚠️  AM 在整个推理过程中压缩了 {final_stats['num_compressions']} 次")
        print("   这意味着 TG 阶段也会触发压缩")
        print("   多次压缩可能导致累积的质量损失！")


if __name__ == "__main__":
    main()
