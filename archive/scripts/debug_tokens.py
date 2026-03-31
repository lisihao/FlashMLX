#!/usr/bin/env python3
"""调试生成的 token ID"""

import sys
sys.path.insert(0, '/Users/lisihao/FlashMLX/mlx-lm-source')

import mlx.core as mx
from mlx_lm import load

def debug_tokens():
    model_path = "/Volumes/toshiba/models/qwen3-8b-mlx"
    prompt = "The capital of France is"

    print("=" * 70)
    print("🔍 调试生成的 Token ID")
    print("=" * 70)

    model, tokenizer = load(model_path)
    tokens = mx.array(tokenizer.encode(prompt))

    print(f"\nPrompt: {prompt}")
    print(f"Prompt tokens: {tokens.tolist()}")

    # 无压缩
    cache = [None] * len(model.model.layers)
    logits = model(tokens[None, :], cache=cache)

    print("\n生成过程:")
    print(f"{'Step':<6} {'Token ID':<10} {'Decoded':<20} {'Is EOS?':<10}")
    print("-" * 70)

    response_ids = []
    for i in range(15):  # 生成 15 个看看
        token_id = mx.argmax(logits[0, -1]).item()
        decoded = tokenizer.decode([token_id])
        is_eos = (token_id == tokenizer.eos_token_id)

        print(f"{i+1:<6} {token_id:<10} {repr(decoded):<20} {'YES ✓' if is_eos else 'no':<10}")

        response_ids.append(token_id)

        # 如果遇到 EOS，应该停止
        if is_eos:
            print("\n⚠️  遇到 EOS token，但测试代码没有停止！")
            break

        logits = model(mx.array([[token_id]]), cache=cache)

    print("\n" + "=" * 70)
    print(f"完整输出: {tokenizer.decode(response_ids)}")
    print(f"EOS token ID: {tokenizer.eos_token_id}")
    print("=" * 70)

if __name__ == "__main__":
    debug_tokens()
