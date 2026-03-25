#!/usr/bin/env python3
"""修复重复字符问题"""

import sys
sys.path.insert(0, '/Users/lisihao/FlashMLX/mlx-lm-source')

import mlx.core as mx
from mlx_lm import load

def test_with_fixes():
    model_path = "/Volumes/toshiba/models/qwen3-8b-mlx"

    print("=" * 70)
    print("🧪 测试修复方案")
    print("=" * 70)

    model, tokenizer = load(model_path)

    # ========== Test 1: 短 prompt + EOS 停止 ==========
    print("\n" + "=" * 70)
    print("Test 1: 短 prompt + EOS 停止条件")
    print("=" * 70)

    prompt = "The capital of France is"
    tokens = mx.array(tokenizer.encode(prompt))
    cache = [None] * len(model.model.layers)
    logits = model(tokens[None, :], cache=cache)

    response = []
    for i in range(50):  # 最多 50 个
        token_id = mx.argmax(logits[0, -1]).item()

        # ✅ 添加 EOS 检查
        if token_id == tokenizer.eos_token_id:
            print(f"  遇到 EOS，第 {i+1} 步停止")
            break

        response.append(token_id)
        logits = model(mx.array([[token_id]]), cache=cache)

    output = tokenizer.decode(response)
    print(f"Prompt: {prompt}")
    print(f"Output: {output}")
    print(f"生成 token 数: {len(response)}")

    # ========== Test 2: 长 prompt + EOS 停止 ==========
    print("\n" + "=" * 70)
    print("Test 2: 长 prompt + EOS 停止条件")
    print("=" * 70)

    long_prompt = """Paris is the capital and largest city of France.
It is located in the north-central part of the country.
Question: What is the capital of France?
Answer:"""

    tokens = mx.array(tokenizer.encode(long_prompt))
    cache = [None] * len(model.model.layers)
    logits = model(tokens[None, :], cache=cache)

    response = []
    for i in range(50):
        token_id = mx.argmax(logits[0, -1]).item()

        if token_id == tokenizer.eos_token_id:
            print(f"  遇到 EOS，第 {i+1} 步停止")
            break

        response.append(token_id)
        logits = model(mx.array([[token_id]]), cache=cache)

    output = tokenizer.decode(response)
    print(f"Output: {output}")
    print(f"生成 token 数: {len(response)}")

    print("\n" + "=" * 70)
    print("✓ 测试完成")
    print("=" * 70)

if __name__ == "__main__":
    test_with_fixes()
