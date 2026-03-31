#!/usr/bin/env python3
"""使用正确的 cache 创建方式"""

import sys
sys.path.insert(0, '/Users/lisihao/FlashMLX/mlx-lm-source')

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models import cache

def test_with_correct_cache():
    model_path = "/Volumes/toshiba/models/qwen3-8b-mlx"

    print("=" * 70)
    print("🧪 使用正确的 cache 创建方式")
    print("=" * 70)

    model, tokenizer = load(model_path)
    prompt = "The capital of France is"
    tokens = mx.array(tokenizer.encode(prompt))

    print(f"\nPrompt: {prompt}")
    print(f"Tokens: {tokens.tolist()}")

    # ✅ 使用官方 cache 创建方式
    prompt_cache = cache.make_prompt_cache(model)

    # Prefill
    logits = model(tokens[None, :], cache=prompt_cache)

    print("\n生成:")
    response = []
    for i in range(20):
        token_id = mx.argmax(logits[0, -1]).item()

        # 检查 EOS
        if token_id in tokenizer.eos_token_ids:
            print(f"\n遇到 EOS，第 {i+1} 步停止")
            break

        decoded = tokenizer.decode([token_id])
        print(f"  Step {i+1}: {repr(decoded)} (ID: {token_id})")

        response.append(token_id)
        logits = model(mx.array([[token_id]]), cache=prompt_cache)

    output = tokenizer.decode(response)
    print(f"\n完整输出: {output}")

    print("\n" + "=" * 70)
    print("✓ 测试完成")
    print("=" * 70)

if __name__ == "__main__":
    test_with_correct_cache()
