#!/usr/bin/env python3
"""验证无压缩路径是否正常"""

import sys
sys.path.insert(0, '/Users/lisihao/FlashMLX/mlx-lm-source')

import mlx.core as mx
from mlx_lm import load

def verify_uncompressed():
    model_path = "/Volumes/toshiba/models/qwen3-8b-mlx"

    prompts = [
        "The capital of France is",
        "1 + 1 = ",
        "The first president of the United States was",
    ]

    print("=" * 70)
    print("🔍 验证无压缩路径")
    print("=" * 70)

    model, tokenizer = load(model_path)

    for prompt in prompts:
        print(f"\n📝 Prompt: {prompt}")

        tokens = mx.array(tokenizer.encode(prompt))

        # 无压缩：cache = None
        cache = [None] * len(model.model.layers)
        logits = model(tokens[None, :], cache=cache)

        # 生成 10 个 token
        response = []
        for i in range(10):
            token = mx.argmax(logits[0, -1]).item()
            response.append(token)
            logits = model(mx.array([[token]]), cache=cache)

        output = tokenizer.decode(response)
        print(f"   Output: {output}")

    print("\n" + "=" * 70)
    print("✓ 验证完成")
    print("=" * 70)

if __name__ == "__main__":
    verify_uncompressed()
