#!/usr/bin/env python3
"""使用官方 generate 函数测试"""

import sys
sys.path.insert(0, '/Users/lisihao/FlashMLX/mlx-lm-source')

from mlx_lm import load, generate

def test_official():
    model_path = "/Volumes/toshiba/models/qwen3-8b-mlx"

    print("=" * 70)
    print("🧪 使用官方 generate 函数")
    print("=" * 70)

    model, tokenizer = load(model_path)

    prompts = [
        "The capital of France is",
        "1 + 1 = ",
        "The first president of the United States was",
    ]

    for prompt in prompts:
        print(f"\n📝 Prompt: {prompt}")

        # 使用官方 generate 函数
        output = generate(
            model,
            tokenizer,
            prompt,
            max_tokens=20,
            verbose=False
        )

        print(f"   Output: {output}")

    print("\n" + "=" * 70)
    print("✓ 测试完成")
    print("=" * 70)

if __name__ == "__main__":
    test_official()
