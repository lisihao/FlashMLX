#!/usr/bin/env python3
"""测试 CompactedKVCache 是否还能正常工作"""

import sys
sys.path.insert(0, '/Users/lisihao/FlashMLX/mlx-lm-source')

from mlx_lm import load, generate

def test_compacted():
    model_path = "/Volumes/toshiba/models/qwen3-8b-mlx"

    prompts = [
        "The capital of France is",
        "1 + 1 = ",
    ]

    print("=" * 70)
    print("🧪 测试 CompactedKVCache（上个版本）")
    print("=" * 70)

    model, tokenizer = load(model_path)

    for prompt in prompts:
        print(f"\n📝 Prompt: {prompt}")
        output = generate(
            model,
            tokenizer,
            prompt,
            max_tokens=30,
            verbose=False
        )
        print(f"   Output: {output}")

    print("\n✓ 测试完成")

if __name__ == "__main__":
    test_compacted()
