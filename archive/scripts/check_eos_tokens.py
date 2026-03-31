#!/usr/bin/env python3
"""检查 tokenizer 的 EOS tokens"""

import sys
sys.path.insert(0, '/Users/lisihao/FlashMLX/mlx-lm-source')

from mlx_lm import load

def check_eos():
    model_path = "/Volumes/toshiba/models/qwen3-8b-mlx"

    print("=" * 70)
    print("🔍 检查 EOS Tokens")
    print("=" * 70)

    model, tokenizer = load(model_path)

    print(f"\neos_token_id (单数): {tokenizer.eos_token_id}")
    print(f"eos_token_ids (复数): {tokenizer.eos_token_ids}")

    # 解码看看这些 token 是什么
    if tokenizer.eos_token_id is not None:
        try:
            decoded = tokenizer.decode([tokenizer.eos_token_id])
            print(f"  解码单数 EOS: {repr(decoded)}")
        except:
            print(f"  无法解码单数 EOS")

    if tokenizer.eos_token_ids:
        print(f"\n所有 EOS tokens:")
        for eos_id in tokenizer.eos_token_ids:
            try:
                decoded = tokenizer.decode([eos_id])
                print(f"  ID {eos_id}: {repr(decoded)}")
            except:
                print(f"  ID {eos_id}: 无法解码")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    check_eos()
