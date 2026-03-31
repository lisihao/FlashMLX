#!/usr/bin/env python3
"""最终质量测试：正确的 cache 方式"""

import sys
sys.path.insert(0, '/Users/lisihao/FlashMLX/mlx-lm-source')

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models import cache
from mlx_lm.models.hybrid_cache import HybridKVCache

def test_quality():
    model_path = "/Volumes/toshiba/models/qwen3-8b-mlx"
    calibration_file = "calibrations/am_calibration_qwen3-8b_2.0x_onpolicy.pkl"

    prompts = [
        "The capital of France is",
        "1 + 1 = ",
        "The first president of the United States was",
    ]

    print("=" * 70)
    print("🧪 最终质量测试（正确的 cache 方式）")
    print("=" * 70)

    for prompt in prompts:
        print(f"\n{'='*70}")
        print(f"📝 Prompt: {prompt}")
        print(f"{'='*70}")

        # ========== Test 1: Uncompressed ==========
        print("\n🔹 无压缩 (baseline):")

        model, tokenizer = load(model_path)
        tokens = mx.array(tokenizer.encode(prompt))

        # 正确的 cache 创建方式
        uncompressed_cache = cache.make_prompt_cache(model)
        logits = model(tokens[None, :], cache=uncompressed_cache)

        response_uncompressed = []
        for i in range(30):
            token_id = mx.argmax(logits[0, -1]).item()
            if token_id in tokenizer.eos_token_ids:
                break
            response_uncompressed.append(token_id)
            logits = model(mx.array([[token_id]]), cache=uncompressed_cache)

        output_uncompressed = tokenizer.decode(response_uncompressed)
        print(f"   {output_uncompressed}")

        # ========== Test 2: Compressed ==========
        print("\n🔹 压缩 (AM 2.0x):")

        model, tokenizer = load(model_path)
        tokens = mx.array(tokenizer.encode(prompt))

        # 使用 HybridKVCache
        compressed_cache = [
            HybridKVCache(
                compression_ratio=2.0,
                calibration_file=calibration_file,
                layer_idx=i
            )
            for i in range(len(model.model.layers))
        ]

        # Prefill
        logits = model(tokens[None, :], cache=compressed_cache)

        # Compress
        for c in compressed_cache:
            c.compress()

        # Generate
        response_compressed = []
        for i in range(30):
            token_id = mx.argmax(logits[0, -1]).item()
            if token_id in tokenizer.eos_token_ids:
                break
            response_compressed.append(token_id)
            logits = model(mx.array([[token_id]]), cache=compressed_cache)

        output_compressed = tokenizer.decode(response_compressed)
        print(f"   {output_compressed}")

        # ========== 对比 ==========
        print(f"\n📊 对比:")
        if output_uncompressed[:20] == output_compressed[:20]:
            print("   ✅ 前 20 字符一致")
        else:
            print("   ⚠️  有差异")
            print(f"   Uncompressed: {output_uncompressed[:50]}...")
            print(f"   Compressed:   {output_compressed[:50]}...")

    print("\n" + "=" * 70)
    print("✓ 测试完成")
    print("=" * 70)

if __name__ == "__main__":
    test_quality()
