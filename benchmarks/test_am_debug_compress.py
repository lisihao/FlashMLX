#!/usr/bin/env python3
"""
Debug _compress 函数
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "mlx-lm-source"))

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.compacted_cache import CompactedKVCache
from mlx_lm.models.cache import ArraysCache


def main():
    print("Debug _compress")

    model_path = "/Volumes/toshiba/models/qwen3-8b-mlx"
    model, tokenizer = load(model_path)

    # 创建 cache
    num_layers = len(model.model.layers)
    cache_list = ArraysCache(size=num_layers)
    for i in range(num_layers):
        cache_list[i] = CompactedKVCache(
            max_size=5,
            enable_compression=True,
            compression_ratio=2.0,
            use_quality_path=True,
        )

    # Prefill
    prompt = "Hello, how are you? I am doing well."
    tokens = mx.array(tokenizer.encode(prompt))
    logits = model(tokens[None], cache=cache_list)

    cache = cache_list[0]
    print(f"\nBefore compress:")
    print(f"  offset: {cache.offset}")
    print(f"  max_size: {cache.max_size}")
    print(f"  compression_ratio: {cache.compression_ratio}")

    # Monkey patch _compress to add debug prints
    original_compress = cache._compress

    def debug_compress(queries):
        print(f"\n[_compress] START")
        print(f"  offset: {cache.offset}")
        print(f"  queries: {queries.shape if queries is not None else 'None'}")

        target_budget = int(cache.offset / cache.compression_ratio)
        print(f"  target_budget: {target_budget}")

        if target_budget < 10:
            print(f"  ⚠️  target_budget < 10, skipping")
            return

        # Call original
        result = original_compress(queries)

        print(f"[_compress] END")
        print(f"  new offset: {cache.offset}")

        return result

    cache._compress = debug_compress

    # Trigger compress
    print(f"\nCalling compact()...")
    success = cache.compact(queries=None)

    print(f"\nAfter compress:")
    print(f"  compact returned: {success}")
    print(f"  offset: {cache.offset}")


if __name__ == "__main__":
    main()
