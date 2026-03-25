#!/usr/bin/env python3
"""
最简单的 AM 压缩测试
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "mlx-lm-source"))

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.compacted_cache import CompactedKVCache
from mlx_lm.models.compaction_engine import CompactionEngine
from mlx_lm.models.cache import ArraysCache


def main():
    print("简单 AM 压缩测试")

    model_path = "/Volumes/toshiba/models/qwen3-8b-mlx"
    model, tokenizer = load(model_path)

    # 创建 cache（max_size 要小，才能触发压缩）
    num_layers = len(model.model.layers)
    cache_list = ArraysCache(size=num_layers)
    for i in range(num_layers):
        cache_list[i] = CompactedKVCache(
            max_size=5,  # ← 必须和 CompactionEngine 的 max_size 一致！
            enable_compression=True,
            compression_ratio=2.0,
            use_quality_path=True,
        )

    # Prefill
    prompt = "Hello, how are you? I am doing well, thank you."
    tokens = mx.array(tokenizer.encode(prompt))
    logits = model(tokens[None], cache=cache_list)

    cache = cache_list[0]
    print(f"\nAfter prefill:")
    print(f"  offset: {cache.offset}")

    # 手动压缩
    engine = CompactionEngine(
        max_size=5,  # offset=14 > 5, 会触发
        compression_ratio=2.0,
        num_queries=7,  # 至少 7 个 queries
    )

    if engine.should_compact(cache):
        print(f"\n压缩前:")
        print(f"  offset: {cache.offset}")

        queries = engine.sample_queries(cache)
        print(f"  queries shape: {queries.shape}")

        # 直接调用 cache.compact
        success = cache.compact(queries)
        print(f"  compact returned: {success}")

        print(f"\n压缩后:")
        print(f"  offset: {cache.offset}")
        print(f"  预期 offset: {int(14 / 2.0)} = 7")

        if cache.offset == 7:
            print(f"  ✅ Offset 正确更新！")
        else:
            print(f"  ❌ Offset 没有更新！")


if __name__ == "__main__":
    main()
