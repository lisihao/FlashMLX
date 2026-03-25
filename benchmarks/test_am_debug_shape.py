#!/usr/bin/env python3
"""
AM 压缩 Debug：检查 shape 和中间值
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
    print("=" * 70)
    print("AM 压缩 Debug：Shape 检查")
    print("=" * 70)

    model_path = "/Volumes/toshiba/models/qwen3-8b-mlx"
    print(f"\n加载模型: {model_path}")
    model, tokenizer = load(model_path)

    # 创建简单的 prompt
    prompt = "Hello, how are you?"
    tokens = mx.array(tokenizer.encode(prompt))

    print(f"\nPrompt: {prompt}")
    print(f"Tokens: {tokens.shape}")

    # 创建 cache（需要为所有层创建）
    num_layers = len(model.model.layers)
    cache_list = ArraysCache(size=num_layers)
    for i in range(num_layers):
        cache_list[i] = CompactedKVCache(
            max_size=4096,
            enable_compression=True,
            compression_ratio=2.0,
            use_quality_path=True,
        )

    # 前向传播（prefill）
    logits = model(tokens[None], cache=cache_list)

    # 使用第 0 层的 cache 进行测试
    cache = cache_list[0]

    print(f"\nAfter prefill:")
    print(f"  cache.keys shape: {cache.keys.shape}")
    print(f"  cache.values shape: {cache.values.shape}")
    print(f"  cache.offset: {cache.offset}")

    # 创建 CompactionEngine
    engine = CompactionEngine(
        max_size=3,  # 很小，强制触发压缩（offset=6 > max_size=3）
        compression_ratio=2.0,
        num_queries=5,  # 少一点，因为 offset 只有 6
    )

    print(f"\nCompactionEngine config:")
    print(f"  max_size: {engine.max_size}")
    print(f"  compression_ratio: {engine.compression_ratio}")
    print(f"  num_queries: {engine.num_queries}")

    # 检查是否需要压缩
    should_compact = engine.should_compact(cache)
    print(f"\nShould compact: {should_compact}")

    if should_compact:
        # 采样 Qref
        print(f"\nSampling Qref...")
        queries = engine.sample_queries(cache, num_queries=5)

        print(f"\n✅ Qref shape:")
        print(f"  queries.shape: {queries.shape}")
        print(f"  Expected shape: (B, n_heads, num_queries, head_dim)")

        B, n_heads, num_queries, head_dim = queries.shape
        print(f"  B={B}, n_heads={n_heads}, num_queries={num_queries}, head_dim={head_dim}")

        # 检查 queries 的值
        print(f"\n✅ Qref stats:")
        print(f"  min: {mx.min(queries)}")
        print(f"  max: {mx.max(queries)}")
        print(f"  mean: {mx.mean(queries)}")
        print(f"  std: {mx.std(queries)}")

        # 手动调用 cache._compress 并打印中间信息
        print(f"\n✅ Calling cache._compress with Qref...")

        # 修改 _compress 以打印中间信息
        original_compress = cache._compress

        def debug_compress(queries_arg):
            print(f"  [DEBUG] _compress called")
            print(f"  [DEBUG] queries_arg shape: {queries_arg.shape if queries_arg is not None else 'None'}")
            print(f"  [DEBUG] current offset: {cache.offset}")
            print(f"  [DEBUG] target budget: {int(cache.offset / cache.compression_ratio)}")

            # 调用原始 _compress
            original_compress(queries_arg)

            print(f"  [DEBUG] after compression:")
            print(f"  [DEBUG]   new offset: {cache.offset}")
            print(f"  [DEBUG]   new keys shape: {cache.keys[..., :cache.offset, :].shape}")

        cache._compress = debug_compress

        # 触发压缩
        cache._compress(queries)

        print(f"\n✅ Compression complete")
        print(f"  Final offset: {cache.offset}")

        # 测试生成
        print(f"\n✅ Testing generation after compression...")
        next_token_logits = logits[:, -1, :]
        next_token = mx.argmax(next_token_logits, axis=-1)

        print(f"  Next token: {next_token}")
        print(f"  Decoded: {tokenizer.decode(next_token.tolist())}")

        # 再生成一个 token
        logits = model(next_token[:, None], cache=cache_list)
        next_token_logits = logits[:, -1, :]
        next_token = mx.argmax(next_token_logits, axis=-1)

        print(f"  Next token: {next_token}")
        print(f"  Decoded: {tokenizer.decode(next_token.tolist())}")

    print(f"\n" + "=" * 70)
    print("Debug complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
