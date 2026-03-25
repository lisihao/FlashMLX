#!/usr/bin/env python3
"""
检查 β 的实际值
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
    print("检查 β 的实际值")

    model_path = "/Volumes/toshiba/models/qwen3-8b-mlx"
    model, tokenizer = load(model_path)

    # 创建 cache
    num_layers = len(model.model.layers)
    cache_list = ArraysCache(size=num_layers)
    for i in range(num_layers):
        cache_list[i] = CompactedKVCache(
            max_size=200,
            enable_compression=True,
            compression_ratio=2.0,
            use_quality_path=True,
        )

    # Prefill（长一点的 prompt）
    prompt = """Machine learning is a branch of artificial intelligence that focuses on building systems that can learn from data. Deep learning uses neural networks with multiple layers to automatically learn hierarchical representations."""
    tokens = mx.array(tokenizer.encode(prompt))
    logits = model(tokens[None], cache=cache_list)

    cache = cache_list[0]
    print(f"\nAfter prefill:")
    print(f"  offset: {cache.offset}")
    print(f"  beta: {cache.beta}")

    # 手动触发压缩
    engine = CompactionEngine(
        max_size=10,  # 强制触发
        compression_ratio=2.0,
        num_queries=16,
    )

    # 直接调用 _compress（绕过 max_size 检查）
    print(f"\n压缩前:")
    print(f"  offset: {cache.offset}")

    queries = engine.sample_queries(cache)
    cache._compress(queries)  # 直接调用

    print(f"\n压缩后:")
    print(f"  offset: {cache.offset}")
    print(f"  beta shape: {cache.beta.shape if cache.beta is not None else 'None'}")

    if cache.beta is not None:
        beta_slice = cache.beta[..., :cache.offset]
        print(f"  beta stats:")
        print(f"    min: {mx.min(beta_slice)}")
        print(f"    max: {mx.max(beta_slice)}")
        print(f"    mean: {mx.mean(beta_slice)}")
        print(f"    std: {mx.std(beta_slice)}")

        # 检查前几个值
        print(f"  beta[0, 0, :5]: {beta_slice[0, 0, :5]}")

        # 检查是否有异常值
        inf_count = mx.sum(mx.isinf(beta_slice))
        nan_count = mx.sum(mx.isnan(beta_slice))
        print(f"  inf count: {inf_count}")
        print(f"  nan count: {nan_count}")


if __name__ == "__main__":
    main()
