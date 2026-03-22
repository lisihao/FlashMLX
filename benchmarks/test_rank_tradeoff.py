#!/usr/bin/env python3
"""
测试不同 Rank 的压缩比-速度权衡

测试 rank=32, 48, 64, 96
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "mlx-lm-source"))

import mlx.core as mx
from mlx_lm import load, generate

sys.path.insert(0, str(Path(__file__).parent.parent / "mlx-lm-source" / "mlx_lm" / "compaction"))
from ssm_state_compressor import LowRankStateCompressor


PROMPT = "介绍机器学习的基本概念和应用场景"
MAX_TOKENS = 50  # 减少 tokens 以加快测试


def classify_layer_type(layer, layer_idx):
    if hasattr(layer, 'linear_attn') or hasattr(layer, 'mamba_block'):
        return "ssm"
    return "attention" if hasattr(layer, 'self_attn') else "unknown"


class CompressedSSMCache:
    def __init__(self, compressor):
        self.compressor = compressor
        self.compressed_state = None
        self.conv_state = None

    def __getitem__(self, index):
        if index == 0:
            return self.conv_state
        elif index == 1:
            return self.compressor.decompress(self.compressed_state) if self.compressed_state else None
        else:
            raise IndexError(f"Invalid cache index: {index}")

    def __setitem__(self, index, value):
        if index == 0:
            self.conv_state = value
        elif index == 1:
            self.compressed_state = self.compressor.compress(value) if value is not None else None
        else:
            raise IndexError(f"Invalid cache index: {index}")

    @property
    def state(self):
        return [self.conv_state, self[1]]

    @state.setter
    def state(self, v):
        if v is not None and len(v) == 2:
            self[0] = v[0]
            self[1] = v[1]


def test_rank(model, tokenizer, ssm_layers, rank):
    """测试特定 rank"""
    print(f"\n{'='*60}")
    print(f"Testing: rank={rank}")
    print(f"{'='*60}\n")

    compressor = LowRankStateCompressor(rank=rank)
    ratio = compressor.get_compression_ratio((1, 64, 128, 192))

    print(f"Compression ratio: {ratio:.2f}x")

    # 创建压缩缓存
    from mlx_lm.models.cache import ArraysCache
    cache = ArraysCache(len(model.layers))

    for i in range(len(model.layers)):
        cache[i] = CompressedSSMCache(compressor) if i in ssm_layers else None

    # 生成
    print("Generating...")
    start_time = time.time()

    response = generate(
        model,
        tokenizer,
        prompt=PROMPT,
        max_tokens=MAX_TOKENS,
        verbose=False,
        prompt_cache=cache
    )

    elapsed = time.time() - start_time
    num_tokens = len(tokenizer.encode(response))

    tps = num_tokens / elapsed

    print(f"✅ rank={rank}: {num_tokens} tokens, {elapsed:.2f}s, {tps:.2f} tok/s")
    print(f"Output preview: {response[:100]}\n")

    # 检查质量
    is_garbage = any(char in response[:100] for char in ['revan', 'gai', 'assemb', 'igest'])

    return {
        'rank': rank,
        'ratio': ratio,
        'tokens': num_tokens,
        'time': elapsed,
        'tps': tps,
        'output': response,
        'quality': 'Normal' if not is_garbage else 'Garbage'
    }


def main():
    print("="*60)
    print("Rank Trade-off Test")
    print("="*60)
    print()

    # 加载模型
    print("Loading model...")
    model, tokenizer = load("/Volumes/toshiba/models/qwen3.5-35b-mlx")

    ssm_layers = [i for i, layer in enumerate(model.layers) if classify_layer_type(layer, i) == "ssm"]
    print(f"Model loaded: {len(model.layers)} layers, {len(ssm_layers)} SSM layers\n")

    # Baseline
    print("="*60)
    print("Baseline (No Compression)")
    print("="*60)
    print("\nGenerating...")
    start_time = time.time()
    baseline_response = generate(model, tokenizer, prompt=PROMPT, max_tokens=MAX_TOKENS, verbose=False)
    baseline_time = time.time() - start_time
    baseline_tokens = len(tokenizer.encode(baseline_response))
    baseline_tps = baseline_tokens / baseline_time
    print(f"✅ Baseline: {baseline_tokens} tokens, {baseline_time:.2f}s, {baseline_tps:.2f} tok/s\n")

    # 测试不同 rank
    ranks = [32, 48, 64, 96]
    results = {}

    for rank in ranks:
        results[rank] = test_rank(model, tokenizer, ssm_layers, rank)

    # 汇总
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print()
    print(f"{'Rank':<10} {'Ratio':<10} {'Speed':<12} {'Slowdown':<12} {'Quality'}")
    print("-"*70)
    print(f"{'Baseline':<10} {'1.0x':<10} {baseline_tps:<12.2f} {'1.0x':<12} Normal")

    for rank in ranks:
        r = results[rank]
        slowdown = baseline_tps / r['tps']
        print(f"{r['rank']:<10} {r['ratio']:<10.2f} {r['tps']:<12.2f} {slowdown:<12.2f} {r['quality']}")

    print()
    print("="*60)
    print("Conclusion")
    print("="*60)
    print()

    # 找到最佳 rank
    best_rank = None
    best_score = 0

    for rank in ranks:
        r = results[rank]
        if r['quality'] == 'Normal':
            # Score = speed / (1 + slowdown)
            # 优先速度，但也考虑压缩比
            score = r['tps'] * r['ratio']
            if score > best_score:
                best_score = score
                best_rank = rank

    if best_rank:
        r = results[best_rank]
        slowdown = baseline_tps / r['tps']
        print(f"✅ Best rank: {best_rank}")
        print(f"   Compression ratio: {r['ratio']:.2f}x")
        print(f"   Speed: {r['tps']:.2f} tok/s")
        print(f"   Slowdown: {slowdown:.2f}x")
        print(f"   Quality: Normal")
        print()

        if slowdown < 5:
            print(f"   ✅ Acceptable slowdown (< 5x)")
        elif slowdown < 10:
            print(f"   ⚠️  Moderate slowdown (5-10x)")
        else:
            print(f"   ❌ High slowdown (> 10x)")
    else:
        print("❌ No rank achieved normal quality")

    print()


if __name__ == "__main__":
    main()
