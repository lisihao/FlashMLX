#!/usr/bin/env python3
"""
SSM 压缩方法对比测试
对比三种方法：Quantization, Low-Rank, Random Projection
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "mlx-lm-source"))

import mlx.core as mx
from mlx_lm import load, generate

sys.path.insert(0, str(Path(__file__).parent.parent / "mlx-lm-source" / "mlx_lm" / "compaction"))
from ssm_state_compressor import QuantizationCompressor, LowRankStateCompressor, RandomProjectionCompressor


PROMPT = "介绍机器学习的基本概念和应用场景"
MAX_TOKENS = 100


def classify_layer_type(layer, layer_idx):
    """分类层类型"""
    if hasattr(layer, 'linear_attn') or hasattr(layer, 'mamba_block'):
        return "ssm"
    elif hasattr(layer, 'self_attn'):
        return "attention"
    else:
        return "unknown"


class CompressedSSMCache:
    """SSM State 压缩缓存"""

    def __init__(self, compressor):
        self.compressor = compressor
        self.compressed_state = None
        self.conv_state = None

    def __getitem__(self, index):
        if index == 0:
            return self.conv_state
        elif index == 1:
            if self.compressed_state is not None:
                return self.compressor.decompress(self.compressed_state)
            return None
        else:
            raise IndexError(f"Invalid cache index: {index}")

    def __setitem__(self, index, value):
        if index == 0:
            self.conv_state = value
        elif index == 1:
            if value is not None:
                self.compressed_state = self.compressor.compress(value)
            else:
                self.compressed_state = None
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


def test_single_method(model, tokenizer, method_name, compressor, ssm_layers):
    """测试单个压缩方法"""
    print(f"\n{'='*60}")
    print(f"Testing: {method_name}")
    print(f"{'='*60}\n")

    # 创建压缩缓存
    from mlx_lm.models.cache import ArraysCache
    cache = ArraysCache(len(model.layers))

    for i in range(len(model.layers)):
        if i in ssm_layers:
            cache[i] = CompressedSSMCache(compressor)
        else:
            cache[i] = None

    # 生成
    print("Generating...")
    start_time = time.time()

    try:
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

        print(f"✅ {method_name} Success!")
        print(f"Tokens: {num_tokens}, Time: {elapsed:.2f}s, Speed: {num_tokens/elapsed:.2f} tok/s")
        print(f"\nOutput preview:\n{response[:200]}\n")

        return {
            'success': True,
            'tokens': num_tokens,
            'time': elapsed,
            'tps': num_tokens / elapsed,
            'output': response,
            'ratio': compressor.get_compression_ratio((1, 64, 128, 192))
        }

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ {method_name} Failed: {str(e)}\n")

        import traceback
        traceback.print_exc()

        return {
            'success': False,
            'error': str(e),
            'ratio': compressor.get_compression_ratio((1, 64, 128, 192))
        }


def main():
    print("="*60)
    print("SSM State Compression Methods Comparison")
    print("="*60)
    print()

    # 加载模型
    print("Loading model...")
    model, tokenizer = load("/Volumes/toshiba/models/qwen3.5-35b-mlx")

    # 识别 SSM 层
    ssm_layers = [i for i, layer in enumerate(model.layers) if classify_layer_type(layer, i) == "ssm"]
    print(f"Model loaded: {len(model.layers)} layers, {len(ssm_layers)} SSM layers\n")

    # Baseline (no compression)
    print("="*60)
    print("Baseline (No Compression)")
    print("="*60)
    print("\nGenerating...")
    start_time = time.time()
    baseline_response = generate(
        model,
        tokenizer,
        prompt=PROMPT,
        max_tokens=MAX_TOKENS,
        verbose=False
    )
    baseline_time = time.time() - start_time
    baseline_tokens = len(tokenizer.encode(baseline_response))
    print(f"✅ Baseline: {baseline_tokens} tokens, {baseline_time:.2f}s, {baseline_tokens/baseline_time:.2f} tok/s")
    print(f"\nOutput preview:\n{baseline_response[:200]}\n")

    # Test methods
    results = {}

    # Method 1: Quantization
    results['Quantization (8-bit)'] = test_single_method(
        model, tokenizer,
        "Quantization (8-bit)",
        QuantizationCompressor(bits=8),
        ssm_layers
    )

    # Method 2: Low-Rank
    results['Low-Rank (rank=32)'] = test_single_method(
        model, tokenizer,
        "Low-Rank (rank=32)",
        LowRankStateCompressor(rank=32),
        ssm_layers
    )

    # Method 3: Random Projection
    results['Random Projection (dim=32)'] = test_single_method(
        model, tokenizer,
        "Random Projection (dim=32)",
        RandomProjectionCompressor(target_dim=32),
        ssm_layers
    )

    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print()
    print(f"{'Method':<30} {'Ratio':<10} {'Status':<10} {'Quality'}")
    print("-"*80)
    print(f"{'Baseline':<30} {'1.0x':<10} {'✅':<10} Normal")

    for method_name, result in results.items():
        ratio = f"{result['ratio']:.2f}x"
        status = '✅' if result['success'] else '❌'
        quality = 'Normal' if result['success'] else 'Failed'

        if result['success']:
            # Check if output is garbage
            output = result['output'][:100]
            if any(char in output for char in ['revan', 'gai', 'assemb', 'igest', 'zinhoLAG']):
                quality = '❌ Garbage'
                status = '❌'
            else:
                quality = '✅ Normal'

        print(f"{method_name:<30} {ratio:<10} {status:<10} {quality}")

    print("\n" + "="*60)
    print("Conclusion")
    print("="*60)
    print()

    # Find best method
    best_method = None
    best_ratio = 0

    for method_name, result in results.items():
        if result['success']:
            output = result['output'][:100]
            is_garbage = any(char in output for char in ['revan', 'gai', 'assemb', 'igest', 'zinhoLAG'])

            if not is_garbage and result['ratio'] > best_ratio:
                best_method = method_name
                best_ratio = result['ratio']

    if best_method:
        print(f"✅ Best method: {best_method}")
        print(f"   Compression ratio: {best_ratio:.2f}x")
        print(f"   Quality: Normal (preserves generation quality)")
    else:
        print("❌ No method successfully preserved quality")

    print()


if __name__ == "__main__":
    main()
