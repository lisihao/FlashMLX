#!/usr/bin/env python3
"""
快速测试 SSM 压缩方法
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "mlx-lm-source"))

import mlx.core as mx
from mlx_lm import load, generate

sys.path.insert(0, str(Path(__file__).parent.parent / "mlx-lm-source" / "mlx_lm" / "compaction"))
from ssm_state_compressor import QuantizationCompressor, LowRankStateCompressor, RandomProjectionCompressor


# 测试prompt
PROMPT = "介绍机器学习的基本概念"


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


def test_method(method_name, compressor):
    """测试单个压缩方法"""
    print(f"\n{'='*60}")
    print(f"Testing: {method_name}")
    print(f"{'='*60}\n")

    # 加载模型
    model, tokenizer = load("/Volumes/toshiba/models/qwen3.5-35b-mlx")

    # 识别 SSM 层
    ssm_layers = [i for i, layer in enumerate(model.layers) if classify_layer_type(layer, i) == "ssm"]
    print(f"SSM layers: {len(ssm_layers)}")

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
    response = generate(
        model,
        tokenizer,
        prompt=PROMPT,
        max_tokens=50,
        verbose=False,
        prompt_cache=cache
    )

    print(f"\n{method_name} Output:")
    print(f"{response[:200]}")
    print()

    return response


if __name__ == "__main__":
    print("Testing 3 SSM compression methods on Qwen3.5-35B\n")

    # Test 1: Quantization (we know this fails)
    # quant_result = test_method("Quantization 8-bit", QuantizationCompressor(bits=8))

    # Test 2: Low-Rank
    lowrank_result = test_method("Low-Rank (rank=32)", LowRankStateCompressor(rank=32))

    # Test 3: Random Projection
    # randproj_result = test_method("Random Projection (dim=32)", RandomProjectionCompressor(target_dim=32))

    print("\n" + "="*60)
    print("Quick Test Complete")
    print("="*60)
