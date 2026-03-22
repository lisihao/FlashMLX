#!/usr/bin/env python3
"""
SSM State Compression 真实模型测试

测试 Quantization 压缩在真实 Qwen3.5 SSM 层上的效果
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "mlx-lm-source"))

import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.models.cache import ArraysCache

# Import SSM compressor
sys.path.insert(0, str(Path(__file__).parent.parent / "mlx-lm-source" / "mlx_lm" / "compaction"))
from ssm_state_compressor import QuantizationCompressor, LowRankStateCompressor


def classify_layer_type(layer, layer_idx):
    """分类层类型"""
    if hasattr(layer, 'linear_attn') or hasattr(layer, 'mamba_block'):
        return "ssm"
    elif hasattr(layer, 'self_attn'):
        return "attention"
    else:
        return "unknown"


class CompressedSSMCache:
    """
    SSM State 压缩缓存

    包装标准 cache，在存储时压缩 SSM state
    GatedDeltaNet cache 结构: [conv_state, ssm_state]
    """

    def __init__(self, compressor):
        """
        Args:
            compressor: SSM state compressor (QuantizationCompressor or LowRankStateCompressor)
        """
        self.compressor = compressor
        self.compressed_state = None
        self.conv_state = None

    def __getitem__(self, index):
        """获取 cache[index]"""
        if index == 0:
            return self.conv_state
        elif index == 1:
            # 解压缩 SSM state
            if self.compressed_state is not None:
                return self.compressor.decompress(self.compressed_state)
            return None
        else:
            raise IndexError(f"Invalid cache index: {index}")

    def __setitem__(self, index, value):
        """设置 cache[index]"""
        if index == 0:
            self.conv_state = value
        elif index == 1:
            # 压缩 SSM state
            if value is not None:
                self.compressed_state = self.compressor.compress(value)
            else:
                self.compressed_state = None
        else:
            raise IndexError(f"Invalid cache index: {index}")

    @property
    def state(self):
        """
        返回 cache state（兼容 mlx_lm cache 接口）
        GatedDeltaNet 期望 [conv_state, ssm_state]
        """
        return [self.conv_state, self[1]]  # 使用 __getitem__ 自动解压缩

    @state.setter
    def state(self, v):
        """
        设置 cache state（兼容 mlx_lm cache 接口）
        """
        if v is not None and len(v) == 2:
            self[0] = v[0]  # conv_state
            self[1] = v[1]  # ssm_state (会自动压缩)


def create_ssm_compressed_cache(model, compress_layers, compressor):
    """
    创建带 SSM 压缩的 cache

    Args:
        model: 模型对象
        compress_layers: 要压缩的 SSM 层索引列表
        compressor: SSM state compressor

    Returns:
        ArraysCache with compressed SSM layers
    """
    num_layers = len(model.layers)
    cache = ArraysCache(size=num_layers)

    for i in range(num_layers):
        if i in compress_layers:
            # 使用压缩缓存
            cache[i] = CompressedSSMCache(compressor)
        else:
            # 标准缓存
            cache[i] = None

    return cache


def test_ssm_compression(
    model_name="/Volumes/toshiba/models/qwen3.5-35b-mlx",
    prompt="介绍机器学习的基本概念和应用场景",
    max_tokens=200,
    compressor_type="quantization",
    compress_layers=None
):
    """
    测试 SSM 压缩

    Args:
        model_name: 模型名称
        prompt: 测试 prompt
        max_tokens: 最大生成 token 数
        compressor_type: 压缩器类型 ('quantization' or 'lowrank')
        compress_layers: 要压缩的层索引，None = 所有 SSM 层

    Returns:
        dict: 测试结果
    """
    print(f"\n{'='*60}")
    print(f"SSM State 压缩测试")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Prompt: '{prompt}'")
    print(f"Max tokens: {max_tokens}")
    print(f"Compressor: {compressor_type}")
    print()

    # 加载模型
    print("Loading model...")
    model, tokenizer = load(model_name)
    print(f"Model loaded: {len(model.layers)} layers\n")

    # 识别 SSM 层
    ssm_layers = []
    attention_layers = []

    for i, layer in enumerate(model.layers):
        layer_type = classify_layer_type(layer, i)
        if layer_type == "ssm":
            ssm_layers.append(i)
        elif layer_type == "attention":
            attention_layers.append(i)

    print(f"SSM layers: {len(ssm_layers)} - {ssm_layers[:5]}..." if len(ssm_layers) > 5 else f"SSM layers: {len(ssm_layers)} - {ssm_layers}")
    print(f"Attention layers: {len(attention_layers)} - {attention_layers}")
    print()

    # 确定要压缩的层
    if compress_layers is None:
        compress_layers = ssm_layers  # 默认压缩所有 SSM 层

    print(f"Compressing {len(compress_layers)} SSM layers")
    print()

    # 创建压缩器
    if compressor_type == "quantization":
        compressor = QuantizationCompressor(bits=8)
        print("Using Quantization (8-bit)")
    elif compressor_type == "lowrank":
        compressor = LowRankStateCompressor(rank=32)
        print("Using Low-Rank (rank=32)")
    else:
        raise ValueError(f"Unknown compressor type: {compressor_type}")

    # 创建压缩缓存
    cache = create_ssm_compressed_cache(model, compress_layers, compressor)

    # 运行生成
    print(f"{'='*60}")
    print("Generating with compressed SSM cache...")
    print(f"{'='*60}\n")

    start_time = time.time()

    try:
        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,
            prompt_cache=cache
        )

        elapsed = time.time() - start_time
        num_tokens = len(tokenizer.encode(response))

        print(f"{'='*60}")
        print("Generated text:")
        print(f"{'='*60}")
        print(response[:500])
        if len(response) > 500:
            print(f"... (truncated, total {len(response)} chars)")
        print(f"{'='*60}\n")

        print(f"✅ Generation Success!")
        print(f"Tokens: {num_tokens}")
        print(f"Time: {elapsed:.2f}s")
        print(f"Speed: {num_tokens / elapsed:.2f} tok/s")

        # 计算压缩统计
        compression_ratio = compressor.get_compression_ratio((1, 64, 128, 192))
        print(f"\nCompression Stats:")
        print(f"  Method: {compressor_type}")
        print(f"  Compressed layers: {len(compress_layers)} / {len(model.layers)}")
        print(f"  Compression ratio per layer: {compression_ratio:.2f}x")

        result = {
            'success': True,
            'tokens': num_tokens,
            'time': elapsed,
            'tps': num_tokens / elapsed,
            'output': response,
            'compressor': compressor_type,
            'compressed_layers': len(compress_layers),
            'compression_ratio': compression_ratio
        }

        return result

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ Generation failed: {str(e)}")

        import traceback
        traceback.print_exc()

        result = {
            'success': False,
            'error': str(e),
            'time': elapsed
        }

        return result


def compare_baseline_vs_compressed(
    model_name="/Volumes/toshiba/models/qwen3.5-35b-mlx",
    prompt="介绍机器学习的基本概念",
    max_tokens=150
):
    """
    对比 Baseline vs SSM 压缩

    Args:
        model_name: 模型名称
        prompt: 测试 prompt
        max_tokens: 最大生成 token 数

    Returns:
        dict: 对比结果
    """
    print(f"\n{'#'*60}")
    print("# Baseline vs SSM Compression 对比")
    print(f"{'#'*60}\n")

    # 加载模型
    print("Loading model...")
    model, tokenizer = load(model_name)
    print(f"Model loaded: {len(model.layers)} layers\n")

    results = {}

    # Test 1: Baseline (无压缩)
    print(f"{'='*60}")
    print("Test 1: Baseline (No Compression)")
    print(f"{'='*60}\n")

    start = time.time()
    try:
        response_baseline = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False
        )
        elapsed_baseline = time.time() - start
        tokens_baseline = len(tokenizer.encode(response_baseline))

        results['baseline'] = {
            'success': True,
            'tokens': tokens_baseline,
            'time': elapsed_baseline,
            'tps': tokens_baseline / elapsed_baseline,
            'output': response_baseline
        }

        print(f"✅ Baseline: {tokens_baseline} tokens, {elapsed_baseline:.2f}s, {results['baseline']['tps']:.2f} tok/s")
        print(f"Output preview: {response_baseline[:100]}...\n")

    except Exception as e:
        results['baseline'] = {'success': False, 'error': str(e)}
        print(f"❌ Baseline failed: {e}\n")

    # Test 2: SSM Compression (Quantization)
    print(f"{'='*60}")
    print("Test 2: SSM Compression (Quantization 8-bit)")
    print(f"{'='*60}\n")

    # 识别 SSM 层
    ssm_layers = [i for i, layer in enumerate(model.layers)
                  if classify_layer_type(layer, i) == "ssm"]

    compressor = QuantizationCompressor(bits=8)
    cache = create_ssm_compressed_cache(model, ssm_layers, compressor)

    start = time.time()
    try:
        response_compressed = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,
            prompt_cache=cache
        )
        elapsed_compressed = time.time() - start
        tokens_compressed = len(tokenizer.encode(response_compressed))

        results['compressed'] = {
            'success': True,
            'tokens': tokens_compressed,
            'time': elapsed_compressed,
            'tps': tokens_compressed / elapsed_compressed,
            'output': response_compressed,
            'compressed_layers': len(ssm_layers)
        }

        print(f"✅ Compressed: {tokens_compressed} tokens, {elapsed_compressed:.2f}s, {results['compressed']['tps']:.2f} tok/s")
        print(f"Output preview: {response_compressed[:100]}...\n")

    except Exception as e:
        results['compressed'] = {'success': False, 'error': str(e)}
        print(f"❌ Compressed failed: {e}\n")

    # 对比
    print(f"{'='*60}")
    print("Comparison Results")
    print(f"{'='*60}\n")

    if results['baseline']['success'] and results['compressed']['success']:
        speedup = results['compressed']['tps'] / results['baseline']['tps']
        token_diff = results['compressed']['tokens'] - results['baseline']['tokens']

        print(f"Baseline:   {results['baseline']['time']:.2f}s, {results['baseline']['tps']:.2f} tok/s")
        print(f"Compressed: {results['compressed']['time']:.2f}s, {results['compressed']['tps']:.2f} tok/s")
        print(f"Speedup: {speedup:.2f}x")
        print(f"Token diff: {token_diff:+d} ({results['compressed']['tokens']} vs {results['baseline']['tokens']})")
        print(f"\nCompressed {results['compressed']['compressed_layers']} SSM layers")

        # 质量对比
        print(f"\n{'='*60}")
        print("Quality Comparison")
        print(f"{'='*60}\n")
        print("Baseline output:")
        print(results['baseline']['output'][:300])
        print("\nCompressed output:")
        print(results['compressed']['output'][:300])

    return results


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="SSM State Compression 测试")
    parser.add_argument(
        '--mode',
        choices=['test', 'compare', 'single'],
        default='compare',
        help='测试模式'
    )
    parser.add_argument(
        '--model',
        default='/Volumes/toshiba/models/qwen3.5-35b-mlx',
        help='模型名称'
    )
    parser.add_argument(
        '--compressor',
        choices=['quantization', 'lowrank'],
        default='quantization',
        help='压缩器类型'
    )
    parser.add_argument(
        '--single-layer',
        type=int,
        help='只压缩单个 SSM 层（用于 ablation）'
    )

    args = parser.parse_args()

    if args.mode == 'test':
        # 单次测试
        compress_layers = [args.single_layer] if args.single_layer is not None else None
        result = test_ssm_compression(
            model_name=args.model,
            compressor_type=args.compressor,
            compress_layers=compress_layers
        )

        # 保存结果
        report_path = Path(__file__).parent.parent / ".solar" / "ssm-compression-test-report.md"
        with open(report_path, "w") as f:
            f.write("# SSM State Compression 测试报告\n\n")
            f.write(f"**日期**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**模型**: {args.model}\n")
            f.write(f"**压缩器**: {args.compressor}\n\n")

            if result['success']:
                f.write("## 测试结果\n\n")
                f.write(f"- ✅ 成功\n")
                f.write(f"- Tokens: {result['tokens']}\n")
                f.write(f"- Time: {result['time']:.2f}s\n")
                f.write(f"- Speed: {result['tps']:.2f} tok/s\n")
                f.write(f"- Compressed layers: {result['compressed_layers']}\n")
                f.write(f"- Compression ratio: {result['compression_ratio']:.2f}x\n\n")

                f.write("## 生成内容\n\n")
                f.write("```\n")
                f.write(result['output'][:500])
                if len(result['output']) > 500:
                    f.write(f"\n... (截断, 总共 {len(result['output'])} 字符)\n")
                f.write("```\n")
            else:
                f.write("## 测试结果\n\n")
                f.write(f"- ❌ 失败\n")
                f.write(f"- Error: {result['error']}\n")

        print(f"\n报告已保存到: {report_path}")

    elif args.mode == 'compare':
        # 对比测试
        results = compare_baseline_vs_compressed(model_name=args.model)

        # 保存对比报告
        report_path = Path(__file__).parent.parent / ".solar" / "ssm-compression-comparison-report.md"
        with open(report_path, "w") as f:
            f.write("# SSM Compression vs Baseline 对比报告\n\n")
            f.write(f"**日期**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**模型**: {args.model}\n\n")

            f.write("## 对比结果\n\n")
            f.write("| 配置 | 成功 | Tokens | 耗时 | 速度 |\n")
            f.write("|------|------|--------|------|------|\n")

            if results['baseline']['success']:
                f.write(f"| Baseline | ✅ | {results['baseline']['tokens']} | {results['baseline']['time']:.2f}s | {results['baseline']['tps']:.2f} tok/s |\n")
            else:
                f.write(f"| Baseline | ❌ | - | - | - |\n")

            if results['compressed']['success']:
                f.write(f"| Compressed | ✅ | {results['compressed']['tokens']} | {results['compressed']['time']:.2f}s | {results['compressed']['tps']:.2f} tok/s |\n")
            else:
                f.write(f"| Compressed | ❌ | - | - | - |\n")

            if results['baseline']['success'] and results['compressed']['success']:
                speedup = results['compressed']['tps'] / results['baseline']['tps']
                f.write(f"\n**Speedup**: {speedup:.2f}x\n\n")

                f.write("## 生成内容对比\n\n")
                f.write("### Baseline\n\n")
                f.write("```\n")
                f.write(results['baseline']['output'][:500])
                f.write("\n```\n\n")

                f.write("### Compressed\n\n")
                f.write("```\n")
                f.write(results['compressed']['output'][:500])
                f.write("\n```\n")

        print(f"\n对比报告已保存到: {report_path}")

    elif args.mode == 'single':
        # 单层测试
        if args.single_layer is None:
            print("Error: --single-layer required for 'single' mode")
            return

        result = test_ssm_compression(
            model_name=args.model,
            compressor_type=args.compressor,
            compress_layers=[args.single_layer]
        )


if __name__ == "__main__":
    main()
