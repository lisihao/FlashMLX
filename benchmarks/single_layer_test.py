#!/usr/bin/env python3
"""
Phase 3 诊断 - 单层压缩测试

只压缩 layer 39 (最后一个 Attention 层)，验证：
1. 单个 Attention 层压缩是否会产生乱码
2. 是否是层间交互导致的问题
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "mlx-lm-source"))

import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.models.cache import ArraysCache
from mlx_lm.models.compacted_cache import CompactedKVCache


def classify_memory_type(layer, layer_idx):
    """分类记忆类型"""
    if hasattr(layer, 'linear_attn') or hasattr(layer, 'mamba_block'):
        return "state_memory"
    elif hasattr(layer, 'self_attn'):
        return "attention_memory"
    else:
        return "unknown"


class SingleLayerCacheManager:
    """单层压缩缓存管理器"""

    def __init__(self, model, target_layer=39, max_size=8192, compression_ratio=2.0):
        self.model = model
        self.target_layer = target_layer
        self.max_size = max_size
        self.compression_ratio = compression_ratio

        self.stats = {
            'attention_memory_count': 0,
            'state_memory_count': 0,
            'unknown_count': 0,
            'compressed_layer': None
        }

        self.caches = self._create_caches()

    def _create_caches(self):
        """创建单层压缩缓存列表"""
        num_layers = len(self.model.layers)
        cache_container = ArraysCache(size=num_layers)

        for i, layer in enumerate(self.model.layers):
            memory_type = classify_memory_type(layer, i)

            if memory_type == "attention_memory":
                self.stats['attention_memory_count'] += 1

                # 只压缩目标层
                if i == self.target_layer:
                    cache = CompactedKVCache(
                        max_size=self.max_size,
                        compression_ratio=self.compression_ratio
                    )
                    self.stats['compressed_layer'] = i
                    print(f"  ✓ Layer {i}: CompactedKVCache (ratio={self.compression_ratio})")
                else:
                    cache = None
                    print(f"  - Layer {i}: No compression")
            elif memory_type == "state_memory":
                cache = None
                self.stats['state_memory_count'] += 1
            else:
                cache = None
                self.stats['unknown_count'] += 1

            cache_container[i] = cache

        return cache_container

    def get_stats(self):
        return self.stats


def test_single_layer(model, tokenizer, prompt, max_tokens, target_layer, compression_ratio):
    """测试单层压缩"""
    print(f"\n{'='*70}")
    print(f"单层压缩测试 - Layer {target_layer}")
    print(f"  compression_ratio={compression_ratio}")
    print(f"{'='*70}")

    start = time.time()

    try:
        cache_manager = SingleLayerCacheManager(
            model,
            target_layer=target_layer,
            max_size=8192,
            compression_ratio=compression_ratio
        )

        print(f"\nCache 配置:")
        print(f"  Total layers: {len(model.layers)}")
        print(f"  Attention layers: {cache_manager.stats['attention_memory_count']}")
        print(f"  State layers: {cache_manager.stats['state_memory_count']}")
        print(f"  Compressed layer: {cache_manager.stats['compressed_layer']}")

        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,
            prompt_cache=cache_manager.caches
        )

        elapsed = time.time() - start
        num_tokens = len(tokenizer.encode(response))

        print(f"\n✅ 成功")
        print(f"Tokens: {num_tokens}")
        print(f"Time: {elapsed:.2f}s")
        print(f"Speed: {num_tokens / elapsed:.2f} tok/s")
        print(f"\n生成内容:")
        print(f"{'-'*70}")
        print(response[:500])
        if len(response) > 500:
            print(f"... (截断, 总共 {len(response)} 字符)")
        print(f"{'-'*70}")

        # 判断是否乱码
        is_garbage = check_garbage(response)
        print(f"\n乱码检测: {'❌ 乱码' if is_garbage else '✅ 正常'}")

        return {
            'success': True,
            'layer': target_layer,
            'compression_ratio': compression_ratio,
            'tokens': num_tokens,
            'time': elapsed,
            'tps': num_tokens / elapsed,
            'is_garbage': is_garbage,
            'output': response
        }

    except Exception as e:
        elapsed = time.time() - start
        print(f"\n❌ 失败: {str(e)}")

        return {
            'success': False,
            'layer': target_layer,
            'compression_ratio': compression_ratio,
            'error': str(e),
            'time': elapsed
        }


def check_garbage(text):
    """检查是否为乱码"""
    # 简单启发式：检查重复的短词
    words = text.split()
    if len(words) < 10:
        return False

    # 检查前 50 个词中是否有大量重复
    first_50 = words[:50]
    unique_ratio = len(set(first_50)) / len(first_50)

    if unique_ratio < 0.3:  # 少于 30% unique words
        return True

    # 检查是否有连续重复
    for i in range(len(words) - 5):
        if words[i] == words[i+1] == words[i+2] == words[i+3] == words[i+4]:
            return True

    return False


def main():
    """主函数"""
    model_name = "/Volumes/toshiba/models/qwen3.5-35b-mlx"
    prompt = "介绍机器学习的基本概念和应用场景"
    max_tokens = 200

    print("="*70)
    print("Phase 3 诊断 - 单层压缩测试")
    print("="*70)
    print(f"\n模型: {model_name}")
    print(f"Prompt: '{prompt}'")
    print(f"Max tokens: {max_tokens}\n")

    # 加载模型
    print("Loading model...")
    model, tokenizer = load(model_name)
    print(f"Model loaded: {len(model.layers)} layers\n")

    # 识别 Attention 层
    attention_layers = []
    for i, layer in enumerate(model.layers):
        memory_type = classify_memory_type(layer, i)
        if memory_type == "attention_memory":
            attention_layers.append(i)

    print(f"Identified Attention layers: {attention_layers}")
    print(f"Total Attention layers: {len(attention_layers)}\n")

    # 测试配置
    configs = [
        ("Baseline (no compression)", None, None),
        ("Layer 39 - Conservative (ratio=2.0)", 39, 2.0),
        ("Layer 39 - Moderate (ratio=3.0)", 39, 3.0),
        ("Layer 39 - Aggressive (ratio=5.0)", 39, 5.0)
    ]

    results = []

    for config_name, target_layer, compression_ratio in configs:
        if target_layer is None:
            # Baseline
            print(f"\n{'='*70}")
            print(f"测试配置: {config_name}")
            print(f"{'='*70}")

            start = time.time()
            response = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                verbose=False
            )
            elapsed = time.time() - start
            num_tokens = len(tokenizer.encode(response))

            print(f"\n✅ 成功")
            print(f"Tokens: {num_tokens}")
            print(f"Time: {elapsed:.2f}s")
            print(f"Speed: {num_tokens / elapsed:.2f} tok/s")
            print(f"\n生成内容:")
            print(f"{'-'*70}")
            print(response[:500])
            if len(response) > 500:
                print(f"... (截断, 总共 {len(response)} 字符)")
            print(f"{'-'*70}")

            result = {
                'success': True,
                'config': config_name,
                'layer': None,
                'compression_ratio': None,
                'tokens': num_tokens,
                'time': elapsed,
                'tps': num_tokens / elapsed,
                'is_garbage': False,
                'output': response
            }
        else:
            result = test_single_layer(
                model, tokenizer, prompt, max_tokens,
                target_layer, compression_ratio
            )
            result['config'] = config_name

        results.append(result)

    # 生成对比报告
    print("\n" + "="*70)
    print("诊断报告")
    print("="*70 + "\n")

    print(f"{'配置':<45} {'成功':<6} {'乱码':<6} {'Tokens':<8} {'速度':<12}")
    print("-"*85)

    for result in results:
        if result['success']:
            garbage_mark = "❌" if result.get('is_garbage', False) else "✅"
            print(f"{result['config']:<45} ✅     {garbage_mark:<6} {result['tokens']:<8} {result['tps']:<12.2f}")
        else:
            print(f"{result['config']:<45} ❌     -      -        -")

    # 保存详细报告
    report_path = Path(__file__).parent.parent / ".solar" / "single-layer-diagnosis.md"
    with open(report_path, "w") as f:
        f.write("# Phase 3 诊断 - 单层压缩测试报告\n\n")
        f.write(f"**日期**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**模型**: {model_name}\n")
        f.write(f"**测试层**: Layer 39 (最后一个 Attention 层)\n")
        f.write(f"**Prompt**: '{prompt}'\n")
        f.write(f"**Max tokens**: {max_tokens}\n\n")

        f.write("## 测试目标\n\n")
        f.write("1. 验证单个 Attention 层压缩是否会产生乱码\n")
        f.write("2. 隔离问题：是 AM 本身失效，还是层间交互导致\n\n")

        f.write("## 对比结果\n\n")
        f.write("| 配置 | 成功 | 乱码 | Tokens | 耗时 | 速度 |\n")
        f.write("|------|------|------|--------|------|------|\n")

        for result in results:
            if result['success']:
                garbage_mark = "❌ 是" if result.get('is_garbage', False) else "✅ 否"
                f.write(f"| {result['config']} | ✅ | {garbage_mark} | {result['tokens']} | {result['time']:.2f}s | {result['tps']:.2f} tok/s |\n")
            else:
                f.write(f"| {result['config']} | ❌ | - | - | - | - |\n")

        f.write("\n## 生成内容对比\n\n")

        for result in results:
            if result['success']:
                f.write(f"### {result['config']}\n\n")
                if result.get('is_garbage', False):
                    f.write("**状态**: ❌ 乱码检测到\n\n")
                else:
                    f.write("**状态**: ✅ 正常\n\n")
                f.write("```\n")
                f.write(result['output'][:500])
                if len(result['output']) > 500:
                    f.write(f"\n... (截断, 总共 {len(result['output'])} 字符)\n")
                f.write("```\n\n")

        f.write("## 结论\n\n")

        # 分析结果
        compressed_results = [r for r in results if r.get('layer') is not None]
        all_garbage = all(r.get('is_garbage', False) for r in compressed_results if r['success'])
        some_garbage = any(r.get('is_garbage', False) for r in compressed_results if r['success'])

        if all_garbage:
            f.write("### 结论 1: AM 在单层压缩时就完全失效\n\n")
            f.write("即使只压缩一个 Attention 层，所有压缩比都产生乱码。\n\n")
            f.write("**推论**:\n")
            f.write("- AM 算法本身在 Qwen3.5 Attention 层上不适用\n")
            f.write("- 问题不是层间交互，而是 AM 与这个架构的基本不兼容\n")
            f.write("- 需要检查：β 值分布、NNLS 求解状态、queries 生成\n\n")
        elif some_garbage:
            f.write("### 结论 2: 激进压缩导致质量下降\n\n")
            f.write("部分压缩比产生乱码，说明压缩过于激进。\n\n")
            f.write("**推论**:\n")
            f.write("- AM 在 Qwen3.5 上可能需要更保守的参数\n")
            f.write("- 需要调优 compression_ratio 和 max_size\n\n")
        else:
            f.write("### 结论 3: 单层压缩正常，问题在层间交互\n\n")
            f.write("单层压缩质量正常，说明 AM 本身可用。\n\n")
            f.write("**推论**:\n")
            f.write("- 问题在于压缩后的 Attention 输出影响了 SSM 层\n")
            f.write("- 需要分析：压缩前后激活值分布、SSM 层输入变化\n")
            f.write("- 可能需要：在 Attention 层输出后添加 LayerNorm 或缩放\n\n")

    print(f"\n详细报告已保存到: {report_path}")


if __name__ == "__main__":
    main()
