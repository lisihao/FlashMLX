#!/usr/bin/env python3
"""
Heterogeneous Memory Compaction 质量验证

对比不同压缩参数下的生成质量
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


class HeterogeneousCacheManager:
    """异构缓存管理器"""

    def __init__(self, model, max_size=8192, compression_ratio=2.0):
        self.model = model
        self.max_size = max_size
        self.compression_ratio = compression_ratio

        self.stats = {
            'attention_memory_count': 0,
            'state_memory_count': 0,
            'unknown_count': 0
        }

        self.caches = self._create_caches()

    def _create_caches(self):
        """创建异构缓存列表"""
        from mlx_lm.models.cache import ArraysCache

        num_layers = len(self.model.layers)
        cache_container = ArraysCache(size=num_layers)

        for i, layer in enumerate(self.model.layers):
            memory_type = classify_memory_type(layer, i)

            if memory_type == "attention_memory":
                cache = CompactedKVCache(
                    max_size=self.max_size,
                    compression_ratio=self.compression_ratio
                )
                self.stats['attention_memory_count'] += 1
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


def test_configuration(model, tokenizer, prompt, max_tokens, config_name, max_size, compression_ratio):
    """测试特定配置"""
    print(f"\n{'='*60}")
    print(f"测试配置: {config_name}")
    print(f"  max_size={max_size}, compression_ratio={compression_ratio}")
    print(f"{'='*60}")

    start = time.time()

    try:
        if config_name == "Baseline":
            # Baseline: 无压缩
            response = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                verbose=False
            )
        else:
            # Heterogeneous: 异构压缩
            cache_manager = HeterogeneousCacheManager(
                model,
                max_size=max_size,
                compression_ratio=compression_ratio
            )

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
        print(f"\n生成内容 (前 200 字符):")
        print(f"{response[:200]}...")

        return {
            'success': True,
            'config': config_name,
            'max_size': max_size,
            'compression_ratio': compression_ratio,
            'tokens': num_tokens,
            'time': elapsed,
            'tps': num_tokens / elapsed,
            'output': response
        }

    except Exception as e:
        elapsed = time.time() - start
        print(f"\n❌ 失败: {str(e)}")

        return {
            'success': False,
            'config': config_name,
            'error': str(e),
            'time': elapsed
        }


def main():
    """主函数"""
    model_name = "/Volumes/toshiba/models/qwen3.5-35b-mlx"
    prompt = "介绍机器学习的基本概念和应用场景"
    max_tokens = 200

    print("="*60)
    print("Heterogeneous Memory Compaction 质量验证")
    print("="*60)
    print(f"\n模型: {model_name}")
    print(f"Prompt: '{prompt}'")
    print(f"Max tokens: {max_tokens}\n")

    # 加载模型
    print("Loading model...")
    model, tokenizer = load(model_name)
    print(f"Model loaded: {len(model.layers)} layers\n")

    # 测试配置
    configs = [
        ("Baseline", None, None),
        ("Conservative (ratio=2.0, size=8192)", 8192, 2.0),
        ("Moderate (ratio=3.0, size=8192)", 8192, 3.0),
        ("Aggressive (ratio=5.0, size=4096)", 4096, 5.0)  # 原始配置
    ]

    results = []

    for config_name, max_size, compression_ratio in configs:
        result = test_configuration(
            model, tokenizer, prompt, max_tokens,
            config_name, max_size, compression_ratio
        )
        results.append(result)

    # 生成对比报告
    print("\n" + "="*60)
    print("对比报告")
    print("="*60 + "\n")

    print(f"{'配置':<40} {'成功':<6} {'Tokens':<8} {'耗时':<10} {'速度':<12}")
    print("-"*80)

    for result in results:
        if result['success']:
            print(f"{result['config']:<40} ✅     {result['tokens']:<8} {result['time']:<10.2f} {result['tps']:<12.2f}")
        else:
            print(f"{result['config']:<40} ❌     -        -          -")

    # 保存详细报告
    report_path = Path(__file__).parent.parent / ".solar" / "hetero-cache-quality-report.md"
    with open(report_path, "w") as f:
        f.write("# Heterogeneous Memory Compaction 质量对比报告\n\n")
        f.write(f"**日期**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**模型**: {model_name}\n")
        f.write(f"**Prompt**: '{prompt}'\n")
        f.write(f"**Max tokens**: {max_tokens}\n\n")

        f.write("## 对比结果\n\n")
        f.write("| 配置 | 成功 | Tokens | 耗时 | 速度 |\n")
        f.write("|------|------|--------|------|------|\n")

        for result in results:
            if result['success']:
                f.write(f"| {result['config']} | ✅ | {result['tokens']} | {result['time']:.2f}s | {result['tps']:.2f} tok/s |\n")
            else:
                f.write(f"| {result['config']} | ❌ | - | - | - |\n")

        f.write("\n## 生成内容对比\n\n")

        for result in results:
            if result['success']:
                f.write(f"### {result['config']}\n\n")
                f.write("```\n")
                f.write(result['output'][:500])
                if len(result['output']) > 500:
                    f.write(f"\n... (截断, 总共 {len(result['output'])} 字符)\n")
                f.write("```\n\n")

    print(f"\n详细报告已保存到: {report_path}")


if __name__ == "__main__":
    main()
