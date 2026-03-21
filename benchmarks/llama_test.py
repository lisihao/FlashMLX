"""
在纯 Transformer 模型（Llama 3.2 3B）上测试 CompactedKVCache

验证：
1. CompactedKVCache 是否能正常工作
2. 输出质量是否正常
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "mlx-lm-source"))

import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.models.compacted_cache import CompactedKVCache
import time

MODEL_PATH = "/Users/lisihao/models/llama-3.2-3b-mlx"

def test_config(config_name, model, tokenizer, cache=None):
    """测试单个配置"""
    print(f"\n{'='*80}")
    print(f"配置: {config_name}")
    print(f"{'='*80}")

    # 测试 prompt
    prompt = "Please explain the concept of machine learning in simple terms."

    # 热身
    mx.eval(mx.zeros(1))

    # 生成
    start_time = time.time()
    generated_text = ""
    token_count = 0

    for token in generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=100,
        prompt_cache=cache,
        verbose=False
    ):
        generated_text += token
        token_count += 1

    total_time = time.time() - start_time

    print(f"\n生成 tokens: {token_count}")
    print(f"总时间: {total_time:.2f}s")
    print(f"速度: {token_count/total_time:.2f} tok/s")
    print(f"\n生成内容:")
    print("-" * 80)
    # 显示前 400 字符
    display_text = generated_text[:400] if len(generated_text) > 400 else generated_text
    print(display_text)
    if len(generated_text) > 400:
        print("...")
    print("-" * 80)

    # 压缩统计
    if cache:
        total_compressions = 0
        for c in cache:
            if c is not None:
                stats = c.get_stats()
                total_compressions += stats['num_compressions']

        if total_compressions > 0:
            print(f"\n压缩统计: 总共 {total_compressions} 次压缩")

    return {
        'config': config_name,
        'tokens': token_count,
        'time': total_time,
        'speed': token_count / total_time if total_time > 0 else 0,
        'text': generated_text
    }

def main():
    print("=" * 80)
    print("Llama 3.2 3B + CompactedKVCache 测试")
    print("=" * 80)

    # 加载模型
    print("\n加载模型...")
    model, tokenizer = load(MODEL_PATH)
    num_layers = len(model.layers)
    print(f"模型加载完成: {num_layers} 层")

    # Test 1: Baseline
    print("\n[Test 1] Baseline (无压缩)")
    result1 = test_config("Baseline", model, tokenizer, cache=None)

    # Test 2: CompactedKVCache (按照文档的标准用法)
    print("\n[Test 2] CompactedKVCache 5x (所有层)")
    cache = [
        CompactedKVCache(
            max_size=4096,
            compression_ratio=5.0,
            use_quality_path=False,
            enable_compression=True
        )
        for _ in range(num_layers)
    ]
    result2 = test_config("CompactedKVCache 5x", model, tokenizer, cache=cache)

    # Test 3: Quality Path
    print("\n[Test 3] CompactedKVCache Quality Path")
    cache_quality = [
        CompactedKVCache(
            max_size=4096,
            compression_ratio=5.0,
            use_quality_path=True,
            enable_compression=True
        )
        for _ in range(num_layers)
    ]
    result3 = test_config("Quality Path 5x", model, tokenizer, cache=cache_quality)

    # 对比分析
    print("\n" + "=" * 80)
    print("对比分析")
    print("=" * 80)
    print(f"  {'配置':<25} {'Tokens':<10} {'速度 (tok/s)':<15} {'Token 差异':<15}")
    print("-" * 80)

    baseline_tokens = result1['tokens']
    for r in [result1, result2, result3]:
        diff = r['tokens'] - baseline_tokens
        diff_pct = (diff / baseline_tokens) * 100 if baseline_tokens > 0 else 0
        print(f"  {r['config']:<25} {r['tokens']:<10} {r['speed']:<15.2f} {diff:+5d} ({diff_pct:+.1f}%)")

    # 质量评估
    print("\n质量评估:")
    print("-" * 80)

    def check_quality(text):
        """简单的质量检查"""
        # 检查是否有重复
        words = text.split()
        if len(words) > 10:
            # 检查是否有连续重复的词
            for i in range(len(words) - 5):
                if words[i] == words[i+1] == words[i+2]:
                    return "❌ 检测到重复"

        # 检查是否有意义
        if len(text.strip()) < 50:
            return "❌ 内容太短"

        return "✅ 正常"

    print(f"  Baseline: {check_quality(result1['text'])}")
    print(f"  CompactedKVCache 5x: {check_quality(result2['text'])}")
    print(f"  Quality Path: {check_quality(result3['text'])}")

    print("\n内容对比（前 150 字符）:")
    print(f"  Baseline: {result1['text'][:150]}...")
    print(f"  CompactedKVCache: {result2['text'][:150]}...")
    print(f"  Quality Path: {result3['text'][:150]}...")

if __name__ == '__main__':
    main()
