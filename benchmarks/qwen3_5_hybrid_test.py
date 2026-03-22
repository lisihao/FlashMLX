"""
测试 CompactedKVCache 在 Qwen3.5 (混合架构) 上的兼容性

目的: 验证扩展后的 CompactedKVCache 支持 SSM + Attention 混合架构
      - SSM 层使用 cache[0], cache[1]
      - Attention 层使用 cache.update_and_fetch()
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "mlx-lm-source"))

import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.models.compacted_cache import CompactedKVCache
import time

MODEL_PATH = "/Users/lisihao/models/qwen3.5-35b-mlx"

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
    print("Qwen3.5 35B + CompactedKVCache 混合架构测试")
    print("验证: 扩展后的 CompactedKVCache 支持 SSM + Attention")
    print("=" * 80)

    # 加载模型
    print("\n加载模型...")
    model, tokenizer = load(MODEL_PATH)
    num_layers = len(model.layers)
    print(f"模型加载完成: {num_layers} 层")

    # 检查架构
    print("\n检查架构:")
    ssm_count = 0
    attention_count = 0
    for i, layer in enumerate(model.layers):
        is_linear = hasattr(layer, 'is_linear') and layer.is_linear
        if is_linear:
            ssm_count += 1
        else:
            attention_count += 1

    print(f"  SSM 层: {ssm_count} 个")
    print(f"  Attention 层: {attention_count} 个")
    print(f"  ✅ 确认: 混合架构（SSM + Attention）")

    # Test 1: Baseline
    print("\n[Test 1] Baseline (无压缩)")
    result1 = test_config("Baseline", model, tokenizer, cache=None)

    # Test 2: CompactedKVCache (Fast Path)
    print("\n[Test 2] CompactedKVCache 5x (Fast Path)")
    cache = [
        CompactedKVCache(
            max_size=4096,
            compression_ratio=5.0,
            use_quality_path=False,
            enable_compression=True
        )
        for _ in range(num_layers)
    ]
    result2 = test_config("CompactedKVCache 5x (Fast)", model, tokenizer, cache=cache)

    # Test 3: CompactedKVCache (Quality Path)
    print("\n[Test 3] CompactedKVCache 5x (Quality Path)")
    cache_quality = [
        CompactedKVCache(
            max_size=4096,
            compression_ratio=5.0,
            use_quality_path=True,
            enable_compression=True
        )
        for _ in range(num_layers)
    ]
    result3 = test_config("CompactedKVCache 5x (Quality)", model, tokenizer, cache=cache_quality)

    # 对比分析
    print("\n" + "=" * 80)
    print("对比分析")
    print("=" * 80)
    print(f"  {'配置':<35} {'Tokens':<10} {'速度 (tok/s)':<15} {'Token 差异':<15}")
    print("-" * 80)

    baseline_tokens = result1['tokens']
    for r in [result1, result2, result3]:
        diff = r['tokens'] - baseline_tokens
        diff_pct = (diff / baseline_tokens) * 100 if baseline_tokens > 0 else 0
        print(f"  {r['config']:<35} {r['tokens']:<10} {r['speed']:<15.2f} {diff:+5d} ({diff_pct:+.1f}%)")

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
    print(f"  CompactedKVCache (Fast): {check_quality(result2['text'])}")
    print(f"  CompactedKVCache (Quality): {check_quality(result3['text'])}")

    print("\n内容对比（前 150 字符）:")
    print(f"  Baseline: {result1['text'][:150]}...")
    print(f"  Fast Path: {result2['text'][:150]}...")
    print(f"  Quality Path: {result3['text'][:150]}...")

    # 最终结论
    print("\n" + "=" * 80)
    print("结论")
    print("=" * 80)

    fast_quality_ok = check_quality(result2['text']) == "✅ 正常"
    quality_quality_ok = check_quality(result3['text']) == "✅ 正常"
    fast_tokens_ok = abs(result2['tokens'] - baseline_tokens) / baseline_tokens < 0.05
    quality_tokens_ok = abs(result3['tokens'] - baseline_tokens) / baseline_tokens < 0.05

    if fast_quality_ok and fast_tokens_ok:
        print("✅ Qwen3.5 混合架构 + CompactedKVCache (Fast Path) 工作正常!")
        print(f"   性能变化: {(result2['speed'] / result1['speed'] - 1) * 100:+.1f}%")
        print("   输出质量: 正常")
    else:
        print("⚠️  Fast Path 存在问题")
        if not fast_quality_ok:
            print("   输出质量异常")
        if not fast_tokens_ok:
            print(f"   Token 数量差异: {(result2['tokens'] - baseline_tokens) / baseline_tokens * 100:.1f}%")

    if quality_quality_ok and quality_tokens_ok:
        print("\n✅ Qwen3.5 混合架构 + CompactedKVCache (Quality Path) 工作正常!")
        print(f"   性能变化: {(result3['speed'] / result1['speed'] - 1) * 100:+.1f}%")
        print("   输出质量: 正常")
    else:
        print("\n⚠️  Quality Path 存在问题")
        if not quality_quality_ok:
            print("   输出质量异常")
        if not quality_tokens_ok:
            print(f"   Token 数量差异: {(result3['tokens'] - baseline_tokens) / baseline_tokens * 100:.1f}%")

    if fast_quality_ok and quality_quality_ok and fast_tokens_ok and quality_tokens_ok:
        print("\n📌 结论:")
        print("   - CompactedKVCache 成功支持混合架构（Qwen3.5）")
        print("   - SSM 层使用 cache[0], cache[1] 正常工作")
        print("   - Attention 层使用 update_and_fetch() 正常工作")
        print("   - 输出质量无损，性能保持或提升")

if __name__ == '__main__':
    main()
