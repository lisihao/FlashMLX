"""
按照文档的正确方式使用 CompactedKVCache

关键修改：为所有层创建 CompactedKVCache，而不是只为 full attention 层
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "mlx-lm-source"))

import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.models.compacted_cache import CompactedKVCache

MODEL_PATH = "/Users/lisihao/models/qwen3.5-35b-mlx"

def test_output(config_name, cache=None):
    """测试输出质量"""
    print(f"\n{'='*80}")
    print(f"配置: {config_name}")
    print(f"{'='*80}")

    # 加载模型
    model, tokenizer = load(MODEL_PATH)

    # 测试 prompt
    prompt = "Please explain machine learning in simple terms."

    # 热身
    mx.eval(mx.zeros(1))

    # 生成
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

    print(f"\n生成 tokens: {token_count}")
    print(f"\n生成内容:")
    print("-" * 80)
    print(generated_text[:500])  # 只显示前 500 字符
    print("-" * 80)

    return {
        'config': config_name,
        'tokens': token_count,
        'text': generated_text
    }

def main():
    print("=" * 80)
    print("正确用法测试：为所有层创建 CompactedKVCache")
    print("=" * 80)

    # Test 1: Baseline
    print("\n[Test 1] Baseline (cache=None)")
    result1 = test_output("Baseline", cache=None)

    # Test 2: 按照文档的正确方式 - 为所有层创建 CompactedKVCache
    print("\n[Test 2] CompactedKVCache for ALL layers (正确用法)")

    # 先加载模型获取层数
    model, tokenizer = load(MODEL_PATH)
    num_layers = len(model.layers)

    # 为所有层创建 CompactedKVCache（按照文档第 90-94 行）
    cache = [
        CompactedKVCache(
            max_size=4096,
            compression_ratio=5.0,
            use_quality_path=False,
            enable_compression=True
        )
        for _ in range(num_layers)  # 所有层！
    ]

    result2 = test_output("CompactedKVCache (All Layers)", cache=cache)

    # 对比
    print("\n" + "=" * 80)
    print("对比结果:")
    print("-" * 80)
    print(f"  {'配置':<30} {'生成 tokens':<15} {'Token 差异':<15}")
    print("-" * 80)
    baseline_tokens = result1['tokens']
    for r in [result1, result2]:
        diff = r['tokens'] - baseline_tokens
        diff_pct = (diff / baseline_tokens) * 100 if baseline_tokens > 0 else 0
        print(f"  {r['config']:<30} {r['tokens']:<15} {diff:+5d} ({diff_pct:+.1f}%)")

    print("\n质量评估:")
    print(f"  Baseline: {result1['text'][:100]}...")
    print(f"  CompactedKVCache: {result2['text'][:100]}...")

if __name__ == '__main__':
    main()
