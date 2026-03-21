"""
输出质量对比测试

对比 Baseline vs CompactedKVCache 的实际输出内容：
1. 输出文本是否有意义
2. 是否提前停止
3. 是否出现重复/退化
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "mlx-lm-source"))

import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.models.compacted_cache import CompactedKVCache
import json

MODEL_PATH = "/Users/lisihao/models/qwen3.5-35b-mlx"

def generate_text(model, tokenizer, prompt, config_name, cache=None, max_tokens=100):
    """生成文本并返回完整输出"""
    print(f"\n{'='*80}")
    print(f"配置: {config_name}")
    print(f"{'='*80}")

    # 热身
    mx.eval(mx.zeros(1))

    # 生成
    generated_text = ""
    token_count = 0
    for token in generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        prompt_cache=cache,
        verbose=False
    ):
        generated_text += token  # 累积每个 token 的文本
        token_count += 1

    # 最终文本
    full_text = generated_text

    print(f"\n生成 tokens: {token_count}")
    print(f"\n生成内容:")
    print("-" * 80)
    print(full_text)
    print("-" * 80)

    # 压缩统计
    if cache:
        active_caches = [c for c in cache if c is not None]
        if active_caches:
            total_compressions = sum(c.get_stats()['num_compressions'] for c in active_caches)
            if total_compressions > 0:
                print(f"\n压缩统计:")
                print(f"  触发次数: {total_compressions}")
                caches_with_compression = [c for c in active_caches if c.get_stats()['num_compressions'] > 0]
                if caches_with_compression:
                    avg_ratio = sum(c.get_stats()['avg_compression_ratio'] for c in caches_with_compression) / len(caches_with_compression)
                    print(f"  平均压缩率: {avg_ratio:.2f}x")

    return {
        'config': config_name,
        'tokens': token_count,
        'text': full_text
    }

def main():
    print("=" * 80)
    print("输出质量对比测试")
    print("=" * 80)

    # 加载模型
    print("\n加载模型...")
    model, tokenizer = load(MODEL_PATH)

    # 获取 layer 信息
    config_path = Path(MODEL_PATH) / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    layer_types = config.get("text_config", {}).get("layer_types", [])
    full_attn_layers = [i for i, t in enumerate(layer_types) if t == "full_attention"]
    num_layers = len(model.layers)

    print(f"模型: {num_layers} 层, Full Attention: {len(full_attn_layers)} 层\n")

    # 测试 prompt - 使用一个明确的问题，便于评估输出质量
    prompt = "Please explain the concept of machine learning in simple terms. "

    print(f"Prompt: {prompt}")

    # Test 1: Baseline
    result1 = generate_text(model, tokenizer, prompt, "Baseline", cache=None, max_tokens=100)

    # Test 2: CompactedKVCache 5x
    cache = []
    for i in range(num_layers):
        if i in full_attn_layers:
            cache.append(CompactedKVCache(
                max_size=4096,
                compression_ratio=5.0,
                use_quality_path=False,
                enable_compression=True
            ))
        else:
            cache.append(None)

    result2 = generate_text(model, tokenizer, prompt, "CompactedKVCache 5x", cache=cache, max_tokens=100)

    # Test 3: CompactedKVCache 2x (降低压缩率)
    cache = []
    for i in range(num_layers):
        if i in full_attn_layers:
            cache.append(CompactedKVCache(
                max_size=4096,
                compression_ratio=2.0,
                use_quality_path=False,
                enable_compression=True
            ))
        else:
            cache.append(None)

    result3 = generate_text(model, tokenizer, prompt, "CompactedKVCache 2x", cache=cache, max_tokens=100)

    # 保存结果
    output_file = Path(__file__).parent / "output_quality_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'prompt': prompt,
            'results': [result1, result2, result3]
        }, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*80}")
    print(f"结果已保存到: {output_file}")
    print(f"{'='*80}")

    # 对比分析
    print("\n对比分析:")
    print("-" * 80)
    print(f"  {'配置':<25} {'生成 tokens':<15} {'Token 差异':<15}")
    print("-" * 80)
    baseline_tokens = result1['tokens']
    for r in [result1, result2, result3]:
        diff = r['tokens'] - baseline_tokens
        diff_pct = (diff / baseline_tokens) * 100 if baseline_tokens > 0 else 0
        print(f"  {r['config']:<25} {r['tokens']:<15} {diff:+5d} ({diff_pct:+.1f}%)")

    # 质量评估提示
    print("\n质量评估:")
    print("  请人工检查上面的生成内容，评估:")
    print("  1. 内容是否完整、连贯？")
    print("  2. 是否提前停止（未完成回答）？")
    print("  3. 是否出现重复或退化？")
    print("  4. 压缩后的输出是否仍然有意义？")

if __name__ == '__main__':
    main()
