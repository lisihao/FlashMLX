"""
最简单的性能测试：只测量总时间和 token 数

不再试图分离 PP 和 TG，只关注：
1. Total latency (越低越好)
2. Average tok/s (越高越好)
3. Generated tokens 数量
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "mlx-lm-source"))

import mlx.core as mx
import time
from mlx_lm import load, generate
from mlx_lm.models.compacted_cache import CompactedKVCache

MODEL_PATH = "/Users/lisihao/models/qwen3.5-35b-mlx"

def simple_test(model, tokenizer, prompt, config_name, cache=None):
    """最简单的测试：只测量总时间"""
    print(f"\n  {config_name}...")

    # 热身
    mx.eval(mx.zeros(1))

    # 生成
    start = time.time()
    generated_tokens = []

    for token in generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=100,
        prompt_cache=cache,
        verbose=False
    ):
        generated_tokens.append(token)

    total_time = time.time() - start

    # 计算
    n_tokens = len(generated_tokens)
    avg_speed = n_tokens / total_time if total_time > 0 else 0

    print(f"    Tokens: {n_tokens}, Time: {total_time:.2f}s, Speed: {avg_speed:.2f} tok/s")

    # 压缩统计
    if cache:
        active_caches = [c for c in cache if c is not None]
        if active_caches:
            total_compressions = sum(c.get_stats()['num_compressions'] for c in active_caches)
            if total_compressions > 0:
                caches_with_compression = [c for c in active_caches if c.get_stats()['num_compressions'] > 0]
                if caches_with_compression:
                    avg_ratio = sum(c.get_stats()['avg_compression_ratio'] for c in caches_with_compression) / len(caches_with_compression)
                    print(f"    Compressions: {total_compressions}, Ratio: {avg_ratio:.2f}x")

    return {
        'config': config_name,
        'tokens': n_tokens,
        'time': total_time,
        'speed': avg_speed
    }

def main():
    print("=" * 80)
    print("简单性能测试：关注总体吞吐量")
    print("=" * 80)

    # 加载模型
    print("\n加载模型...")
    model, tokenizer = load(MODEL_PATH)

    # 获取 layer 信息
    import json
    config_path = Path(MODEL_PATH) / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    layer_types = config.get("text_config", {}).get("layer_types", [])
    full_attn_layers = [i for i, t in enumerate(layer_types) if t == "full_attention"]
    num_layers = len(model.layers)

    print(f"模型: {num_layers} 层, Full Attention: {len(full_attn_layers)} 层\n")

    # 测试 prompt
    prompt = "The history of artificial intelligence began in antiquity. " * 200

    print("场景: Medium (2K tokens)")
    print("-" * 80)

    # Test 1: Baseline
    result1 = simple_test(model, tokenizer, prompt, "Baseline", cache=None)

    # Test 2: CompactedKVCache
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

    result2 = simple_test(model, tokenizer, prompt, "CompactedKVCache 5x", cache=cache)

    # 对比
    print("\n" + "=" * 80)
    print("对比结果:")
    print("-" * 80)
    print(f"  {'配置':<25} {'Tokens':<10} {'Time (s)':<12} {'Speed (tok/s)':<15}")
    print("-" * 80)
    print(f"  {result1['config']:<25} {result1['tokens']:<10} {result1['time']:<12.2f} {result1['speed']:<15.2f}")
    print(f"  {result2['config']:<25} {result2['tokens']:<10} {result2['time']:<12.2f} {result2['speed']:<15.2f}")

    # 差异分析
    print("\n差异分析:")
    token_diff = result2['tokens'] - result1['tokens']
    token_diff_pct = (token_diff / result1['tokens']) * 100 if result1['tokens'] > 0 else 0
    speed_diff_pct = ((result2['speed'] / result1['speed']) - 1) * 100 if result1['speed'] > 0 else 0

    print(f"  Token 数量差异: {token_diff:+d} ({token_diff_pct:+.1f}%)")
    print(f"  速度差异: {speed_diff_pct:+.1f}%")

    if abs(token_diff_pct) > 10:
        print(f"\n  ⚠️  生成的 token 数量差异较大 ({token_diff_pct:+.1f}%)")
        print(f"      这说明 CompactedKVCache 影响了模型的输出行为")
        print(f"      可能原因：压缩改变了 attention 分布，导致 EOS 更早/晚出现")

if __name__ == '__main__':
    main()
