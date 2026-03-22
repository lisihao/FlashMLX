"""
简化的 CompactedKVCache 性能测试

直接硬编码模型路径，避免每次搜索
"""

import mlx.core as mx
import time
import json
from pathlib import Path
from datetime import datetime
from mlx_lm import load, generate
from mlx_lm.models.compacted_cache import CompactedKVCache

# 硬编码模型路径
MODEL_PATH = "/Users/lisihao/models/qwen3.5-35b-mlx"

def run_single_test(model, tokenizer, prompt, cache_config_name, cache=None):
    """运行单次测试"""
    print(f"  {cache_config_name}...")

    # 计算 prompt tokens
    prompt_tokens = len(tokenizer.encode(prompt))

    # 生成
    start_time = time.time()
    mx.eval(mx.zeros(1))  # 热身

    # 收集生成的 tokens
    generated_tokens_list = []
    for token in generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=100,
        prompt_cache=cache,
        verbose=False
    ):
        generated_tokens_list.append(token)

    total_time = time.time() - start_time

    # 计算指标
    generated_tokens = len(generated_tokens_list)

    pp_speed = prompt_tokens / total_time if total_time > 0 else 0
    tg_speed = generated_tokens / total_time if total_time > 0 and generated_tokens > 0 else 0

    print(f"    PP: {pp_speed:.2f} tok/s, TG: {tg_speed:.2f} tok/s, Total: {total_time:.2f}s")

    # 压缩统计 (只统计非 None 的 cache)
    if cache:
        active_caches = [c for c in cache if c is not None]
        if active_caches:
            total_compressions = sum(c.get_stats()['num_compressions'] for c in active_caches)
            if total_compressions > 0:
                caches_with_compression = [c for c in active_caches if c.get_stats()['num_compressions'] > 0]
                if caches_with_compression:
                    avg_ratio = sum(c.get_stats()['avg_compression_ratio'] for c in caches_with_compression) / len(caches_with_compression)
                    print(f"    Compressions: {total_compressions}, Avg ratio: {avg_ratio:.2f}x")

    return {
        'config': cache_config_name,
        'pp_speed': pp_speed,
        'tg_speed': tg_speed,
        'total_time': total_time,
        'prompt_tokens': prompt_tokens,
        'generated_tokens': generated_tokens
    }

def main():
    print("=" * 80)
    print("CompactedKVCache 简化性能测试")
    print(f"模型: {MODEL_PATH}")
    print("=" * 80)

    # 加载模型
    print("\n加载模型...")
    model, tokenizer = load(MODEL_PATH)
    num_layers = len(model.layers)
    print(f"模型加载完成: {num_layers} 层")

    # 检查 layer types (Qwen 3.5 混合架构)
    config_path = Path(MODEL_PATH) / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    layer_types = config.get("text_config", {}).get("layer_types", [])
    full_attn_layers = [i for i, t in enumerate(layer_types) if t == "full_attention"]
    linear_attn_layers = [i for i, t in enumerate(layer_types) if t == "linear_attention"]

    print(f"Full Attention 层 ({len(full_attn_layers)}): {full_attn_layers}")
    print(f"Linear Attention 层 ({len(linear_attn_layers)}): 略\n")

    # 测试场景
    scenarios = [
        ("Short (512 tokens)", "The history of artificial intelligence began in antiquity. " * 50),
        ("Medium (2K tokens)", "The history of artificial intelligence began in antiquity. " * 200),
        ("Long (8K tokens)", "The history of artificial intelligence began in antiquity. " * 800),
    ]

    all_results = []

    for scenario_name, prompt in scenarios:
        print(f"\n{'=' * 80}")
        print(f"场景: {scenario_name}")
        print(f"{'=' * 80}\n")

        # 1. Baseline (无 cache，让模型自己管理)
        result = run_single_test(model, tokenizer, prompt, "Baseline (Standard KVCache)", cache=None)
        result['scenario'] = scenario_name
        all_results.append(result)

        # 2. Fast Path 5x (仅 full_attention 层)
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
                cache.append(None)  # linear_attention 层不使用 cache

        result = run_single_test(model, tokenizer, prompt, "Fast Path 5x", cache=cache)
        result['scenario'] = scenario_name
        all_results.append(result)

        # 3. Quality Path 5x (仅 full_attention 层)
        cache = []
        for i in range(num_layers):
            if i in full_attn_layers:
                cache.append(CompactedKVCache(
                    max_size=4096,
                    compression_ratio=5.0,
                    use_quality_path=True,
                    quality_fit_beta=True,
                    quality_fit_c2=True,
                    enable_compression=True
                ))
            else:
                cache.append(None)

        result = run_single_test(model, tokenizer, prompt, "Quality Path 5x", cache=cache)
        result['scenario'] = scenario_name
        all_results.append(result)

        # 4. Fast Path 10x (仅 full_attention 层)
        cache = []
        for i in range(num_layers):
            if i in full_attn_layers:
                cache.append(CompactedKVCache(
                    max_size=4096,
                    compression_ratio=10.0,
                    use_quality_path=False,
                    enable_compression=True
                ))
            else:
                cache.append(None)

        result = run_single_test(model, tokenizer, prompt, "Fast Path 10x", cache=cache)
        result['scenario'] = scenario_name
        all_results.append(result)

    # 保存结果
    output_file = Path(__file__).parent / "simple_benchmark_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'model': MODEL_PATH,
            'timestamp': datetime.now().isoformat(),
            'num_layers': num_layers,
            'results': all_results
        }, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"结果已保存到: {output_file}")
    print(f"{'=' * 80}\n")

    # 打印总结
    print("总结:")
    for scenario in ["Short (512 tokens)", "Medium (2K tokens)", "Long (8K tokens)"]:
        print(f"\n{scenario}:")
        baseline = next((r for r in all_results if r['scenario'] == scenario and 'Baseline' in r['config']), None)
        if baseline:
            print(f"  {'配置':<25} {'PP (tok/s)':<15} {'TG (tok/s)':<15}")
            print("  " + "-" * 55)
            for r in all_results:
                if r['scenario'] == scenario:
                    print(f"  {r['config']:<25} {r['pp_speed']:<15.2f} {r['tg_speed']:<15.2f}")

if __name__ == '__main__':
    main()
