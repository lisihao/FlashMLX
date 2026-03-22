"""
准确的 CompactedKVCache 性能测试

修复指标计算错误：
- PP speed: 仅计算 prompt processing 时间（Time To First Token）
- TG speed: 仅计算 token generation 时间（平均每个 token 的生成时间）
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "mlx-lm-source"))

import mlx.core as mx
import time
import json
from datetime import datetime
from mlx_lm import load, generate
from mlx_lm.models.compacted_cache import CompactedKVCache

# 硬编码模型路径
MODEL_PATH = "/Users/lisihao/models/qwen3.5-35b-mlx"

def run_single_test(model, tokenizer, prompt, cache_config_name, cache=None):
    """运行单次测试，正确分离 PP 和 TG 时间"""
    print(f"  {cache_config_name}...")

    # 计算 prompt tokens
    prompt_tokens = len(tokenizer.encode(prompt))

    # 热身
    mx.eval(mx.zeros(1))

    # 生成，记录每个 token 的时间
    start_time = time.time()
    token_times = []
    generated_tokens_list = []

    for token in generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=100,
        prompt_cache=cache,
        verbose=False
    ):
        token_time = time.time()
        token_times.append(token_time)
        generated_tokens_list.append(token)

    total_time = time.time() - start_time

    # 计算指标
    generated_tokens = len(generated_tokens_list)

    if generated_tokens > 0 and len(token_times) > 0:
        # PP time: Time To First Token (TTFT)
        ttft = token_times[0] - start_time
        pp_speed = prompt_tokens / ttft if ttft > 0 else 0

        # TG time: 后续 tokens 的平均生成时间
        if generated_tokens > 1:
            # 后续 tokens 的总时间
            tg_total_time = token_times[-1] - token_times[0]
            tg_speed = (generated_tokens - 1) / tg_total_time if tg_total_time > 0 else 0
        else:
            tg_speed = 0
    else:
        pp_speed = 0
        tg_speed = 0
        ttft = 0

    print(f"    TTFT: {ttft:.2f}s, PP: {pp_speed:.2f} tok/s, TG: {tg_speed:.2f} tok/s, Total: {total_time:.2f}s")

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
        'ttft': ttft,
        'pp_speed': pp_speed,
        'tg_speed': tg_speed,
        'total_time': total_time,
        'prompt_tokens': prompt_tokens,
        'generated_tokens': generated_tokens
    }

def main():
    print("=" * 80)
    print("CompactedKVCache 准确性能测试")
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

    # 测试场景 - 只测试一个场景来快速验证
    scenarios = [
        ("Medium (2K tokens)", "The history of artificial intelligence began in antiquity. " * 200),
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
                cache.append(None)

        result = run_single_test(model, tokenizer, prompt, "Fast Path 5x", cache=cache)
        result['scenario'] = scenario_name
        all_results.append(result)

    # 保存结果
    output_file = Path(__file__).parent / "accurate_benchmark_results.json"
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

    # 打印对比
    print("对比结果:")
    for scenario in ["Medium (2K tokens)"]:
        print(f"\n{scenario}:")
        print(f"  {'配置':<30} {'TTFT (s)':<12} {'PP (tok/s)':<15} {'TG (tok/s)':<15}")
        print("  " + "-" * 72)
        for r in all_results:
            if r['scenario'] == scenario:
                print(f"  {r['config']:<30} {r['ttft']:<12.2f} {r['pp_speed']:<15.2f} {r['tg_speed']:<15.2f}")

if __name__ == '__main__':
    main()
