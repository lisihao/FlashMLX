#!/usr/bin/env python3
"""
StreamingLLM MVP - Task #61 Day 1
验证 RotatingKVCache 在 Qwen3.5-35B 上的 StreamingLLM 效果
"""

import mlx.core as mx
import mlx_lm
from mlx_lm import load, generate
from mlx_lm.models.cache import RotatingKVCache
import time
import json
from pathlib import Path

# StreamingLLM 配置
STREAMING_CONFIG = {
    "attention_sink": 4,      # 保留前 4 个 tokens
    "sliding_window": 2044,   # 滑动窗口 2044 tokens
    "max_cache_size": 2048    # 总缓存 = sink + window
}

# 使用基线测试的四场景
TEST_SCENARIOS = {
    "chinese": {
        "prompt": "什么是人工智能？请用中文回答。",
        "max_tokens": 50,
        "description": "纯中文生成"
    },
    "think_tag": {
        "prompt": "请使用 <think> 标签思考：什么是深度学习？",
        "max_tokens": 50,
        "description": "<think> 标签控制"
    },
    "format_control": {
        "prompt": "列出三个机器学习的应用场景，使用列表格式。",
        "max_tokens": 50,
        "description": "格式化输出（列表）"
    },
    "mixed_language": {
        "prompt": "Please answer in Chinese: What is deep learning? Answer (Chinese):",
        "max_tokens": 50,
        "description": "英文+中文混合"
    }
}

def measure_generation_with_streaming(model, tokenizer, prompt, max_tokens=50):
    """使用 RotatingKVCache 测量生成质量和性能"""
    print(f"  使用 RotatingKVCache (keep={STREAMING_CONFIG['attention_sink']}, "
          f"max_size={STREAMING_CONFIG['max_cache_size']})...", end="", flush=True)

    start_time = time.perf_counter()

    # 生成文本 - generate() 会自动使用模型的 cache 设置
    output = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        verbose=False
    )

    end_time = time.perf_counter()

    # 计算指标
    total_time = (end_time - start_time) * 1000  # ms
    output_tokens = len(output.split())  # 简单估算
    tbt = total_time / output_tokens if output_tokens > 0 else 0

    print(f" 完成 ({output_tokens} tokens, {tbt:.2f} ms/token)")

    return {
        "output": output,
        "total_time_ms": total_time,
        "tbt_ms": tbt,
        "tokens_generated": output_tokens
    }

def run_mvp_test(model_path):
    """运行 MVP 测试"""
    print("=" * 60)
    print("StreamingLLM MVP 测试 - Task #61 Day 1")
    print("=" * 60)

    # 加载模型
    print(f"\n加载模型: {model_path}")
    model, tokenizer = load(model_path)
    print("✅ 模型加载完成\n")

    # 获取模型配置
    print("📊 模型配置:")
    print(f"  模型类型: {type(model).__name__}")

    # 检查模型是否支持 cache
    if hasattr(model, 'make_cache'):
        print("  ✅ 模型支持自定义 cache")

        # 创建 RotatingKVCache
        print(f"\n🔧 注入 RotatingKVCache:")
        print(f"  attention_sink: {STREAMING_CONFIG['attention_sink']}")
        print(f"  sliding_window: {STREAMING_CONFIG['sliding_window']}")
        print(f"  max_cache_size: {STREAMING_CONFIG['max_cache_size']}")

        # Monkey patch: 覆盖 make_cache() 方法
        original_make_cache = model.language_model.make_cache

        def make_streaming_cache():
            """创建 StreamingLLM cache: Attention 层用 RotatingKVCache，SSM 层用 ArraysCache"""
            from mlx_lm.models.cache import ArraysCache

            layers = model.language_model.model.layers
            cache = []
            for l in layers:
                if l.is_linear:
                    # SSM 层 - 使用原始的 ArraysCache
                    cache.append(ArraysCache(size=2))
                else:
                    # Attention 层 - 使用 RotatingKVCache (StreamingLLM)
                    cache.append(RotatingKVCache(
                        max_size=STREAMING_CONFIG['max_cache_size'],
                        keep=STREAMING_CONFIG['attention_sink']
                    ))
            return cache

        # 注入
        model.language_model.make_cache = make_streaming_cache
        print("  ✅ RotatingKVCache 已注入到 Attention 层")
        print(f"  ✅ SSM 层保持使用 ArraysCache (固定大小)")

    else:
        print("  ⚠️  模型不支持自定义 cache，将使用默认缓存")

    print()

    # 测试四场景
    all_results = {}

    for scenario_name, scenario_config in TEST_SCENARIOS.items():
        print(f"{'=' * 60}")
        print(f"场景: {scenario_name} - {scenario_config['description']}")
        print(f"{'=' * 60}")
        print(f"Prompt: {scenario_config['prompt']}")
        print()

        results = measure_generation_with_streaming(
            model,
            tokenizer,
            prompt=scenario_config['prompt'],
            max_tokens=scenario_config['max_tokens']
        )

        print(f"\n📊 结果:")
        print(f"  TBT: {results['tbt_ms']:.2f} ms/token")
        print(f"  总时间: {results['total_time_ms']:.2f} ms")
        print(f"  生成 tokens: {results['tokens_generated']}")
        print(f"\n  输出:")
        print(f"  {results['output'][:100]}...")
        print()

        all_results[scenario_name] = {
            "config": scenario_config,
            "performance": {
                "tbt_ms": results["tbt_ms"],
                "total_time_ms": results["total_time_ms"],
                "tokens": results["tokens_generated"]
            },
            "output": results["output"]
        }

    # 保存结果
    output_file = Path(".solar/streaming-llm-mvp-results.json")
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(f"✅ MVP 测试完成，结果已保存到: {output_file}")
    print("=" * 60)

    # 对比基线
    print("\n📊 与基线对比:")
    print(f"{'场景':<20} {'基线 TBT':<15} {'MVP TBT':<15} {'变化':<10}")
    print("-" * 60)

    baseline_results = {
        "chinese": 37.78,
        "think_tag": 77.42,
        "format_control": 27.40,
        "mixed_language": 26.77
    }

    for scenario_name, baseline_tbt in baseline_results.items():
        if scenario_name in all_results:
            mvp_tbt = all_results[scenario_name]["performance"]["tbt_ms"]
            change = ((mvp_tbt - baseline_tbt) / baseline_tbt) * 100
            change_str = f"{change:+.1f}%"
            print(f"{scenario_name:<20} {baseline_tbt:<15.2f} {mvp_tbt:<15.2f} {change_str:<10}")

    return all_results

if __name__ == "__main__":
    # 使用 Qwen3.5-35B
    MODEL_PATH = "/Volumes/toshiba/models/qwen3.5-35b-mlx"

    results = run_mvp_test(MODEL_PATH)
