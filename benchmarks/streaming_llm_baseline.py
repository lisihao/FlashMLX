#!/usr/bin/env python3
"""
StreamingLLM 基线性能测试 - Task #61 Day 1
测试四场景（Chinese/Think/Format/Mixed）的质量、TBT、内存占用
"""

import mlx.core as mx
import mlx_lm
from mlx_lm import load, generate
import time
import json
from pathlib import Path

# 四场景测试（与 SSM 压缩研究一致）
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

def measure_generation(model, tokenizer, prompt, max_tokens=50, repetitions=3):
    """测量生成质量和性能"""
    results = {
        "outputs": [],
        "tbt_ms": [],
        "total_time_ms": [],
        "tokens_generated": []
    }

    for i in range(repetitions):
        print(f"  重复 {i+1}/{repetitions}...", end="", flush=True)

        start_time = time.perf_counter()

        # 生成文本
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

        # 简单估算 token 数量（真实应该用 tokenizer.encode）
        output_tokens = len(output.split())
        tbt = total_time / output_tokens if output_tokens > 0 else 0

        results["outputs"].append(output)
        results["total_time_ms"].append(total_time)
        results["tbt_ms"].append(tbt)
        results["tokens_generated"].append(output_tokens)

        print(f" 完成 ({output_tokens} tokens, {tbt:.2f} ms/token)")

    # 计算平均值
    avg_results = {
        "avg_tbt_ms": sum(results["tbt_ms"]) / len(results["tbt_ms"]),
        "avg_total_time_ms": sum(results["total_time_ms"]) / len(results["total_time_ms"]),
        "avg_tokens": sum(results["tokens_generated"]) / len(results["tokens_generated"]),
        "outputs": results["outputs"]
    }

    return avg_results

def estimate_memory_usage():
    """估算当前内存使用（简化版）"""
    # MLX 不提供直接的内存查询 API，这里返回占位符
    return {
        "gpu_memory_mb": "N/A (MLX 不提供直接查询)",
        "note": "需要使用 Activity Monitor 或 Metal System Trace 查看"
    }

def run_baseline_test(model_path, repetitions=3):
    """运行基线性能测试"""
    print("=" * 60)
    print("StreamingLLM 基线性能测试 - Task #61 Day 1")
    print("=" * 60)

    # 加载模型
    print(f"\n加载模型: {model_path}")
    model, tokenizer = load(model_path)
    print("✅ 模型加载完成\n")

    # 测试四场景
    all_results = {}

    for scenario_name, scenario_config in TEST_SCENARIOS.items():
        print(f"{'=' * 60}")
        print(f"场景: {scenario_name} - {scenario_config['description']}")
        print(f"{'=' * 60}")
        print(f"Prompt: {scenario_config['prompt']}")
        print()

        results = measure_generation(
            model,
            tokenizer,
            prompt=scenario_config['prompt'],
            max_tokens=scenario_config['max_tokens'],
            repetitions=repetitions
        )

        print(f"\n📊 结果:")
        print(f"  平均 TBT: {results['avg_tbt_ms']:.2f} ms/token")
        print(f"  平均总时间: {results['avg_total_time_ms']:.2f} ms")
        print(f"  平均生成 tokens: {results['avg_tokens']:.1f}")
        print(f"\n  示例输出:")
        print(f"  {results['outputs'][0][:100]}...")
        print()

        all_results[scenario_name] = {
            "config": scenario_config,
            "performance": {
                "avg_tbt_ms": results["avg_tbt_ms"],
                "avg_total_time_ms": results["avg_total_time_ms"],
                "avg_tokens": results["avg_tokens"]
            },
            "outputs": results["outputs"]
        }

    # 内存使用
    memory_info = estimate_memory_usage()
    all_results["memory"] = memory_info

    # 保存结果
    output_file = Path(".solar/streaming-llm-baseline-results.json")
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(f"✅ 基线测试完成，结果已保存到: {output_file}")
    print("=" * 60)

    # 打印汇总
    print("\n📊 四场景汇总:")
    print(f"{'场景':<20} {'平均 TBT (ms/token)':<25} {'平均 tokens':<15}")
    print("-" * 60)
    for scenario_name, data in all_results.items():
        if scenario_name != "memory":
            perf = data["performance"]
            print(f"{scenario_name:<20} {perf['avg_tbt_ms']:<25.2f} {perf['avg_tokens']:<15.1f}")

    return all_results

if __name__ == "__main__":
    # 使用 Qwen3.5-35B（混合架构）
    MODEL_PATH = "/Volumes/toshiba/models/qwen3.5-35b-mlx"

    results = run_baseline_test(MODEL_PATH, repetitions=1)  # 先测试 1 次，快速验证
