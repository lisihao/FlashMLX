#!/usr/bin/env python3
"""
完整质量测试 - Qwen3.5 35B 真实文本
测试 Attention Matching 在真实模型上的表现

测试场景：
1. 中文生成 - 测试中文语义理解
2. Think mode - 测试推理链质量
3. 逻辑推理 - 测试逻辑一致性
4. 长 Context - 测试长序列处理

Compression ratios: 2.0x, 2.5x, 3.0x

验收标准：
- Token overlap ≥ 50%
- 生成质量无明显下降
- 性能开销 ≤ 10%
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate

from flashmlx.cache.simple_injection import inject_attention_matching


# ==================== Test Scenarios ====================

TEST_SCENARIOS = {
    "中文生成": {
        "prompt": "请用中文写一首关于春天的诗，要求押韵，每句7字，共4句。",
        "max_tokens": 100,
        "description": "测试中文语义理解和押韵能力"
    },
    "Think模式": {
        "prompt": """请一步步思考：如果一个房间里有3只猫，每只猫抓到2只老鼠，那么房间里现在有多少只动物？

让我们按步骤思考：""",
        "max_tokens": 150,
        "description": "测试逻辑推理链完整性"
    },
    "逻辑推理": {
        "prompt": """问题：小明有5个苹果，给了小红2个，又买了3个，请问小明现在有几个苹果？

回答：让我们计算一下：""",
        "max_tokens": 80,
        "description": "测试数学逻辑准确性"
    },
    "长Context": {
        "prompt": """请阅读以下故事并回答问题：

从前有一个勇敢的骑士，他住在一座古老的城堡里。有一天，国王派他去寻找传说中的魔法宝石。骑士骑着他的白马，穿过黑暗的森林，来到了一座高山脚下。山上住着一条龙，守护着宝石。骑士与龙激战三天三夜，最终用智慧战胜了龙，取得了宝石。

问题：骑士用什么战胜了龙？
回答：""",
        "max_tokens": 50,
        "description": "测试长文本理解和记忆"
    }
}


# ==================== Metrics Calculation ====================

def calculate_token_overlap(tokens1: List[int], tokens2: List[int]) -> float:
    """计算 token overlap（前缀匹配率）"""
    min_len = min(len(tokens1), len(tokens2))
    if min_len == 0:
        return 0.0

    # Count matching tokens
    matches = sum(1 for i in range(min_len) if tokens1[i] == tokens2[i])
    return matches / min_len


def decode_tokens(tokens: List[int], tokenizer) -> str:
    """解码 tokens 为文本"""
    try:
        # Convert to mx.array if needed
        if isinstance(tokens, list):
            tokens = mx.array(tokens)
        return tokenizer.decode(tokens.tolist())
    except Exception as e:
        return f"<decode_error: {e}>"


# ==================== Testing Functions ====================

def run_baseline(model, tokenizer, prompt: str, max_tokens: int) -> Tuple[List[int], float, float]:
    """运行 baseline（无压缩）"""
    print(f"  🔵 Running baseline...")

    # Generate
    start_time = time.time()
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        verbose=False
    )
    end_time = time.time()

    # Extract tokens (approximate - we'll use the response text)
    # Note: In production, we'd capture actual token IDs
    response_text = response.strip()
    tokens = tokenizer.encode(response_text)

    # Calculate metrics
    total_time = end_time - start_time
    tg = len(tokens) / total_time if total_time > 0 else 0

    return tokens, tg, total_time


def run_compressed(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int,
    compression_ratio: float
) -> Tuple[List[int], float, float, Dict]:
    """运行压缩版本"""
    print(f"  🟢 Running with compression_ratio={compression_ratio}...")

    # Inject compression
    try:
        restore_fn = inject_attention_matching(
            model,
            compression_ratio=compression_ratio,
            num_queries=100,  # Optimal value from Task #91
            verbose=False
        )
    except Exception as e:
        print(f"    ❌ Injection failed: {e}")
        return [], 0, 0, {"error": str(e)}

    # Generate
    start_time = time.time()
    try:
        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False
        )
        end_time = time.time()

        # Extract tokens
        response_text = response.strip()
        tokens = tokenizer.encode(response_text)

        # Calculate metrics
        total_time = end_time - start_time
        tg = len(tokens) / total_time if total_time > 0 else 0

        # Get compression stats
        stats = {"success": True}

    except Exception as e:
        print(f"    ❌ Generation failed: {e}")
        tokens = []
        tg = 0
        total_time = 0
        stats = {"error": str(e)}
    finally:
        # Restore
        restore_fn()

    return tokens, tg, total_time, stats


def run_scenario_test(
    model,
    tokenizer,
    scenario_name: str,
    scenario_config: Dict,
    compression_ratios: List[float]
) -> Dict:
    """运行单个场景的所有测试"""
    print(f"\n{'='*60}")
    print(f"📋 Scenario: {scenario_name}")
    print(f"   {scenario_config['description']}")
    print(f"{'='*60}")

    prompt = scenario_config["prompt"]
    max_tokens = scenario_config["max_tokens"]

    results = {
        "scenario": scenario_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "baseline": {},
        "compressed": []
    }

    # Run baseline
    baseline_tokens, baseline_tg, baseline_time = run_baseline(
        model, tokenizer, prompt, max_tokens
    )
    baseline_text = decode_tokens(baseline_tokens, tokenizer)

    results["baseline"] = {
        "tokens": baseline_tokens,
        "text": baseline_text,
        "tg": baseline_tg,
        "time": baseline_time,
        "num_tokens": len(baseline_tokens)
    }

    print(f"  ✅ Baseline: {len(baseline_tokens)} tokens, {baseline_tg:.2f} tok/s")
    print(f"     Output: {baseline_text[:100]}...")

    # Run compressed versions
    for ratio in compression_ratios:
        compressed_tokens, compressed_tg, compressed_time, stats = run_compressed(
            model, tokenizer, prompt, max_tokens, ratio
        )

        if "error" in stats:
            print(f"  ❌ Compression {ratio}x failed")
            results["compressed"].append({
                "compression_ratio": ratio,
                "error": stats["error"]
            })
            continue

        compressed_text = decode_tokens(compressed_tokens, tokenizer)

        # Calculate token overlap
        token_overlap = calculate_token_overlap(baseline_tokens, compressed_tokens)

        # Calculate performance change
        tg_change = (compressed_tg - baseline_tg) / baseline_tg * 100 if baseline_tg > 0 else 0
        time_overhead = (compressed_time - baseline_time) / baseline_time * 100 if baseline_time > 0 else 0

        results["compressed"].append({
            "compression_ratio": ratio,
            "tokens": compressed_tokens,
            "text": compressed_text,
            "tg": compressed_tg,
            "time": compressed_time,
            "num_tokens": len(compressed_tokens),
            "token_overlap": token_overlap,
            "tg_change": tg_change,
            "time_overhead": time_overhead
        })

        # Print results
        print(f"  🟢 Compression {ratio}x:")
        print(f"     Token overlap: {token_overlap*100:.1f}% {'✅' if token_overlap >= 0.5 else '⚠️'}")
        print(f"     TG: {compressed_tg:.2f} tok/s ({tg_change:+.1f}%)")
        print(f"     Time overhead: {time_overhead:+.1f}% {'✅' if abs(time_overhead) <= 10 else '⚠️'}")
        print(f"     Output: {compressed_text[:100]}...")

    return results


# ==================== Main Test ====================

def main():
    """运行完整质量测试"""
    print("\n" + "="*60)
    print("🧪 FlashMLX - Attention Matching 真实文本质量测试")
    print("="*60)

    # Config
    model_path = "/Volumes/toshiba/models/qwen3.5-35b-mlx"
    compression_ratios = [2.0, 2.5, 3.0]

    print(f"\n📦 Model: {model_path}")
    print(f"🔧 Compression ratios: {compression_ratios}")
    print(f"🎯 Target: Token overlap ≥ 50%, Time overhead ≤ 10%")

    # Load model
    print(f"\n⏳ Loading model...")
    start_load = time.time()
    model, tokenizer = load(model_path)
    load_time = time.time() - start_load
    print(f"✅ Model loaded in {load_time:.1f}s")

    # Run all scenarios
    all_results = []
    for scenario_name, scenario_config in TEST_SCENARIOS.items():
        result = run_scenario_test(
            model,
            tokenizer,
            scenario_name,
            scenario_config,
            compression_ratios
        )
        all_results.append(result)

    # ==================== Summary ====================

    print("\n" + "="*60)
    print("📊 SUMMARY - Token Overlap Results")
    print("="*60)

    for result in all_results:
        print(f"\n{result['scenario']}:")
        for comp in result["compressed"]:
            if "error" in comp:
                print(f"  {comp['compression_ratio']}x: ❌ ERROR")
            else:
                overlap = comp["token_overlap"]
                status = "✅ PASS" if overlap >= 0.5 else "⚠️ FAIL"
                print(f"  {comp['compression_ratio']}x: {overlap*100:.1f}% {status}")

    print("\n" + "="*60)
    print("📊 SUMMARY - Performance Results")
    print("="*60)

    for result in all_results:
        print(f"\n{result['scenario']}:")
        baseline_tg = result["baseline"]["tg"]
        print(f"  Baseline: {baseline_tg:.2f} tok/s")
        for comp in result["compressed"]:
            if "error" not in comp:
                tg_change = comp["tg_change"]
                time_overhead = comp["time_overhead"]
                overhead_status = "✅" if abs(time_overhead) <= 10 else "⚠️"
                print(f"  {comp['compression_ratio']}x: {comp['tg']:.2f} tok/s ({tg_change:+.1f}%), "
                      f"overhead {time_overhead:+.1f}% {overhead_status}")

    # Save results
    output_file = "quality_test_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Results saved to {output_file}")

    # ==================== Final Verdict ====================

    print("\n" + "="*60)
    print("🎯 FINAL VERDICT")
    print("="*60)

    # Calculate overall stats
    total_tests = 0
    passed_overlap = 0
    passed_overhead = 0

    for result in all_results:
        for comp in result["compressed"]:
            if "error" not in comp:
                total_tests += 1
                if comp["token_overlap"] >= 0.5:
                    passed_overlap += 1
                if abs(comp["time_overhead"]) <= 10:
                    passed_overhead += 1

    overlap_rate = passed_overlap / total_tests * 100 if total_tests > 0 else 0
    overhead_rate = passed_overhead / total_tests * 100 if total_tests > 0 else 0

    print(f"\nTotal tests: {total_tests}")
    print(f"Token overlap ≥50%: {passed_overlap}/{total_tests} ({overlap_rate:.1f}%) "
          f"{'✅ PASS' if overlap_rate >= 80 else '⚠️ NEEDS IMPROVEMENT'}")
    print(f"Time overhead ≤10%: {passed_overhead}/{total_tests} ({overhead_rate:.1f}%) "
          f"{'✅ PASS' if overhead_rate >= 80 else '⚠️ NEEDS IMPROVEMENT'}")

    overall_pass = overlap_rate >= 80 and overhead_rate >= 80
    print(f"\n{'🎉 OVERALL: PASS' if overall_pass else '⚠️ OVERALL: NEEDS WORK'}")


if __name__ == "__main__":
    main()
