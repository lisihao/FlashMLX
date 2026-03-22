#!/usr/bin/env python3
"""
对照实验 - Qwen3-8B (纯 Transformer)
验证 Attention Matching 在纯 Transformer 架构上的表现

目的：
- 如果 Qwen3-8B 上工作正常 → 问题出在 Qwen3.5 混合架构
- 如果 Qwen3-8B 上也失效 → 问题可能是 Qwen 系列特殊实现
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

# 使用更简单的测试场景（减少生成时间）
TEST_SCENARIOS = {
    "简单数学": {
        "prompt": "问题：3 + 5 = ?\n回答：",
        "max_tokens": 30,
        "description": "测试基本逻辑能力"
    },
    "中文理解": {
        "prompt": "请用一句话描述春天的特点。\n回答：",
        "max_tokens": 50,
        "description": "测试中文语义理解"
    },
    "简单推理": {
        "prompt": "如果小明有5个苹果，吃了2个，还剩几个？\n回答：",
        "max_tokens": 40,
        "description": "测试简单推理"
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
        if isinstance(tokens, list):
            tokens = mx.array(tokens)
        return tokenizer.decode(tokens.tolist())
    except Exception as e:
        return f"<decode_error: {e}>"


# ==================== Testing Functions ====================

def run_baseline(model, tokenizer, prompt: str, max_tokens: int) -> Tuple[List[int], float, str]:
    """运行 baseline（无压缩）"""
    print(f"  🔵 Baseline...")

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

    # Extract tokens
    response_text = response.strip()
    tokens = tokenizer.encode(response_text)

    # Calculate metrics
    total_time = end_time - start_time
    tg = len(tokens) / total_time if total_time > 0 else 0

    return tokens, tg, response_text


def run_compressed(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int,
    compression_ratio: float
) -> Tuple[List[int], float, str, Dict]:
    """运行压缩版本"""
    print(f"  🟢 Compression {compression_ratio}x...")

    # Inject compression
    try:
        restore_fn = inject_attention_matching(
            model,
            compression_ratio=compression_ratio,
            num_queries=100,
            verbose=False
        )
    except Exception as e:
        print(f"    ❌ Injection failed: {e}")
        return [], 0, "", {"error": str(e)}

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

        stats = {"success": True}

    except Exception as e:
        print(f"    ❌ Generation failed: {e}")
        tokens = []
        tg = 0
        response_text = ""
        stats = {"error": str(e)}
    finally:
        # Restore
        restore_fn()

    return tokens, tg, response_text, stats


def run_scenario_test(
    model,
    tokenizer,
    scenario_name: str,
    scenario_config: Dict,
    compression_ratios: List[float]
) -> Dict:
    """运行单个场景的所有测试"""
    print(f"\n{'='*60}")
    print(f"📋 {scenario_name}: {scenario_config['description']}")
    print(f"{'='*60}")

    prompt = scenario_config["prompt"]
    max_tokens = scenario_config["max_tokens"]

    results = {
        "scenario": scenario_name,
        "prompt": prompt,
        "baseline": {},
        "compressed": []
    }

    # Run baseline
    baseline_tokens, baseline_tg, baseline_text = run_baseline(
        model, tokenizer, prompt, max_tokens
    )

    results["baseline"] = {
        "text": baseline_text,
        "tg": baseline_tg,
        "num_tokens": len(baseline_tokens)
    }

    print(f"  ✅ {len(baseline_tokens)} tokens, {baseline_tg:.2f} tok/s")
    print(f"     \"{baseline_text[:80]}...\"")

    # Run compressed versions
    for ratio in compression_ratios:
        compressed_tokens, compressed_tg, compressed_text, stats = run_compressed(
            model, tokenizer, prompt, max_tokens, ratio
        )

        if "error" in stats:
            print(f"  ❌ Failed")
            results["compressed"].append({
                "compression_ratio": ratio,
                "error": stats["error"]
            })
            continue

        # Calculate token overlap
        token_overlap = calculate_token_overlap(baseline_tokens, compressed_tokens)

        # Calculate performance change
        tg_change = (compressed_tg - baseline_tg) / baseline_tg * 100 if baseline_tg > 0 else 0

        results["compressed"].append({
            "compression_ratio": ratio,
            "text": compressed_text,
            "tg": compressed_tg,
            "num_tokens": len(compressed_tokens),
            "token_overlap": token_overlap,
            "tg_change": tg_change
        })

        # Print results
        overlap_status = "✅" if token_overlap >= 0.5 else "⚠️"
        tg_status = "✅" if abs(tg_change) <= 20 else "⚠️"

        print(f"  🟢 {ratio}x: Overlap {token_overlap*100:.1f}% {overlap_status}, "
              f"TG {compressed_tg:.2f} ({tg_change:+.1f}%) {tg_status}")
        print(f"     \"{compressed_text[:80]}...\"")

    return results


# ==================== Main Test ====================

def main():
    """运行对照实验"""
    print("\n" + "="*60)
    print("🧪 对照实验 - Qwen3-8B (纯 Transformer)")
    print("="*60)

    # Config
    model_path = "/Volumes/toshiba/models/qwen3-8b-mlx"
    compression_ratios = [2.0, 2.5, 3.0]

    print(f"\n📦 Model: Qwen3-8B (纯 Transformer)")
    print(f"🔧 Compression ratios: {compression_ratios}")
    print(f"🎯 Target: Token overlap ≥ 50%")

    # Load model
    print(f"\n⏳ Loading model...")
    start_load = time.time()
    model, tokenizer = load(model_path)
    load_time = time.time() - start_load
    print(f"✅ Model loaded in {load_time:.1f}s")

    # Check architecture
    print(f"\n🔍 Architecture check:")
    if hasattr(model, 'layers'):
        num_layers = len(model.layers)
        print(f"   Layers: {num_layers}")

        # Check first layer type
        first_layer = model.layers[0]
        has_self_attn = hasattr(first_layer, 'self_attn')
        has_linear_attn = hasattr(first_layer, 'is_linear')

        if has_linear_attn:
            print(f"   ⚠️  WARNING: Model has 'is_linear' attribute (混合架构?)")
        elif has_self_attn:
            print(f"   ✅ Pure Transformer (self_attn found)")
        else:
            print(f"   ⚠️  Unknown architecture")

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
    print("📊 SUMMARY")
    print("="*60)

    total_tests = 0
    passed_overlap = 0
    passed_tg = 0

    for result in all_results:
        print(f"\n{result['scenario']}:")
        print(f"  Baseline: \"{result['baseline']['text'][:60]}...\"")

        for comp in result["compressed"]:
            if "error" not in comp:
                total_tests += 1
                ratio = comp["compression_ratio"]
                overlap = comp["token_overlap"]
                tg_change = comp["tg_change"]

                if overlap >= 0.5:
                    passed_overlap += 1
                if abs(tg_change) <= 20:
                    passed_tg += 1

                overlap_status = "✅ PASS" if overlap >= 0.5 else "⚠️ FAIL"
                tg_status = "✅" if abs(tg_change) <= 20 else "⚠️"

                print(f"  {ratio}x: Overlap {overlap*100:.1f}% {overlap_status}, "
                      f"TG {tg_change:+.1f}% {tg_status}")
                print(f"       \"{comp['text'][:60]}...\"")

    # Final stats
    print("\n" + "="*60)
    print("🎯 FINAL RESULT")
    print("="*60)

    if total_tests > 0:
        overlap_rate = passed_overlap / total_tests * 100
        tg_rate = passed_tg / total_tests * 100

        print(f"\nToken overlap ≥50%: {passed_overlap}/{total_tests} ({overlap_rate:.1f}%)")
        print(f"TG change ≤20%: {passed_tg}/{total_tests} ({tg_rate:.1f}%)")

        if overlap_rate >= 80:
            print(f"\n🎉 PASS - Attention Matching 在纯 Transformer 上工作正常！")
            print(f"   结论：问题确实出在 Qwen3.5 混合架构上")
        elif overlap_rate >= 50:
            print(f"\n⚠️ PARTIAL PASS - 部分场景通过")
            print(f"   结论：需要进一步分析")
        else:
            print(f"\n❌ FAIL - Attention Matching 在 Qwen3 上也失效")
            print(f"   结论：可能是 Qwen 系列模型的特殊实现问题")
    else:
        print(f"\n❌ No tests completed")

    # Save results
    output_file = "qwen3_pure_transformer_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Results saved to {output_file}")


if __name__ == "__main__":
    main()
