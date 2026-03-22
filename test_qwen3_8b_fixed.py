#!/usr/bin/env python3
"""
Qwen3-8B Attention Matching 修复验证
测试 bfloat16 转换修复后的效果
"""

import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate

from flashmlx.cache.simple_injection import inject_attention_matching


# ==================== Test Scenarios ====================

TEST_SCENARIOS = {
    "简单数学": {
        "prompt": "问题：3 + 5 = ?\n回答：",
        "max_tokens": 30,
    },
    "中文理解": {
        "prompt": "请用一句话描述春天的特点。\n回答：",
        "max_tokens": 50,
    },
    "简单推理": {
        "prompt": "如果小明有5个苹果，吃了2个，还剩几个？\n回答：",
        "max_tokens": 40,
    },
}


# ==================== Metrics ====================

def calculate_token_overlap(tokens1: list, tokens2: list) -> float:
    """计算 token overlap（前缀匹配率）"""
    min_len = min(len(tokens1), len(tokens2))
    if min_len == 0:
        return 0.0

    matches = sum(1 for i in range(min_len) if tokens1[i] == tokens2[i])
    return matches / min_len


# ==================== Main ====================

def main():
    print("=" * 80)
    print("Qwen3-8B Attention Matching 修复验证")
    print("=" * 80)

    # Load model from toshiba disk
    model_path = "/Volumes/toshiba/models/qwen3-8b-mlx"
    print(f"\n🔄 加载模型: {model_path}")

    try:
        model, tokenizer = load(model_path)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    print(f"✓ 模型加载成功")
    # 检测模型结构
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        num_layers = len(model.model.layers)
        print(f"  - 层数: {num_layers}")
    elif hasattr(model, 'layers'):
        num_layers = len(model.layers)
        print(f"  - 层数: {num_layers}")
    else:
        print(f"  - 层数: 未知")

    # Test compression ratios
    compression_ratios = [1.5, 2.0, 2.5]

    results = []

    for ratio in compression_ratios:
        print(f"\n{'=' * 80}")
        print(f"测试压缩比例: {ratio}x")
        print(f"{'=' * 80}")

        # Inject Attention Matching
        print(f"\n🔧 注入 Attention Matching (ratio={ratio}x)...")
        try:
            inject_attention_matching(
                model,
                compression_ratio=ratio,
                num_queries=100,
            )
            print(f"✓ Attention Matching 注入成功")
        except Exception as e:
            print(f"❌ 注入失败: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Run tests
        ratio_results = []

        for scenario_name, scenario_config in TEST_SCENARIOS.items():
            print(f"\n📝 测试场景: {scenario_name}")
            prompt = scenario_config["prompt"]
            max_tokens = scenario_config["max_tokens"]

            try:
                # Generate with compression
                print(f"  🔄 生成中...")
                compressed_output = generate(
                    model,
                    tokenizer,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    verbose=False,
                )

                # Tokenize outputs
                compressed_tokens = tokenizer.encode(compressed_output)

                print(f"  ✓ 生成完成")
                print(f"    输出长度: {len(compressed_tokens)} tokens")
                print(f"    输出预览: {compressed_output[:100]}...")

                # Check for common failure patterns
                is_garbage = (
                    compressed_output.count('S') > len(compressed_output) * 0.5 or
                    compressed_output.count(')') > len(compressed_output) * 0.3 or
                    compressed_output.count('!') > len(compressed_output) * 0.3
                )

                status = "❌ 乱码" if is_garbage else "✓ 正常"
                print(f"    状态: {status}")

                ratio_results.append({
                    "scenario": scenario_name,
                    "prompt": prompt,
                    "compressed_output": compressed_output,
                    "compressed_tokens": len(compressed_tokens),
                    "is_garbage": is_garbage,
                })

            except Exception as e:
                print(f"  ❌ 生成失败: {e}")
                import traceback
                traceback.print_exc()
                ratio_results.append({
                    "scenario": scenario_name,
                    "error": str(e),
                })

        results.append({
            "compression_ratio": ratio,
            "scenarios": ratio_results,
        })

        # Reload model for next ratio (清理 injection)
        if ratio != compression_ratios[-1]:
            print(f"\n🔄 重新加载模型...")
            model, tokenizer = load(model_path)

    # ==================== Summary ====================

    print(f"\n{'=' * 80}")
    print("测试总结")
    print(f"{'=' * 80}")

    for result in results:
        ratio = result["compression_ratio"]
        scenarios = result["scenarios"]

        successful = sum(1 for s in scenarios if not s.get("is_garbage", True) and "error" not in s)
        total = len(scenarios)

        print(f"\n压缩比例 {ratio}x: {successful}/{total} 成功")

        for scenario in scenarios:
            name = scenario["scenario"]
            if "error" in scenario:
                print(f"  - {name}: ❌ 错误 ({scenario['error'][:50]})")
            elif scenario.get("is_garbage"):
                print(f"  - {name}: ❌ 乱码")
            else:
                print(f"  - {name}: ✓ 正常")

    # Save results
    output_file = "test_qwen3_8b_fixed_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✓ 结果已保存到: {output_file}")


if __name__ == "__main__":
    main()
