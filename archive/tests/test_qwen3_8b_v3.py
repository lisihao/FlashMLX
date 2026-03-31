#!/usr/bin/env python3
"""
Qwen3-8B Attention Matching V3 测试
正确应用 Beta 到 attention scores
"""

import json
from mlx_lm import load, generate
from flashmlx.cache.simple_injection_v3 import inject_attention_matching_v3


TEST_SCENARIOS = {
    "简单数学": "问题：3 + 5 = ?\n回答：",
    "中文": "请用一句话描述春天。\n回答：",
    "推理": "小明有5个苹果，吃了2个，还剩几个？\n回答：",
}


def main():
    print("=" * 80)
    print("Qwen3-8B Attention Matching V3 - 正确应用 Beta")
    print("=" * 80)

    model_path = "/Volumes/toshiba/models/qwen3-8b-mlx"
    print(f"\n🔄 加载模型...")
    model, tokenizer = load(model_path)
    print(f"✓ 模型加载成功")

    compression_ratio = 2.0
    print(f"\n🔧 注入 Attention Matching V3 (ratio={compression_ratio}x)...")

    compressor = inject_attention_matching_v3(
        model,
        compression_ratio=compression_ratio,
        num_queries=100,
    )

    print()

    # 测试
    results = {}
    for name, prompt in TEST_SCENARIOS.items():
        print(f"\n📝 {name}")
        try:
            output = generate(model, tokenizer, prompt=prompt, max_tokens=50, verbose=False)
            print(f"  输出: {output[:100]}...")

            # 检查是否乱码
            is_garbage = (
                output.count('S') > len(output) * 0.5 or
                output.count(')') > len(output) * 0.3 or
                output.count('!') > len(output) * 0.3 or
                len(set(output[:50])) < 5
            )

            status = "❌ 乱码" if is_garbage else "✓ 正常"
            print(f"  状态: {status}")
            results[name] = {"output": output, "is_garbage": is_garbage}

        except Exception as e:
            print(f"  ❌ 错误: {e}")
            import traceback
            traceback.print_exc()
            results[name] = {"error": str(e)}

    # 总结
    print(f"\n{'=' * 80}")
    successful = sum(1 for r in results.values() if not r.get("is_garbage", True) and "error" not in r)
    print(f"结果: {successful}/{len(results)} 成功")

    # 压缩统计
    stats = compressor.get_compression_stats()
    print(f"\n压缩统计:")
    print(f"  - 总压缩次数: {stats['total_compressions']}")
    print(f"  - 压缩前 tokens: {stats['total_keys_before']}")
    print(f"  - 压缩后 tokens: {stats['total_keys_after']}")
    print(f"  - 实际压缩比: {stats['avg_compression_ratio']:.2f}x")

    with open("test_v3_results.json", "w") as f:
        json.dump({"results": results, "stats": stats}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
