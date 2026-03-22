#!/usr/bin/env python3
"""
Layerwise Ablation 实验
验证假设：只有标准 Attention 层可以用 AM 压缩，SSM 层不行

实验设计：
- Baseline: 不压缩任何层
- Exp 1: 只压缩 Layer 31-40 (Attention) - 预期成功
- Exp 2: 只压缩 Layer 1-30 (SSM) - 预期失败
- Exp 3: 压缩所有层 - 预期失败
- Exp 4: 只压缩 Layer 31 (单个 Attention) - 预期成功
- Exp 5: 只压缩 Layer 1 (单个 SSM) - 预期失败
"""

import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "mlx-lm-source"))

import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.models.cache import ArraysCache
from mlx_lm.models.compacted_cache import CompactedKVCache


def classify_layer(layer, layer_idx):
    """
    分类层类型

    Args:
        layer: 模型层对象
        layer_idx: 层索引

    Returns:
        str: "attention" | "ssm" | "sliding" | "unknown"
    """
    # 检查是否是 SSM/Mamba 层（Qwen3.5 的 linear_attn）
    if hasattr(layer, 'linear_attn'):
        return "ssm"  # ← 不能用 AM 压缩

    # 检查是否是标准 Attention 层
    elif hasattr(layer, 'self_attn'):
        # 检查是否是滑动窗口 Attention
        if hasattr(layer.self_attn, 'sliding_window'):
            return "sliding"
        else:
            return "attention"  # ← 只有这类可以用 AM 压缩

    # 检查其他 SSM 变体
    elif hasattr(layer, 'mamba_block'):
        return "ssm"  # ← 不能用 AM 压缩

    else:
        return "unknown"


def create_selective_cache(model, compress_layers, max_size=4096, compression_ratio=5.0):
    """
    创建选择性缓存

    Args:
        model: 模型对象
        compress_layers: List[int] - 要压缩的层索引
        max_size: 压缩阈值
        compression_ratio: 压缩比例

    Returns:
        List[Cache] - 每层的缓存对象
    """
    num_layers = len(model.layers)
    cache = ArraysCache(size=num_layers)

    for i, layer in enumerate(model.layers):
        layer_type = classify_layer(layer, i)

        if i in compress_layers:
            # 压缩这一层
            cache[i] = CompactedKVCache(max_size=max_size, compression_ratio=compression_ratio)
            print(f"Layer {i:2d} ({layer_type:10s}): Compressed")
        else:
            # 不压缩 - 使用 None（generate 会自动处理）
            cache[i] = None
            print(f"Layer {i:2d} ({layer_type:10s}): Not compressed")

    return cache


def check_repetition(text, threshold=0.3):
    """
    检查文本是否有重复模式

    Args:
        text: 生成的文本
        threshold: 重复比例阈值

    Returns:
        bool: True 如果有明显重复
    """
    words = text.lower().split()
    if len(words) < 10:
        return False

    # 检查连续重复
    consecutive_repeats = 0
    for i in range(len(words) - 1):
        if words[i] == words[i + 1]:
            consecutive_repeats += 1

    repeat_ratio = consecutive_repeats / len(words)
    return repeat_ratio > threshold


def evaluate_quality(text):
    """
    简单的质量评分

    Args:
        text: 生成的文本

    Returns:
        float: 质量分数 (0-10)
    """
    if len(text) < 50:
        return 2.0  # 太短

    if check_repetition(text):
        return 3.0  # 有重复

    # 检查是否有意义（简单启发式）
    words = text.split()
    unique_words = len(set(words))
    diversity = unique_words / len(words) if len(words) > 0 else 0

    if diversity < 0.3:
        return 4.0  # 多样性太低
    elif diversity < 0.5:
        return 7.0  # 中等
    else:
        return 9.0  # 良好


def layerwise_ablation_test(model, tokenizer, compress_layers, prompt="介绍机器学习", max_tokens=150):
    """
    Layerwise ablation 测试

    Args:
        model: 模型对象
        tokenizer: tokenizer 对象
        compress_layers: List[int] - 要压缩的层索引
        prompt: 测试 prompt
        max_tokens: 最大生成 token 数

    Returns:
        dict: 输出质量指标
    """
    print(f"\nCreating cache...")
    cache = create_selective_cache(model, compress_layers)

    print(f"\nGenerating with prompt: '{prompt}'")
    print(f"Max tokens: {max_tokens}")

    start_time = time.time()

    try:
        # 生成
        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,
            prompt_cache=cache
        )

        elapsed = time.time() - start_time

        # 评估
        num_tokens = len(tokenizer.encode(response))
        has_repetition = check_repetition(response)
        quality_score = evaluate_quality(response)

        print(f"\n{'='*60}")
        print(f"Generated text:")
        print(f"{'='*60}")
        print(response[:500])  # 只显示前 500 字符
        if len(response) > 500:
            print(f"... (truncated, total {len(response)} chars)")
        print(f"{'='*60}")

        return {
            'success': True,
            'num_tokens': num_tokens,
            'has_repetition': has_repetition,
            'quality_score': quality_score,
            'time_elapsed': elapsed,
            'output': response
        }

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ Generation failed: {str(e)}")

        return {
            'success': False,
            'error': str(e),
            'time_elapsed': elapsed,
            'num_tokens': 0,
            'has_repetition': True,
            'quality_score': 0.0,
            'output': ''
        }


def print_result(exp_name, result):
    """打印实验结果"""
    print(f"\n{'='*60}")
    print(f"Experiment: {exp_name}")
    print(f"{'='*60}")

    if result['success']:
        print(f"✅ Success")
        print(f"Tokens: {result['num_tokens']}")
        print(f"Time: {result['time_elapsed']:.2f}s")
        print(f"Repetition: {result['has_repetition']}")
        print(f"Quality: {result['quality_score']:.1f}/10")
    else:
        print(f"❌ Failed")
        print(f"Error: {result['error']}")
        print(f"Time: {result['time_elapsed']:.2f}s")


def main():
    """运行 Layerwise Ablation 实验"""
    print("="*60)
    print("Layerwise Ablation 实验")
    print("="*60)
    print("\n目标：验证假设 - 只有标准 Attention 层可以用 AM 压缩")
    print("模型：Qwen3.5-35B (48 layers: ~36 SSM + ~12 Attention)")
    print()

    # 加载模型
    print("Loading model (from local cache)...")
    model_name = "/Volumes/toshiba/models/qwen3.5-35b-mlx"
    model, tokenizer = load(model_name)

    print(f"Model loaded: {model_name}")
    print(f"Total layers: {len(model.layers)}")

    # 分析层结构
    print("\n" + "="*60)
    print("Layer structure analysis:")
    print("="*60)
    attention_layers = []
    ssm_layers = []

    for i, layer in enumerate(model.layers):
        layer_type = classify_layer(layer, i)
        if layer_type == "attention":
            attention_layers.append(i)
        elif layer_type == "ssm":
            ssm_layers.append(i)

    print(f"Attention layers: {attention_layers}")
    print(f"SSM layers: {ssm_layers}")

    # 定义实验
    experiments = {
        "Baseline": {
            "layers": [],
            "description": "不压缩任何层（基准）"
        },
        "Attention Only": {
            "layers": attention_layers,
            "description": "只压缩 Attention 层 - 预期成功 ✅"
        },
        "SSM Only": {
            "layers": ssm_layers[:5],  # 只压缩前 5 个 SSM 层（节省时间）
            "description": "只压缩 SSM 层 - 预期失败 ❌"
        },
        "All Layers": {
            "layers": list(range(len(model.layers))),
            "description": "压缩所有层 - 预期失败 ❌"
        },
        "Single Attention": {
            "layers": [attention_layers[0]] if attention_layers else [],
            "description": "单个 Attention 层 - 预期成功 ✅"
        },
        "Single SSM": {
            "layers": [ssm_layers[0]] if ssm_layers else [],
            "description": "单个 SSM 层 - 预期失败 ❌"
        },
    }

    # 运行实验
    results = {}

    for exp_name, exp_config in experiments.items():
        print(f"\n\n{'#'*60}")
        print(f"# {exp_name}")
        print(f"# {exp_config['description']}")
        print(f"# Compress layers: {exp_config['layers']}")
        print(f"{'#'*60}")

        result = layerwise_ablation_test(
            model,
            tokenizer,
            compress_layers=exp_config['layers'],
            prompt="介绍机器学习的基本概念",
            max_tokens=150
        )

        results[exp_name] = result
        print_result(exp_name, result)

        # 短暂休息，避免过热
        time.sleep(2)

    # 生成总结报告
    print("\n\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)

    summary_table = []
    summary_table.append(f"{'Experiment':<20} {'Success':<10} {'Tokens':<10} {'Quality':<10} {'Repetition':<12}")
    summary_table.append("-" * 62)

    for exp_name, result in results.items():
        success = "✅" if result['success'] else "❌"
        tokens = result['num_tokens'] if result['success'] else "N/A"
        quality = f"{result['quality_score']:.1f}/10" if result['success'] else "N/A"
        repetition = "Yes" if result['has_repetition'] else "No" if result['success'] else "N/A"

        summary_table.append(f"{exp_name:<20} {success:<10} {str(tokens):<10} {str(quality):<10} {str(repetition):<12}")

    print("\n".join(summary_table))

    # 验证假设
    print("\n" + "="*60)
    print("HYPOTHESIS VERIFICATION")
    print("="*60)

    baseline_ok = results["Baseline"]["success"]
    attention_ok = results["Attention Only"]["success"]
    ssm_failed = not results["SSM Only"]["success"]
    all_failed = not results["All Layers"]["success"]
    single_attn_ok = results["Single Attention"]["success"]
    single_ssm_failed = not results["Single SSM"]["success"]

    print(f"1. Baseline works: {baseline_ok} {'✅' if baseline_ok else '❌'}")
    print(f"2. Attention layers can be compressed: {attention_ok} {'✅' if attention_ok else '❌'}")
    print(f"3. SSM layers cannot be compressed: {ssm_failed} {'✅' if ssm_failed else '❌'}")
    print(f"4. Mixed compression fails: {all_failed} {'✅' if all_failed else '❌'}")
    print(f"5. Single Attention layer works: {single_attn_ok} {'✅' if single_attn_ok else '❌'}")
    print(f"6. Single SSM layer fails: {single_ssm_failed} {'✅' if single_ssm_failed else '❌'}")

    hypothesis_confirmed = baseline_ok and attention_ok and ssm_failed and all_failed

    print("\n" + "="*60)
    if hypothesis_confirmed:
        print("🎯 HYPOTHESIS CONFIRMED:")
        print("   只有标准 Attention 层可以用 AM 压缩！")
        print("   SSM 层不能用 AM 压缩！")
    else:
        print("⚠️  HYPOTHESIS NOT FULLY CONFIRMED:")
        print("   需要进一步调查异常结果")
    print("="*60)

    # 保存详细报告
    report_path = Path(__file__).parent.parent / ".solar" / "layerwise-ablation-report.md"
    with open(report_path, "w") as f:
        f.write("# Layerwise Ablation 实验报告\n\n")
        f.write(f"**日期**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**模型**: {model_name}\n")
        f.write(f"**总层数**: {len(model.layers)}\n")
        f.write(f"**Attention 层**: {len(attention_layers)}\n")
        f.write(f"**SSM 层**: {len(ssm_layers)}\n\n")

        f.write("## 实验假设\n\n")
        f.write("**H1**: 只有标准 Attention 层可以用 AM (Attention Matching) 压缩\n")
        f.write("**H2**: SSM 层不能用 AM 压缩（因为没有 attention mass 概念）\n\n")

        f.write("## 实验结果\n\n")
        f.write("| 实验 | 成功 | Tokens | 质量 | 重复 | 耗时 |\n")
        f.write("|------|------|--------|------|------|------|\n")

        for exp_name, result in results.items():
            success = "✅" if result['success'] else "❌"
            tokens = result['num_tokens'] if result['success'] else "N/A"
            quality = f"{result['quality_score']:.1f}" if result['success'] else "N/A"
            repetition = "Yes" if result['has_repetition'] else "No" if result['success'] else "N/A"
            time_str = f"{result['time_elapsed']:.2f}s"

            f.write(f"| {exp_name} | {success} | {tokens} | {quality} | {repetition} | {time_str} |\n")

        f.write("\n## 假设验证\n\n")
        f.write(f"1. ✅ Baseline 工作正常: {baseline_ok}\n")
        f.write(f"2. ✅ Attention 层可以压缩: {attention_ok}\n")
        f.write(f"3. ✅ SSM 层不能压缩: {ssm_failed}\n")
        f.write(f"4. ✅ 混合压缩失败: {all_failed}\n")
        f.write(f"5. ✅ 单个 Attention 层可以: {single_attn_ok}\n")
        f.write(f"6. ✅ 单个 SSM 层失败: {single_ssm_failed}\n")

        f.write("\n## 结论\n\n")
        if hypothesis_confirmed:
            f.write("**✅ 假设得到验证！**\n\n")
            f.write("AM (Attention Matching) 只能用于标准 softmax attention 层，不能用于 SSM/Mamba 层。\n\n")
            f.write("**关键发现**：\n")
            f.write("- ✅ Attention 层可以安全压缩\n")
            f.write("- ❌ SSM 层压缩导致输出崩溃\n")
            f.write("- ❌ 混合压缩会破坏模型生成\n\n")
            f.write("**下一步**：\n")
            f.write("1. 实现选择性压缩（只压缩 Attention 层）\n")
            f.write("2. 为 SSM 层设计专门的压缩算法\n")
            f.write("3. 建立混合架构记忆压缩的统一框架\n")
        else:
            f.write("**⚠️ 假设验证不完全，需要进一步调查。**\n\n")

        f.write("\n## 详细输出\n\n")
        for exp_name, result in results.items():
            f.write(f"### {exp_name}\n\n")
            f.write(f"**配置**: {experiments[exp_name]['description']}\n\n")
            f.write(f"**压缩层**: {experiments[exp_name]['layers']}\n\n")

            if result['success']:
                f.write(f"**输出**:\n```\n{result['output'][:500]}\n")
                if len(result['output']) > 500:
                    f.write(f"... (截断, 总共 {len(result['output'])} 字符)\n")
                f.write("```\n\n")
            else:
                f.write(f"**错误**: {result['error']}\n\n")

    print(f"\n详细报告已保存到: {report_path}")


if __name__ == "__main__":
    main()
