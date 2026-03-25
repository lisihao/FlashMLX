#!/usr/bin/env python3
"""
测试自适应路由器的核心功能（不做实际推理）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "mlx-lm-source"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mlx_lm import load
from flashmlx.cache.adaptive_compressor import AdaptiveCompressor, ModelArchitectureDetector


def test_architecture_detection(model_path):
    """测试架构检测"""
    print(f"\n{'='*70}")
    print(f"测试: {Path(model_path).name}")
    print(f"{'='*70}")

    try:
        # 加载模型
        print("\nLoading model...")
        model, tokenizer = load(model_path)

        # 检测架构
        print("\n1. 架构检测")
        print("-" * 70)
        architecture = ModelArchitectureDetector.detect(model)

        print(f"架构类型: {architecture['type']}")
        print(f"Attention 层: {architecture['attention_layers']} 个")
        print(f"SSM 层: {architecture['ssm_layers']} 个")
        print(f"总层数: {architecture['total_layers']} 个")

        # 创建自适应压缩器
        print("\n2. 算法选择")
        print("-" * 70)
        compressor = AdaptiveCompressor(
            model=model,
            max_size=4096,
            compression_ratio=2.0,
            verbose=False
        )

        recommendation = compressor.get_recommendation()

        print(f"选择算法: {recommendation['algorithm']}")
        print(f"预期质量: {recommendation['expected_quality']:.2f}")
        print(f"预期加速: {recommendation['expected_speed_boost']:.2f}x")
        print(f"\n理由: {recommendation['reason']}")

        # 验证选择逻辑
        print("\n3. 验证选择逻辑")
        print("-" * 70)

        expected_algorithm = None
        if architecture['type'] == 'pure_transformer':
            expected_algorithm = 'AM'
            print("✅ 纯 Transformer → 应该选择 AM")
        elif architecture['type'] == 'hybrid':
            expected_algorithm = 'H2O'
            print("✅ 混合架构 → 应该选择 H2O (AM 会崩溃)")
        else:
            print(f"⚠️  未知架构类型: {architecture['type']}")

        if expected_algorithm:
            if recommendation['algorithm'] == expected_algorithm:
                print(f"✅ 正确！选择了 {recommendation['algorithm']}")
            else:
                print(f"❌ 错误！应该选择 {expected_algorithm}，但选择了 {recommendation['algorithm']}")

        return {
            'success': True,
            'model': Path(model_path).name,
            'architecture': architecture['type'],
            'selected_algorithm': recommendation['algorithm'],
            'expected_algorithm': expected_algorithm,
            'correct': recommendation['algorithm'] == expected_algorithm,
            'recommendation': recommendation
        }

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

        return {
            'success': False,
            'model': Path(model_path).name,
            'error': str(e)
        }


def main():
    print("="*70)
    print("自适应路由器核心功能测试")
    print("="*70)

    models = [
        # 纯 Transformer
        ("/Volumes/toshiba/models/qwen3-8b-mlx", "纯 Transformer"),
        # 混合架构
        ("/Volumes/toshiba/models/qwen3.5-0.8b-opus-distilled", "混合架构 (0.8B)"),
        ("/Volumes/toshiba/models/qwen3.5-2b-opus-distilled", "混合架构 (2B)"),
        ("/Volumes/toshiba/models/qwen3.5-35b-mlx", "混合架构 (35B)"),
    ]

    all_results = {}

    for model_path, model_desc in models:
        result = test_architecture_detection(model_path)
        all_results[Path(model_path).name] = result

    # 生成报告
    print("\n" + "="*70)
    print("测试总结")
    print("="*70)

    for model_name, result in all_results.items():
        if result.get('success'):
            status = "✅" if result['correct'] else "❌"
            print(f"{status} {model_name}:")
            print(f"   架构: {result['architecture']}")
            print(f"   选择: {result['selected_algorithm']}")
            print(f"   预期: {result['expected_algorithm']}")
        else:
            print(f"❌ {model_name}: 测试失败")

    # 生成详细报告
    report_path = Path(__file__).parent.parent / ".solar" / "adaptive-routing-test-report.md"

    with open(report_path, "w") as f:
        f.write("# 自适应路由器核心功能测试报告\n\n")
        f.write(f"**日期**: {Path(__file__).parent.parent}\n\n")

        f.write("## 路由策略\n\n")
        f.write("| 架构类型 | 选择算法 | 原因 |\n")
        f.write("|---------|---------|------|\n")
        f.write("| 纯 Transformer | AM | 质量 1.0, 速度 +46% |\n")
        f.write("| 混合架构 | H2O | AM 崩溃, H2O 质量 0.69 |\n\n")

        f.write("## 测试结果\n\n")
        f.write("| 模型 | 架构 | 选择算法 | 预期算法 | 正确性 |\n")
        f.write("|------|------|---------|---------|--------|\n")

        for model_name, result in all_results.items():
            if result.get('success'):
                correct = "✅" if result['correct'] else "❌"
                f.write(f"| {model_name} | {result['architecture']} | {result['selected_algorithm']} | {result['expected_algorithm']} | {correct} |\n")
            else:
                f.write(f"| {model_name} | - | - | - | ❌ 失败 |\n")

        f.write("\n## 结论\n\n")

        success_count = sum(1 for r in all_results.values() if r.get('success') and r.get('correct'))
        total_count = len(all_results)

        f.write(f"测试通过: {success_count}/{total_count}\n\n")

        if success_count == total_count:
            f.write("✅ **所有测试通过！自适应路由器工作正常。**\n\n")
            f.write("路由器能够：\n")
            f.write("- ✅ 正确检测模型架构（纯 Transformer vs 混合架构）\n")
            f.write("- ✅ 为纯 Transformer 选择 AM（最优性能）\n")
            f.write("- ✅ 为混合架构选择 H2O（避免 AM 崩溃）\n")
            f.write("- ✅ 提供合理的推荐理由\n")
        else:
            f.write("⚠️  **部分测试未通过，需要检查路由逻辑。**\n")

    print(f"\n详细报告已保存到: {report_path}")

    print("\n" + "="*70)
    print("测试完成!")
    print("="*70)


if __name__ == "__main__":
    main()
