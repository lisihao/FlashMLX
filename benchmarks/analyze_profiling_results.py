#!/usr/bin/env python3
"""
分析 Critical Channels Profiling 结果

分析内容：
1. 通道功能分布模式
2. 跨层通道重现规律
3. 层级重要性分布
4. 压缩策略优化建议
"""

import json
import os
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np


def load_all_calibrations(calibration_dir: str = ".solar/calibration"):
    """加载所有 calibration 文件"""
    calibrations = {}

    for file in sorted(os.listdir(calibration_dir)):
        if file.endswith('.json'):
            layer_idx = int(file.split('_')[1])
            with open(os.path.join(calibration_dir, file), 'r') as f:
                calibrations[layer_idx] = json.load(f)

    return calibrations


def analyze_channel_frequency(calibrations):
    """分析通道在多层中的出现频率"""
    channel_frequency = Counter()
    channel_layers = defaultdict(list)

    for layer_idx, cal in calibrations.items():
        for ch in cal['critical_channels']:
            channel_frequency[ch] += 1
            channel_layers[ch].append(layer_idx)

    # 按频率排序
    sorted_channels = sorted(channel_frequency.items(), key=lambda x: x[1], reverse=True)

    return sorted_channels, dict(channel_layers)


def analyze_importance_distribution(calibrations):
    """分析重要性分数分布"""
    layer_max_scores = {}
    layer_mean_scores = {}
    layer_critical_scores = {}

    for layer_idx, cal in calibrations.items():
        scores = cal['profiling_metadata']['importance_scores']
        critical_channels = cal['critical_channels']

        layer_max_scores[layer_idx] = max(scores)
        layer_mean_scores[layer_idx] = np.mean(scores)
        layer_critical_scores[layer_idx] = [scores[ch] for ch in critical_channels]

    return layer_max_scores, layer_mean_scores, layer_critical_scores


def analyze_layer_position_effect(calibrations):
    """分析层位置对重要性的影响"""
    early_layers = []  # 0-12
    middle_layers = []  # 13-25
    late_layers = []    # 26-38

    for layer_idx, cal in calibrations.items():
        scores = cal['profiling_metadata']['importance_scores']
        max_score = max(scores)

        if layer_idx <= 12:
            early_layers.append(max_score)
        elif layer_idx <= 25:
            middle_layers.append(max_score)
        else:
            late_layers.append(max_score)

    return {
        'early': {'mean': np.mean(early_layers), 'max': max(early_layers), 'layers': len(early_layers)},
        'middle': {'mean': np.mean(middle_layers), 'max': max(middle_layers), 'layers': len(middle_layers)},
        'late': {'mean': np.mean(late_layers), 'max': max(late_layers), 'layers': len(late_layers)}
    }


def find_top_channels_per_metric(calibrations):
    """找出每个指标下的 top channels"""
    # 这需要读取 profiling 时记录的各项指标
    # 目前只有 overall_score，未来可以扩展

    top_channels_global = {}

    for layer_idx, cal in calibrations.items():
        scores = cal['profiling_metadata']['importance_scores']
        critical = cal['critical_channels']

        for ch in critical:
            if ch not in top_channels_global:
                top_channels_global[ch] = []
            top_channels_global[ch].append({
                'layer': layer_idx,
                'score': scores[ch]
            })

    return top_channels_global


def generate_compression_ratio_analysis(calibrations):
    """分析当前压缩比和建议"""
    total_channels = 0
    total_critical = 0

    for cal in calibrations.values():
        num_channels = cal['profiling_metadata']['num_channels']
        num_critical = len(cal['critical_channels'])
        total_channels += num_channels
        total_critical += num_critical

    current_ratio = total_critical / total_channels

    return {
        'current_ratio': current_ratio,
        'current_percentage': current_ratio * 100,
        'total_channels': total_channels,
        'total_critical': total_critical,
        'channels_per_layer': 128,
        'critical_per_layer': 6
    }


def main():
    print("=" * 60)
    print("Critical Channels Profiling 结果分析")
    print("=" * 60)
    print()

    # 1. 加载所有 calibration 文件
    print("1. 加载 calibration 文件...")
    calibrations = load_all_calibrations()
    print(f"   ✅ 加载了 {len(calibrations)} 层的数据")
    print()

    # 2. 分析通道频率
    print("2. 分析跨层通道重现...")
    sorted_channels, channel_layers = analyze_channel_frequency(calibrations)

    print(f"\n   📊 高频出现通道 (出现在 2+ 层):")
    for ch, freq in sorted_channels[:20]:
        if freq >= 2:
            layers = channel_layers[ch]
            print(f"      Channel {ch:3d}: {freq} 层 → {layers}")
    print()

    # 3. 分析重要性分布
    print("3. 分析重要性分数分布...")
    layer_max_scores, layer_mean_scores, layer_critical_scores = analyze_importance_distribution(calibrations)

    print(f"\n   📈 Top 10 最高分数 (跨所有层):")
    all_max_scores = [(layer, score) for layer, score in layer_max_scores.items()]
    all_max_scores.sort(key=lambda x: x[1], reverse=True)

    for layer, score in all_max_scores[:10]:
        critical = calibrations[layer]['critical_channels']
        scores = calibrations[layer]['profiling_metadata']['importance_scores']
        top_ch = max(range(len(scores)), key=lambda i: scores[i])
        print(f"      Layer {layer:2d}, Channel {top_ch:3d}: {score:.4f}")
    print()

    # 4. 分析层位置效应
    print("4. 分析层位置对重要性的影响...")
    position_effect = analyze_layer_position_effect(calibrations)

    print(f"\n   🎯 早期层 (0-12):")
    print(f"      平均最高分: {position_effect['early']['mean']:.4f}")
    print(f"      最高分: {position_effect['early']['max']:.4f}")
    print(f"      层数: {position_effect['early']['layers']}")

    print(f"\n   🎯 中期层 (13-25):")
    print(f"      平均最高分: {position_effect['middle']['mean']:.4f}")
    print(f"      最高分: {position_effect['middle']['max']:.4f}")
    print(f"      层数: {position_effect['middle']['layers']}")

    print(f"\n   🎯 后期层 (26-38):")
    print(f"      平均最高分: {position_effect['late']['mean']:.4f}")
    print(f"      最高分: {position_effect['late']['max']:.4f}")
    print(f"      层数: {position_effect['late']['layers']}")
    print()

    # 5. 压缩比分析
    print("5. 压缩比分析...")
    compression = generate_compression_ratio_analysis(calibrations)

    print(f"\n   📦 当前压缩设置:")
    print(f"      每层通道数: {compression['channels_per_layer']}")
    print(f"      每层保留: {compression['critical_per_layer']}")
    print(f"      压缩比: {compression['current_percentage']:.2f}%")
    print(f"      总通道数: {compression['total_channels']}")
    print(f"      总保留: {compression['total_critical']}")
    print()

    # 6. 生成报告
    print("6. 生成分析报告...")

    report = {
        'summary': {
            'total_layers': len(calibrations),
            'channels_per_layer': 128,
            'critical_per_layer': 6,
            'compression_ratio': compression['current_percentage']
        },
        'high_frequency_channels': {
            ch: {'frequency': freq, 'layers': channel_layers[ch]}
            for ch, freq in sorted_channels if freq >= 2
        },
        'layer_max_scores': layer_max_scores,
        'position_effect': position_effect,
        'compression_analysis': compression
    }

    # 保存报告
    output_file = ".solar/profiling-analysis-report.json"
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"   ✅ 报告已保存: {output_file}")
    print()

    # 7. 关键发现总结
    print("=" * 60)
    print("关键发现总结")
    print("=" * 60)
    print()

    print("🔍 发现 1: 跨层通道重现")
    high_freq = [(ch, freq) for ch, freq in sorted_channels if freq >= 3]
    if high_freq:
        print(f"   有 {len(high_freq)} 个通道出现在 3+ 层")
        print(f"   最高频: Channel {high_freq[0][0]} ({high_freq[0][1]} 层)")
        print("   → 推测: 这些通道控制全局特征（语言、格式等）")
    else:
        print("   没有通道出现在 3+ 层")
        print("   → 推测: 通道功能高度层级专用化")
    print()

    print("📈 发现 2: 层位置效应")
    if position_effect['late']['mean'] > position_effect['early']['mean']:
        diff = position_effect['late']['mean'] - position_effect['early']['mean']
        pct = (diff / position_effect['early']['mean']) * 100
        print(f"   后期层平均分比早期层高 {pct:.1f}%")
        print("   → 推测: 后期层控制最终输出，更关键")
    else:
        print("   早期层和后期层重要性相近")
        print("   → 推测: 所有层同等重要")
    print()

    print("📊 发现 3: 压缩比评估")
    print(f"   当前: {compression['current_percentage']:.2f}% 保留")
    print(f"   即: 每层 128 通道中保留 {compression['critical_per_layer']} 个")
    if compression['current_percentage'] < 10:
        print("   → 评估: 压缩比激进（<10%），需验证质量")
    else:
        print("   → 评估: 压缩比保守（≥10%），质量风险低")
    print()

    print("=" * 60)
    print("建议下一步")
    print("=" * 60)
    print()
    print("1. ✅ 执行 Task #60: 测试 critical channels 质量")
    print("   → 验证 5% 压缩比是否保持输出质量")
    print()
    print("2. 🔬 深入分析高频通道功能")
    print("   → 对 Channel 77, 70, 86 等做专项测试")
    print("   → 确定它们控制什么功能（语言/格式/<think>）")
    print()
    print("3. 📐 优化压缩比")
    print("   → 如果质量测试失败，增加到 10% 或 15%")
    print("   → 如果质量测试通过，可尝试降低到 3%")
    print()


if __name__ == "__main__":
    main()
