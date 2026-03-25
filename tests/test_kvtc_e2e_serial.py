#!/usr/bin/env python3
"""KVTC 端到端测试 - 串行执行，环境隔离

测试方案：
1. DCT-Fixed (65x, 0.89) - Baseline
2. 方案 A (56x, 0.83) - 高压缩率
3. Balanced+ (34x, 0.77) - 平衡
4. 方案 A++ (12x, 0.35) - 高精度
5. No Compression (1x, 0.00) - 无压缩（参考）

测试指标：
- 推理速度 (tok/s)
- 首 token 延迟 (ms)
- 编解码时间 (ms)
- 内存占用 (MB)
- 生成质量（定性）

串行隔离机制：
- 测完一个立即写盘（JSON）
- 检测并杀死之前的模型进程
- 避免环境波动
"""

import sys
sys.path.insert(0, '.')

import os
import json
import time
import psutil
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step, stream_generate
from mlx_lm.models.cache import KVTCPromptCache, make_prompt_cache
from mlx_lm.models.kvtc_codec import KVTCCodecConfig
from mlx_lm.models.kvtc_dct_codec import fit_dct_shared_calibration
from mlx_lm.models.kvtc_pca_codec import fit_pca_calibration
from mlx_lm.models.kvtc_magnitude_pruning import (
    KVTCMagnitudeTieredConfig,
    fit_magnitude_tiered_calibration,
)


# ============================================================================
# 环境隔离工具
# ============================================================================

def kill_model_processes():
    """杀死所有可能占用 GPU/内存的模型进程"""
    print("🔍 检测并清理模型进程...")

    killed = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info['cmdline'] or [])

            # 检测可能的模型进程
            if any(keyword in cmdline.lower() for keyword in [
                'llama-server', 'llama-cli', 'llama.cpp',
                'vllm', 'tgi', 'text-generation',
                'mlx_lm.server', 'mlx_lm.generate'
            ]):
                # 跳过当前脚本
                if 'test_kvtc_e2e_serial' in cmdline:
                    continue

                print(f"   终止进程: {proc.info['pid']} - {proc.info['name']}")
                proc.kill()
                killed.append(proc.info['pid'])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    if killed:
        print(f"   ✅ 已终止 {len(killed)} 个进程")
        time.sleep(2)  # 等待进程清理
    else:
        print("   ✅ 无需清理")
    print()


def get_memory_usage():
    """获取当前内存使用（MB）"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def clear_mlx_cache():
    """清理 MLX 缓存"""
    mx.metal.clear_cache()
    import gc
    gc.collect()


# ============================================================================
# 测试配置
# ============================================================================

class KVTCTestConfig:
    """KVTC 测试配置"""

    def __init__(self, name, description, compression_ratio, accuracy,
                 calibration_fn, calibration_config):
        self.name = name
        self.description = description
        self.compression_ratio = compression_ratio  # 预期压缩率
        self.accuracy = accuracy  # 预期精度（相对误差）
        self.calibration_fn = calibration_fn  # 校准函数
        self.calibration_config = calibration_config  # 配置参数


def get_test_configs():
    """获取所有测试配置"""

    configs = []

    # 1. DCT-Fixed (Baseline)
    configs.append(KVTCTestConfig(
        name="DCT-Fixed",
        description="频域假设，分层量化 (Baseline)",
        compression_ratio=65,
        accuracy=0.89,
        calibration_fn=lambda kv: fit_dct_shared_calibration(
            [kv['keys']], [kv['values']],
            KVTCCodecConfig(rank=8, bits=4, group_size=16),
            use_fixed_allocation=True
        ),
        calibration_config={'rank': 8, 'bits': 4}
    ))

    # 2. 方案 A (Balanced 25%)
    configs.append(KVTCTestConfig(
        name="Plan-A-Balanced",
        description="Magnitude + Tiered (保留 25%, 4/3/2/0 bits)",
        compression_ratio=56,
        accuracy=0.83,
        calibration_fn=lambda kv: fit_magnitude_tiered_calibration(
            [kv['keys']], [kv['values']],
            KVTCCodecConfig(rank=8, bits=4, group_size=16),
            KVTCMagnitudeTieredConfig(
                tier_ratios=(0.10, 0.20, 0.25),
                tier_bits=(4, 3, 2, 0),
                pruning_method="l2"
            )
        ),
        calibration_config={'keep': 0.25, 'bits': '4/3/2/0'}
    ))

    # 3. Balanced+ (30%)
    configs.append(KVTCTestConfig(
        name="Balanced-Plus",
        description="Magnitude + Tiered (保留 30%, 6/5/4/3/2/0 bits)",
        compression_ratio=34,
        accuracy=0.77,
        calibration_fn=lambda kv: fit_magnitude_tiered_calibration(
            [kv['keys']], [kv['values']],
            KVTCCodecConfig(rank=8, bits=6, group_size=16),
            KVTCMagnitudeTieredConfig(
                tier_ratios=(0.05, 0.10, 0.20, 0.25, 0.30),
                tier_bits=(6, 5, 4, 3, 2, 0),
                pruning_method="l2"
            )
        ),
        calibration_config={'keep': 0.30, 'bits': '6/5/4/3/2/0'}
    ))

    # 4. 方案 A++ (Super Precision 80%)
    configs.append(KVTCTestConfig(
        name="Plan-A-Plus-Plus",
        description="Magnitude + Tiered (保留 80%, 5/4/3/0 bits)",
        compression_ratio=12,
        accuracy=0.35,
        calibration_fn=lambda kv: fit_magnitude_tiered_calibration(
            [kv['keys']], [kv['values']],
            KVTCCodecConfig(rank=8, bits=5, group_size=16),
            KVTCMagnitudeTieredConfig(
                tier_ratios=(0.30, 0.60, 0.80),
                tier_bits=(5, 4, 3, 0),
                pruning_method="l2"
            )
        ),
        calibration_config={'keep': 0.80, 'bits': '5/4/3/0'}
    ))

    # 5. PCA-8 (Data-driven, rank=8)
    configs.append(KVTCTestConfig(
        name="PCA-8",
        description="PCA 数据驱动 (rank=8, 4-bit)",
        compression_ratio=60,  # Expected
        accuracy=0.20,  # Expected (from unit test)
        calibration_fn=lambda kv: fit_pca_calibration(
            [kv['keys']], [kv['values']],
            KVTCCodecConfig(rank=8, bits=4, group_size=16)
        ),
        calibration_config={'rank': 8, 'bits': 4}
    ))

    # 6. PCA-16 (Data-driven, rank=16)
    configs.append(KVTCTestConfig(
        name="PCA-16",
        description="PCA 数据驱动 (rank=16, 4-bit)",
        compression_ratio=40,  # Expected
        accuracy=0.20,  # Expected
        calibration_fn=lambda kv: fit_pca_calibration(
            [kv['keys']], [kv['values']],
            KVTCCodecConfig(rank=16, bits=4, group_size=16)
        ),
        calibration_config={'rank': 16, 'bits': 4}
    ))

    # 7. No Compression (参考)
    configs.append(KVTCTestConfig(
        name="No-Compression",
        description="无压缩（参考速度上限）",
        compression_ratio=1,
        accuracy=0.0,
        calibration_fn=None,
        calibration_config={}
    ))

    return configs


# ============================================================================
# KVTC 应用辅助函数
# ============================================================================

def _is_kvtc_supported(layer_cache):
    """Check if a layer cache supports KVTC compression."""
    state = layer_cache.state
    if not isinstance(state, tuple) or len(state) != 2:
        return False
    keys, values = state
    return getattr(keys, "ndim", 0) == 4 and getattr(values, "ndim", 0) == 4


def _prefill_prompt(model, tokenizer, prompt_text, max_kv_size=None):
    """Prefill the model with a prompt and return the KV cache."""
    # Tokenize prompt
    if tokenizer.has_chat_template:
        messages = [{"role": "user", "content": prompt_text}]
        prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            continue_final_message=True,
        )
    else:
        prompt = tokenizer.encode(prompt_text)

    # Create cache and prefill
    cache = make_prompt_cache(model, max_kv_size)
    y = mx.array(prompt)
    for _ in generate_step(y, model, max_tokens=0, prompt_cache=cache):
        pass

    return cache, prompt


def _fit_calibrations(cache, calibration_fn):
    """Fit calibrations using the provided calibration function.

    Args:
        cache: List of layer caches
        calibration_fn: Function that takes (keys_list, values_list) and returns calibration

    Returns:
        Dictionary mapping (key_dim, value_dim) -> calibration
    """
    if calibration_fn is None:
        return None

    groups = {}
    for layer_cache in cache:
        if layer_cache.empty() or not _is_kvtc_supported(layer_cache):
            continue
        keys, values = layer_cache.state
        group_key = (keys.shape[-1], values.shape[-1])
        groups.setdefault(group_key, {"keys": [], "values": []})
        groups[group_key]["keys"].append(keys.reshape(-1, keys.shape[-1]))
        groups[group_key]["values"].append(values.reshape(-1, values.shape[-1]))

    calibrations = {}
    for group_key, group in groups.items():
        calibrations[group_key] = calibration_fn({
            'keys': group["keys"][0],  # Use first layer as representative
            'values': group["values"][0]
        })

    return calibrations


def _layer_calibration(layer_cache, calibrations):
    """Get calibration for a specific layer."""
    if calibrations is None:
        return None
    keys, values = layer_cache.state
    return calibrations[(keys.shape[-1], values.shape[-1])]


def _apply_kvtc_compression(cache, calibrations):
    """Apply KVTC compression to cache using calibrations.

    Args:
        cache: List of layer caches
        calibrations: Dictionary of calibrations or None (for no compression)

    Returns:
        List of (possibly compressed) caches
    """
    if calibrations is None:
        return cache

    compressed_cache = []
    for c in cache:
        if _is_kvtc_supported(c):
            calibration = _layer_calibration(c, calibrations)
            compressed_cache.append(
                KVTCPromptCache.from_cache(c, calibration=calibration)
            )
        else:
            compressed_cache.append(c)

    return compressed_cache


# ============================================================================
# 测试执行
# ============================================================================

def run_single_test(config, model_path, output_dir):
    """运行单个配置的测试

    Args:
        config: KVTCTestConfig
        model_path: 模型路径
        output_dir: 结果输出目录

    Returns:
        测试结果字典
    """
    print("=" * 70)
    print(f"测试配置: {config.name}")
    print(f"说明: {config.description}")
    print(f"预期: 压缩率 {config.compression_ratio}x, 精度误差 {config.accuracy}")
    print("=" * 70)
    print()

    # 清理环境
    kill_model_processes()
    clear_mlx_cache()

    mem_before = get_memory_usage()

    # 加载模型
    print("📦 加载模型...")
    t0 = time.time()
    model, tokenizer = load(model_path)
    load_time = time.time() - t0
    print(f"   ✅ 加载完成 ({load_time:.2f}s)")
    print()

    mem_after_load = get_memory_usage()

    # 应用 KVTC 配置
    print("🔧 应用 KVTC 配置...")
    calibrations = None
    compression_applied = False

    if config.calibration_fn is not None:
        # 使用第一个 prompt 进行校准
        calibration_prompt = "The future of artificial intelligence is"
        print(f"   校准 prompt: {calibration_prompt[:60]}...")

        # Prefill to get cache
        t0 = time.time()
        cache, _ = _prefill_prompt(model, tokenizer, calibration_prompt)
        prefill_time = time.time() - t0

        # Fit calibrations
        t0 = time.time()
        calibrations = _fit_calibrations(cache, config.calibration_fn)
        calibration_time = time.time() - t0

        print(f"   ✅ Prefill: {prefill_time:.2f}s, Calibration: {calibration_time:.2f}s")
        compression_applied = True
    else:
        print(f"   ℹ️  无压缩模式 (No Compression)")

    # 测试生成
    print("🚀 推理测试...")
    test_prompts = [
        "The future of artificial intelligence is",
        "In the year 2050, technology will",
        "Explain quantum computing in simple terms:",
    ]

    results = {
        'config': {
            'name': config.name,
            'description': config.description,
            'compression_ratio': config.compression_ratio,
            'accuracy': config.accuracy,
            'calibration_config': config.calibration_config,
        },
        'memory': {
            'before_load_mb': mem_before,
            'after_load_mb': mem_after_load,
            'model_size_mb': mem_after_load - mem_before,
        },
        'load_time_s': load_time,
        'compression_applied': compression_applied,
        'generation_tests': [],
    }

    for i, prompt in enumerate(test_prompts):
        print(f"\n   测试 {i+1}/{len(test_prompts)}: {prompt[:50]}...")

        # Prefill with compression
        t0 = time.time()
        cache, prompt_tokens = _prefill_prompt(model, tokenizer, prompt)

        if compression_applied:
            cache = _apply_kvtc_compression(cache, calibrations)

        prefill_time = time.time() - t0

        # 生成测试（使用 stream_generate 以使用 prompt_cache）
        t0 = time.time()
        first_token_time = None
        generated_tokens = 0
        response_text = ""

        for gen_response in stream_generate(
            model,
            tokenizer,
            prompt_tokens,
            max_tokens=100,
            prompt_cache=cache,
        ):
            if first_token_time is None:
                first_token_time = time.time() - t0
            generated_tokens += 1
            response_text += gen_response.text

        gen_time = time.time() - t0
        tok_per_sec = generated_tokens / gen_time if gen_time > 0 else 0

        print(f"   ⏱️  Prefill: {prefill_time*1000:.1f}ms, First token: {first_token_time*1000:.1f}ms")
        print(f"   ⏱️  生成时间: {gen_time:.2f}s, {tok_per_sec:.2f} tok/s")
        print(f"   📝 生成长度: {generated_tokens} tokens")

        results['generation_tests'].append({
            'prompt': prompt,
            'response': response_text,
            'tokens': generated_tokens,
            'time_s': gen_time,
            'tok_per_s': tok_per_sec,
            'prefill_time_s': prefill_time,
            'first_token_time_s': first_token_time,
        })

    # 计算平均速度
    avg_tok_per_s = np.mean([t['tok_per_s'] for t in results['generation_tests']])
    results['avg_tok_per_s'] = avg_tok_per_s

    print()
    print("📊 **测试结果**:")
    print(f"   平均速度: {avg_tok_per_s:.2f} tok/s")
    print(f"   内存占用: {results['memory']['model_size_mb']:.2f} MB")
    print()

    # 立即写盘
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = output_dir / f"{config.name}_{timestamp}.json"

    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"💾 结果已保存: {result_file}")
    print()

    # 清理模型
    del model, tokenizer
    clear_mlx_cache()

    return results


def run_serial_tests(model_path, output_dir):
    """串行运行所有测试"""

    print()
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 18 + "KVTC 端到端测试 - 串行执行" + " " * 23 + "║")
    print("╚" + "=" * 68 + "╝")
    print()

    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取测试配置
    configs = get_test_configs()

    print(f"📋 测试计划: {len(configs)} 个配置")
    for i, cfg in enumerate(configs, 1):
        print(f"   {i}. {cfg.name} - {cfg.description}")
    print()

    # 串行执行
    all_results = []

    for i, config in enumerate(configs, 1):
        print(f"\n{'='*70}")
        print(f"进度: {i}/{len(configs)}")
        print(f"{'='*70}\n")

        try:
            result = run_single_test(config, model_path, output_dir)
            all_results.append(result)
        except Exception as e:
            print(f"\n❌ 测试失败: {config.name}")
            print(f"   错误: {e}")
            import traceback
            traceback.print_exc()

            # 记录失败
            all_results.append({
                'config': {'name': config.name},
                'error': str(e),
                'status': 'failed'
            })

        # 测试间隔，确保环境清理
        if i < len(configs):
            print("⏳ 等待 3 秒，确保环境清理...")
            time.sleep(3)

    # 汇总结果
    print("\n" + "=" * 70)
    print("测试完成！汇总结果")
    print("=" * 70)
    print()

    summary_file = output_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(summary_file, 'w') as f:
        json.dump({
            'total_tests': len(configs),
            'successful': sum(1 for r in all_results if 'error' not in r),
            'failed': sum(1 for r in all_results if 'error' in r),
            'results': all_results,
        }, f, indent=2)

    print(f"📊 汇总结果已保存: {summary_file}")
    print()

    # 打印性能对比
    print("📊 **性能对比**:")
    print(f"   {'配置':25s}  {'压缩率':>10s}  {'速度 (tok/s)':>15s}  {'内存 (MB)':>12s}")
    print(f"   {'-'*25}  {'-'*10}  {'-'*15}  {'-'*12}")

    for result in all_results:
        if 'error' in result:
            continue

        name = result['config']['name']
        comp = result['config']['compression_ratio']
        speed = result.get('avg_tok_per_s', 0)
        mem = result['memory']['model_size_mb']

        print(f"   {name:25s}  {comp:9.0f}x  {speed:14.2f}  {mem:11.2f}")

    print()


# ============================================================================
# 主程序
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='KVTC 端到端测试 - 串行执行')
    parser.add_argument(
        '--model',
        type=str,
        default='mlx-community/Qwen2.5-0.5B-Instruct-4bit',
        help='模型路径（默认: Qwen2.5-0.5B，用于快速测试）'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./kvtc_e2e_results',
        help='结果输出目录'
    )

    args = parser.parse_args()

    print(f"\n🎯 模型: {args.model}")
    print(f"📁 输出目录: {args.output}\n")

    # 提示：使用小模型快速测试
    if '0.5B' not in args.model and '1B' not in args.model:
        print("⚠️  注意: 使用较大模型进行测试")
        # 自动继续，不需要交互确认

    run_serial_tests(args.model, args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
