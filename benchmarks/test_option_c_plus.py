#!/usr/bin/env python3
"""
测试 Option C+ (增加 Rank)

对比：
- Option C: rank=16/32/48
- Option C+: rank=32/64/96

目标: Think/Format 场景 < 20% 差异率
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "mlx-lm-source"))

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models import cache as cache_module
import json
import os
from typing import Dict, List, Tuple


# 测试 prompts
TEST_SCENARIOS = {
    'chinese': """请用中文回答以下问题：

问题：什么是人工智能？

回答：""",

    'think_tag': """请用中文回答，并在回答前先思考：

<think>
首先我需要理解...
</think>

问题：机器学习和深度学习的区别是什么？

回答：""",

    'format_control': """请用列表形式回答以下问题：

问题：机器学习有哪些主要类型？

回答：
1.""",

    'mixed_language': """Please answer in both English and Chinese:

Question: What is deep learning?

Answer (English):"""
}


class OptionCPlusTester:
    """Option C+ 测试器"""

    def __init__(self, model, tokenizer, calibration_dir: str = ".solar/calibration"):
        self.model = model
        self.tokenizer = tokenizer
        self.calibration_dir = calibration_dir
        self.calibrations = self._load_calibrations()

    def _load_calibrations(self) -> Dict[int, dict]:
        """加载所有 calibration 文件"""
        calibrations = {}

        for file in os.listdir(self.calibration_dir):
            if file.endswith('.json'):
                layer_idx = int(file.split('_')[1])
                with open(os.path.join(self.calibration_dir, file), 'r') as f:
                    calibrations[layer_idx] = json.load(f)

        return calibrations

    def compress_option_c(
        self,
        state: mx.array,
        layer_idx: int,
        use_plus: bool = False
    ) -> Tuple[mx.array, dict]:
        """
        Option C / C+ 分层压缩

        Args:
            use_plus: True = Option C+, False = Option C
        """
        B, Hv, Dv, Dk = state.shape

        # 确定 rank
        if use_plus:
            # Option C+: 更高的 rank
            if layer_idx <= 12:
                rank = 32  # 早期层
            elif layer_idx <= 25:
                rank = 64  # 中期层
            else:
                rank = 96  # 后期层
        else:
            # Option C: 原始 rank
            if layer_idx <= 12:
                rank = 16
            elif layer_idx <= 25:
                rank = 32
            else:
                rank = 48

        # 对整个 state 做低秩近似
        state_flat = state.reshape(-1, Dk)

        # SVD (先转 float32)
        state_flat_f32 = state_flat.astype(mx.float32)
        state_np = np.array(state_flat_f32)
        U, S, Vt = np.linalg.svd(state_np, full_matrices=False)

        # 保留 top rank
        U_r = U[:, :rank]
        S_r = S[:rank]
        Vt_r = Vt[:rank, :]

        # 重构
        state_approx_np = U_r @ np.diag(S_r) @ Vt_r
        compressed_state = mx.array(state_approx_np).astype(state.dtype).reshape(B, Hv, Dv, Dk)

        metadata = {
            'method': 'option_c_plus' if use_plus else 'option_c',
            'rank': rank,
            'layer_position': 'early' if layer_idx <= 12 else ('middle' if layer_idx <= 25 else 'late'),
            'compression_ratio': rank / Dv
        }

        return compressed_state, metadata

    def generate_with_compression(
        self,
        prompt: str,
        use_plus: bool = False,
        max_tokens: int = 50
    ) -> Tuple[str, List[int], mx.array, List[dict]]:
        """
        使用压缩后的 state 生成文本

        Args:
            use_plus: True = Option C+, False = Option C
        """
        # 创建 cache
        cache = cache_module.make_prompt_cache(self.model)

        # Encode prompt
        tokens = mx.array(self.tokenizer.encode(prompt))

        # Prefill
        logits = self.model(tokens[None], cache=cache)
        mx.eval(logits)

        # Decode one token to populate cache
        next_token = mx.argmax(logits[0, -1, :], keepdims=True)
        logits = self.model(next_token[None], cache=cache)
        mx.eval(logits, cache)

        # 对所有 SSM 层应用压缩
        compression_metadata = []

        for layer_idx in self.calibrations.keys():
            layer_cache = cache[layer_idx]
            if hasattr(layer_cache, 'state') and layer_cache.state is not None:
                current_state = layer_cache.state
                if isinstance(current_state, list) and len(current_state) == 2:
                    ssm_state = current_state[1]

                    # 应用压缩
                    compressed_state, metadata = self.compress_option_c(
                        ssm_state, layer_idx, use_plus=use_plus
                    )

                    layer_cache.state = [current_state[0], compressed_state]
                    compression_metadata.append({
                        'layer': layer_idx,
                        **metadata
                    })

        # 继续生成
        all_logits = []
        token_ids = [int(next_token.item())]
        current_tokens = next_token.reshape(1, 1)

        for _ in range(max_tokens - 1):
            logits = self.model(current_tokens, cache=cache)
            mx.eval(logits)

            all_logits.append(logits[0, -1, :])

            next_token = mx.argmax(logits[0, -1, :], axis=-1)
            token_ids.append(int(next_token.item()))

            current_tokens = next_token.reshape(1, 1)

        text = self.tokenizer.decode(token_ids)
        all_logits = mx.stack(all_logits) if all_logits else mx.zeros((0, logits.shape[-1]))

        return text, token_ids, all_logits, compression_metadata

    def generate_baseline(
        self,
        prompt: str,
        max_tokens: int = 50
    ) -> Tuple[str, List[int], mx.array]:
        """生成 baseline (无压缩)"""
        cache = cache_module.make_prompt_cache(self.model)

        tokens = mx.array(self.tokenizer.encode(prompt))

        logits = self.model(tokens[None], cache=cache)
        mx.eval(logits)

        next_token = mx.argmax(logits[0, -1, :], keepdims=True)
        logits = self.model(next_token[None], cache=cache)
        mx.eval(logits, cache)

        all_logits = []
        token_ids = [int(next_token.item())]
        current_tokens = next_token.reshape(1, 1)

        for _ in range(max_tokens - 1):
            logits = self.model(current_tokens, cache=cache)
            mx.eval(logits)

            all_logits.append(logits[0, -1, :])

            next_token = mx.argmax(logits[0, -1, :], axis=-1)
            token_ids.append(int(next_token.item()))

            current_tokens = next_token.reshape(1, 1)

        text = self.tokenizer.decode(token_ids)
        all_logits = mx.stack(all_logits) if all_logits else mx.zeros((0, logits.shape[-1]))

        return text, token_ids, all_logits

    def compare_outputs(
        self,
        baseline_tokens: List[int],
        compressed_tokens: List[int],
        baseline_logits: mx.array,
        compressed_logits: mx.array
    ) -> Dict[str, float]:
        """对比输出质量"""
        min_len = min(len(baseline_tokens), len(compressed_tokens))
        token_diff_count = sum(1 for i in range(min_len)
                               if baseline_tokens[i] != compressed_tokens[i])
        token_diff_rate = token_diff_count / min_len if min_len > 0 else 0

        if baseline_logits.size > 0 and compressed_logits.size > 0:
            min_logits_len = min(baseline_logits.shape[0], compressed_logits.shape[0])
            if min_logits_len > 0:
                baseline_probs = mx.softmax(baseline_logits[:min_logits_len], axis=-1)
                compressed_probs = mx.softmax(compressed_logits[:min_logits_len], axis=-1)

                kl = (baseline_probs * mx.log(baseline_probs / (compressed_probs + 1e-10))).sum(axis=-1).mean()
                kl_divergence = float(kl.item())
            else:
                kl_divergence = 0.0
        else:
            kl_divergence = 0.0

        return {
            'token_diff_rate': token_diff_rate,
            'token_diff_count': token_diff_count,
            'kl_divergence': kl_divergence,
            'baseline_length': len(baseline_tokens),
            'compressed_length': len(compressed_tokens)
        }


def main():
    print("=" * 60)
    print("Option C+ 优化测试")
    print("对比 Option C vs Option C+")
    print("=" * 60)
    print()

    # Load model
    print("加载模型...")
    model_path = "/Volumes/toshiba/models/qwen3.5-35b-mlx"
    model, tokenizer = load(model_path)
    print(f"✅ 模型加载完成")
    print()

    # Create tester
    tester = OptionCPlusTester(model, tokenizer)
    print(f"✅ 加载了 {len(tester.calibrations)} 层 calibration")
    print()

    # Rank 设置
    print("Rank 设置对比:")
    print("  Option C:  早期=16, 中期=32, 后期=48")
    print("  Option C+: 早期=32, 中期=64, 后期=96")
    print()

    all_results = []

    # 测试每个场景
    for scenario_name, prompt in TEST_SCENARIOS.items():
        print(f"\n{'='*60}")
        print(f"测试场景: {scenario_name}")
        print(f"{'='*60}")
        print()

        # Baseline
        print("  生成 baseline...")
        baseline_text, baseline_tokens, baseline_logits = tester.generate_baseline(prompt)

        # Option C
        print("  生成 Option C...")
        optC_text, optC_tokens, optC_logits, optC_meta = tester.generate_with_compression(
            prompt, use_plus=False
        )

        # Option C+
        print("  生成 Option C+...")
        optCplus_text, optCplus_tokens, optCplus_logits, optCplus_meta = tester.generate_with_compression(
            prompt, use_plus=True
        )

        # Compare
        optC_comp = tester.compare_outputs(baseline_tokens, optC_tokens, baseline_logits, optC_logits)
        optCplus_comp = tester.compare_outputs(baseline_tokens, optCplus_tokens, baseline_logits, optCplus_logits)

        result = {
            'scenario': scenario_name,
            'baseline': {
                'text': baseline_text,
                'tokens': baseline_tokens,
                'length': len(baseline_tokens)
            },
            'option_c': {
                'text': optC_text,
                'tokens': optC_tokens,
                'length': len(optC_tokens),
                'comparison': optC_comp,
                'metadata': optC_meta
            },
            'option_c_plus': {
                'text': optCplus_text,
                'tokens': optCplus_tokens,
                'length': len(optCplus_tokens),
                'comparison': optCplus_comp,
                'metadata': optCplus_meta
            }
        }

        all_results.append(result)

        # Print comparison
        print(f"\n  📊 对比结果:")
        print(f"     Option C:  Token 差异率 {optC_comp['token_diff_rate'] * 100:.1f}%, KL {optC_comp['kl_divergence']:.4f}")
        print(f"     Option C+: Token 差异率 {optCplus_comp['token_diff_rate'] * 100:.1f}%, KL {optCplus_comp['kl_divergence']:.4f}")

        improvement = (optC_comp['token_diff_rate'] - optCplus_comp['token_diff_rate']) * 100
        if improvement > 0:
            print(f"     ✅ 改善: {improvement:.1f}%")
        else:
            print(f"     ⚠️  未改善 ({improvement:.1f}%)")

    # Save results
    print(f"\n{'='*60}")
    print("保存测试结果...")
    print(f"{'='*60}")

    output_file = ".solar/option-c-plus-results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"✅ 结果已保存: {output_file}")
    print()

    # Generate summary
    print(f"{'='*60}")
    print("总结对比")
    print(f"{'='*60}")
    print()

    optC_avg_diff = np.mean([r['option_c']['comparison']['token_diff_rate'] for r in all_results])
    optC_avg_kl = np.mean([r['option_c']['comparison']['kl_divergence'] for r in all_results])

    optCplus_avg_diff = np.mean([r['option_c_plus']['comparison']['token_diff_rate'] for r in all_results])
    optCplus_avg_kl = np.mean([r['option_c_plus']['comparison']['kl_divergence'] for r in all_results])

    print("📊 Option C (原始):")
    print(f"   平均 Token 差异率: {optC_avg_diff * 100:.2f}%")
    print(f"   平均 KL Divergence: {optC_avg_kl:.4f}")

    if all_results[0]['option_c']['metadata']:
        optC_compression = np.mean([
            m['compression_ratio']
            for r in all_results
            for m in r['option_c']['metadata']
        ])
        print(f"   平均压缩比: {optC_compression * 100:.2f}% 保留")

    print()

    print("📊 Option C+ (优化):")
    print(f"   平均 Token 差异率: {optCplus_avg_diff * 100:.2f}%")
    print(f"   平均 KL Divergence: {optCplus_avg_kl:.4f}")

    if all_results[0]['option_c_plus']['metadata']:
        optCplus_compression = np.mean([
            m['compression_ratio']
            for r in all_results
            for m in r['option_c_plus']['metadata']
        ])
        print(f"   平均压缩比: {optCplus_compression * 100:.2f}% 保留")

    print()

    # Improvement
    improvement = (optC_avg_diff - optCplus_avg_diff) * 100
    print(f"📈 改善幅度: {improvement:.2f}%")
    print()

    # Quality assessment per scenario
    print(f"{'='*60}")
    print("分场景评估")
    print(f"{'='*60}")
    print()

    for r in all_results:
        scenario = r['scenario']
        cplus_diff = r['option_c_plus']['comparison']['token_diff_rate'] * 100

        print(f"{scenario}:")
        print(f"  Option C+: {cplus_diff:.1f}%")

        if cplus_diff < 15:
            quality = "✅ 优秀"
        elif cplus_diff < 20:
            quality = "🟢 良好"
        elif cplus_diff < 50:
            quality = "🟡 一般"
        else:
            quality = "🔴 差"

        print(f"  质量: {quality}")
        print()

    # Final decision
    print(f"{'='*60}")
    print("最终决策")
    print(f"{'='*60}")
    print()

    # Check targets
    chinese_ok = all_results[0]['option_c_plus']['comparison']['token_diff_rate'] < 0.15
    think_ok = all_results[1]['option_c_plus']['comparison']['token_diff_rate'] < 0.20
    format_ok = all_results[2]['option_c_plus']['comparison']['token_diff_rate'] < 0.20
    mixed_ok = all_results[3]['option_c_plus']['comparison']['token_diff_rate'] < 0.25

    all_ok = chinese_ok and think_ok and format_ok and mixed_ok

    if all_ok:
        print("✅ Go - 所有场景达标！")
        print()
        print("可以继续 Phase 2:")
        print("- 使用 Option C+ 作为 Warm 压缩方法")
        print("- 实现三段式缓存 (Hot/Warm/Cold)")
        print("- 集成到 generate() 函数")
    else:
        print("⚠️  部分场景未达标")
        print()
        print("未达标场景:")
        if not chinese_ok:
            print(f"  - Chinese: {all_results[0]['option_c_plus']['comparison']['token_diff_rate'] * 100:.1f}% (目标 <15%)")
        if not think_ok:
            print(f"  - Think Tag: {all_results[1]['option_c_plus']['comparison']['token_diff_rate'] * 100:.1f}% (目标 <20%)")
        if not format_ok:
            print(f"  - Format: {all_results[2]['option_c_plus']['comparison']['token_diff_rate'] * 100:.1f}% (目标 <20%)")
        if not mixed_ok:
            print(f"  - Mixed: {all_results[3]['option_c_plus']['comparison']['token_diff_rate'] * 100:.1f}% (目标 <25%)")
        print()
        print("建议:")
        print("1. 进一步增加 rank (如 64/96/128)")
        print("2. 或接受当前质量，继续 Phase 2")
        print("3. 或重新评估压缩方向")


if __name__ == "__main__":
    main()
