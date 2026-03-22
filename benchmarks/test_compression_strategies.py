#!/usr/bin/env python3
"""
对比压缩策略：Option A vs Option C

Option A: 混合压缩 (Critical 全精度 + Bulk 低秩)
Option C: 分层压缩 (早期/中期/后期不同 rank)
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


class CompressionTester:
    """压缩策略测试器"""

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

    def compress_option_a(
        self,
        state: mx.array,
        layer_idx: int,
        rank: int = 32
    ) -> Tuple[mx.array, dict]:
        """
        Option A: 混合压缩
        Critical channels 全精度 + Bulk channels 低秩近似

        Returns:
            compressed_state: 压缩后的 state
            metadata: 压缩元数据
        """
        cal = self.calibrations.get(layer_idx)
        if cal is None:
            return state, {'method': 'none'}

        critical_channels = cal['critical_channels']
        B, Hv, Dv, Dk = state.shape

        # 1. 分离 critical 和 bulk
        # MLX 不支持 boolean indexing，使用整数索引
        critical_indices = [int(i) for i in critical_channels]  # Convert to Python ints
        all_indices = list(range(Dv))
        bulk_indices = [i for i in all_indices if i not in critical_indices]

        # 提取 critical channels
        critical_state = mx.concatenate([
            state[:, :, i:i+1, :] for i in critical_indices
        ], axis=2)  # (B, Hv, 6, Dk)

        # 提取 bulk channels
        bulk_state = mx.concatenate([
            state[:, :, i:i+1, :] for i in bulk_indices
        ], axis=2)  # (B, Hv, 122, Dk)

        # 2. 对 bulk 做 SVD 低秩近似
        # Reshape: (B, Hv, 122, Dk) -> (B*Hv*122, Dk)
        bulk_flat = bulk_state.reshape(-1, Dk)

        # SVD (使用 numpy 因为 MLX 的 SVD 可能有限制)
        # 先转 float32 再转 numpy (bfloat16 不能直接转 numpy)
        bulk_flat_f32 = bulk_flat.astype(mx.float32)
        bulk_np = np.array(bulk_flat_f32)
        U, S, Vt = np.linalg.svd(bulk_np, full_matrices=False)

        # 保留 top rank
        U_r = U[:, :rank]
        S_r = S[:rank]
        Vt_r = Vt[:rank, :]

        # 重构
        bulk_approx_np = U_r @ np.diag(S_r) @ Vt_r
        bulk_approx = mx.array(bulk_approx_np).astype(state.dtype).reshape(B, Hv, -1, Dk)

        # 3. 合并 critical 和 bulk
        # 创建完整 state - 需要按原始顺序重组
        # 使用字典存储每个 channel 的数据
        channel_dict = {}

        # 添加 critical channels
        for i, ch_idx in enumerate(critical_indices):
            channel_dict[ch_idx] = critical_state[:, :, i:i+1, :]

        # 添加 bulk channels (近似后的)
        for i, ch_idx in enumerate(bulk_indices):
            channel_dict[ch_idx] = bulk_approx[:, :, i:i+1, :]

        # 按顺序重组
        compressed_state = mx.concatenate([
            channel_dict[i] for i in range(Dv)
        ], axis=2)

        metadata = {
            'method': 'option_a',
            'critical_channels': len(critical_channels),
            'bulk_channels': Dv - len(critical_channels),
            'rank': rank,
            'compression_ratio': (len(critical_channels) + rank * (Dv - len(critical_channels)) / Dk) / Dv
        }

        return compressed_state, metadata

    def compress_option_c(
        self,
        state: mx.array,
        layer_idx: int
    ) -> Tuple[mx.array, dict]:
        """
        Option C: 分层压缩
        根据层位置动态调整 rank
        """
        # 确定 rank
        if layer_idx <= 12:
            rank = 16  # 早期层
        elif layer_idx <= 25:
            rank = 32  # 中期层
        else:
            rank = 48  # 后期层

        B, Hv, Dv, Dk = state.shape

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
            'method': 'option_c',
            'rank': rank,
            'layer_position': 'early' if layer_idx <= 12 else ('middle' if layer_idx <= 25 else 'late'),
            'compression_ratio': rank / Dv
        }

        return compressed_state, metadata

    def generate_with_compression(
        self,
        prompt: str,
        compression_method: str = 'option_a',
        max_tokens: int = 50
    ) -> Tuple[str, List[int], mx.array, List[dict]]:
        """
        使用压缩后的 state 生成文本

        Args:
            compression_method: 'option_a' 或 'option_c'

        Returns:
            text, token_ids, logits, compression_metadata
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
                    if compression_method == 'option_a':
                        compressed_state, metadata = self.compress_option_a(ssm_state, layer_idx)
                    elif compression_method == 'option_c':
                        compressed_state, metadata = self.compress_option_c(ssm_state, layer_idx)
                    else:
                        compressed_state = ssm_state
                        metadata = {'method': 'none'}

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
    print("压缩策略对比测试")
    print("Option A vs Option C")
    print("=" * 60)
    print()

    # Load model
    print("加载模型...")
    model_path = "/Volumes/toshiba/models/qwen3.5-35b-mlx"
    model, tokenizer = load(model_path)
    print(f"✅ 模型加载完成")
    print()

    # Create tester
    tester = CompressionTester(model, tokenizer)
    print(f"✅ 加载了 {len(tester.calibrations)} 层 calibration")
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

        # Option A
        print("  生成 Option A (混合压缩)...")
        optA_text, optA_tokens, optA_logits, optA_meta = tester.generate_with_compression(
            prompt, 'option_a'
        )

        # Option C
        print("  生成 Option C (分层压缩)...")
        optC_text, optC_tokens, optC_logits, optC_meta = tester.generate_with_compression(
            prompt, 'option_c'
        )

        # Compare
        optA_comp = tester.compare_outputs(baseline_tokens, optA_tokens, baseline_logits, optA_logits)
        optC_comp = tester.compare_outputs(baseline_tokens, optC_tokens, baseline_logits, optC_logits)

        result = {
            'scenario': scenario_name,
            'baseline': {
                'text': baseline_text,
                'tokens': baseline_tokens,
                'length': len(baseline_tokens)
            },
            'option_a': {
                'text': optA_text,
                'tokens': optA_tokens,
                'length': len(optA_tokens),
                'comparison': optA_comp,
                'metadata': optA_meta
            },
            'option_c': {
                'text': optC_text,
                'tokens': optC_tokens,
                'length': len(optC_tokens),
                'comparison': optC_comp,
                'metadata': optC_meta
            }
        }

        all_results.append(result)

        # Print comparison
        print(f"\n  📊 对比结果:")
        print(f"     Option A - Token 差异率: {optA_comp['token_diff_rate'] * 100:.1f}%, KL: {optA_comp['kl_divergence']:.4f}")
        print(f"     Option C - Token 差异率: {optC_comp['token_diff_rate'] * 100:.1f}%, KL: {optC_comp['kl_divergence']:.4f}")

    # Save results
    print(f"\n{'='*60}")
    print("保存测试结果...")
    print(f"{'='*60}")

    output_file = ".solar/compression-comparison-results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"✅ 结果已保存: {output_file}")
    print()

    # Generate summary
    print(f"{'='*60}")
    print("总结对比")
    print(f"{'='*60}")
    print()

    optA_avg_diff = np.mean([r['option_a']['comparison']['token_diff_rate'] for r in all_results])
    optA_avg_kl = np.mean([r['option_a']['comparison']['kl_divergence'] for r in all_results])

    optC_avg_diff = np.mean([r['option_c']['comparison']['token_diff_rate'] for r in all_results])
    optC_avg_kl = np.mean([r['option_c']['comparison']['kl_divergence'] for r in all_results])

    print("📊 Option A (混合压缩):")
    print(f"   平均 Token 差异率: {optA_avg_diff * 100:.2f}%")
    print(f"   平均 KL Divergence: {optA_avg_kl:.4f}")

    # 计算压缩比
    if all_results[0]['option_a']['metadata']:
        optA_compression = np.mean([
            m['compression_ratio']
            for r in all_results
            for m in r['option_a']['metadata']
        ])
        print(f"   平均压缩比: {optA_compression * 100:.2f}% 保留")

    print()

    print("📊 Option C (分层压缩):")
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

    # Winner
    print(f"{'='*60}")
    print("结论")
    print(f"{'='*60}")
    print()

    if optA_avg_diff < optC_avg_diff:
        winner = "Option A"
        winner_diff = optA_avg_diff
        winner_kl = optA_avg_kl
    else:
        winner = "Option C"
        winner_diff = optC_avg_diff
        winner_kl = optC_avg_kl

    print(f"🏆 优胜者: {winner}")
    print(f"   Token 差异率: {winner_diff * 100:.2f}%")
    print(f"   KL Divergence: {winner_kl:.4f}")
    print()

    # Quality assessment
    if winner_diff < 0.05:
        quality = "✅ 优秀 - 可以用于生产"
    elif winner_diff < 0.10:
        quality = "🟢 良好 - 质量可接受"
    elif winner_diff < 0.20:
        quality = "🟡 一般 - 需要进一步优化"
    else:
        quality = "🔴 差 - 不推荐使用"

    print(f"质量评估: {quality}")
    print()

    if winner_diff < 0.10:
        print("✅ Go - 可以继续 Phase 2 实现")
        print(f"   推荐方案: {winner}")
    else:
        print("⚠️  需要进一步优化")
        print(f"   当前最佳: {winner} (差异率 {winner_diff * 100:.1f}%)")


if __name__ == "__main__":
    main()
