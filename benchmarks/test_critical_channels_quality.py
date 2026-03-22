#!/usr/bin/env python3
"""
Critical Channels 质量验证测试

验证只保留 critical channels 是否保持输出质量

测试方案：
1. Selective state masking (保留 critical, 清零其他)
2. 4 个质量测试场景
3. 对比分析 (完整 vs critical-only)
4. 不同压缩比测试
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "mlx-lm-source"))

import mlx.core as mx
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


class QualityTester:
    """质量验证测试器"""

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

    def mask_non_critical_channels(
        self,
        state: mx.array,
        layer_idx: int,
        compression_ratio: float = 0.05
    ) -> mx.array:
        """
        保留 critical channels，清零其他通道

        Args:
            state: (B, Hv, Dv, Dk) - SSM state
            layer_idx: 层索引
            compression_ratio: 压缩比 (决定保留多少通道)

        Returns:
            masked_state: 只保留 critical channels 的 state
        """
        # 获取该层的 calibration
        cal = self.calibrations.get(layer_idx)
        if cal is None:
            print(f"⚠️  Layer {layer_idx} 没有 calibration，返回原 state")
            return state

        # 如果需要调整压缩比，重新选择 critical channels
        num_channels = cal['profiling_metadata']['num_channels']
        num_critical = max(1, int(num_channels * compression_ratio))

        # 获取 importance scores 并选择 top N
        importance_scores = cal['profiling_metadata']['importance_scores']
        sorted_indices = sorted(range(len(importance_scores)),
                                key=lambda i: importance_scores[i],
                                reverse=True)
        critical_channels = sorted_indices[:num_critical]

        # 创建 mask (B, Hv, Dv, Dk)
        B, Hv, Dv, Dk = state.shape
        mask = mx.zeros((B, Hv, Dv, Dk), dtype=state.dtype)

        # 只保留 critical channels
        for ch in critical_channels:
            mask[:, :, ch, :] = 1.0

        # 应用 mask
        masked_state = state * mask

        return masked_state

    def generate_with_masked_state(
        self,
        prompt: str,
        compression_ratio: float = 0.05,
        max_tokens: int = 50
    ) -> Tuple[str, List[int], mx.array]:
        """
        使用 masked state 生成文本

        Returns:
            text: 生成的文本
            token_ids: token IDs
            logits: 所有 token 的 logits
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

        # 现在对所有 SSM 层应用 masking
        for layer_idx in self.calibrations.keys():
            layer_cache = cache[layer_idx]
            if hasattr(layer_cache, 'state') and layer_cache.state is not None:
                current_state = layer_cache.state
                if isinstance(current_state, list) and len(current_state) == 2:
                    # Mask SSM state
                    ssm_state = current_state[1]
                    masked_ssm_state = self.mask_non_critical_channels(
                        ssm_state, layer_idx, compression_ratio
                    )
                    layer_cache.state = [current_state[0], masked_ssm_state]

        # 继续生成
        all_logits = []
        token_ids = [int(next_token.item())]
        current_tokens = next_token.reshape(1, 1)

        for _ in range(max_tokens - 1):
            # Forward pass
            logits = self.model(current_tokens, cache=cache)
            mx.eval(logits)

            # Store logits
            all_logits.append(logits[0, -1, :])

            # Sample next token (greedy)
            next_token = mx.argmax(logits[0, -1, :], axis=-1)
            token_ids.append(int(next_token.item()))

            # Update
            current_tokens = next_token.reshape(1, 1)

        # Decode
        text = self.tokenizer.decode(token_ids)

        # Stack logits
        all_logits = mx.stack(all_logits) if all_logits else mx.zeros((0, logits.shape[-1]))

        return text, token_ids, all_logits

    def generate_baseline(
        self,
        prompt: str,
        max_tokens: int = 50
    ) -> Tuple[str, List[int], mx.array]:
        """
        使用完整 state 生成文本 (baseline)
        """
        cache = cache_module.make_prompt_cache(self.model)

        tokens = mx.array(self.tokenizer.encode(prompt))

        # Prefill
        logits = self.model(tokens[None], cache=cache)
        mx.eval(logits)

        # Decode one token
        next_token = mx.argmax(logits[0, -1, :], keepdims=True)
        logits = self.model(next_token[None], cache=cache)
        mx.eval(logits, cache)

        # Continue generation
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
        masked_tokens: List[int],
        baseline_logits: mx.array,
        masked_logits: mx.array
    ) -> Dict[str, float]:
        """对比两个输出"""
        # Token 差异率
        min_len = min(len(baseline_tokens), len(masked_tokens))
        token_diff_count = sum(1 for i in range(min_len)
                               if baseline_tokens[i] != masked_tokens[i])
        token_diff_rate = token_diff_count / min_len if min_len > 0 else 0

        # Logits KL divergence (如果有 logits)
        if baseline_logits.size > 0 and masked_logits.size > 0:
            min_logits_len = min(baseline_logits.shape[0], masked_logits.shape[0])
            if min_logits_len > 0:
                baseline_probs = mx.softmax(baseline_logits[:min_logits_len], axis=-1)
                masked_probs = mx.softmax(masked_logits[:min_logits_len], axis=-1)

                kl = (baseline_probs * mx.log(baseline_probs / (masked_probs + 1e-10))).sum(axis=-1).mean()
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
            'masked_length': len(masked_tokens)
        }

    def run_quality_test(
        self,
        scenario_name: str,
        prompt: str,
        compression_ratio: float = 0.05
    ) -> Dict:
        """运行单个质量测试"""
        print(f"\n  测试场景: {scenario_name}")
        print(f"  压缩比: {compression_ratio * 100:.1f}%")

        # Baseline
        print("    生成 baseline...")
        baseline_text, baseline_tokens, baseline_logits = self.generate_baseline(prompt)

        # Masked
        print("    生成 masked...")
        masked_text, masked_tokens, masked_logits = self.generate_with_masked_state(
            prompt, compression_ratio
        )

        # Compare
        comparison = self.compare_outputs(
            baseline_tokens, masked_tokens, baseline_logits, masked_logits
        )

        return {
            'scenario': scenario_name,
            'compression_ratio': compression_ratio,
            'baseline': {
                'text': baseline_text,
                'tokens': baseline_tokens,
                'length': len(baseline_tokens)
            },
            'masked': {
                'text': masked_text,
                'tokens': masked_tokens,
                'length': len(masked_tokens)
            },
            'comparison': comparison
        }


def main():
    print("=" * 60)
    print("Critical Channels 质量验证测试")
    print("=" * 60)
    print()

    # Load model
    print("加载模型...")
    model_path = "/Volumes/toshiba/models/qwen3.5-35b-mlx"
    model, tokenizer = load(model_path)
    print(f"✅ 模型加载完成")
    print()

    # Create tester
    tester = QualityTester(model, tokenizer)
    print(f"✅ 加载了 {len(tester.calibrations)} 层 calibration")
    print()

    # Test different compression ratios
    compression_ratios = [0.05, 0.10, 0.15]  # 5%, 10%, 15%

    all_results = []

    for ratio in compression_ratios:
        print(f"\n{'='*60}")
        print(f"测试压缩比: {ratio * 100:.1f}%")
        print(f"{'='*60}")

        for scenario_name, prompt in TEST_SCENARIOS.items():
            result = tester.run_quality_test(scenario_name, prompt, ratio)
            all_results.append(result)

            # Print summary
            comp = result['comparison']
            print(f"    ✅ 完成")
            print(f"       Token 差异率: {comp['token_diff_rate'] * 100:.1f}%")
            print(f"       KL Divergence: {comp['kl_divergence']:.4f}")

    # Save results
    print(f"\n{'='*60}")
    print("保存测试结果...")
    print(f"{'='*60}")

    output_file = ".solar/quality-test-results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"✅ 结果已保存: {output_file}")
    print()

    # Generate summary report
    print(f"{'='*60}")
    print("质量测试总结")
    print(f"{'='*60}")
    print()

    for ratio in compression_ratios:
        print(f"\n📊 压缩比 {ratio * 100:.1f}%:")

        ratio_results = [r for r in all_results if r['compression_ratio'] == ratio]

        avg_token_diff = sum(r['comparison']['token_diff_rate'] for r in ratio_results) / len(ratio_results)
        avg_kl = sum(r['comparison']['kl_divergence'] for r in ratio_results) / len(ratio_results)

        print(f"   平均 Token 差异率: {avg_token_diff * 100:.2f}%")
        print(f"   平均 KL Divergence: {avg_kl:.4f}")

        # Quality assessment
        if avg_token_diff < 0.05 and avg_kl < 0.5:
            quality = "✅ 优秀 - 质量几乎无损"
        elif avg_token_diff < 0.15 and avg_kl < 1.0:
            quality = "🟢 良好 - 质量可接受"
        elif avg_token_diff < 0.30 and avg_kl < 2.0:
            quality = "🟡 一般 - 有明显差异"
        else:
            quality = "🔴 差 - 质量显著下降"

        print(f"   质量评估: {quality}")

    print()
    print(f"{'='*60}")
    print("建议")
    print(f"{'='*60}")
    print()

    # Find best ratio
    best_ratio = None
    best_quality = None

    for ratio in compression_ratios:
        ratio_results = [r for r in all_results if r['compression_ratio'] == ratio]
        avg_token_diff = sum(r['comparison']['token_diff_rate'] for r in ratio_results) / len(ratio_results)
        avg_kl = sum(r['comparison']['kl_divergence'] for r in ratio_results) / len(ratio_results)

        quality_score = avg_token_diff + avg_kl * 0.1  # Weighted score

        if best_quality is None or quality_score < best_quality:
            best_quality = quality_score
            best_ratio = ratio

    print(f"🎯 推荐压缩比: {best_ratio * 100:.1f}%")
    print()

    # Go/No-Go decision
    best_results = [r for r in all_results if r['compression_ratio'] == best_ratio]
    avg_diff = sum(r['comparison']['token_diff_rate'] for r in best_results) / len(best_results)

    if avg_diff < 0.10:
        print("✅ Go - 可以继续 Phase 2 实现")
        print(f"   使用压缩比: {best_ratio * 100:.1f}%")
    else:
        print("⚠️  No-Go - 需要调整策略")
        print("   建议:")
        print("   1. 增加压缩比到 15-20%")
        print("   2. 实现分层压缩策略")
        print("   3. 重新 profile 更多通道")


if __name__ == "__main__":
    main()
