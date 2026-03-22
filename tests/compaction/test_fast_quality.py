"""
A.5: 多场景质量测试和可视化分析

测试 Fast Path 在不同数据分布下的质量。
"""

import unittest
import numpy as np
import mlx.core as mx

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'mlx-lm-source')))

from mlx_lm.compaction.fast import (
    compact_single_head_fast_with_queries,
)
from mlx_lm.compaction.fast_v2 import (
    compact_single_head_fast_v2_with_queries,
)
from mlx_lm.compaction.base import (
    visualize_key_selection,
)


def generate_attention_like_data(seq_len, head_dim, n_queries, locality_strength=1.0):
    """
    生成带 attention 局部性的合成数据

    Parameters
    ----------
    seq_len : int
    head_dim : int
    n_queries : int
    locality_strength : float
        局部性强度:
        - 1.0: 强局部性 (recent tokens 很重要)
        - 0.5: 中等局部性 (混合分布)
        - 0.0: 完全随机 (worst case)

    Returns
    -------
    K, V, queries : mx.array
    """
    # 生成 Keys 和 Values
    K = mx.random.normal((seq_len, head_dim))
    V = mx.random.normal((seq_len, head_dim))

    # 生成 Queries，使其与 Keys 有相关性
    # 策略：Queries 主要关注 recent tokens
    if locality_strength > 0:
        # 选择一些 recent keys 作为 query 的基础
        recent_ratio = 0.1 + 0.2 * locality_strength  # 10%-30%
        n_recent = int(seq_len * recent_ratio)
        recent_start = seq_len - n_recent

        queries_list = []
        for _ in range(n_queries):
            # 随机选择一个 recent key
            idx = np.random.randint(recent_start, seq_len)
            base_key = K[idx]

            # 添加噪声（噪声大小与局部性反相关）
            noise_scale = 1.0 - locality_strength * 0.5
            noise = mx.random.normal((head_dim,)) * noise_scale

            query = base_key + noise
            queries_list.append(query[None, :])

        queries = mx.concatenate(queries_list, axis=0)
    else:
        # 完全随机 queries
        queries = mx.random.normal((n_queries, head_dim))

    return K, V, queries


class TestFastPathQuality(unittest.TestCase):
    """测试 Fast Path 在不同数据分布下的质量"""

    def test_ideal_data_strong_locality(self):
        """测试理想数据：强局部性"""
        seq_len = 1000
        head_dim = 128
        budget = 200  # 5x compression
        n_queries = 20

        mx.random.seed(42)
        K, V, queries = generate_attention_like_data(
            seq_len, head_dim, n_queries, locality_strength=1.0
        )

        result = compact_single_head_fast_with_queries(
            K, V, queries, budget
        )

        relative_error = result['metrics']['relative_error']
        print(f"\n理想数据 (strong locality): relative_error = {relative_error:.4f}")

        # 理想数据下应该质量很好
        self.assertLess(relative_error, 0.15)  # < 15%

    def test_medium_data_partial_locality(self):
        """测试中等数据：部分局部性"""
        seq_len = 1000
        head_dim = 128
        budget = 200  # 5x compression
        n_queries = 20

        mx.random.seed(42)
        K, V, queries = generate_attention_like_data(
            seq_len, head_dim, n_queries, locality_strength=0.5
        )

        result = compact_single_head_fast_with_queries(
            K, V, queries, budget
        )

        relative_error = result['metrics']['relative_error']
        print(f"中等数据 (partial locality): relative_error = {relative_error:.4f}")

        # 中等局部性，质量会下降但应该可接受
        self.assertLess(relative_error, 0.5)  # < 50%

    def test_random_data_no_locality(self):
        """测试随机数据：无局部性（worst case）- v1 vs v2 对比"""
        seq_len = 1000
        head_dim = 128
        budget = 200  # 5x compression
        n_queries = 20

        mx.random.seed(42)
        K, V, queries = generate_attention_like_data(
            seq_len, head_dim, n_queries, locality_strength=0.0
        )

        # Test v1
        result_v1 = compact_single_head_fast_with_queries(
            K, V, queries, budget
        )
        error_v1 = result_v1['metrics']['relative_error']

        # Test v2
        result_v2 = compact_single_head_fast_v2_with_queries(
            K, V, queries, budget, n_query_samples=32
        )
        error_v2 = result_v2['metrics']['relative_error']

        print(f"\n随机数据 (no locality):")
        print(f"  v1 (Recent+Stride): {error_v1:.4f}")
        print(f"  v2 (Attention-aware): {error_v2:.4f}")
        print(f"  Improvement: {(error_v1 - error_v2) / error_v1 * 100:.1f}%")

        # v2 应该显著好于 v1
        self.assertLess(error_v2, error_v1 * 0.8)  # 至少改善 20%

    def test_compression_ratio_vs_quality(self):
        """测试压缩比 vs 质量的权衡"""
        seq_len = 1000
        head_dim = 128
        n_queries = 20

        mx.random.seed(42)
        K, V, queries = generate_attention_like_data(
            seq_len, head_dim, n_queries, locality_strength=1.0
        )

        results = []
        compression_ratios = [2, 3, 5, 10]

        print(f"\n压缩比 vs 质量曲线 (strong locality):")
        for ratio in compression_ratios:
            budget = seq_len // ratio
            result = compact_single_head_fast_with_queries(
                K, V, queries, budget
            )

            relative_error = result['metrics']['relative_error']
            results.append((ratio, relative_error))
            print(f"  {ratio}x: relative_error = {relative_error:.4f}")

        # 验证：压缩比越高，误差越大
        errors = [r[1] for r in results]
        for i in range(len(errors) - 1):
            self.assertLessEqual(errors[i], errors[i+1] * 1.5)  # 允许一些波动

    def test_key_selection_visualization(self):
        """可视化 key selection 模式"""
        seq_len = 1000
        head_dim = 128
        budget = 200

        mx.random.seed(42)
        K, V, queries = generate_attention_like_data(
            seq_len, head_dim, 20, locality_strength=1.0
        )

        # 获取 indices
        from mlx_lm.compaction.fast import compact_single_head_fast
        C1, beta, C2, indices = compact_single_head_fast(
            K, V, budget, return_indices=True
        )

        # 计算 attention scores（用于对比）
        from mlx_lm.compaction.base import compute_attention_output
        scores = queries @ K.T / (head_dim ** 0.5)
        attn_weights = mx.softmax(scores, axis=1)
        avg_attn = mx.mean(attn_weights, axis=0)  # 平均 attention

        # 可视化
        vis_str = visualize_key_selection(seq_len, indices, avg_attn)
        print(f"\nKey Selection 可视化:")
        print(vis_str)

        # 验证选中的 keys 覆盖了大部分 attention mass
        # （从 vis_str 中提取 coverage）
        lines = vis_str.split('\n')
        for line in lines:
            if 'Attention coverage' in line:
                coverage_str = line.split(':')[1].strip().rstrip('%')
                coverage = float(coverage_str)
                print(f"Attention coverage: {coverage:.1f}%")

                # Fast Path 应该覆盖大部分 attention mass
                # 但在最优策略下应该能达到 70%+
                # 这里设置保守阈值
                self.assertGreater(coverage, 40.0)  # > 40%


if __name__ == '__main__':
    unittest.main(verbosity=2)
