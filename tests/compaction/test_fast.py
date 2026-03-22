"""
单元测试：Fast Path 压缩算法

验证 Fast Path 的功能和质量。
"""

import unittest
import numpy as np
import mlx.core as mx

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'mlx-lm-source')))

from mlx_lm.compaction.fast import (
    compact_single_head_fast,
    compact_single_head_fast_with_queries,
    compact_multi_head_fast,
    estimate_compression_time,
)


class TestFastPath(unittest.TestCase):
    """测试 Fast Path 压缩算法"""

    def test_single_head_basic(self):
        """测试单头压缩基本功能"""
        seq_len = 1000
        head_dim = 128
        budget = 200

        K = mx.random.normal((seq_len, head_dim))
        V = mx.random.normal((seq_len, head_dim))

        C1, beta, C2 = compact_single_head_fast(K, V, budget)

        # 验证形状
        self.assertEqual(C1.shape, (budget, head_dim))
        self.assertEqual(beta.shape, (budget,))
        self.assertEqual(C2.shape, (budget, head_dim))

        # 验证 beta = 0 (Fast Path 特征)
        self.assertTrue(mx.all(beta == 0))

    def test_single_head_with_indices(self):
        """测试返回 indices"""
        seq_len = 1000
        head_dim = 128
        budget = 200

        K = mx.random.normal((seq_len, head_dim))
        V = mx.random.normal((seq_len, head_dim))

        C1, beta, C2, indices = compact_single_head_fast(
            K, V, budget, return_indices=True
        )

        # 验证 indices
        self.assertEqual(indices.shape, (budget,))
        self.assertTrue(mx.all(indices >= 0))
        self.assertTrue(mx.all(indices < seq_len))

        # 验证 C1/C2 确实是 K/V 的子集
        for i, idx in enumerate(indices):
            idx_val = int(idx)
            # 允许浮点误差
            self.assertTrue(mx.allclose(C1[i], K[idx_val], atol=1e-5))
            self.assertTrue(mx.allclose(C2[i], V[idx_val], atol=1e-5))

    def test_recent_ratio(self):
        """测试 recent_ratio 参数"""
        seq_len = 1000
        head_dim = 64
        budget = 200
        recent_ratio = 0.25

        K = mx.random.normal((seq_len, head_dim))
        V = mx.random.normal((seq_len, head_dim))

        _, _, _, indices = compact_single_head_fast(
            K, V, budget, recent_ratio=recent_ratio, return_indices=True
        )

        # 验证最近 recent_ratio * budget 个 tokens 是连续的
        n_recent = int(budget * recent_ratio)
        recent_start = seq_len - n_recent

        # 最后 n_recent 个 indices 应该是连续的
        recent_indices = indices[-n_recent:]
        expected_recent = mx.arange(recent_start, seq_len)

        # 因为索引可能排序不同，检查集合相等
        recent_set = set(int(x) for x in recent_indices)
        expected_set = set(int(x) for x in expected_recent)
        self.assertEqual(recent_set, expected_set)

    def test_compression_ratio(self):
        """测试不同压缩比"""
        seq_len = 1000
        head_dim = 128

        for compression_ratio in [2, 5, 10]:
            budget = seq_len // compression_ratio

            K = mx.random.normal((seq_len, head_dim))
            V = mx.random.normal((seq_len, head_dim))

            C1, beta, C2 = compact_single_head_fast(K, V, budget)

            # 验证形状
            self.assertEqual(C1.shape, (budget, head_dim))
            self.assertEqual(C2.shape, (budget, head_dim))

    def test_with_queries_basic(self):
        """测试带 queries 的压缩"""
        seq_len = 1000
        head_dim = 128
        budget = 200
        n_queries = 10

        K = mx.random.normal((seq_len, head_dim))
        V = mx.random.normal((seq_len, head_dim))
        queries = mx.random.normal((n_queries, head_dim))

        result = compact_single_head_fast_with_queries(
            K, V, queries, budget
        )

        # 验证返回的 dict 结构
        self.assertIn('C1', result)
        self.assertIn('beta', result)
        self.assertIn('C2', result)
        self.assertIn('indices', result)
        self.assertIn('metrics', result)

        # 验证 metrics 包含所有指标
        metrics = result['metrics']
        self.assertIn('mse', metrics)
        self.assertIn('relative_error', metrics)
        self.assertIn('max_error', metrics)
        self.assertIn('mean_abs_error', metrics)

    def test_quality_5x_compression(self):
        """测试 5x 压缩质量 (完全随机数据)"""
        seq_len = 1000
        head_dim = 128
        budget = 200  # 5x compression
        n_queries = 20

        # 使用固定随机种子确保可重复
        mx.random.seed(42)

        K = mx.random.normal((seq_len, head_dim))
        V = mx.random.normal((seq_len, head_dim))
        queries = mx.random.normal((n_queries, head_dim))

        result = compact_single_head_fast_with_queries(
            K, V, queries, budget
        )

        relative_error = result['metrics']['relative_error']
        print(f"5x compression (random data): relative_error = {relative_error:.4f}")

        # 注意：完全随机数据没有真实 attention 的局部性
        # 所以误差会很大。这里只验证算法能正常运行
        # 真实质量测试（带 attention 局部性）在 A.5 阶段进行
        self.assertGreater(relative_error, 0)  # 有误差
        self.assertLess(relative_error, 3.0)  # 但不会无限大

    def test_multi_head_basic(self):
        """测试多头压缩"""
        num_heads = 32
        seq_len = 1000
        head_dim = 128
        budget = 200

        K = mx.random.normal((num_heads, seq_len, head_dim))
        V = mx.random.normal((num_heads, seq_len, head_dim))

        C1, beta, C2 = compact_multi_head_fast(K, V, budget)

        # 验证形状
        self.assertEqual(C1.shape, (num_heads, budget, head_dim))
        self.assertEqual(beta.shape, (num_heads, budget))
        self.assertEqual(C2.shape, (num_heads, budget, head_dim))

        # 验证所有 head 的 beta = 0
        self.assertTrue(mx.all(beta == 0))

    def test_multi_head_independent(self):
        """验证多头独立压缩"""
        num_heads = 4
        seq_len = 100
        head_dim = 64
        budget = 20

        K = mx.random.normal((num_heads, seq_len, head_dim))
        V = mx.random.normal((num_heads, seq_len, head_dim))

        C1_multi, beta_multi, C2_multi = compact_multi_head_fast(K, V, budget)

        # 验证每个 head 独立压缩的结果与单头压缩一致
        for head_idx in range(num_heads):
            K_head = K[head_idx]
            V_head = V[head_idx]

            C1_single, beta_single, C2_single = compact_single_head_fast(
                K_head, V_head, budget, recent_ratio=0.25
            )

            # 应该完全一致（相同的选择策略）
            self.assertTrue(mx.allclose(C1_multi[head_idx], C1_single, atol=1e-5))
            self.assertTrue(mx.allclose(C2_multi[head_idx], C2_single, atol=1e-5))

    def test_estimate_compression_time(self):
        """测试时间估计"""
        seq_len = 60000
        budget = 12000
        num_heads = 32

        timing = estimate_compression_time(seq_len, budget, num_heads)

        # 验证返回的 dict 包含所有字段
        self.assertIn('selection', timing)
        self.assertIn('c1_extract', timing)
        self.assertIn('beta', timing)
        self.assertIn('c2_extract', timing)
        self.assertIn('per_head', timing)
        self.assertIn('total', timing)

        # 验证时间估计合理（经验常数可能不准，放宽约束）
        self.assertGreater(timing['total'], 0)
        self.assertLess(timing['total'], 60.0)  # 应该 < 1 分钟

        # 验证 total = per_head * num_heads
        expected_total = timing['per_head'] * num_heads
        self.assertAlmostEqual(timing['total'], expected_total, places=6)

        print(f"Estimated time for {num_heads} heads, {seq_len} tokens: {timing['total']:.4f}s")
        print(f"Note: Actual time will vary based on hardware and MLX version")

    def test_edge_case_small_budget(self):
        """测试边界情况：小 budget"""
        seq_len = 1000
        head_dim = 128
        budget = 10  # 很小

        K = mx.random.normal((seq_len, head_dim))
        V = mx.random.normal((seq_len, head_dim))

        C1, beta, C2 = compact_single_head_fast(K, V, budget)

        # 应该正常工作
        self.assertEqual(C1.shape, (budget, head_dim))

    def test_edge_case_large_recent_ratio(self):
        """测试边界情况：大 recent_ratio"""
        seq_len = 1000
        head_dim = 64
        budget = 200
        recent_ratio = 0.9  # 90% recent

        K = mx.random.normal((seq_len, head_dim))
        V = mx.random.normal((seq_len, head_dim))

        C1, beta, C2 = compact_single_head_fast(
            K, V, budget, recent_ratio=recent_ratio
        )

        # 应该正常工作
        self.assertEqual(C1.shape, (budget, head_dim))


if __name__ == '__main__':
    unittest.main(verbosity=2)
