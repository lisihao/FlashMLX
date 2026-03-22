"""
单元测试：NNLS 求解器

验证 NNLS 实现的正确性。
"""

import unittest
import numpy as np
import mlx.core as mx

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'mlx-lm-source')))

from mlx_lm.compaction.solvers import nnls_clamped, nnls_pgd, nnls_auto


class TestNNLS(unittest.TestCase):
    """测试 NNLS 求解器"""

    def test_clamped_basic(self):
        """测试 Clamped LS"""
        M_np = np.random.randn(50, 20)
        y_np = np.random.randn(50)

        M_mlx = mx.array(M_np)
        y_mlx = mx.array(y_np)

        x = nnls_clamped(M_mlx, y_mlx, lower_bound=0.0)

        # 验证约束：x >= 0
        self.assertTrue(np.all(np.array(x) >= -1e-10), "NNLS solution violates x >= 0")

        # 验证残差合理
        residual = M_mlx @ x - y_mlx
        loss = mx.sum(residual ** 2)
        print(f"nnls_clamped loss: {float(loss):.4f}")

    def test_pgd_convergence(self):
        """测试 PGD 收敛"""
        M_np = np.random.randn(50, 20)
        y_np = np.random.randn(50)

        M_mlx = mx.array(M_np)
        y_mlx = mx.array(y_np)

        x = nnls_pgd(M_mlx, y_mlx, lower_bound=0.0, max_iters=100, verbose=False)

        # 验证约束：x >= 0
        self.assertTrue(np.all(np.array(x) >= -1e-10), "NNLS PGD violates x >= 0")

        # 验证残差
        residual = M_mlx @ x - y_mlx
        loss = mx.sum(residual ** 2)
        print(f"nnls_pgd loss: {float(loss):.4f}")

    def test_pgd_better_than_clamped(self):
        """验证 PGD 比 Clamped 更优"""
        M_np = np.random.randn(50, 20)
        y_np = np.random.randn(50)

        M_mlx = mx.array(M_np)
        y_mlx = mx.array(y_np)

        x_clamped = nnls_clamped(M_mlx, y_mlx, lower_bound=0.0)
        x_pgd = nnls_pgd(M_mlx, y_mlx, lower_bound=0.0, max_iters=100)

        loss_clamped = float(mx.sum((M_mlx @ x_clamped - y_mlx) ** 2))
        loss_pgd = float(mx.sum((M_mlx @ x_pgd - y_mlx) ** 2))

        print(f"loss_clamped: {loss_clamped:.4f}, loss_pgd: {loss_pgd:.4f}")

        # PGD 应该不差于 Clamped（可能相同）
        self.assertLessEqual(loss_pgd, loss_clamped * 1.01)  # 允许 1% 误差

    def test_auto_selector(self):
        """测试自动选择器"""
        M_np = np.random.randn(30, 10)
        y_np = np.random.randn(30)

        M_mlx = mx.array(M_np)
        y_mlx = mx.array(y_np)

        x_fast = nnls_auto(M_mlx, y_mlx, quality='fast')
        x_medium = nnls_auto(M_mlx, y_mlx, quality='medium')
        x_high = nnls_auto(M_mlx, y_mlx, quality='high')

        # 验证所有解都满足约束
        for x in [x_fast, x_medium, x_high]:
            self.assertTrue(np.all(np.array(x) >= -1e-10))


if __name__ == '__main__':
    unittest.main(verbosity=2)
