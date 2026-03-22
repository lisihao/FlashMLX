"""
单元测试：LSQ 求解器

验证 C2 求解器的正确性。
"""

import unittest
import numpy as np
import mlx.core as mx

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'mlx-lm-source')))

from mlx_lm.compaction.solvers import (
    compute_C2_lstsq,
    compute_C2_cholesky,
    compute_C2_pinv,
    compute_C2_auto,
    evaluate_C2_quality,
)


class TestLSQ(unittest.TestCase):
    """测试 LSQ 求解器"""

    def test_lstsq_basic(self):
        """测试标准最小二乘"""
        X_np = np.random.randn(100, 20)
        Y_np = np.random.randn(100, 128)

        X_mlx = mx.array(X_np)
        Y_mlx = mx.array(Y_np)

        C2 = compute_C2_lstsq(X_mlx, Y_mlx)

        # 验证形状
        self.assertEqual(C2.shape, (20, 128))

        # 验证拟合质量
        metrics = evaluate_C2_quality(X_mlx, Y_mlx, C2)
        print(f"lstsq: relative_error = {metrics['relative_error']:.4f}")

        # 最小二乘不保证完美拟合（欠参数化时）
        # 只要不是完全错误即可
        self.assertLess(metrics['relative_error'], 1.0)

    def test_cholesky_basic(self):
        """测试 Cholesky 求解"""
        X_np = np.random.randn(100, 20)
        Y_np = np.random.randn(100, 128)

        X_mlx = mx.array(X_np)
        Y_mlx = mx.array(Y_np)

        C2 = compute_C2_cholesky(X_mlx, Y_mlx, ridge_lambda=0.0)

        # 验证形状
        self.assertEqual(C2.shape, (20, 128))

        # 验证拟合质量
        metrics = evaluate_C2_quality(X_mlx, Y_mlx, C2)
        print(f"cholesky: relative_error = {metrics['relative_error']:.4f}")

        self.assertLess(metrics['relative_error'], 1.0)

    def test_cholesky_with_ridge(self):
        """测试 Ridge regression"""
        X_np = np.random.randn(100, 20)
        Y_np = np.random.randn(100, 128)

        X_mlx = mx.array(X_np)
        Y_mlx = mx.array(Y_np)

        C2 = compute_C2_cholesky(X_mlx, Y_mlx, ridge_lambda=1e-6)

        # Ridge 应该略微降低拟合（但更稳定）
        metrics = evaluate_C2_quality(X_mlx, Y_mlx, C2)
        print(f"cholesky+ridge: relative_error = {metrics['relative_error']:.4f}")

        # 拟合应该仍然不错
        self.assertLess(metrics['relative_error'], 1.0)

    def test_pinv(self):
        """测试伪逆"""
        X_np = np.random.randn(100, 20)
        Y_np = np.random.randn(100, 128)

        X_mlx = mx.array(X_np)
        Y_mlx = mx.array(Y_np)

        C2 = compute_C2_pinv(X_mlx, Y_mlx)

        # 验证形状
        self.assertEqual(C2.shape, (20, 128))

        # 验证拟合质量
        metrics = evaluate_C2_quality(X_mlx, Y_mlx, C2)
        print(f"pinv: relative_error = {metrics['relative_error']:.4f}")

        self.assertLess(metrics['relative_error'], 1.0)

    def test_methods_consistency(self):
        """验证三种方法的一致性"""
        X_np = np.random.randn(100, 20)
        Y_np = np.random.randn(100, 128)

        X_mlx = mx.array(X_np)
        Y_mlx = mx.array(Y_np)

        C2_lstsq = compute_C2_lstsq(X_mlx, Y_mlx)
        C2_cholesky = compute_C2_cholesky(X_mlx, Y_mlx, ridge_lambda=0.0)
        C2_pinv = compute_C2_pinv(X_mlx, Y_mlx)

        # 三种方法应该给出相似结果（数值误差内）
        diff_lstsq_cholesky = mx.linalg.norm(C2_lstsq - C2_cholesky) / mx.linalg.norm(C2_lstsq)
        diff_lstsq_pinv = mx.linalg.norm(C2_lstsq - C2_pinv) / mx.linalg.norm(C2_lstsq)

        print(f"lstsq vs cholesky: {float(diff_lstsq_cholesky):.4f}")
        print(f"lstsq vs pinv: {float(diff_lstsq_pinv):.4f}")

        # 允许 5% 数值误差
        self.assertLess(float(diff_lstsq_cholesky), 0.05)
        self.assertLess(float(diff_lstsq_pinv), 0.05)

    def test_auto_selector(self):
        """测试自动选择器"""
        X_np = np.random.randn(50, 10)
        Y_np = np.random.randn(50, 64)

        X_mlx = mx.array(X_np)
        Y_mlx = mx.array(Y_np)

        C2_lstsq = compute_C2_auto(X_mlx, Y_mlx, method='lstsq')
        C2_cholesky = compute_C2_auto(X_mlx, Y_mlx, method='cholesky')
        C2_pinv = compute_C2_auto(X_mlx, Y_mlx, method='pinv')

        # 验证所有方法都能工作
        for C2 in [C2_lstsq, C2_cholesky, C2_pinv]:
            self.assertEqual(C2.shape, (10, 64))


if __name__ == '__main__':
    unittest.main(verbosity=2)
