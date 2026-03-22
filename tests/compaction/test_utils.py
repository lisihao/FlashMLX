"""
单元测试：MLX 缺失函数

验证 MLX 实现与 NumPy/PyTorch 参考实现对齐。
"""

import unittest
import numpy as np
import mlx.core as mx

import sys
import os

# Add mlx-lm-source to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'mlx-lm-source')))

from mlx_lm.compaction.solvers import (
    cholesky_solve_mlx,
    spectral_norm_mlx,
    spectral_norm_squared_mlx,
    lstsq_mlx,
    clip_mlx,
    safe_softmax,
)


class TestCholeskySolve(unittest.TestCase):
    """测试 Cholesky solve"""

    def test_basic(self):
        """基本功能测试"""
        # Create positive definite matrix
        A_np = np.random.rand(50, 50)
        A_np = A_np @ A_np.T + np.eye(50) * 0.1  # Ensure positive definite
        b_np = np.random.rand(50)

        # NumPy reference
        x_np = np.linalg.solve(A_np, b_np)

        # MLX implementation
        L_np = np.linalg.cholesky(A_np)
        L_mlx = mx.array(L_np)
        b_mlx = mx.array(b_np)
        x_mlx = cholesky_solve_mlx(L_mlx, b_mlx)

        # Compare
        error = np.linalg.norm(x_np - np.array(x_mlx)) / np.linalg.norm(x_np)
        print(f"cholesky_solve relative error: {error:.2e}")
        self.assertLess(error, 1e-5, "Cholesky solve error too large")

    def test_multiple_rhs(self):
        """测试多个右侧向量（矩阵形式）"""
        A_np = np.random.rand(30, 30)
        A_np = A_np @ A_np.T + np.eye(30) * 0.1
        B_np = np.random.rand(30, 5)

        # NumPy reference
        X_np = np.linalg.solve(A_np, B_np)

        # MLX implementation
        L_np = np.linalg.cholesky(A_np)
        L_mlx = mx.array(L_np)
        B_mlx = mx.array(B_np)
        X_mlx = cholesky_solve_mlx(L_mlx, B_mlx)

        # Compare
        error = np.linalg.norm(X_np - np.array(X_mlx)) / np.linalg.norm(X_np)
        print(f"cholesky_solve (multiple RHS) relative error: {error:.2e}")
        self.assertLess(error, 1e-5)


class TestSpectralNorm(unittest.TestCase):
    """测试 Spectral Norm 估计"""

    def test_basic(self):
        """基本功能测试"""
        M_np = np.random.randn(100, 50)

        # NumPy reference (via SVD)
        _, s, _ = np.linalg.svd(M_np, full_matrices=False)
        sigma_true = s[0]

        # MLX implementation
        M_mlx = mx.array(M_np)
        sigma_mlx = float(spectral_norm_mlx(M_mlx, n_iters=10))

        # Compare
        error = abs(sigma_true - sigma_mlx) / sigma_true
        print(f"spectral_norm relative error: {error:.2e}")
        # Power iteration 是近似算法，5% 误差可接受
        self.assertLess(error, 0.05, "Spectral norm error > 5%")

    def test_squared(self):
        """测试 spectral_norm_squared"""
        M_np = np.random.randn(80, 40)

        _, s, _ = np.linalg.svd(M_np, full_matrices=False)
        sigma_squared_true = s[0] ** 2

        M_mlx = mx.array(M_np)
        sigma_squared_mlx = float(spectral_norm_squared_mlx(M_mlx, n_iters=10))

        error = abs(sigma_squared_true - sigma_squared_mlx) / sigma_squared_true
        print(f"spectral_norm_squared relative error: {error:.2e}")
        self.assertLess(error, 0.01)


class TestLstsq(unittest.TestCase):
    """测试 Least Squares"""

    def test_basic(self):
        """基本功能测试"""
        A_np = np.random.randn(100, 50)
        b_np = np.random.randn(100)

        # NumPy reference
        x_np = np.linalg.lstsq(A_np, b_np, rcond=None)[0]

        # MLX implementation
        A_mlx = mx.array(A_np)
        b_mlx = mx.array(b_np)
        x_mlx = lstsq_mlx(A_mlx, b_mlx)

        # Compare
        error = np.linalg.norm(x_np - np.array(x_mlx)) / np.linalg.norm(x_np)
        print(f"lstsq relative error: {error:.2e}")
        self.assertLess(error, 1e-5)


class TestClip(unittest.TestCase):
    """测试 Clip"""

    def test_basic(self):
        """基本功能测试"""
        x_np = np.random.randn(100)

        # NumPy reference
        clipped_np = np.clip(x_np, 0.0, 1.0)

        # MLX implementation
        x_mlx = mx.array(x_np)
        clipped_mlx = clip_mlx(x_mlx, 0.0, 1.0)

        # Compare
        np.testing.assert_allclose(clipped_np, np.array(clipped_mlx), rtol=1e-6)

    def test_min_only(self):
        """测试只有 min_val"""
        x_np = np.random.randn(100)
        clipped_np = np.maximum(x_np, 0.0)

        x_mlx = mx.array(x_np)
        clipped_mlx = clip_mlx(x_mlx, 0.0, None)

        np.testing.assert_allclose(clipped_np, np.array(clipped_mlx), rtol=1e-6)


class TestSafeSoftmax(unittest.TestCase):
    """测试数值稳定的 Softmax"""

    def test_basic(self):
        """基本功能测试"""
        x_np = np.random.randn(10, 20)

        # NumPy reference
        exp_x = np.exp(x_np - np.max(x_np, axis=1, keepdims=True))
        softmax_np = exp_x / np.sum(exp_x, axis=1, keepdims=True)

        # MLX implementation
        x_mlx = mx.array(x_np)
        softmax_mlx = safe_softmax(x_mlx, axis=1)

        # Compare
        np.testing.assert_allclose(softmax_np, np.array(softmax_mlx), rtol=1e-5)

    def test_large_values(self):
        """测试大数值稳定性"""
        x_np = np.array([[1000.0, 1001.0], [0.0, 1.0]])

        # NumPy with stability trick
        exp_x = np.exp(x_np - np.max(x_np, axis=1, keepdims=True))
        softmax_np = exp_x / np.sum(exp_x, axis=1, keepdims=True)

        # MLX implementation
        x_mlx = mx.array(x_np)
        softmax_mlx = safe_softmax(x_mlx, axis=1)

        # Should not produce NaN or Inf
        self.assertFalse(np.any(np.isnan(np.array(softmax_mlx))))
        self.assertFalse(np.any(np.isinf(np.array(softmax_mlx))))

        # Compare
        np.testing.assert_allclose(softmax_np, np.array(softmax_mlx), rtol=1e-5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
