"""
Non-Negative Least Squares (NNLS) Solvers

用于拟合 attention bias (beta) 的约束优化求解器。

Problem: min ||Mx - y||^2, s.t. lower_bound <= x <= upper_bound

实现两种方法：
1. Clamped Least Squares - 快速近似
2. Projected Gradient Descent - 迭代优化
"""

import mlx.core as mx
from .utils import lstsq_mlx, spectral_norm_squared_mlx, clip_mlx


def nnls_clamped(
    M: mx.array,
    y: mx.array,
    lower_bound: float = 1e-12,
    upper_bound: float = None
) -> mx.array:
    """
    Clamped Least Squares - 快速NNLS近似

    算法：
    1. 无约束最小二乘: x = lstsq(M, y)
    2. 投影到约束: x = clip(x, lower_bound, upper_bound)

    Parameters
    ----------
    M : mx.array, shape (n, m)
        系数矩阵
    y : mx.array, shape (n,)
        目标向量
    lower_bound : float, default=1e-12
        下界约束
    upper_bound : float, optional
        上界约束

    Returns
    -------
    x : mx.array, shape (m,)
        解向量

    Time Complexity
    ---------------
    O(nm^2) - QR 分解

    Examples
    --------
    >>> M = mx.random.normal((50, 20))
    >>> y = mx.random.normal((50,))
    >>> x = nnls_clamped(M, y, lower_bound=0.0)
    >>> # Verify: ||Mx - y|| is minimized and x >= 0
    """
    # 无约束最小二乘
    x = lstsq_mlx(M, y)

    # 投影到约束
    x = clip_mlx(x, lower_bound, upper_bound)

    return x


def nnls_pgd(
    M: mx.array,
    y: mx.array,
    lower_bound: float = 1e-12,
    upper_bound: float = None,
    max_iters: int = 100,
    tol: float = 1e-6,
    verbose: bool = False
) -> mx.array:
    """
    Projected Gradient Descent - 迭代NNLS优化

    算法：
    1. 初始解: x = clamped_lstsq(M, y)
    2. 迭代:
        - 计算梯度: grad = M^T (Mx - y)
        - 梯度下降: x = x - eta * grad
        - 投影: x = clip(x, lower_bound, upper_bound)
    3. 收敛判据: ||grad|| < tol

    Parameters
    ----------
    M : mx.array, shape (n, m)
        系数矩阵
    y : mx.array, shape (n,)
        目标向量
    lower_bound : float, default=1e-12
        下界约束
    upper_bound : float, optional
        上界约束
    max_iters : int, default=100
        最大迭代次数
    tol : float, default=1e-6
        收敛阈值（梯度范数）
    verbose : bool, default=False
        是否打印调试信息

    Returns
    -------
    x : mx.array, shape (m,)
        解向量

    Time Complexity
    ---------------
    O(nm * max_iters) - 梯度计算主导

    References
    ----------
    - Projected Gradient Descent for Constrained Optimization
    - Lipschitz constant: L = ||M||_2^2

    Examples
    --------
    >>> M = mx.random.normal((50, 20))
    >>> y = mx.random.normal((50,))
    >>> x = nnls_pgd(M, y, max_iters=100, verbose=True)
    """
    # 初始解：Clamped LS
    x = nnls_clamped(M, y, lower_bound, upper_bound)

    # 计算步长：eta = 1 / L, L = ||M||_2^2
    L = spectral_norm_squared_mlx(M, n_iters=10)
    eta = 1.0 / (L + 1e-12)  # 避免除零

    if verbose:
        print(f"NNLS PGD: L = {float(L):.4f}, eta = {float(eta):.6f}")

    # 投影梯度下降
    for iter in range(max_iters):
        # 计算梯度: grad = M^T (Mx - y)
        Mx = M @ x
        residual = Mx - y
        grad = M.T @ residual

        # 梯度范数（收敛检查）
        grad_norm = mx.linalg.norm(grad)

        if verbose and iter % 20 == 0:
            loss = mx.sum(residual ** 2)
            print(f"  Iter {iter}: loss = {float(loss):.6f}, ||grad|| = {float(grad_norm):.6f}")

        # 收敛检查
        if grad_norm < tol:
            if verbose:
                print(f"  Converged at iter {iter}")
            break

        # 梯度下降
        x = x - eta * grad

        # 投影到约束
        x = clip_mlx(x, lower_bound, upper_bound)

    return x


def nnls_auto(
    M: mx.array,
    y: mx.array,
    lower_bound: float = 1e-12,
    upper_bound: float = None,
    quality: str = 'fast'
) -> mx.array:
    """
    自动选择NNLS求解器

    Parameters
    ----------
    M : mx.array, shape (n, m)
    y : mx.array, shape (n,)
    lower_bound : float
    upper_bound : float, optional
    quality : str, {'fast', 'medium', 'high'}
        质量档位:
        - 'fast': Clamped LS（~2ms）
        - 'medium': PGD 20 iters（~20ms）
        - 'high': PGD 100 iters（~100ms）

    Returns
    -------
    x : mx.array, shape (m,)

    Examples
    --------
    >>> M = mx.random.normal((50, 20))
    >>> y = mx.random.normal((50,))
    >>> x = nnls_auto(M, y, quality='fast')
    """
    if quality == 'fast':
        return nnls_clamped(M, y, lower_bound, upper_bound)
    elif quality == 'medium':
        return nnls_pgd(M, y, lower_bound, upper_bound, max_iters=20)
    elif quality == 'high':
        return nnls_pgd(M, y, lower_bound, upper_bound, max_iters=100)
    else:
        raise ValueError(f"Unknown quality: {quality}")
