"""
Least Squares (LSQ) Solvers for C2

用于拟合压缩后的 values (C2) 的求解器。

Problem: X @ C2 = Y

其中:
- X: (n_queries, budget) - 压缩后的 attention weights
- Y: (n_queries, head_dim) - 目标 attention outputs
- C2: (budget, head_dim) - 压缩后的 values（待求解）

实现三种方法：
1. Lstsq - 标准最小二乘（QR 分解）
2. Cholesky - Ridge regression（更稳定）
3. Pinv - 伪逆（慢但稳定）
"""

import mlx.core as mx
from .utils import lstsq_mlx, cholesky_solve_mlx


def compute_C2_lstsq(
    X: mx.array,
    Y: mx.array
) -> mx.array:
    """
    标准最小二乘求解 C2

    算法: C2 = lstsq(X, Y)

    Parameters
    ----------
    X : mx.array, shape (n_queries, budget)
        压缩后的 attention weights
    Y : mx.array, shape (n_queries, head_dim)
        目标 attention outputs

    Returns
    -------
    C2 : mx.array, shape (budget, head_dim)
        压缩后的 values

    Time Complexity
    ---------------
    O(n * budget * head_dim) - QR 分解

    Examples
    --------
    >>> X = mx.random.normal((100, 20))
    >>> Y = mx.random.normal((100, 128))
    >>> C2 = compute_C2_lstsq(X, Y)
    >>> # Verify: X @ C2 ≈ Y
    """
    # 对每一列独立求解（head_dim 维度）
    # lstsq_mlx 支持向量右侧，也支持矩阵右侧
    C2 = lstsq_mlx(X, Y)
    return C2


def compute_C2_cholesky(
    X: mx.array,
    Y: mx.array,
    ridge_lambda: float = 0.0
) -> mx.array:
    """
    Cholesky 分解求解 C2（支持 Ridge regression）

    算法:
    1. XtX = X^T @ X + lambda * I  (Ridge regularization)
    2. XtY = X^T @ Y
    3. L = cholesky(XtX)
    4. C2 = cholesky_solve(L, XtY)

    Parameters
    ----------
    X : mx.array, shape (n_queries, budget)
        压缩后的 attention weights
    Y : mx.array, shape (n_queries, head_dim)
        目标 attention outputs
    ridge_lambda : float, default=0.0
        Ridge 正则化系数（防止过拟合）
        如果为 0，会根据条件数自动设置

    Returns
    -------
    C2 : mx.array, shape (budget, head_dim)
        压缩后的 values

    Time Complexity
    ---------------
    O(budget^2 * (n_queries + head_dim)) - 矩阵乘法 + Cholesky

    Notes
    -----
    - Ridge regularization 可以提高数值稳定性
    - 当 X 接近奇异时，ridge_lambda > 0 有帮助
    - 推荐用于 Fast Path（更稳定）

    Examples
    --------
    >>> X = mx.random.normal((100, 20))
    >>> Y = mx.random.normal((100, 128))
    >>> C2 = compute_C2_cholesky(X, Y, ridge_lambda=1e-6)
    """
    n_queries, budget = X.shape
    _, head_dim = Y.shape

    # Compute X^T X and X^T Y
    XtX = X.T @ X  # (budget, budget)
    XtY = X.T @ Y  # (budget, head_dim)

    # Auto-select ridge regularization if not provided
    if ridge_lambda == 0.0:
        # Use condition number to determine regularization
        # For attention weights, typical range is 1e-8 to 1e-4
        max_val = float(mx.max(mx.abs(XtX)))
        ridge_lambda = max_val * 1e-6  # Adaptive regularization

    # Ridge regularization (always apply for numerical stability)
    XtX = XtX + mx.eye(budget) * ridge_lambda

    # Cholesky 分解需要在 CPU 上运行
    try:
        with mx.stream(mx.cpu):
            L = mx.linalg.cholesky(XtX)
        # 求解 L L^T C2 = X^T Y
        C2 = cholesky_solve_mlx(L, XtY)
    except Exception as e:
        # If Cholesky fails, fall back to pinv
        import warnings
        warnings.warn(f"Cholesky failed ({str(e)}), falling back to pinv")
        C2 = compute_C2_pinv(X, Y)

    return C2


def compute_C2_pinv(
    X: mx.array,
    Y: mx.array
) -> mx.array:
    """
    伪逆求解 C2

    算法: C2 = pinv(X) @ Y

    Parameters
    ----------
    X : mx.array, shape (n_queries, budget)
    Y : mx.array, shape (n_queries, head_dim)

    Returns
    -------
    C2 : mx.array, shape (budget, head_dim)

    Time Complexity
    ---------------
    O(n * budget * min(n, budget)) - SVD

    Notes
    -----
    - 最稳定但最慢
    - 适合调试和对照
    - 生产环境推荐 Cholesky

    Examples
    --------
    >>> X = mx.random.normal((100, 20))
    >>> Y = mx.random.normal((100, 128))
    >>> C2 = compute_C2_pinv(X, Y)
    """
    # MLX pinv 实现
    # pinv(X) = V @ diag(1/s) @ U^T，其中 X = U @ diag(s) @ V^T
    with mx.stream(mx.cpu):
        U, s, Vt = mx.linalg.svd(X)  # SVD 需在 CPU 上

    # 计算伪逆: X^+ = V @ diag(1/s) @ U^T
    # V = Vt^T, U = U
    # X (n x m) = U (n x n) @ diag(s) (n x m) @ Vt (m x m)
    # X^+ (m x n) = V (m x m) @ diag(1/s) (m x n) @ U^T (n x n)
    n, m = X.shape
    s_inv = 1.0 / (s + 1e-12)  # 避免除零

    # 构建对角矩阵 diag(1/s): (m, n)
    diag_s_inv = mx.zeros((m, n))
    min_dim = min(m, n)
    for i in range(min_dim):
        diag_s_inv[i, i] = s_inv[i]

    # X_pinv = Vt^T @ diag_s_inv @ U^T
    X_pinv = Vt.T @ diag_s_inv @ U.T

    # C2 = X_pinv @ Y
    C2 = X_pinv @ Y

    return C2


def compute_C2_auto(
    X: mx.array,
    Y: mx.array,
    method: str = 'cholesky',
    ridge_lambda: float = 1e-6
) -> mx.array:
    """
    自动选择 C2 求解器

    Parameters
    ----------
    X : mx.array, shape (n_queries, budget)
    Y : mx.array, shape (n_queries, head_dim)
    method : str, {'lstsq', 'cholesky', 'pinv'}
        求解方法:
        - 'lstsq': 标准最小二乘（~50ms）
        - 'cholesky': Ridge regression（~30ms，推荐）
        - 'pinv': 伪逆（~100ms，最稳定）
    ridge_lambda : float, default=1e-6
        Ridge 正则化系数（仅用于 cholesky）

    Returns
    -------
    C2 : mx.array, shape (budget, head_dim)

    Raises
    ------
    ValueError
        如果 method 不在支持列表中

    Examples
    --------
    >>> X = mx.random.normal((100, 20))
    >>> Y = mx.random.normal((100, 128))
    >>> C2 = compute_C2_auto(X, Y, method='cholesky')
    """
    if method == 'lstsq':
        return compute_C2_lstsq(X, Y)
    elif method == 'cholesky':
        return compute_C2_cholesky(X, Y, ridge_lambda)
    elif method == 'pinv':
        return compute_C2_pinv(X, Y)
    else:
        raise ValueError(f"Unknown method: {method}. Choose from: lstsq, cholesky, pinv")


def evaluate_C2_quality(
    X: mx.array,
    Y: mx.array,
    C2: mx.array
) -> dict:
    """
    评估 C2 的拟合质量

    Parameters
    ----------
    X : mx.array, shape (n_queries, budget)
    Y : mx.array, shape (n_queries, head_dim)
    C2 : mx.array, shape (budget, head_dim)

    Returns
    -------
    metrics : dict
        包含以下指标:
        - 'mse': 均方误差
        - 'relative_error': 相对误差
        - 'max_error': 最大误差

    Examples
    --------
    >>> X = mx.random.normal((100, 20))
    >>> Y = mx.random.normal((100, 128))
    >>> C2 = compute_C2_cholesky(X, Y)
    >>> metrics = evaluate_C2_quality(X, Y, C2)
    >>> print(f"MSE: {metrics['mse']:.4f}")
    """
    # Predict
    Y_pred = X @ C2

    # Compute errors
    diff = Y - Y_pred
    mse = float(mx.mean(diff ** 2))
    relative_error = float(mx.linalg.norm(diff) / mx.linalg.norm(Y))
    max_error = float(mx.max(mx.abs(diff)))

    return {
        'mse': mse,
        'relative_error': relative_error,
        'max_error': max_error
    }
