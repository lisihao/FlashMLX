"""
MLX 数学工具函数

提供 PyTorch 中存在但 MLX 中缺失的数学函数实现。
"""

import mlx.core as mx
import numpy as np


def cholesky_solve_mlx(L: mx.array, b: mx.array) -> mx.array:
    """
    解方程 L L^T x = b，其中 L 是 Cholesky 分解的下三角矩阵

    PyTorch 有 torch.cholesky_solve()，MLX 需要手动实现。

    Parameters
    ----------
    L : mx.array, shape (n, n)
        Cholesky 分解的下三角矩阵
    b : mx.array, shape (n,) or (n, k)
        右侧向量或矩阵

    Returns
    -------
    x : mx.array, shape (n,) or (n, k)
        解向量或矩阵

    Algorithm
    ---------
    1. Forward substitution: L y = b  → y = L^{-1} b
    2. Backward substitution: L^T x = y  → x = (L^T)^{-1} y

    Note: mx.linalg.solve 目前不支持 GPU，需要在 CPU 上运行。

    Examples
    --------
    >>> A = mx.array([[4., 2.], [2., 3.]])
    >>> L = mx.linalg.cholesky(A)
    >>> b = mx.array([1., 2.])
    >>> x = cholesky_solve_mlx(L, b)
    >>> # Verify: A @ x ≈ b
    """
    # MLX solve 需要在 CPU 上运行
    # Forward substitution: L y = b
    with mx.stream(mx.cpu):
        y = mx.linalg.solve(L, b)
        # Backward substitution: L^T x = y
        x = mx.linalg.solve(L.T, y)

    return x


def spectral_norm_mlx(M: mx.array, n_iters: int = 10) -> mx.array:
    """
    估计矩阵的 spectral norm (最大奇异值) ||M||_2

    PyTorch 有 torch.linalg.matrix_norm(M, ord=2)，MLX 需要手动实现。
    使用 Power Iteration 方法快速估计。

    Parameters
    ----------
    M : mx.array, shape (m, n)
        输入矩阵
    n_iters : int, default=3
        Power iteration 迭代次数，越多越精确

    Returns
    -------
    sigma : mx.array, scalar
        估计的最大奇异值

    Algorithm
    ---------
    Power iteration for M^T M:
    1. 随机初始化 u
    2. For k iterations:
        v = M @ u / ||M @ u||
        u = M^T @ v / ||M^T @ v||
    3. sigma = sqrt(u^T M^T M u)

    Examples
    --------
    >>> M = mx.random.normal((100, 50))
    >>> sigma = spectral_norm_mlx(M)
    >>> # Compare with SVD: sigma ≈ mx.linalg.svd(M)[1][0]
    """
    m, n = M.shape

    # 随机初始化 u
    u = mx.random.normal((n,))
    u = u / (mx.linalg.norm(u) + 1e-12)

    # Power iteration
    for _ in range(n_iters):
        # v = M @ u / ||M @ u||
        v = M @ u
        v = v / (mx.linalg.norm(v) + 1e-12)

        # u = M^T @ v / ||M^T @ v||
        u = M.T @ v
        u = u / (mx.linalg.norm(u) + 1e-12)

    # Rayleigh quotient: sigma^2 = u^T (M^T M) u
    MtMu = M.T @ (M @ u)
    sigma_squared = mx.sum(u * MtMu)
    sigma = mx.sqrt(mx.maximum(sigma_squared, 0.0))  # Ensure non-negative

    return sigma


def spectral_norm_squared_mlx(M: mx.array, n_iters: int = 10) -> mx.array:
    """
    估计 ||M||_2^2，避免 sqrt 计算

    用于 NNLS 中计算 Lipschitz constant: L = ||M||_2^2

    Parameters
    ----------
    M : mx.array, shape (m, n)
        输入矩阵
    n_iters : int, default=3
        Power iteration 迭代次数

    Returns
    -------
    sigma_squared : mx.array, scalar
        ||M||_2^2
    """
    m, n = M.shape

    u = mx.random.normal((n,))
    u = u / (mx.linalg.norm(u) + 1e-12)

    for _ in range(n_iters):
        v = M @ u
        v = v / (mx.linalg.norm(v) + 1e-12)
        u = M.T @ v
        u = u / (mx.linalg.norm(u) + 1e-12)

    MtMu = M.T @ (M @ u)
    sigma_squared = mx.sum(u * MtMu)

    return mx.maximum(sigma_squared, 0.0)


# ==================== API Wrappers ====================

def lstsq_mlx(A: mx.array, b: mx.array) -> mx.array:
    """
    torch.linalg.lstsq 兼容 wrapper

    MLX 没有 lstsq，需要根据系统类型选择不同方法：
    - Overdetermined (m > n): QR 分解
    - Underdetermined (m <= n): 最小范数解 A^T (A A^T)^{-1} b

    Parameters
    ----------
    A : mx.array, shape (m, n)
    b : mx.array, shape (m,) or (m, k)

    Returns
    -------
    solution : mx.array, shape (n,) or (n, k)
    """
    m, n = A.shape

    # Convert to float32 if needed (pinv doesn't support bfloat16)
    original_dtype = A.dtype
    if A.dtype == mx.bfloat16:
        A = A.astype(mx.float32)
        b = b.astype(mx.float32)

    # 在 CPU 上运行
    try:
        with mx.stream(mx.cpu):
            if m >= n:
                # Overdetermined system: use QR decomposition
                # A = Q R, A x = b → R x = Q^T b
                Q, R = mx.linalg.qr(A)
                Qtb = Q.T @ b

                # Add regularization to R diagonal for numerical stability
                R_diag = mx.diagonal(R)
                max_diag = float(mx.max(mx.abs(R_diag)))
                regularization = max_diag * 1e-8
                R = R + mx.eye(n) * regularization

                x = mx.linalg.solve(R, Qtb)
            else:
                # Underdetermined system: use minimum norm solution
                # x = A^T (A A^T)^{-1} b
                AAt = A @ A.T  # (m, m)
                # Adaptive regularization
                max_val = float(mx.max(mx.abs(AAt)))
                regularization = max_val * 1e-6
                AAt = AAt + mx.eye(m) * regularization
                # Solve (A A^T) z = b
                z = mx.linalg.solve(AAt, b)
                # x = A^T z
                x = A.T @ z
    except Exception as e:
        # If solve fails, fall back to pinv
        import warnings
        warnings.warn(f"lstsq_mlx failed ({str(e)}), falling back to pinv")
        with mx.stream(mx.cpu):
            A_pinv = mx.linalg.pinv(A)
            x = A_pinv @ b

    # Convert back to original dtype if needed
    if original_dtype == mx.bfloat16:
        x = x.astype(original_dtype)

    return x


def softmax_mlx(x: mx.array, axis: int) -> mx.array:
    """
    torch.softmax 兼容 wrapper

    PyTorch 使用 dim，MLX 使用 axis

    Parameters
    ----------
    x : mx.array
    axis : int
        Softmax dimension

    Returns
    -------
    out : mx.array
    """
    return mx.softmax(x, axis=axis)


def clip_mlx(x: mx.array, min_val: float, max_val: float = None) -> mx.array:
    """
    torch.clamp 兼容 wrapper

    Parameters
    ----------
    x : mx.array
    min_val : float
    max_val : float, optional

    Returns
    -------
    out : mx.array
    """
    if max_val is None:
        return mx.maximum(x, min_val)
    else:
        return mx.clip(x, min_val, max_val)


def safe_softmax(scores: mx.array, axis: int = -1) -> mx.array:
    """
    数值稳定的 softmax（log-sum-exp trick）

    避免 exp(large_number) overflow

    Parameters
    ----------
    scores : mx.array
    axis : int
        Softmax dimension

    Returns
    -------
    out : mx.array
    """
    max_scores = mx.max(scores, axis=axis, keepdims=True)
    exp_scores = mx.exp(scores - max_scores)
    return exp_scores / mx.sum(exp_scores, axis=axis, keepdims=True)


def compute_in_fp32(func):
    """
    Decorator: 强制在 fp32 下计算，然后转回原始 dtype

    用于数值敏感的数学计算（NNLS, LSQ）

    Examples
    --------
    @compute_in_fp32
    def my_solver(A, b):
        return mx.linalg.lstsq(A, b)[0]
    """
    def wrapper(*args, **kwargs):
        # Upcast to fp32
        original_dtype = None
        args_fp32 = []
        for arg in args:
            if isinstance(arg, mx.array):
                if original_dtype is None:
                    original_dtype = arg.dtype
                args_fp32.append(arg.astype(mx.float32))
            else:
                args_fp32.append(arg)

        # Compute
        result = func(*args_fp32, **kwargs)

        # Downcast to original dtype
        if isinstance(result, mx.array) and original_dtype is not None and original_dtype != mx.float32:
            result = result.astype(original_dtype)

        return result
    return wrapper
