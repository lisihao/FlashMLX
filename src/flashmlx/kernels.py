"""
FlashMLX Kernels - Custom Metal kernel wrappers
"""

from typing import Optional
import mlx.core as mx


def optimized_gemv(
    weight: mx.array,
    input: mx.array,
    bias: Optional[mx.array] = None,
) -> mx.array:
    """
    Optimized GEMV (matrix-vector multiplication)

    Args:
        weight: Weight matrix [out_features, in_features]
        input: Input vector [in_features]
        bias: Optional bias [out_features]

    Returns:
        Output [out_features]
    """
    # TODO: Implement optimized GEMV kernel
    # Currently falls back to MLX's native implementation
    output = weight @ input
    if bias is not None:
        output = output + bias
    return output


def optimized_matmul(
    a: mx.array,
    b: mx.array,
) -> mx.array:
    """
    Optimized matrix multiplication

    Args:
        a: First matrix
        b: Second matrix

    Returns:
        Product matrix
    """
    # TODO: Implement optimized MatMul kernel
    # Currently falls back to MLX's native implementation
    return a @ b
