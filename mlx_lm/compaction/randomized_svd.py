"""
Randomized SVD for GPU acceleration

Implements randomized SVD that runs mostly on GPU, avoiding the CPU bottleneck
of traditional SVD in MLX.

Reference: "Finding Structure with Randomness" (Halko et al., 2011)

Author: FlashMLX Research
Date: 2026-03-21
"""

import mlx.core as mx
from typing import Tuple


def randomized_svd(
    A: mx.array,
    rank: int,
    n_oversamples: int = 10,
    n_iter: int = 2,
    random_state: int = 0
) -> Tuple[mx.array, mx.array, mx.array]:
    """
    Compute approximate SVD using randomized algorithm.

    Most computations run on GPU (matmul, QR), only final small SVD on CPU.

    Args:
        A: Input matrix (m, n)
        rank: Target rank (number of singular values to keep)
        n_oversamples: Additional samples for accuracy (default: 10)
        n_iter: Number of power iterations for accuracy (default: 2)
        random_state: Random seed

    Returns:
        U: (m, rank) left singular vectors
        S: (rank,) singular values
        Vt: (rank, n) right singular vectors
    """
    m, n = A.shape
    k = min(rank + n_oversamples, min(m, n))

    # Step 1: Random projection (GPU)
    mx.random.seed(random_state)
    Omega = mx.random.normal(shape=(n, k))

    # Step 2: Compute Y = A @ Omega (GPU matmul)
    Y = mx.matmul(A, Omega)

    # Step 3: Power iteration for better accuracy (optional, GPU)
    for _ in range(n_iter):
        Y = mx.matmul(A, mx.matmul(A.T, Y))

    # Step 4: QR decomposition (CPU - MLX doesn't support GPU QR yet)
    with mx.stream(mx.cpu):
        Y_cpu = Y.astype(mx.float32)
        Q, _ = mx.linalg.qr(Y_cpu)

    # Step 5: Project A onto Q's span: B = Q^T @ A (GPU matmul)
    B = mx.matmul(Q.T, A)  # Shape: (k, n)

    # Step 6: SVD on small matrix B (CPU, but B is much smaller!)
    with mx.stream(mx.cpu):
        # Convert to float32 for SVD
        B_f32 = B.astype(mx.float32)
        Ub, S, Vt = mx.linalg.svd(B_f32)

    # Step 7: Reconstruct U = Q @ Ub (GPU matmul)
    U = mx.matmul(Q, Ub[:, :rank])

    # Keep only top-rank components
    S = S[:rank]
    Vt = Vt[:rank, :]

    return U, S, Vt


def test_randomized_svd():
    """Test randomized SVD vs exact SVD"""
    import time

    # Test matrix (similar to SSM state slice)
    m, n = 128, 192
    A = mx.random.normal(shape=(m, n)).astype(mx.bfloat16)

    rank = 32

    print("Testing Randomized SVD vs Exact SVD")
    print(f"Matrix shape: {A.shape}")
    print(f"Rank: {rank}")
    print()

    # Test 1: Exact SVD (CPU only)
    print("Exact SVD (CPU):")
    start = time.time()
    with mx.stream(mx.cpu):
        A_f32 = A.astype(mx.float32)
        U_exact, S_exact, Vt_exact = mx.linalg.svd(A_f32)
        U_exact = U_exact[:, :rank]
        S_exact = S_exact[:rank]
        Vt_exact = Vt_exact[:rank, :]
    mx.eval(U_exact, S_exact, Vt_exact)
    exact_time = time.time() - start
    print(f"  Time: {exact_time*1000:.2f}ms")
    print()

    # Test 2: Randomized SVD (mostly GPU)
    print("Randomized SVD (GPU):")
    start = time.time()
    U_rand, S_rand, Vt_rand = randomized_svd(A.astype(mx.float32), rank=rank)
    mx.eval(U_rand, S_rand, Vt_rand)
    rand_time = time.time() - start
    print(f"  Time: {rand_time*1000:.2f}ms")
    print(f"  Speedup: {exact_time/rand_time:.2f}x")
    print()

    # Test 3: Accuracy check
    print("Accuracy check:")

    # Reconstruct matrices
    A_exact_recon = U_exact @ mx.diag(S_exact) @ Vt_exact
    A_rand_recon = U_rand @ mx.diag(S_rand) @ Vt_rand

    # Compute reconstruction errors
    error_exact = mx.linalg.norm(A.astype(mx.float32) - A_exact_recon) / mx.linalg.norm(A.astype(mx.float32))
    error_rand = mx.linalg.norm(A.astype(mx.float32) - A_rand_recon) / mx.linalg.norm(A.astype(mx.float32))

    print(f"  Exact SVD error: {error_exact.item():.6f}")
    print(f"  Randomized SVD error: {error_rand.item():.6f}")
    print()

    # Compare singular values
    S_diff = mx.abs(S_exact - S_rand).mean()
    print(f"  Singular values diff (mean): {S_diff.item():.6f}")
    print()

    if rand_time < exact_time and error_rand.item() < 0.01:
        print("✅ Randomized SVD is faster and accurate!")
    elif rand_time < exact_time:
        print("⚠️  Randomized SVD is faster but less accurate")
    else:
        print("❌ Randomized SVD is not faster")


if __name__ == "__main__":
    test_randomized_svd()
