"""
SSM State Compression

Compression methods specifically designed for State-Memory (SSM/Mamba/Linear Attention)
that do not use Attention Matching.

Author: FlashMLX Research
Date: 2026-03-21
Task: #53 - State-Memory专用压缩算法
"""

import mlx.core as mx
from typing import Dict, Optional, Tuple


class LowRankStateCompressor:
    """
    Low-Rank Approximation for SSM State Compression

    Uses SVD to compress each (Dv, Dk) slice of the state tensor.

    State shape: (B, Hv, Dv, Dk)
    Compressed: {'U': (B, Hv, Dv, rank), 'S': (B, Hv, rank), 'Vt': (B, Hv, rank, Dk)}
    """

    def __init__(self, rank: int = 32):
        """
        Args:
            rank: Number of singular values to keep (compression rank)
        """
        self.rank = rank

    def compress(self, state: mx.array) -> Dict[str, mx.array]:
        """
        Compress SSM state using low-rank SVD approximation

        Args:
            state: (B, Hv, Dv, Dk) SSM state tensor

        Returns:
            compressed: Dict with keys 'U', 'S', 'Vt'
        """
        if state is None:
            return None

        B, Hv, Dv, Dk = state.shape
        original_dtype = state.dtype

        # Storage for compressed components
        U_list, S_list, Vt_list = [], [], []

        for b in range(B):
            U_batch, S_batch, Vt_batch = [], [], []

            for h in range(Hv):
                # Convert to float32 for SVD (MLX requirement)
                state_slice = state[b, h].astype(mx.float32)

                # SVD on (Dv, Dk) slice (must run on CPU)
                with mx.stream(mx.cpu):
                    U, S, Vt = mx.linalg.svd(state_slice)

                # Keep top-rank components
                U_batch.append(U[:, :self.rank])      # (Dv, rank)
                S_batch.append(S[:self.rank])         # (rank,)
                Vt_batch.append(Vt[:self.rank, :])    # (rank, Dk)

            U_list.append(mx.stack(U_batch))
            S_list.append(mx.stack(S_batch))
            Vt_list.append(mx.stack(Vt_batch))

        compressed = {
            'U': mx.stack(U_list),      # (B, Hv, Dv, rank)
            'S': mx.stack(S_list),      # (B, Hv, rank)
            'Vt': mx.stack(Vt_list),    # (B, Hv, rank, Dk)
            'rank': self.rank,
            'original_shape': (B, Hv, Dv, Dk),
            'original_dtype': str(original_dtype)
        }

        return compressed

    def decompress(self, compressed: Dict[str, mx.array]) -> mx.array:
        """
        Reconstruct SSM state from compressed representation

        Args:
            compressed: Dict with keys 'U', 'S', 'Vt' from compress()

        Returns:
            state: (B, Hv, Dv, Dk) reconstructed state
        """
        if compressed is None:
            return None

        U = compressed['U']      # (B, Hv, Dv, rank)
        S = compressed['S']      # (B, Hv, rank)
        Vt = compressed['Vt']    # (B, Hv, rank, Dk)

        B, Hv, Dv, rank = U.shape
        Dk = Vt.shape[-1]

        # Reconstruct per slice
        state_list = []

        for b in range(B):
            state_batch = []

            for h in range(Hv):
                # Reconstruct: state[b,h] = U @ diag(S) @ Vt
                reconstructed = U[b, h] @ mx.diag(S[b, h]) @ Vt[b, h]
                state_batch.append(reconstructed)

            state_list.append(mx.stack(state_batch))

        state = mx.stack(state_list)

        # Convert back to original dtype if specified
        if 'original_dtype' in compressed:
            dtype_str = compressed['original_dtype']
            if 'bfloat16' in dtype_str:
                state = state.astype(mx.bfloat16)
            elif 'float16' in dtype_str:
                state = state.astype(mx.float16)
            # float32 is default, no conversion needed

        return state

    def get_compression_ratio(self, original_shape: Tuple[int, ...]) -> float:
        """
        Calculate compression ratio

        Args:
            original_shape: (B, Hv, Dv, Dk)

        Returns:
            compression_ratio: original_size / compressed_size
        """
        B, Hv, Dv, Dk = original_shape

        original_size = B * Hv * Dv * Dk
        compressed_size = (
            B * Hv * Dv * self.rank +  # U
            B * Hv * self.rank +        # S
            B * Hv * self.rank * Dk     # Vt
        )

        return original_size / compressed_size


class RandomProjectionCompressor:
    """
    Random Projection for SSM State Compression

    Uses a fixed random matrix to project the state to lower dimension.
    Much faster than SVD but slightly less accurate.
    """

    def __init__(self, target_dim: int = 32, seed: int = 0):
        """
        Args:
            target_dim: Target dimension after projection
            seed: Random seed for reproducibility
        """
        self.target_dim = target_dim
        self.seed = seed
        self.projection_matrix = None

    def _init_projection_matrix(self, Dk: int):
        """Initialize Gaussian random projection matrix"""
        if self.projection_matrix is None or self.projection_matrix.shape[0] != Dk:
            mx.random.seed(self.seed)
            # Gaussian random matrix with variance normalization
            self.projection_matrix = mx.random.normal(
                shape=(Dk, self.target_dim),
                scale=1.0 / (self.target_dim ** 0.5)
            )

    def compress(self, state: mx.array) -> Dict[str, mx.array]:
        """
        Compress SSM state using random projection

        Args:
            state: (B, Hv, Dv, Dk)

        Returns:
            compressed: Dict with 'state' and 'projection_matrix'
        """
        if state is None:
            return None

        B, Hv, Dv, Dk = state.shape

        # Initialize projection matrix if needed
        self._init_projection_matrix(Dk)

        # Project: (B, Hv, Dv, Dk) @ (Dk, target_dim) → (B, Hv, Dv, target_dim)
        compressed_state = mx.matmul(state, self.projection_matrix)

        return {
            'state': compressed_state,
            'projection_matrix': self.projection_matrix,
            'original_shape': (B, Hv, Dv, Dk)
        }

    def decompress(self, compressed: Dict[str, mx.array]) -> mx.array:
        """
        Reconstruct SSM state using pseudo-inverse

        Args:
            compressed: Dict from compress()

        Returns:
            state: (B, Hv, Dv, Dk)
        """
        if compressed is None:
            return None

        compressed_state = compressed['state']  # (B, Hv, Dv, target_dim)
        proj_matrix = compressed['projection_matrix']  # (Dk, target_dim)
        original_dtype = compressed_state.dtype

        # Convert to float32 for pinv (MLX requirement)
        proj_matrix_f32 = proj_matrix.astype(mx.float32)

        # Compute pseudo-inverse (must run on CPU)
        with mx.stream(mx.cpu):
            proj_pinv = mx.linalg.pinv(proj_matrix_f32)  # (target_dim, Dk)

        # Reconstruct: (B, Hv, Dv, target_dim) @ (target_dim, Dk) → (B, Hv, Dv, Dk)
        state = mx.matmul(compressed_state.astype(mx.float32), proj_pinv)

        # Convert back to original dtype
        state = state.astype(original_dtype)

        return state

    def get_compression_ratio(self, original_shape: Tuple[int, ...]) -> float:
        """Calculate compression ratio"""
        B, Hv, Dv, Dk = original_shape
        original_size = B * Hv * Dv * Dk
        compressed_size = B * Hv * Dv * self.target_dim
        # Note: projection_matrix is shared and amortized across all compressions
        return original_size / compressed_size


class QuantizationCompressor:
    """
    Quantization-based SSM State Compression

    Quantizes FP16 state to INT8 or INT4.
    Simplest method with lowest compression ratio but very fast.
    """

    def __init__(self, bits: int = 8):
        """
        Args:
            bits: Quantization bits (4 or 8)
        """
        if bits not in [4, 8]:
            raise ValueError(f"bits must be 4 or 8, got {bits}")
        self.bits = bits
        self.qmin = 0
        self.qmax = (1 << bits) - 1  # 15 for 4-bit, 255 for 8-bit

    def compress(self, state: mx.array) -> Dict[str, mx.array]:
        """
        Quantize SSM state

        Args:
            state: (B, Hv, Dv, Dk) in FP16

        Returns:
            compressed: Dict with 'quantized', 'scale', 'zero_point'
        """
        if state is None:
            return None

        B, Hv, Dv, Dk = state.shape

        # Per-head quantization
        state_flat = state.reshape(B, Hv, -1)  # (B, Hv, Dv*Dk)

        # Compute scale and zero_point
        min_val = state_flat.min(axis=-1, keepdims=True)  # (B, Hv, 1)
        max_val = state_flat.max(axis=-1, keepdims=True)

        scale = (max_val - min_val) / (self.qmax - self.qmin)
        zero_point = self.qmin - min_val / scale

        # Quantize
        quantized = mx.clip(
            mx.round((state_flat - min_val) / scale),
            self.qmin,
            self.qmax
        )

        # Cast to uint8 (MLX doesn't have uint4, so we use uint8 for both)
        quantized = quantized.astype(mx.uint8)

        return {
            'quantized': quantized.reshape(B, Hv, Dv, Dk),
            'scale': scale.squeeze(-1),        # (B, Hv)
            'zero_point': zero_point.squeeze(-1),  # (B, Hv)
            'bits': self.bits,
            'original_shape': (B, Hv, Dv, Dk)
        }

    def decompress(self, compressed: Dict[str, mx.array]) -> mx.array:
        """
        Dequantize SSM state

        Args:
            compressed: Dict from compress()

        Returns:
            state: (B, Hv, Dv, Dk) in FP16
        """
        if compressed is None:
            return None

        quantized = compressed['quantized']  # (B, Hv, Dv, Dk) uint8
        scale = compressed['scale'][..., None, None]  # (B, Hv, 1, 1)
        zero_point = compressed['zero_point'][..., None, None]

        # Dequantize
        state = (quantized.astype(mx.float16) - zero_point) * scale

        return state

    def get_compression_ratio(self, original_shape: Tuple[int, ...]) -> float:
        """Calculate compression ratio"""
        # FP16 = 16 bits, compressed = self.bits
        return 16.0 / self.bits  # 2x for INT8, 4x for INT4


def test_compressors():
    """Simple test to verify compressors work"""
    # Simulate Qwen3.5 SSM state
    B, Hv, Dv, Dk = 1, 64, 128, 192
    state = mx.random.normal(shape=(B, Hv, Dv, Dk))

    print("Testing SSM State Compressors")
    print(f"Original state shape: {state.shape}")
    print(f"Original size: {state.size} elements\n")

    # Test Low-Rank
    print("=" * 60)
    print("Method 1: Low-Rank Approximation (rank=32)")
    print("=" * 60)
    lr_compressor = LowRankStateCompressor(rank=32)
    compressed_lr = lr_compressor.compress(state)
    reconstructed_lr = lr_compressor.decompress(compressed_lr)
    error_lr = mx.abs(state - reconstructed_lr).mean()
    ratio_lr = lr_compressor.get_compression_ratio(state.shape)
    print(f"Compression ratio: {ratio_lr:.2f}x")
    print(f"Reconstruction error (mean abs): {error_lr.item():.6f}\n")

    # Test Random Projection
    print("=" * 60)
    print("Method 2: Random Projection (target_dim=32)")
    print("=" * 60)
    rp_compressor = RandomProjectionCompressor(target_dim=32)
    compressed_rp = rp_compressor.compress(state)
    reconstructed_rp = rp_compressor.decompress(compressed_rp)
    error_rp = mx.abs(state - reconstructed_rp).mean()
    ratio_rp = rp_compressor.get_compression_ratio(state.shape)
    print(f"Compression ratio: {ratio_rp:.2f}x")
    print(f"Reconstruction error (mean abs): {error_rp.item():.6f}\n")

    # Test Quantization
    print("=" * 60)
    print("Method 3: Quantization (8-bit)")
    print("=" * 60)
    quant_compressor = QuantizationCompressor(bits=8)
    compressed_quant = quant_compressor.compress(state)
    reconstructed_quant = quant_compressor.decompress(compressed_quant)
    error_quant = mx.abs(state - reconstructed_quant).mean()
    ratio_quant = quant_compressor.get_compression_ratio(state.shape)
    print(f"Compression ratio: {ratio_quant:.2f}x")
    print(f"Reconstruction error (mean abs): {error_quant.item():.6f}\n")

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Method':<30} {'Ratio':<10} {'Error':<15}")
    print("-" * 60)
    print(f"{'Low-Rank (rank=32)':<30} {ratio_lr:<10.2f} {error_lr.item():<15.6f}")
    print(f"{'Random Projection (dim=32)':<30} {ratio_rp:<10.2f} {error_rp.item():<15.6f}")
    print(f"{'Quantization (8-bit)':<30} {ratio_quant:<10.2f} {error_quant.item():<15.6f}")


if __name__ == "__main__":
    test_compressors()
