"""
SharedH0Transport — POSIX shared memory transport for h^(0) residual stream.

Enables two independent model instances to communicate via the initial
residual h^(0) = embed_tokens(input_tokens), avoiding the need to transfer
full KV cache (9× larger for Qwen3-8B, 24× for 1T MoE).

Shared memory layout (64-byte header + payload):

  Offset  Size   Field              Description
  0       4      state_flag         uint32: 0=empty, 1=writing, 2=ready, 3=read, 4=error
  4       4      dtype_code         uint32: 0=bf16, 1=f32, 2=f16
  8       4      n_tokens           uint32: number of tokens
  12      4      d_hidden           uint32: hidden dimension
  16      8      data_nbytes        uint64: h^(0) payload size in bytes
  24      4      first_token_id     uint32: first token (sanity check)
  28      4      checksum           uint32: CRC32 of payload
  32      32     reserved           padding to 64-byte alignment
  64      ...    h^(0) payload      raw bytes (bf16 as uint16)

bf16 transfer: numpy doesn't support bfloat16 natively. We use raw bytes
transfer — mx.array bytes are written directly, then reconstructed via
np.frombuffer(dtype=np.uint16) + mx.array view-cast to bfloat16.
This is bit-exact and avoids any dtype conversion overhead.
"""

from __future__ import annotations

import atexit
import struct
import time
import zlib
from multiprocessing import shared_memory
from typing import Optional, Tuple

import numpy as np

try:
    import mlx.core as mx
except ImportError:
    mx = None  # allow import for type checking without MLX

# Header format: 7 uint32 fields + 1 uint64 field, padded to 64 bytes
# state(u32) dtype(u32) n_tokens(u32) d_hidden(u32) data_nbytes(u64) first_token(u32) checksum(u32)
HEADER_FORMAT = "<IIII Q II"
HEADER_SIZE = 64  # padded
HEADER_PACKED_SIZE = struct.calcsize(HEADER_FORMAT)  # 32 bytes, rest is reserved

# State flags
STATE_EMPTY = 0
STATE_WRITING = 1
STATE_READY = 2
STATE_READ = 3
STATE_ERROR = 4

# Dtype codes
DTYPE_BF16 = 0
DTYPE_F32 = 1
DTYPE_F16 = 2

_DTYPE_TO_CODE = {
    "bfloat16": DTYPE_BF16,
    "float32": DTYPE_F32,
    "float16": DTYPE_F16,
}

_CODE_TO_MX_DTYPE = {}
if mx is not None:
    _CODE_TO_MX_DTYPE = {
        DTYPE_BF16: mx.bfloat16,
        DTYPE_F32: mx.float32,
        DTYPE_F16: mx.float16,
    }

_CODE_TO_BYTES_PER_ELEM = {
    DTYPE_BF16: 2,
    DTYPE_F32: 4,
    DTYPE_F16: 2,
}


class SharedH0Transport:
    """POSIX shared memory transport for h^(0) residual stream.

    Usage:
        # Instance A (writer/creator):
        transport = SharedH0Transport(max_tokens=4096, d_hidden=4096, create=True)
        transport.write_h0(h0_array)
        transport.wait_for_read()
        transport.close()

        # Instance B (reader):
        transport = SharedH0Transport(create=False)
        h0 = transport.read_h0(timeout_s=30.0)
        transport.close()
    """

    DEFAULT_SHM_NAME = "flashmlx_h0_bridge"

    def __init__(
        self,
        max_tokens: int = 8192,
        d_hidden: int = 4096,
        create: bool = False,
        shm_name: str = DEFAULT_SHM_NAME,
    ):
        self._create = create
        self._shm_name = shm_name
        self._shm: Optional[shared_memory.SharedMemory] = None

        if create:
            # Calculate max payload size (bf16 = 2 bytes per element)
            # Shape: (1, max_tokens, d_hidden) — batch dim always 1 for this experiment
            max_payload = max_tokens * d_hidden * 4  # use f32 size as upper bound
            total_size = HEADER_SIZE + max_payload

            # Clean up any stale segment with the same name
            try:
                stale = shared_memory.SharedMemory(name=shm_name)
                stale.close()
                stale.unlink()
            except FileNotFoundError:
                pass

            self._shm = shared_memory.SharedMemory(
                name=shm_name, create=True, size=total_size
            )
            # Initialize header to empty
            self._write_header(STATE_EMPTY, 0, 0, 0, 0, 0, 0)
            atexit.register(self._cleanup)
        else:
            # Connect to existing segment
            self._shm = shared_memory.SharedMemory(name=shm_name, create=False)

    def _write_header(
        self,
        state: int,
        dtype_code: int,
        n_tokens: int,
        d_hidden: int,
        data_nbytes: int,
        first_token_id: int,
        checksum: int,
    ) -> None:
        packed = struct.pack(
            HEADER_FORMAT,
            state, dtype_code, n_tokens, d_hidden,
            data_nbytes, first_token_id, checksum,
        )
        self._shm.buf[:HEADER_PACKED_SIZE] = packed

    def _read_header(self) -> Tuple[int, int, int, int, int, int, int]:
        packed = bytes(self._shm.buf[:HEADER_PACKED_SIZE])
        return struct.unpack(HEADER_FORMAT, packed)

    def _get_state(self) -> int:
        return struct.unpack("<I", bytes(self._shm.buf[:4]))[0]

    def _set_state(self, state: int) -> None:
        self._shm.buf[:4] = struct.pack("<I", state)

    def write_h0(self, h0: "mx.array", first_token_id: int = 0) -> int:
        """Write h^(0) to shared memory.

        Args:
            h0: mx.array of shape (B, N, d_hidden) in bfloat16/float32/float16.
            first_token_id: First token ID for sanity checking.

        Returns:
            Number of bytes written (payload only).
        """
        assert mx is not None, "MLX required"
        assert self._create, "Only creator can write"
        assert self._shm is not None

        # Ensure h0 is evaluated
        mx.eval(h0)

        # Get shape — squeeze batch dim if needed
        if h0.ndim == 3:
            assert h0.shape[0] == 1, f"Batch size must be 1, got {h0.shape[0]}"
            n_tokens = h0.shape[1]
            d_hidden = h0.shape[2]
        elif h0.ndim == 2:
            n_tokens = h0.shape[0]
            d_hidden = h0.shape[1]
        else:
            raise ValueError(f"Expected 2D or 3D h0, got {h0.ndim}D")

        # Determine dtype
        if h0.dtype == mx.bfloat16:
            dtype_code = DTYPE_BF16
        elif h0.dtype == mx.float32:
            dtype_code = DTYPE_F32
        elif h0.dtype == mx.float16:
            dtype_code = DTYPE_F16
        else:
            raise ValueError(f"Unsupported dtype: {h0.dtype}. Use bfloat16/float32/float16")
        bpe = _CODE_TO_BYTES_PER_ELEM[dtype_code]

        # Signal writing
        self._set_state(STATE_WRITING)

        # Extract raw bytes — this is the key: no dtype conversion for bf16
        h0_flat = h0.reshape(-1)
        raw_bytes = bytes(h0_flat)
        data_nbytes = len(raw_bytes)

        expected_nbytes = n_tokens * d_hidden * bpe
        assert data_nbytes == expected_nbytes, (
            f"Size mismatch: got {data_nbytes}, expected {expected_nbytes}"
        )

        # Check capacity
        capacity = len(self._shm.buf) - HEADER_SIZE
        if data_nbytes > capacity:
            self._set_state(STATE_ERROR)
            raise ValueError(
                f"h^(0) payload ({data_nbytes} bytes, {n_tokens} tokens) "
                f"exceeds shared memory capacity ({capacity} bytes)"
            )

        # Compute checksum
        checksum = zlib.crc32(raw_bytes) & 0xFFFFFFFF

        # Write payload
        self._shm.buf[HEADER_SIZE:HEADER_SIZE + data_nbytes] = raw_bytes

        # Write header (atomic-ish: state goes to READY last via _set_state)
        self._write_header(
            STATE_WRITING, dtype_code, n_tokens, d_hidden,
            data_nbytes, first_token_id, checksum,
        )
        # Signal ready
        self._set_state(STATE_READY)

        return data_nbytes

    def read_h0(self, timeout_s: float = 30.0) -> "mx.array":
        """Poll and read h^(0) from shared memory.

        Blocks until data is ready or timeout.

        Args:
            timeout_s: Maximum wait time in seconds.

        Returns:
            mx.array of shape (1, N, d_hidden) in original dtype.
        """
        assert mx is not None, "MLX required"
        assert self._shm is not None

        # Poll for READY state
        start = time.monotonic()
        while True:
            state = self._get_state()
            if state == STATE_READY:
                break
            if state == STATE_ERROR:
                raise RuntimeError("Writer signaled error")
            elapsed = time.monotonic() - start
            if elapsed > timeout_s:
                raise TimeoutError(
                    f"Timed out waiting for h^(0) after {timeout_s:.1f}s "
                    f"(state={state})"
                )
            time.sleep(0.001)  # 1ms poll interval

        # Read header
        _, dtype_code, n_tokens, d_hidden, data_nbytes, first_token_id, checksum = (
            self._read_header()
        )

        # Read payload
        raw_bytes = bytes(self._shm.buf[HEADER_SIZE:HEADER_SIZE + data_nbytes])

        # Verify checksum
        actual_checksum = zlib.crc32(raw_bytes) & 0xFFFFFFFF
        if actual_checksum != checksum:
            self._set_state(STATE_ERROR)
            raise RuntimeError(
                f"Checksum mismatch: expected {checksum:#x}, got {actual_checksum:#x}"
            )

        # Signal that we've read the data
        self._set_state(STATE_READ)

        # Reconstruct mx.array
        mx_dtype = _CODE_TO_MX_DTYPE[dtype_code]
        bpe = _CODE_TO_BYTES_PER_ELEM[dtype_code]

        if dtype_code == DTYPE_BF16:
            # bf16: interpret as uint16, create mx.array, view-cast
            np_arr = np.frombuffer(raw_bytes, dtype=np.uint16).reshape(1, n_tokens, d_hidden)
            h0 = mx.array(np_arr)
            h0 = h0.view(mx.bfloat16)
        elif dtype_code == DTYPE_F32:
            np_arr = np.frombuffer(raw_bytes, dtype=np.float32).reshape(1, n_tokens, d_hidden)
            h0 = mx.array(np_arr)
        elif dtype_code == DTYPE_F16:
            np_arr = np.frombuffer(raw_bytes, dtype=np.float16).reshape(1, n_tokens, d_hidden)
            h0 = mx.array(np_arr)
        else:
            raise ValueError(f"Unknown dtype code: {dtype_code}")

        return h0

    def wait_for_read(self, timeout_s: float = 300.0) -> bool:
        """Wait until the reader signals it has read the data.

        Args:
            timeout_s: Maximum wait time.

        Returns:
            True if reader confirmed, False on timeout.
        """
        start = time.monotonic()
        while True:
            state = self._get_state()
            if state == STATE_READ:
                return True
            if state == STATE_ERROR:
                raise RuntimeError("Reader signaled error")
            if time.monotonic() - start > timeout_s:
                return False
            time.sleep(0.01)

    def reset(self) -> None:
        """Reset to empty state for next transfer."""
        assert self._create, "Only creator can reset"
        self._set_state(STATE_EMPTY)

    def _cleanup(self) -> None:
        if self._shm is not None:
            try:
                self._shm.close()
                if self._create:
                    self._shm.unlink()
            except Exception:
                pass
            self._shm = None

    def close(self) -> None:
        """Close the shared memory connection."""
        self._cleanup()

    @property
    def shm_name(self) -> str:
        return self._shm_name

    @staticmethod
    def compute_sizes(n_tokens: int, d_hidden: int, n_layers: int, n_kv_heads: int, head_dim: int) -> dict:
        """Compute h^(0) size vs equivalent KV cache size.

        Returns dict with h0_bytes, kv_bytes, compression_ratio.
        """
        h0_bytes = n_tokens * d_hidden * 2  # bf16
        kv_bytes = 2 * n_layers * n_kv_heads * head_dim * 2 * n_tokens  # K+V, bf16
        return {
            "h0_bytes": h0_bytes,
            "kv_bytes": kv_bytes,
            "compression_ratio": kv_bytes / h0_bytes if h0_bytes > 0 else 0,
            "h0_mb": h0_bytes / (1024 * 1024),
            "kv_mb": kv_bytes / (1024 * 1024),
        }

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


if __name__ == "__main__":
    # Self-test: write and read back a random bf16 array
    import mlx.core as mx

    n_tokens, d_hidden = 1024, 4096
    h0_orig = mx.random.normal((1, n_tokens, d_hidden)).astype(mx.bfloat16)
    mx.eval(h0_orig)

    print(f"Self-test: {n_tokens} tokens, d={d_hidden}, dtype=bf16")

    # Write
    writer = SharedH0Transport(max_tokens=n_tokens, d_hidden=d_hidden, create=True, shm_name="h0_selftest")
    nbytes = writer.write_h0(h0_orig, first_token_id=42)
    print(f"  Written: {nbytes} bytes ({nbytes/1024:.1f} KB)")

    # Read
    reader = SharedH0Transport(create=False, shm_name="h0_selftest")
    h0_read = reader.read_h0(timeout_s=5.0)

    # Verify bit-exact
    diff = mx.abs(h0_orig.astype(mx.float32) - h0_read.astype(mx.float32))
    max_diff = mx.max(diff).item()
    print(f"  Shape: {h0_read.shape}, dtype: {h0_read.dtype}")
    print(f"  Max diff: {max_diff}")
    assert max_diff == 0.0, f"NOT bit-exact! max_diff={max_diff}"
    print("  PASS: bit-exact round-trip")

    # Size comparison
    sizes = SharedH0Transport.compute_sizes(
        n_tokens=n_tokens, d_hidden=d_hidden,
        n_layers=36, n_kv_heads=4, head_dim=128,
    )
    print(f"  h^(0): {sizes['h0_mb']:.2f} MB")
    print(f"  KV:    {sizes['kv_mb']:.2f} MB")
    print(f"  Ratio: {sizes['compression_ratio']:.1f}×")

    reader.close()
    writer.close()
    print("  Cleanup OK")
