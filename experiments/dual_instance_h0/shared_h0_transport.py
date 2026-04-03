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
STATE_APPENDING = 5

# Extended header format (lives in reserved region, bytes 32-63)
# n_existing_tokens(u32), append_count(u32), n_total_chunks(u32),
# n_chunks_ready(u32), chunk_size_tokens(u32)
EXTENDED_FORMAT = "<IIIII"
EXTENDED_OFFSET = HEADER_PACKED_SIZE  # 32
EXTENDED_SIZE = struct.calcsize(EXTENDED_FORMAT)  # 20 bytes

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

    def _write_extended(
        self,
        n_existing_tokens: int = 0,
        append_count: int = 0,
        n_total_chunks: int = 0,
        n_chunks_ready: int = 0,
        chunk_size_tokens: int = 0,
    ) -> None:
        packed = struct.pack(
            EXTENDED_FORMAT,
            n_existing_tokens, append_count, n_total_chunks,
            n_chunks_ready, chunk_size_tokens,
        )
        self._shm.buf[EXTENDED_OFFSET:EXTENDED_OFFSET + EXTENDED_SIZE] = packed

    def _read_extended(self) -> tuple[int, int, int, int, int]:
        packed = bytes(self._shm.buf[EXTENDED_OFFSET:EXTENDED_OFFSET + EXTENDED_SIZE])
        return struct.unpack(EXTENDED_FORMAT, packed)

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

    def write_h0_append(self, h0_new: "mx.array", first_token_id: int = 0) -> int:
        """Append new tokens' h^(0) to existing data in shared memory.

        For incremental/multi-turn use: only the delta is transferred.
        The header is updated with the new total token count.

        Args:
            h0_new: New h^(0) to append, shape (1, N_new, d_hidden) or (N_new, d_hidden).

        Returns:
            Number of new bytes written.
        """
        assert mx is not None, "MLX required"
        assert self._create, "Only creator can write"
        assert self._shm is not None

        mx.eval(h0_new)

        # Get shape
        if h0_new.ndim == 3:
            assert h0_new.shape[0] == 1
            n_new = h0_new.shape[1]
            d_hidden_new = h0_new.shape[2]
        elif h0_new.ndim == 2:
            n_new = h0_new.shape[0]
            d_hidden_new = h0_new.shape[1]
        else:
            raise ValueError(f"Expected 2D or 3D h0_new, got {h0_new.ndim}D")

        # Read current header to get existing state
        _, dtype_code_existing, n_existing, d_hidden_existing, existing_nbytes, _, _ = (
            self._read_header()
        )
        n_existing_before, append_count, _, _, _ = self._read_extended()

        # Validate compatibility
        if n_existing > 0:
            assert d_hidden_new == d_hidden_existing, (
                f"d_hidden mismatch: existing={d_hidden_existing}, new={d_hidden_new}"
            )

        # Determine dtype
        if h0_new.dtype == mx.bfloat16:
            dtype_code = DTYPE_BF16
        elif h0_new.dtype == mx.float32:
            dtype_code = DTYPE_F32
        elif h0_new.dtype == mx.float16:
            dtype_code = DTYPE_F16
        else:
            raise ValueError(f"Unsupported dtype: {h0_new.dtype}")
        bpe = _CODE_TO_BYTES_PER_ELEM[dtype_code]

        # Signal appending
        self._set_state(STATE_APPENDING)

        # Extract raw bytes for new data
        new_flat = h0_new.reshape(-1)
        new_bytes = bytes(new_flat)
        new_nbytes = len(new_bytes)

        # Write new payload after existing data
        write_offset = HEADER_SIZE + existing_nbytes
        capacity = len(self._shm.buf) - write_offset
        if new_nbytes > capacity:
            self._set_state(STATE_ERROR)
            raise ValueError(
                f"Append payload ({new_nbytes} bytes) exceeds remaining capacity ({capacity} bytes)"
            )
        self._shm.buf[write_offset:write_offset + new_nbytes] = new_bytes

        # Update totals
        total_nbytes = existing_nbytes + new_nbytes
        total_tokens = n_existing + n_new
        d_hidden = d_hidden_new if d_hidden_existing == 0 else d_hidden_existing

        # Recompute checksum over entire payload (incremental CRC)
        checksum = zlib.crc32(
            bytes(self._shm.buf[HEADER_SIZE:HEADER_SIZE + total_nbytes])
        ) & 0xFFFFFFFF

        # Write updated header
        self._write_header(
            STATE_APPENDING, dtype_code, total_tokens, d_hidden,
            total_nbytes, first_token_id, checksum,
        )
        # Write extended fields
        self._write_extended(
            n_existing_tokens=n_existing,
            append_count=append_count + 1,
        )
        # Signal ready
        self._set_state(STATE_READY)

        return new_nbytes

    def read_h0_delta(self, n_existing_tokens: int, timeout_s: float = 30.0) -> "mx.array":
        """Read only the NEW tokens' h^(0) (delta since last read).

        Args:
            n_existing_tokens: Number of tokens the reader already has.
            timeout_s: Maximum wait time.

        Returns:
            mx.array of shape (1, N_new, d_hidden) containing only new tokens.
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
                raise TimeoutError(f"Timed out after {timeout_s:.1f}s (state={state})")
            time.sleep(0.001)

        # Read header
        _, dtype_code, n_total, d_hidden, total_nbytes, _, checksum = self._read_header()

        if n_total <= n_existing_tokens:
            # No new data
            self._set_state(STATE_READ)
            return None

        n_new = n_total - n_existing_tokens
        bpe = _CODE_TO_BYTES_PER_ELEM[dtype_code]
        existing_data_bytes = n_existing_tokens * d_hidden * bpe
        new_data_bytes = n_new * d_hidden * bpe

        # Read only the delta
        delta_start = HEADER_SIZE + existing_data_bytes
        raw_bytes = bytes(self._shm.buf[delta_start:delta_start + new_data_bytes])

        # Signal read
        self._set_state(STATE_READ)

        # Reconstruct mx.array
        if dtype_code == DTYPE_BF16:
            np_arr = np.frombuffer(raw_bytes, dtype=np.uint16).reshape(1, n_new, d_hidden)
            h0 = mx.array(np_arr).view(mx.bfloat16)
        elif dtype_code == DTYPE_F32:
            np_arr = np.frombuffer(raw_bytes, dtype=np.float32).reshape(1, n_new, d_hidden)
            h0 = mx.array(np_arr)
        elif dtype_code == DTYPE_F16:
            np_arr = np.frombuffer(raw_bytes, dtype=np.float16).reshape(1, n_new, d_hidden)
            h0 = mx.array(np_arr)
        else:
            raise ValueError(f"Unknown dtype code: {dtype_code}")

        return h0

    def read_h0_no_ack(self, timeout_s: float = 30.0) -> "mx.array":
        """Read h^(0) without acknowledging — for multi-reader fan-out.

        Same as read_h0() but does NOT set STATE_READ, so the data
        stays in READY state for other readers. Used with --fan-out.
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
            time.sleep(0.001)

        # Read header
        _, dtype_code, n_tokens, d_hidden, data_nbytes, first_token_id, checksum = (
            self._read_header()
        )

        # Read payload
        raw_bytes = bytes(self._shm.buf[HEADER_SIZE:HEADER_SIZE + data_nbytes])

        # Verify checksum
        actual_checksum = zlib.crc32(raw_bytes) & 0xFFFFFFFF
        if actual_checksum != checksum:
            raise RuntimeError(
                f"Checksum mismatch: expected {checksum:#x}, got {actual_checksum:#x}"
            )

        # DO NOT set STATE_READ — leave READY for other readers

        # Reconstruct mx.array (same as read_h0)
        if dtype_code == DTYPE_BF16:
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

    def begin_streaming(
        self,
        n_total_tokens: int,
        d_hidden: int,
        dtype_code: int,
        chunk_size: int = 512,
    ) -> None:
        """Initialize header for chunked streaming mode.

        Instance A calls this before writing chunks. Instance B polls
        n_chunks_ready to know when each chunk is available.
        """
        assert self._create, "Only creator can begin streaming"
        n_chunks = (n_total_tokens + chunk_size - 1) // chunk_size
        self._set_state(STATE_WRITING)
        # Write header with totals (payload will be filled chunk by chunk)
        bpe = _CODE_TO_BYTES_PER_ELEM[dtype_code]
        total_nbytes = n_total_tokens * d_hidden * bpe
        self._write_header(
            STATE_WRITING, dtype_code, n_total_tokens, d_hidden,
            total_nbytes, 0, 0,  # checksum computed at end
        )
        self._write_extended(
            n_existing_tokens=0,
            append_count=0,
            n_total_chunks=n_chunks,
            n_chunks_ready=0,
            chunk_size_tokens=chunk_size,
        )

    def write_h0_chunk(self, h0_chunk: "mx.array", chunk_idx: int, is_last: bool = False) -> int:
        """Write one chunk of h^(0) to shared memory at the correct offset.

        Args:
            h0_chunk: Shape (1, chunk_tokens, d_hidden) or (chunk_tokens, d_hidden).
            chunk_idx: 0-based chunk index.
            is_last: If True, finalize the streaming (compute checksum, set READY).

        Returns:
            Number of bytes written for this chunk.
        """
        assert mx is not None
        assert self._create
        mx.eval(h0_chunk)

        if h0_chunk.ndim == 3:
            n_chunk_tokens = h0_chunk.shape[1]
            d_hidden = h0_chunk.shape[2]
        elif h0_chunk.ndim == 2:
            n_chunk_tokens = h0_chunk.shape[0]
            d_hidden = h0_chunk.shape[1]
        else:
            raise ValueError(f"Expected 2D or 3D, got {h0_chunk.ndim}D")

        # Read extended to get chunk_size_tokens
        _, _, n_total_chunks, n_chunks_ready, chunk_size_tokens = self._read_extended()
        _, dtype_code, n_total, _, total_nbytes, _, _ = self._read_header()
        bpe = _CODE_TO_BYTES_PER_ELEM[dtype_code]

        # Compute offset
        offset = HEADER_SIZE + chunk_idx * chunk_size_tokens * d_hidden * bpe
        chunk_bytes = bytes(h0_chunk.reshape(-1))
        self._shm.buf[offset:offset + len(chunk_bytes)] = chunk_bytes

        # Update n_chunks_ready
        new_ready = chunk_idx + 1
        self._write_extended(
            n_existing_tokens=0,
            append_count=0,
            n_total_chunks=n_total_chunks,
            n_chunks_ready=new_ready,
            chunk_size_tokens=chunk_size_tokens,
        )

        if is_last:
            # Compute full checksum
            all_raw = bytes(self._shm.buf[HEADER_SIZE:HEADER_SIZE + total_nbytes])
            checksum = zlib.crc32(all_raw) & 0xFFFFFFFF
            self._write_header(
                STATE_READY, dtype_code, n_total, d_hidden,
                total_nbytes, 0, checksum,
            )

        return len(chunk_bytes)

    def read_h0_streaming(self, timeout_s: float = 30.0):
        """Generator that yields h^(0) chunks as they become available.

        Yields:
            mx.array of shape (1, chunk_tokens, d_hidden) per chunk.
        """
        assert mx is not None
        assert self._shm is not None

        # Wait for header to be initialized (n_total_chunks > 0)
        start = time.monotonic()
        while True:
            _, _, n_total_chunks, _, chunk_size_tokens = self._read_extended()
            if n_total_chunks > 0:
                break
            if time.monotonic() - start > timeout_s:
                raise TimeoutError("Timed out waiting for streaming header")
            time.sleep(0.001)

        _, dtype_code, n_total, d_hidden, total_nbytes, _, _ = self._read_header()
        bpe = _CODE_TO_BYTES_PER_ELEM[dtype_code]

        chunks_read = 0
        while chunks_read < n_total_chunks:
            # Poll for next chunk
            while True:
                _, _, _, n_chunks_ready, _ = self._read_extended()
                if n_chunks_ready > chunks_read:
                    break
                if time.monotonic() - start > timeout_s:
                    raise TimeoutError(
                        f"Timed out waiting for chunk {chunks_read} "
                        f"(ready={n_chunks_ready}/{n_total_chunks})"
                    )
                time.sleep(0.0005)  # 500us poll

            # Read this chunk
            chunk_offset = HEADER_SIZE + chunks_read * chunk_size_tokens * d_hidden * bpe
            # Last chunk may be smaller
            is_last_chunk = (chunks_read == n_total_chunks - 1)
            if is_last_chunk:
                remaining_tokens = n_total - chunks_read * chunk_size_tokens
                chunk_tokens = remaining_tokens
            else:
                chunk_tokens = chunk_size_tokens
            chunk_nbytes = chunk_tokens * d_hidden * bpe
            raw_bytes = bytes(self._shm.buf[chunk_offset:chunk_offset + chunk_nbytes])

            # Reconstruct mx.array
            if dtype_code == DTYPE_BF16:
                np_arr = np.frombuffer(raw_bytes, dtype=np.uint16).reshape(1, chunk_tokens, d_hidden)
                h0_chunk = mx.array(np_arr).view(mx.bfloat16)
            elif dtype_code == DTYPE_F32:
                np_arr = np.frombuffer(raw_bytes, dtype=np.float32).reshape(1, chunk_tokens, d_hidden)
                h0_chunk = mx.array(np_arr)
            elif dtype_code == DTYPE_F16:
                np_arr = np.frombuffer(raw_bytes, dtype=np.float16).reshape(1, chunk_tokens, d_hidden)
                h0_chunk = mx.array(np_arr)
            else:
                raise ValueError(f"Unknown dtype code: {dtype_code}")

            yield h0_chunk
            chunks_read += 1

        # After reading all chunks, ACK
        self._set_state(STATE_READ)

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
