"""Lightweight KV transform-coding helpers.

This module implements a practical approximation of the paper's KVTC
pipeline:

* fit a shared PCA calibration on representative KV tensors
* compute a shared bit-allocation plan with dynamic programming
* encode coefficients block-by-block with quantization + DEFLATE

The intent is to keep the implementation MLX-friendly and easy to wire into
prompt-cache persistence, while moving the structure closer to the paper.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import zlib
from typing import Iterable, Optional, Sequence

import mlx.core as mx
import numpy as np


@dataclass(frozen=True)
class KVTCCodecConfig:
    """Codec knobs for KVTC calibration and encoding."""

    energy: float = 0.995
    rank: Optional[int] = None
    bits: int = 4
    group_size: int = 64
    sample_limit: int = 4096
    seed: int = 0
    allowed_block_sizes: tuple[int, ...] = (1, 16, 64, 256, 1024)
    allowed_bits: tuple[int, ...] = (0, 2, 4, 8)
    scale_overhead_bits: int = 64
    zero_bit_penalty_bits: int = 8
    zero_bit_energy_fraction: float = 0.015


@dataclass
class KVTCTransformPlan:
    """Shared transform and bit-allocation plan for a single tensor family."""

    mean: np.ndarray
    basis: np.ndarray
    block_meta: np.ndarray
    config: KVTCCodecConfig

    @property
    def state(self):
        return self.mean, self.basis, self.block_meta

    @property
    def meta_state(self):
        return json.dumps(asdict(self.config))

    @classmethod
    def from_state(cls, state, meta_state):
        if isinstance(meta_state, str):
            config = KVTCCodecConfig(**json.loads(meta_state))
        else:
            config = KVTCCodecConfig(**dict(meta_state))
        mean, basis, block_meta = state
        return cls(
            mean=_to_numpy(mean).astype(np.float32, copy=False),
            basis=_to_numpy(basis).astype(np.float32, copy=False),
            block_meta=_to_numpy(block_meta).astype(np.int32, copy=False),
            config=config,
        )

    def encode(self, x: np.ndarray):
        x = _to_numpy(x).astype(np.float32, copy=False)
        coeffs = project(x, self.mean, self.basis)
        payloads = []
        shifts = []
        scales = []
        q_shapes = []

        for start, width, bits in self.block_meta:
            block = coeffs[:, start : start + width]
            if int(bits) == 0:
                payloads.append(np.zeros(1, dtype=np.uint8))
                shifts.append(np.zeros(1, dtype=np.float32))
                scales.append(np.zeros(1, dtype=np.float32))
                q_shapes.append(np.asarray(block.shape, dtype=np.int32))
                continue
            group_size = min(self.config.group_size, int(width))
            payload, block_shifts, block_scales, q_shape = quantize_groups(
                block, int(bits), group_size
            )
            payloads.append(payload)
            shifts.append(block_shifts)
            scales.append(block_scales)
            q_shapes.append(np.asarray(q_shape, dtype=np.int32))

        return (
            tuple(payloads),
            tuple(shifts),
            tuple(scales),
            tuple(q_shapes),
            np.asarray(x.shape, dtype=np.int32),
        )

    def decode(self, encoded):
        if len(encoded) == 4:
            payloads, scales, q_shapes, orig_shape = encoded
            shifts = tuple(np.zeros_like(_to_numpy(scale), dtype=np.float32) for scale in scales)
        else:
            payloads, shifts, scales, q_shapes, orig_shape = encoded
        coeffs = np.zeros((int(orig_shape[0]), self.basis.shape[1]), dtype=np.float32)

        for idx, (start, width, bits) in enumerate(self.block_meta):
            if int(bits) == 0:
                continue
            group_size = min(self.config.group_size, int(width))
            block = dequantize_groups(
                payloads[idx],
                q_shapes[idx],
                shifts[idx],
                scales[idx],
                int(bits),
                group_size,
            )
            coeffs[:, start : start + width] = block

        reconstructed = reconstruct(coeffs, self.mean, self.basis)
        return reconstructed.reshape(tuple(int(v) for v in _to_numpy(orig_shape).tolist()))

    def fingerprint(self) -> str:
        """Stable-ish identifier for sharing calibration across caches."""

        h = hashlib.sha1()
        h.update(np.asarray(self.mean.shape, dtype=np.int32).tobytes())
        h.update(np.asarray(self.basis.shape, dtype=np.int32).tobytes())
        h.update(np.asarray(self.block_meta.shape, dtype=np.int32).tobytes())
        h.update(self.mean.tobytes())
        h.update(self.basis.tobytes())
        h.update(self.block_meta.tobytes())
        h.update(self.meta_state.encode("utf-8"))
        return h.hexdigest()


@dataclass
class KVTCSharedCalibration:
    """Shared key/value calibration used by a prompt-cache group."""

    keys: KVTCTransformPlan
    values: KVTCTransformPlan

    @property
    def state(self):
        return {"keys": self.keys.state, "values": self.values.state}

    @property
    def meta_state(self):
        return {
            "keys": self.keys.meta_state,
            "values": self.values.meta_state,
        }

    @classmethod
    def from_state(cls, state, meta_state):
        return cls(
            keys=KVTCTransformPlan.from_state(state["keys"], meta_state["keys"]),
            values=KVTCTransformPlan.from_state(
                state["values"], meta_state["values"]
            ),
        )

    def fingerprint(self) -> str:
        return hashlib.sha1(
            (self.keys.fingerprint() + "::" + self.values.fingerprint()).encode("utf-8")
        ).hexdigest()

    def encode(self, keys, values):
        return self.keys.encode(keys), self.values.encode(values)

    def decode(self, encoded_keys, encoded_values):
        return self.keys.decode(encoded_keys), self.values.decode(encoded_values)


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    try:
        return np.asarray(x)
    except Exception:
        if hasattr(x, "astype"):
            try:
                return np.asarray(x.astype(mx.float32))
            except Exception:
                pass
        if hasattr(x, "tolist"):
            return np.asarray(x.tolist())
        raise


def _subsample_rows(x: np.ndarray, limit: int, seed: int) -> np.ndarray:
    if x.shape[0] <= limit:
        return x
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(x.shape[0], size=limit, replace=False))
    return x[indices]


def fit_pca_basis(x, config: KVTCCodecConfig):
    """Fit a PCA basis for a 2D array."""

    x = _to_numpy(x).astype(np.float32, copy=False)
    if x.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape {x.shape}")

    x_fit = _subsample_rows(x, config.sample_limit, config.seed)
    mean = x_fit.mean(axis=0, keepdims=True)
    centered = x_fit - mean
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)

    if config.rank is not None:
        rank = max(1, min(config.rank, vt.shape[0]))
    else:
        total = np.sum(singular_values**2)
        if total == 0:
            rank = 1
        else:
            energy = np.cumsum(singular_values**2) / total
            rank = int(np.searchsorted(energy, config.energy)) + 1
            rank = max(1, min(rank, vt.shape[0]))

    basis = vt[:rank].T.astype(np.float32, copy=False)
    return mean.astype(np.float32, copy=False), basis


def project(x, mean: np.ndarray, basis: np.ndarray) -> np.ndarray:
    x = _to_numpy(x).astype(np.float32, copy=False)
    return (x - mean) @ basis


def reconstruct(coeffs, mean: np.ndarray, basis: np.ndarray) -> np.ndarray:
    coeffs = _to_numpy(coeffs).astype(np.float32, copy=False)
    return coeffs @ basis.T + mean


def quantize_groups(x, bits: int, group_size: int):
    """Per-group affine quantization plus DEFLATE compression."""

    if bits < 2 or bits > 8:
        raise ValueError(f"Unsupported bit-width: {bits}")
    x = _to_numpy(x).astype(np.float32, copy=False)
    if x.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape {x.shape}")

    qmax = (1 << (bits - 1)) - 1
    if qmax <= 0:
        raise ValueError(f"Unsupported bit-width: {bits}")

    n_cols = x.shape[1]
    n_groups = (n_cols + group_size - 1) // group_size
    q = np.zeros_like(x, dtype=np.int8)
    shifts = np.zeros(n_groups, dtype=np.float32)
    scales = np.zeros(n_groups, dtype=np.float32)

    for g in range(n_groups):
        start = g * group_size
        end = min(n_cols, start + group_size)
        chunk = x[:, start:end]
        shift = float(np.mean(chunk))
        centered = chunk - shift
        scale = float(np.max(np.abs(centered)))
        scale = scale / qmax if scale > 0 else 1.0
        shifts[g] = shift
        scales[g] = scale
        q[:, start:end] = np.clip(
            np.rint(centered / scale), -qmax - 1, qmax
        ).astype(np.int8)

    payload = zlib.compress(q.tobytes(), level=9)
    return (
        np.frombuffer(payload, dtype=np.uint8).copy(),
        shifts,
        scales,
        np.asarray(q.shape, dtype=np.int32),
    )


def _quantize_groups_raw(x, bits: int, group_size: int):
    """Quantize groups without the DEFLATE step.

    This is used by the DP planner, where exact compressed payload size is
    much less important than avoiding thousands of zlib round-trips.
    """

    if bits < 2 or bits > 8:
        raise ValueError(f"Unsupported bit-width: {bits}")
    x = _to_numpy(x).astype(np.float32, copy=False)
    if x.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape {x.shape}")

    qmax = (1 << (bits - 1)) - 1
    if qmax <= 0:
        raise ValueError(f"Unsupported bit-width: {bits}")

    n_cols = x.shape[1]
    n_groups = (n_cols + group_size - 1) // group_size
    q = np.zeros_like(x, dtype=np.int8)
    shifts = np.zeros(n_groups, dtype=np.float32)
    scales = np.zeros(n_groups, dtype=np.float32)

    for g in range(n_groups):
        start = g * group_size
        end = min(n_cols, start + group_size)
        chunk = x[:, start:end]
        shift = float(np.mean(chunk))
        centered = chunk - shift
        scale = float(np.max(np.abs(centered)))
        scale = scale / qmax if scale > 0 else 1.0
        shifts[g] = shift
        scales[g] = scale
        q[:, start:end] = np.clip(
            np.rint(centered / scale), -qmax - 1, qmax
        ).astype(np.int8)

    return q, shifts, scales, np.asarray(q.shape, dtype=np.int32)


def dequantize_groups(
    payload, q_shape, shifts, scales, bits: int, group_size: int
) -> np.ndarray:
    if bits < 2 or bits > 8:
        raise ValueError(f"Unsupported bit-width: {bits}")

    raw = zlib.decompress(_to_numpy(payload).astype(np.uint8, copy=False).tobytes())
    q_shape = tuple(int(v) for v in _to_numpy(q_shape).tolist())
    shifts = _to_numpy(shifts).astype(np.float32, copy=False)
    q = np.frombuffer(raw, dtype=np.int8).reshape(q_shape)
    q = q.astype(np.float32, copy=False)

    n_cols = q.shape[1]
    n_groups = (n_cols + group_size - 1) // group_size
    out = np.zeros_like(q, dtype=np.float32)
    for g in range(n_groups):
        start = g * group_size
        end = min(n_cols, start + group_size)
        out[:, start:end] = q[:, start:end] * float(scales[g]) + float(shifts[g])
    return out


def _collect_rows(matrices: Sequence[np.ndarray], config: KVTCCodecConfig) -> np.ndarray:
    rows = []
    for i, matrix in enumerate(matrices):
        mat = _to_numpy(matrix).astype(np.float32, copy=False)
        if mat.ndim != 2:
            raise ValueError(f"Expected a 2D matrix, got shape {mat.shape}")
        sampled = _subsample_rows(
            mat,
            max(1, config.sample_limit // max(1, len(matrices))),
            config.seed + i,
        )
        rows.append(sampled)
    if not rows:
        raise ValueError("No matrices were provided for calibration")
    return np.concatenate(rows, axis=0)


def _quantize_block_error(block: np.ndarray, bits: int, group_size: int) -> float:
    if bits == 0:
        return float(np.sum(block * block))
    q, shifts, scales, q_shape = _quantize_groups_raw(block, bits, group_size)
    recon = np.zeros_like(q, dtype=np.float32)
    q_float = q.astype(np.float32, copy=False)
    n_cols = q.shape[1]
    n_groups = (n_cols + group_size - 1) // group_size
    for g in range(n_groups):
        start = g * group_size
        end = min(n_cols, start + group_size)
        recon[:, start:end] = q_float[:, start:end] * float(scales[g]) + float(shifts[g])
    diff = block - recon
    return float(np.sum(diff * diff))


def _estimate_block_cost(block: np.ndarray, bits: int, group_size: int, config: KVTCCodecConfig) -> int:
    if bits == 0:
        penalty_bytes = int(np.ceil(config.zero_bit_penalty_bits / 8))
        return max(1, penalty_bytes)
    q, _, _, q_shape = _quantize_groups_raw(block, bits, group_size)
    n_groups = int(np.ceil(q_shape[1] / group_size))
    payload_bytes = int(np.ceil(q.size * bits / 8.0))
    return int(payload_bytes + n_groups * max(1, config.scale_overhead_bits // 8))


def plan_bit_allocation(coeffs: np.ndarray, config: KVTCCodecConfig) -> np.ndarray:
    """Compute a DP-based blockwise precision plan.

    The plan is stored as ``(start, width, bits)`` rows in ascending order.
    """

    coeffs = _to_numpy(coeffs).astype(np.float32, copy=False)
    if coeffs.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape {coeffs.shape}")

    rank = coeffs.shape[1]
    if rank == 0:
        return np.zeros((0, 3), dtype=np.int32)

    allowed_block_sizes = sorted({int(s) for s in config.allowed_block_sizes if int(s) > 0})
    allowed_block_sizes = [s for s in allowed_block_sizes if s <= rank]
    if not allowed_block_sizes:
        allowed_block_sizes = [1]

    allowed_bits = sorted({int(b) for b in config.allowed_bits if int(b) >= 0})
    if 0 not in allowed_bits:
        allowed_bits = [0] + allowed_bits

    budget = max(
        1,
        int(rank * max(1, config.bits + config.scale_overhead_bits // 8)),
    )
    best_error = np.full((rank + 1, budget + 1), np.inf, dtype=np.float64)
    prev_i = np.full((rank + 1, budget + 1), -1, dtype=np.int32)
    prev_b = np.full((rank + 1, budget + 1), -1, dtype=np.int32)
    choice_width = np.zeros((rank + 1, budget + 1), dtype=np.int32)
    choice_bits = np.zeros((rank + 1, budget + 1), dtype=np.int32)

    best_error[0, :] = 0.0

    total_energy = float(np.sum(coeffs * coeffs))

    for i in range(1, rank + 1):
        for b in range(0, budget + 1):
            if b > 0 and best_error[i, b - 1] <= best_error[i, b]:
                best_error[i, b] = best_error[i, b - 1]
                prev_i[i, b] = prev_i[i, b - 1]
                prev_b[i, b] = prev_b[i, b - 1]
                choice_width[i, b] = choice_width[i, b - 1]
                choice_bits[i, b] = choice_bits[i, b - 1]

            for width in allowed_block_sizes:
                if width > i:
                    continue
                start = i - width
                block = coeffs[:, start:i]
                block_energy = float(np.sum(block * block))

                for bits in allowed_bits:
                    if (
                        bits == 0
                        and total_energy > 0
                        and block_energy
                        > config.zero_bit_energy_fraction * total_energy
                    ):
                        continue
                    group_size = min(config.group_size, width)
                    cost = _estimate_block_cost(block, bits, group_size, config)
                    if cost > b:
                        continue
                    if bits == 0:
                        error = block_energy
                    else:
                        error = _quantize_block_error(block, bits, group_size)

                    candidate = best_error[start, b - cost] + error
                    if candidate < best_error[i, b]:
                        best_error[i, b] = candidate
                        prev_i[i, b] = start
                        prev_b[i, b] = b - cost
                        choice_width[i, b] = width
                        choice_bits[i, b] = bits

    # Reconstruct the plan from the strongest budget.
    b = budget
    while b > 0 and choice_width[rank, b] == 0 and prev_i[rank, b] < 0:
        b -= 1

    blocks = []
    i = rank
    while i > 0:
        if choice_width[i, b] == 0 and prev_i[i, b] < 0:
            b -= 1
            if b < 0:
                break
            continue
        width = int(choice_width[i, b])
        bits = int(choice_bits[i, b])
        start = int(prev_i[i, b])
        if width <= 0 or start < 0:
            break
        blocks.append((start, width, bits))
        next_i = int(prev_i[i, b])
        next_b = int(prev_b[i, b])
        i = next_i
        b = next_b

    if not blocks:
        return np.asarray([(0, rank, 0)], dtype=np.int32)

    blocks.sort(key=lambda row: row[0])
    return np.asarray(blocks, dtype=np.int32)


def fit_transform_plan(matrices: Sequence[np.ndarray], config: KVTCCodecConfig) -> KVTCTransformPlan:
    """Fit a shared plan from one or more 2D calibration matrices."""

    combined = _collect_rows(matrices, config)
    mean, basis = fit_pca_basis(combined, config)
    coeffs = project(combined, mean, basis)
    block_meta = plan_bit_allocation(coeffs, config)
    return KVTCTransformPlan(mean=mean, basis=basis, block_meta=block_meta, config=config)


def fit_shared_calibration(
    key_matrices: Sequence[np.ndarray],
    value_matrices: Sequence[np.ndarray],
    config: KVTCCodecConfig,
) -> KVTCSharedCalibration:
    """Fit a shared key/value calibration that can be reused across layers."""

    key_plan = fit_transform_plan(key_matrices, config)
    value_plan = fit_transform_plan(value_matrices, config)
    return KVTCSharedCalibration(keys=key_plan, values=value_plan)


def encode_tensor(x, plan: KVTCTransformPlan):
    """Encode a 2D matrix with a fitted transform plan."""

    return plan.encode(x)


def decode_tensor(encoded, plan: KVTCTransformPlan) -> np.ndarray:
    """Decode a tensor encoded with :func:`encode_tensor`."""

    return plan.decode(encoded)
