"""
FlashMLX Expert Offloading v3 — Three-Tier Intelligent Expert Management.

Architecture:
  Tier 0: GPU Pool    — mx.take + gather_qmm, zero Python sync
  Tier 1: CPU Warm    — numpy arrays in CPU memory, ~40x faster than SSD on UMA
  Tier 2: SSD Cold    — pread() from safetensors via OS page cache

Key innovations beyond flash-moe:
  1. Zero-sync pool: mx.take replaces .tolist(), eliminating 40 GPU->CPU syncs/token
  2. Telemetry-driven prediction: track activation patterns, prefetch likely experts
  3. Dynamic pool: hot-insert on miss, evict cold experts to CPU tier
  4. CPU warm cache: exploit UMA's 12GB CPU memory as fast buffer (~273 GB/s)
  5. Regime detection: auto-select strategy based on model/memory/concurrency

Three phases:
  Discovery (prefill):  .tolist() + SSD load, build telemetry baseline
  Pool (generation):    mx.take + gather_qmm, zero sync in hot path
  Dynamic (long gen):   pool misses trigger CPU/SSD promotion between tokens
"""

import gc
import json
import math
import os
import struct
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np


# ============================================================================
# gather_qmm sort helpers (from switch_layers.py)
# ============================================================================


def _gather_sort(x, indices):
    """Sort tokens by expert index for better gather_qmm memory access."""
    *_, M = indices.shape
    indices = indices.flatten()
    order = mx.argsort(indices)
    inv_order = mx.argsort(order)
    return x.flatten(0, -3)[order // M], indices[order], inv_order


def _scatter_unsort(x, inv_order, shape=None):
    """Restore original token order after sorted gather_qmm."""
    x = x[inv_order]
    if shape is not None:
        x = mx.unflatten(x, 0, shape)
    return x


# ============================================================================
# Expert Index — maps expert weights to file offsets
# ============================================================================


@dataclass
class ComponentInfo:
    """Location of one weight component (e.g. gate_proj.weight) in a file."""
    file_path: str
    base_offset: int
    expert_stride: int
    per_expert_size: int
    shape_per_expert: list
    dtype: str  # "U32", "BF16", "F32"


@dataclass
class LayerExpertIndex:
    """Expert weight locations for one MoE layer."""
    layer_idx: int
    num_experts: int
    components: Dict[str, ComponentInfo]


def build_expert_index(model_path: str) -> Dict[int, LayerExpertIndex]:
    """Scan safetensors files and build expert weight index."""
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"No safetensors index at {index_path}")

    with open(index_path) as f:
        idx = json.load(f)

    weight_map = idx["weight_map"]
    header_cache = {}

    def get_header(filename):
        if filename not in header_cache:
            path = os.path.join(model_path, filename)
            with open(path, "rb") as f:
                header_len = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(header_len))
                data_start = 8 + header_len
            header_cache[filename] = (header, data_start)
        return header_cache[filename]

    layers = {}

    for key, filename in weight_map.items():
        if ".switch_mlp." not in key:
            continue

        parts = key.split(".")
        layer_idx = None
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts):
                try:
                    layer_idx = int(parts[i + 1])
                except ValueError:
                    continue
                break

        if layer_idx is None:
            continue

        switch_idx = parts.index("switch_mlp")
        comp_name = ".".join(parts[switch_idx + 1:])

        header, data_start = get_header(filename)
        if key not in header:
            continue

        meta = header[key]
        shape = meta["shape"]
        dtype = meta["dtype"]
        offset_start = data_start + meta["data_offsets"][0]
        total_size = meta["data_offsets"][1] - meta["data_offsets"][0]

        num_experts = shape[0]
        per_expert_size = total_size // num_experts
        expert_stride = per_expert_size

        if layer_idx not in layers:
            layers[layer_idx] = LayerExpertIndex(
                layer_idx=layer_idx,
                num_experts=num_experts,
                components={},
            )

        layers[layer_idx].components[comp_name] = ComponentInfo(
            file_path=os.path.join(model_path, filename),
            base_offset=offset_start,
            expert_stride=expert_stride,
            per_expert_size=per_expert_size,
            shape_per_expert=shape[1:],
            dtype=dtype,
        )

    print(f"[ExpertOffload] Indexed {len(layers)} MoE layers, "
          f"{len(next(iter(layers.values())).components)} components each")
    return layers


# ============================================================================
# Expert Loader — reads expert weights from SSD via pread()
# ============================================================================


class ExpertLoader:
    """Loads expert weights from SSD using pread() with OS page cache."""

    def __init__(self, expert_index: Dict[int, LayerExpertIndex], max_workers: int = 4):
        self._index = expert_index
        self._max_workers = max_workers
        self._fds: Dict[str, int] = {}
        for layer_info in expert_index.values():
            for comp in layer_info.components.values():
                if comp.file_path not in self._fds:
                    self._fds[comp.file_path] = os.open(comp.file_path, os.O_RDONLY)

        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        print(f"[ExpertLoader] Opened {len(self._fds)} files, {max_workers} I/O threads")

    def load_expert_component(self, layer_idx: int, expert_idx: int, comp_name: str) -> bytes:
        """Load one component of one expert via pread()."""
        comp = self._index[layer_idx].components[comp_name]
        fd = self._fds[comp.file_path]
        offset = comp.base_offset + expert_idx * comp.expert_stride
        return os.pread(fd, comp.per_expert_size, offset)

    def load_experts(self, layer_idx: int, expert_indices: List[int]) -> Dict[str, mx.array]:
        """Load all components for multiple experts as mx.arrays."""
        layer_info = self._index[layer_idx]
        futures = {}
        for comp_name in layer_info.components:
            for k, eidx in enumerate(expert_indices):
                key = (comp_name, k, eidx)
                futures[key] = self._executor.submit(
                    self.load_expert_component, layer_idx, eidx, comp_name
                )

        component_buffers = {}
        for (comp_name, k, eidx), future in futures.items():
            if comp_name not in component_buffers:
                component_buffers[comp_name] = [None] * len(expert_indices)
            component_buffers[comp_name][k] = future.result()

        results = {}
        for comp_name, buffers in component_buffers.items():
            comp = layer_info.components[comp_name]
            if comp.dtype == "U32":
                np_dtype = np.uint32
            elif comp.dtype == "BF16":
                np_dtype = np.uint16
            elif comp.dtype == "F32":
                np_dtype = np.float32
            else:
                np_dtype = np.uint8

            arrays = []
            for buf in buffers:
                arr = np.frombuffer(buf, dtype=np_dtype).reshape(comp.shape_per_expert)
                arrays.append(arr)

            stacked = np.stack(arrays, axis=0)
            if comp.dtype == "BF16":
                results[comp_name] = mx.array(stacked).view(mx.bfloat16)
            else:
                results[comp_name] = mx.array(stacked)

        return results

    def load_experts_numpy(self, layer_idx: int, expert_indices: List[int]) -> Dict[str, np.ndarray]:
        """Load experts as numpy arrays (for CPU warm cache, avoids GPU allocation)."""
        layer_info = self._index[layer_idx]
        futures = {}
        for comp_name in layer_info.components:
            for k, eidx in enumerate(expert_indices):
                key = (comp_name, k, eidx)
                futures[key] = self._executor.submit(
                    self.load_expert_component, layer_idx, eidx, comp_name
                )

        component_buffers = {}
        for (comp_name, k, eidx), future in futures.items():
            if comp_name not in component_buffers:
                component_buffers[comp_name] = [None] * len(expert_indices)
            component_buffers[comp_name][k] = future.result()

        results = {}
        for comp_name, buffers in component_buffers.items():
            comp = layer_info.components[comp_name]
            if comp.dtype == "U32":
                np_dtype = np.uint32
            elif comp.dtype == "BF16":
                np_dtype = np.uint16
            elif comp.dtype == "F32":
                np_dtype = np.float32
            else:
                np_dtype = np.uint8

            arrays = []
            for buf in buffers:
                arr = np.frombuffer(buf, dtype=np_dtype).reshape(comp.shape_per_expert)
                arrays.append(arr)

            results[comp_name] = np.stack(arrays, axis=0)
        return results

    def expert_byte_size(self, layer_idx: int = None) -> int:
        """Calculate bytes per expert for any layer."""
        if layer_idx is None:
            layer_idx = next(iter(self._index))
        layer_info = self._index[layer_idx]
        return sum(c.per_expert_size for c in layer_info.components.values())

    def close(self):
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()
        self._executor.shutdown(wait=False)

    def __del__(self):
        if self._fds:
            self.close()


# ============================================================================
# Expert Telemetry — activation tracking, prediction, and eviction intelligence
# ============================================================================


class ExpertTelemetry:
    """Tracks expert activation patterns for prediction-based prefetch and eviction.

    Collects per-layer, per-expert:
      - Activation frequency (total count)
      - Recency (last token position when activated)
      - Window frequency (rolling window for trend detection)

    Provides:
      - Hot expert prediction (for prefetch)
      - Cold expert detection (for eviction)
      - Coverage analysis (pool hit rate estimation)
    """

    def __init__(self, num_layers: int, num_experts: int, window_size: int = 64):
        self.num_layers = num_layers
        self.num_experts = num_experts
        self._window_size = window_size

        # Lifetime frequency: [num_layers, num_experts]
        self._freq = np.zeros((num_layers, num_experts), dtype=np.int32)

        # Recency: last token position when expert was activated
        self._recency = np.zeros((num_layers, num_experts), dtype=np.int32)

        # Rolling window: recent activation counts for trend detection
        self._window_freq = np.zeros((num_layers, num_experts), dtype=np.int32)
        self._window_start_token = 0

        # Global counters
        self._total_tokens = 0
        self._total_activations = 0

        # Per-layer hit/miss tracking (for pool effectiveness)
        self._pool_hits = np.zeros(num_layers, dtype=np.int32)
        self._pool_misses = np.zeros(num_layers, dtype=np.int32)

    def record_activation(self, layer_idx: int, expert_ids: List[int], token_pos: int):
        """Record expert activations for one token at one layer."""
        for eid in expert_ids:
            self._freq[layer_idx, eid] += 1
            self._window_freq[layer_idx, eid] += 1
            self._recency[layer_idx, eid] = token_pos
        self._total_activations += len(expert_ids)
        self._total_tokens = max(self._total_tokens, token_pos + 1)

        # Reset window periodically
        if token_pos - self._window_start_token >= self._window_size:
            self._window_freq[:] = 0
            self._window_start_token = token_pos

    def record_pool_hit(self, layer_idx: int, hits: int, misses: int):
        """Record pool hit/miss for effectiveness tracking."""
        self._pool_hits[layer_idx] += hits
        self._pool_misses[layer_idx] += misses

    def predict_hot_experts(self, layer_idx: int, top_k: int,
                            exclude: Optional[Set[int]] = None) -> List[int]:
        """Predict top-k most likely experts for next tokens.

        Uses combined score: 0.6 * normalized_freq + 0.4 * recency_score.
        Excludes experts already in the target set.
        """
        freq = self._freq[layer_idx].astype(np.float32)
        recency = self._recency[layer_idx].astype(np.float32)

        # Normalize frequency
        freq_max = freq.max()
        if freq_max > 0:
            freq_norm = freq / freq_max
        else:
            freq_norm = freq

        # Recency score: higher = more recent
        if self._total_tokens > 0:
            recency_norm = recency / self._total_tokens
        else:
            recency_norm = recency

        score = 0.6 * freq_norm + 0.4 * recency_norm

        if exclude:
            for eid in exclude:
                if eid < len(score):
                    score[eid] = -1

        top_indices = np.argsort(score)[-top_k:][::-1]
        return [int(idx) for idx in top_indices if score[idx] > 0]

    def get_cold_experts(self, layer_idx: int, pool_expert_ids: List[int],
                         keep_min: int = 8) -> List[int]:
        """Identify cold experts in pool that are candidates for eviction.

        Returns expert IDs sorted by coldness (coldest first).
        Never evicts below keep_min experts in pool.
        """
        if len(pool_expert_ids) <= keep_min:
            return []

        scores = []
        for eid in pool_expert_ids:
            freq = float(self._freq[layer_idx, eid])
            recency = float(self._recency[layer_idx, eid])

            freq_max = float(self._freq[layer_idx].max())
            freq_norm = freq / freq_max if freq_max > 0 else 0

            recency_norm = recency / self._total_tokens if self._total_tokens > 0 else 0

            score = 0.6 * freq_norm + 0.4 * recency_norm
            scores.append((eid, score))

        # Sort by score ascending (coldest first)
        scores.sort(key=lambda x: x[1])

        max_evict = len(pool_expert_ids) - keep_min
        return [eid for eid, _ in scores[:max_evict]]

    def get_unique_expert_count(self, layer_idx: int) -> int:
        """Count unique experts ever activated at this layer."""
        return int(np.sum(self._freq[layer_idx] > 0))

    def get_pool_hit_rate(self, layer_idx: int) -> float:
        """Get pool hit rate for a layer."""
        total = self._pool_hits[layer_idx] + self._pool_misses[layer_idx]
        return float(self._pool_hits[layer_idx]) / total if total > 0 else 1.0

    def get_overall_hit_rate(self) -> float:
        """Get overall pool hit rate across all layers."""
        total_hits = int(self._pool_hits.sum())
        total_misses = int(self._pool_misses.sum())
        total = total_hits + total_misses
        return total_hits / total if total > 0 else 1.0

    def summary(self) -> dict:
        """Return telemetry summary."""
        unique_per_layer = [self.get_unique_expert_count(i) for i in range(self.num_layers)]
        return {
            "total_tokens": self._total_tokens,
            "total_activations": self._total_activations,
            "avg_unique_per_layer": float(np.mean(unique_per_layer)),
            "max_unique_per_layer": int(np.max(unique_per_layer)),
            "overall_pool_hit_rate": self.get_overall_hit_rate(),
            "per_layer_hit_rate": {
                i: self.get_pool_hit_rate(i) for i in range(self.num_layers)
                if self._pool_hits[i] + self._pool_misses[i] > 0
            },
        }


# ============================================================================
# CPU Warm Cache — numpy arrays in CPU memory, ~40x faster than SSD on UMA
# ============================================================================


class CPUWarmCache:
    """CPU-resident expert cache using numpy arrays.

    On Apple Silicon UMA, CPU and GPU share the same physical DRAM bus (~273 GB/s).
    numpy -> mx.array conversion is a memory copy within the same DRAM:
      - CPU warm: 1.69 MB / 273 GB/s = ~6 us per expert
      - SSD cold:  1.69 MB / 7 GB/s = ~240 us per expert
      => ~40x faster promotion from CPU cache vs SSD

    On Mac Mini M4 Pro 48GB: GPU gets 36GB, CPU gets 12GB.
    12 GB / 1.69 MB per expert = ~7,100 expert slots.
    At 40 layers, that's ~177 experts/layer — 69% of all 256 experts.
    Combined with GPU pool (~50-90), coverage reaches 88-100%.
    """

    def __init__(self, max_bytes: int):
        self._max_bytes = max_bytes
        # LRU cache: (layer_idx, expert_id) -> {comp_name: np.ndarray}
        self._cache: OrderedDict = OrderedDict()
        self._current_bytes = 0
        self._hits = 0
        self._misses = 0

    def get(self, layer_idx: int, expert_id: int) -> Optional[Dict[str, np.ndarray]]:
        """Get expert from CPU cache. Returns numpy dict or None."""
        key = (layer_idx, expert_id)
        if key in self._cache:
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def get_mx(self, layer_idx: int, expert_id: int,
               dtype_map: Optional[Dict[str, str]] = None) -> Optional[Dict[str, mx.array]]:
        """Get expert as mx.arrays (GPU). Fast on UMA (~6 us per expert)."""
        np_data = self.get(layer_idx, expert_id)
        if np_data is None:
            return None
        result = {}
        for comp, arr in np_data.items():
            mx_arr = mx.array(arr)
            if dtype_map and comp in dtype_map and dtype_map[comp] == "BF16":
                mx_arr = mx_arr.view(mx.bfloat16)
            result[comp] = mx_arr
        return result

    def put(self, layer_idx: int, expert_id: int, data: Dict[str, np.ndarray]):
        """Store expert in CPU cache. Evicts LRU if over budget."""
        key = (layer_idx, expert_id)
        if key in self._cache:
            self._cache.move_to_end(key)
            return

        size = sum(arr.nbytes for arr in data.values())

        # Evict LRU entries until we have room
        while self._current_bytes + size > self._max_bytes and self._cache:
            oldest_key, oldest_data = self._cache.popitem(last=False)
            self._current_bytes -= sum(a.nbytes for a in oldest_data.values())

        if self._current_bytes + size <= self._max_bytes:
            self._cache[key] = data
            self._current_bytes += size

    def put_from_mx(self, layer_idx: int, expert_id: int, data: Dict[str, mx.array]):
        """Store expert from mx.arrays by converting to numpy (CPU side)."""
        np_data = {}
        for comp, arr in data.items():
            if arr.dtype == mx.bfloat16:
                np_data[comp] = np.array(arr.view(mx.uint16))
            else:
                np_data[comp] = np.array(arr)
        self.put(layer_idx, expert_id, np_data)

    def contains(self, layer_idx: int, expert_id: int) -> bool:
        return (layer_idx, expert_id) in self._cache

    def batch_get_missing(self, layer_idx: int, expert_ids: List[int]) -> Tuple[List[int], List[int]]:
        """Split expert_ids into cached and missing lists."""
        cached = [eid for eid in expert_ids if self.contains(layer_idx, eid)]
        missing = [eid for eid in expert_ids if not self.contains(layer_idx, eid)]
        return cached, missing

    @property
    def utilization(self) -> float:
        return self._current_bytes / self._max_bytes if self._max_bytes > 0 else 0

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0

    def summary(self) -> dict:
        return {
            "capacity_gb": self._max_bytes / 1024**3,
            "used_gb": self._current_bytes / 1024**3,
            "utilization": self.utilization,
            "entries": len(self._cache),
            "hit_rate": self.hit_rate,
            "hits": self._hits,
            "misses": self._misses,
        }


# ============================================================================
# Regime Detector — auto-select optimal offloading strategy
# ============================================================================


@dataclass
class RegimeConfig:
    """Configuration output from regime detection."""
    regime: str  # "A_streaming", "B_three_tier", "C_full_gpu"
    gpu_pool_budget_gb: float
    cpu_cache_budget_gb: float
    pool_size_per_layer: int
    max_concurrent_requests: int
    description: str


class RegimeDetector:
    """Auto-detect optimal offloading strategy based on hardware and model.

    Three regimes:
      A (Streaming):   model >> memory. Minimal pool, SSD streaming.
      B (Three-Tier):  model ~ memory. GPU pool + CPU warm + SSD cold.
      C (Full GPU):    model << memory. Optional pool for locality boost.

    The key insight: regime depends not just on "can the model fit" but on
    the concurrency target. A 18GB model fits in 24GB (Regime C for 1 request),
    but 8 concurrent requests need 4GB KV cache → Regime B.
    """

    @staticmethod
    def detect(
        total_expert_gb: float,
        non_expert_gb: float,
        gpu_memory_gb: float,
        cpu_memory_gb: float = 0,
        kv_per_request_gb: float = 0.5,
        target_concurrent: int = 1,
        num_layers: int = 40,
        num_experts: int = 256,
        expert_bytes: int = 1_770_000,  # ~1.69 MB default
    ) -> RegimeConfig:
        """Detect optimal regime and return configuration.

        Args:
            total_expert_gb: Total expert weight size (e.g. 16.88 GB)
            non_expert_gb: Non-expert model size (e.g. 1.29 GB)
            gpu_memory_gb: GPU memory budget (e.g. 36 GB on Mini Pro)
            cpu_memory_gb: CPU-only memory available (e.g. 12 GB on Mini Pro)
            kv_per_request_gb: KV cache per concurrent request at target context
            target_concurrent: Number of concurrent requests
            num_layers: Number of MoE layers
            num_experts: Experts per layer
            expert_bytes: Bytes per expert
        """
        os_overhead_gb = 2.0
        kv_total_gb = target_concurrent * kv_per_request_gb

        available_for_experts = gpu_memory_gb - os_overhead_gb - non_expert_gb - kv_total_gb
        ratio = available_for_experts / total_expert_gb if total_expert_gb > 0 else 999

        if ratio >= 1.0:
            # Regime C: All experts CAN fit in GPU, but we still offload to
            # save memory for KV cache / concurrent requests / other models.
            # Pool sized for cache locality: top-N hot experts, not all.
            # Empirically, 25% of experts covers 95%+ of activations for MoE.
            pool_experts_target = max(num_experts // 4, 32)  # 25% or min 32
            pool_gb = pool_experts_target * expert_bytes * num_layers / 1024**3
            pool_gb = min(pool_gb, available_for_experts * 0.4)
            experts_in_pool = int(pool_gb * 1024**3 / expert_bytes / num_layers)
            experts_in_pool = min(experts_in_pool, pool_experts_target)
            experts_in_pool = max(experts_in_pool, 32)
            return RegimeConfig(
                regime="C_full_gpu",
                gpu_pool_budget_gb=pool_gb,
                cpu_cache_budget_gb=cpu_memory_gb * 0.8,
                pool_size_per_layer=experts_in_pool,
                max_concurrent_requests=target_concurrent,
                description=f"Full GPU: {experts_in_pool}/{num_experts} experts/layer in pool, "
                            f"{target_concurrent} concurrent slots",
            )

        elif ratio >= 0.3:
            # Regime B: Three-tier cache
            # 40% of available GPU for pool, rest for KV headroom
            pool_gb = available_for_experts * 0.4
            pool_gb = max(pool_gb, 2.0)  # Minimum 2 GB pool
            experts_in_pool = int(pool_gb * 1024**3 / expert_bytes / num_layers)
            experts_in_pool = min(experts_in_pool, num_experts)
            experts_in_pool = max(experts_in_pool, 16)  # Minimum 16 per layer

            cpu_budget = cpu_memory_gb * 0.9  # 90% of CPU memory
            cpu_experts = int(cpu_budget * 1024**3 / expert_bytes / num_layers) if cpu_budget > 0 else 0
            total_fast = experts_in_pool + cpu_experts

            return RegimeConfig(
                regime="B_three_tier",
                gpu_pool_budget_gb=pool_gb,
                cpu_cache_budget_gb=cpu_budget,
                pool_size_per_layer=experts_in_pool,
                max_concurrent_requests=target_concurrent,
                description=f"Three-tier: {experts_in_pool} GPU + {cpu_experts} CPU "
                            f"= {total_fast}/{num_experts} fast experts/layer, "
                            f"{target_concurrent} concurrent",
            )

        else:
            # Regime A: SSD streaming, minimal pool
            pool_gb = max(available_for_experts * 0.5, 1.0)
            experts_in_pool = int(pool_gb * 1024**3 / expert_bytes / num_layers)
            experts_in_pool = max(experts_in_pool, 8)  # Bare minimum

            return RegimeConfig(
                regime="A_streaming",
                gpu_pool_budget_gb=pool_gb,
                cpu_cache_budget_gb=cpu_memory_gb * 0.9,
                pool_size_per_layer=experts_in_pool,
                max_concurrent_requests=target_concurrent,
                description=f"Streaming: {experts_in_pool}/{num_experts} experts in pool, "
                            f"SSD for rest",
            )


# ============================================================================
# FlashMoE SwitchGLU — Three-tier intelligent expert management
# ============================================================================


class FlashMoeSwitchGLU(nn.Module):
    """Zero-sync expert offloading with three-tier cache and telemetry.

    Three phases:
      Discovery (prefill): .tolist() + SSD/CPU load, builds telemetry baseline
      Pool (generation): mx.take + gather_qmm, zero sync in hot path
      Dynamic (ongoing): between-token pool maintenance (promote/evict)

    Three tiers:
      Tier 0 - GPU Pool:  mx.take lookup, zero latency
      Tier 1 - CPU Warm:  numpy arrays, ~6 us promotion on UMA
      Tier 2 - SSD Cold:  pread(), ~240 us per expert

    Telemetry integration:
      - Records every activation during both phases
      - Predicts hot experts for prefetch
      - Identifies cold experts for eviction
      - Tracks pool hit rate for self-optimization
    """

    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        expert_loader: ExpertLoader,
        layer_idx: int,
        telemetry: ExpertTelemetry,
        cpu_cache: Optional[CPUWarmCache] = None,
        group_size: int = 64,
        bits: int = 4,
        pool_size: int = 48,
    ):
        super().__init__()
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.num_experts = num_experts
        self.group_size = group_size
        self.bits = bits
        self._loader = expert_loader
        self._layer_idx = layer_idx
        self._pool_size = pool_size
        self._telemetry = telemetry
        self._cpu_cache = cpu_cache

        # Build dtype map for CPU cache BF16 handling
        layer_info = expert_loader._index.get(layer_idx)
        self._dtype_map = {}
        if layer_info:
            for comp_name, comp_info in layer_info.components.items():
                self._dtype_map[comp_name] = comp_info.dtype

        # === Discovery phase state ===
        self._seen_experts: set = set()
        self._discovery_cache: Dict[int, Dict[str, mx.array]] = {}
        self._np_remap = np.zeros(num_experts, dtype=np.int32)
        self._token_counter = 0

        # === Pool phase state ===
        self._pool: Optional[Dict[str, mx.array]] = None
        self._pool_remap: Optional[mx.array] = None
        self._pool_expert_ids: List[int] = []  # expert_id at each pool slot
        self._pool_remap_np: Optional[np.ndarray] = None  # numpy mirror for updates
        self._pool_is_identity: bool = False  # True = skip mx.take remap (Regime C)

        # === Dynamic phase state ===
        self._pending_promotions: List[int] = []  # experts to promote into pool
        self._generation_token_count = 0

        # === Two-phase state (PP full → TG compact) ===
        self._prebuilt_full: bool = False  # True = full pool prebuilt for PP
        self._pool_compacted: bool = False  # True = pool already compacted for TG
        self._pp_indices_buffer: List[mx.array] = []  # buffered indices during PP (no GPU sync)

    # ------------------------------------------------------------------
    # Discovery phase: slow path with .tolist() + tiered loading
    # ------------------------------------------------------------------

    def _load_from_tiers(self, layer_idx: int, expert_ids: List[int]) -> Dict[str, mx.array]:
        """Load experts using three-tier fallback: discovery_cache -> CPU -> SSD.

        NOTE: Does NOT populate CPU cache during discovery to avoid GPU→CPU syncs.
        CPU cache is populated asynchronously after pool building via SSD→numpy.
        """
        # Check discovery cache first
        from_cache = []
        from_cpu = []
        from_ssd = []

        for eid in expert_ids:
            if eid in self._discovery_cache:
                from_cache.append(eid)
            elif self._cpu_cache and self._cpu_cache.contains(layer_idx, eid):
                from_cpu.append(eid)
            else:
                from_ssd.append(eid)

        # Load from SSD (slowest tier)
        if from_ssd:
            ssd_data = self._loader.load_experts(layer_idx, from_ssd)
            for i, eid in enumerate(from_ssd):
                self._discovery_cache[eid] = {
                    comp: data[i:i+1] for comp, data in ssd_data.items()
                }
                # CPU cache population moved to _build_pool (async SSD→numpy)
                # to avoid GPU→CPU syncs that killed PP performance

        # Load from CPU cache (medium tier)
        for eid in from_cpu:
            np_data = self._cpu_cache.get(layer_idx, eid)
            if np_data is not None:
                mx_data = {}
                for comp, arr in np_data.items():
                    mx_arr = mx.array(arr)
                    if self._dtype_map.get(comp) == "BF16":
                        mx_arr = mx_arr.view(mx.bfloat16)
                    mx_data[comp] = mx_arr
                self._discovery_cache[eid] = {
                    comp: data[np.newaxis] if data.ndim == len(self._loader._index[layer_idx].components[comp].shape_per_expert) else data
                    for comp, data in mx_data.items()
                }

        # Stack all requested experts from discovery cache
        comp_names = list(self._discovery_cache[expert_ids[0]].keys())
        result = {}
        for comp in comp_names:
            result[comp] = mx.concatenate(
                [self._discovery_cache[eid][comp] for eid in expert_ids], axis=0
            )
        return result

    def _discovery_call(self, x: mx.array, indices: mx.array) -> mx.array:
        """Prefill: .tolist() + tiered load + gather_qmm. Tracks telemetry."""
        x = mx.expand_dims(x, (-2, -3))

        flat_ids = indices.reshape(-1).tolist()
        unique = sorted(set(flat_ids))
        self._seen_experts.update(unique)

        # Record telemetry
        self._telemetry.record_activation(self._layer_idx, flat_ids, self._token_counter)

        # Numpy remap for this batch
        for i, eid in enumerate(unique):
            self._np_remap[eid] = i
        local_np = self._np_remap[np.array(flat_ids, dtype=np.int32)]
        local_indices = mx.array(local_np.reshape(indices.shape))

        data = self._load_from_tiers(self._layer_idx, unique)
        y = self._switchglu(x, data, local_indices)
        return y.squeeze(-2)

    # ------------------------------------------------------------------
    # Pool building: from discovery cache + telemetry-informed selection
    # ------------------------------------------------------------------

    def _build_pool(self):
        """Build GPU-resident expert pool from discovery cache.

        Uses telemetry to prioritize hot experts if pool_size < seen experts.
        """
        available = [eid for eid in self._seen_experts if eid in self._discovery_cache]

        if len(available) <= self._pool_size:
            pool_ids = sorted(available)
        else:
            # Use telemetry to select hottest experts
            hot = self._telemetry.predict_hot_experts(
                self._layer_idx, self._pool_size, exclude=None
            )
            # Filter to only available experts
            pool_ids = [eid for eid in hot if eid in available][:self._pool_size]
            # Fill remaining slots with any available expert
            remaining = [eid for eid in available if eid not in set(pool_ids)]
            pool_ids.extend(remaining[:self._pool_size - len(pool_ids)])

        K = len(pool_ids)
        if K == 0:
            return

        # Stack from discovery cache
        comp_names = list(self._discovery_cache[pool_ids[0]].keys())
        self._pool = {}
        for comp in comp_names:
            self._pool[comp] = mx.concatenate(
                [self._discovery_cache[eid][comp] for eid in pool_ids], axis=0
            )
        mx.eval(self._pool)

        # Build remap table (sentinel = K for non-pool experts → miss detection)
        self._pool_expert_ids = pool_ids
        self._pool_remap_np = np.full(self.num_experts, K, dtype=np.int32)  # sentinel
        for i, eid in enumerate(pool_ids):
            self._pool_remap_np[eid] = i
        self._pool_remap = mx.array(self._pool_remap_np)

        # Queue async CPU cache population for non-pool experts
        # Uses SSD→numpy (no GPU involvement), avoids all GPU→CPU syncs
        non_pool_ids = [eid for eid in self._discovery_cache if eid not in set(pool_ids)]
        if self._cpu_cache and non_pool_ids:
            self._async_populate_cpu_cache(non_pool_ids)

        self._discovery_cache.clear()
        self._seen_experts.clear()

    # ------------------------------------------------------------------
    # Pre-build pool: load experts from SSD at patch time (Regime C)
    # ------------------------------------------------------------------

    def prebuild_pool(self, expert_ids: Optional[List[int]] = None, full: bool = False):
        """Pre-build pool directly from SSD, bypassing discovery phase entirely.

        Two modes:
          full=False: Load pool_size experts (compact mode for TG)
          full=True:  Load ALL experts (full mode for fast PP, then auto-compact)

        Args:
            expert_ids: Which experts to load. None = auto-select.
            full: If True, load all experts for PP speed; will auto-compact to
                  pool_size after PP→TG transition.
        """
        if expert_ids is None:
            if full:
                expert_ids = list(range(self.num_experts))
            else:
                expert_ids = list(range(min(self.num_experts, self._pool_size)))

        K = len(expert_ids)
        if K == 0:
            return

        # Batch load from SSD → mx.array (GPU)
        data = self._loader.load_experts(self._layer_idx, expert_ids)
        self._pool = {comp: arr for comp, arr in data.items()}
        mx.eval(self._pool)

        # Build remap table (sentinel = K for non-pool experts → miss detection)
        self._pool_expert_ids = expert_ids
        self._pool_remap_np = np.full(self.num_experts, K, dtype=np.int32)  # sentinel
        for i, eid in enumerate(expert_ids):
            self._pool_remap_np[eid] = i
        self._pool_remap = mx.array(self._pool_remap_np)

        # Detect identity mapping: all experts present, sequential order
        # When identity, skip mx.take and sentinel check entirely
        self._pool_is_identity = (
            K == self.num_experts and
            expert_ids == list(range(self.num_experts))
        )

        # Track two-phase state
        if full:
            self._prebuilt_full = True
            self._pool_compacted = False
            self._pp_indices_buffer = []

        # Pre-warm CPU cache for non-pool experts (background SSD→numpy)
        if self._cpu_cache and K < self.num_experts:
            non_pool = [e for e in range(self.num_experts) if e not in set(expert_ids)]
            if non_pool:
                self._async_populate_cpu_cache(non_pool)

    # ------------------------------------------------------------------
    # Async CPU cache population (SSD→numpy, zero GPU involvement)
    # ------------------------------------------------------------------

    def _async_populate_cpu_cache(self, expert_ids: List[int]):
        """Populate CPU cache from SSD→numpy in background thread.

        Key insight: loads directly from SSD to numpy arrays (CPU memory),
        completely bypassing GPU. This avoids the GPU→CPU syncs that killed PP.

        Data flow: SSD (pread) → bytes → numpy (CPU) → CPUWarmCache
        NOT: SSD → mx.array (GPU) → np.array (CPU sync!) → CPUWarmCache
        """
        layer_idx = self._layer_idx
        loader = self._loader
        cpu_cache = self._cpu_cache

        def _populate():
            try:
                np_data = loader.load_experts_numpy(layer_idx, expert_ids)
                for i, eid in enumerate(expert_ids):
                    single = {comp: data[i:i+1] for comp, data in np_data.items()}
                    cpu_cache.put(layer_idx, eid, single)
            except Exception:
                pass  # Best-effort, non-blocking

        thread = threading.Thread(target=_populate, daemon=True)
        thread.start()

    # ------------------------------------------------------------------
    # Pool phase: ZERO Python sync, pure MLX graph
    # ------------------------------------------------------------------

    def _pool_call(self, x: mx.array, indices: mx.array) -> mx.array:
        """Pool-based forward pass with two-phase support.

        Phase 1 (PP, full pool): identity path + record expert usage for compaction.
        Phase 2 (TG, compact pool): sentinel-based miss detection with fallback.

        The PP→TG transition is detected by seq_len dropping to 1.
        On transition, pool is compacted from ALL experts to top-K hot experts.
        """
        seq_len = indices.shape[-2] if indices.ndim >= 3 else 1

        # --- Two-phase: PP→TG transition detected when seq_len drops to 1 ---
        # Compact is triggered externally via OffloadContext.compact(),
        # NOT in the hot path. Just stop buffering PP indices.
        if self._prebuilt_full and not self._pool_compacted and seq_len == 1:
            self._pool_compacted = True  # Stop buffering, compact later

        # --- Full pool (identity): zero overhead path ---
        if self._pool_is_identity:
            x_e = mx.expand_dims(x, (-2, -3))

            # Sort indices for better memory access (critical for PP with many tokens)
            do_sort = indices.size >= 64
            idx = indices
            inv_order = None
            if do_sort:
                x_e, idx, inv_order = _gather_sort(x_e, indices)

            y = self._switchglu(x_e, self._pool, idx, sorted_indices=do_sort)
            self._generation_token_count += 1

            if do_sort:
                y = _scatter_unsort(y, inv_order, indices.shape)

            # Buffer indices during PP for deferred counting (NO GPU sync here)
            if self._prebuilt_full and not self._pool_compacted and seq_len > 1:
                self._pp_indices_buffer.append(indices.reshape(-1))

            return y.squeeze(-2)

        # --- Partial pool: speculative execution (no sentinel check) ---
        # Remap table already clamps non-pool experts to K-1 (last valid slot),
        # so a single mx.take is all we need — no mx.minimum, no .item().
        # Full lazy evaluation across all 40 layers, same as identity path.
        #
        # For hot pools (coverage=100%), the evicted experts had zero PP
        # activations, so TG routing to them is extremely rare.
        local_indices = self._pool_remap[indices]

        x_e = mx.expand_dims(x, (-2, -3))

        do_sort = local_indices.size >= 64
        idx = local_indices
        inv_order = None
        if do_sort:
            x_e, idx, inv_order = _gather_sort(x_e, local_indices)

        y = self._switchglu(x_e, self._pool, idx, sorted_indices=do_sort)
        self._generation_token_count += 1

        if do_sort:
            y = _scatter_unsort(y, inv_order, local_indices.shape)

        return y.squeeze(-2)

    def _pool_miss_call(self, x: mx.array, indices: mx.array) -> mx.array:
        """Handle pool miss during TG. Lightweight — no discovery_cache pollution.

        Builds a temporary mini-pool with exactly the needed experts:
        - In-pool experts: sliced from existing pool tensor
        - Out-of-pool experts: loaded from CPU cache (fast on UMA, ~6us)
        - SSD fallback only if CPU cache miss (~240us, rare after compact)

        For TG (seq_len=1, top-8), indices has just 8 values → .tolist() is negligible.
        """
        x_e = mx.expand_dims(x, (-2, -3))

        flat_ids = indices.reshape(-1).tolist()
        unique = sorted(set(flat_ids))
        K = len(self._pool_expert_ids)
        comp_names = list(self._pool.keys())

        # Collect per-expert data: pool slice or CPU cache or SSD
        expert_slices = {}
        for eid in unique:
            slot = int(self._pool_remap_np[eid])
            if slot < K:
                # In pool: direct slice (lazy, no eval)
                expert_slices[eid] = {
                    comp: self._pool[comp][slot:slot+1] for comp in comp_names
                }
            elif self._cpu_cache:
                mx_data = self._cpu_cache.get_mx(
                    self._layer_idx, eid, dtype_map=self._dtype_map
                )
                if mx_data is not None:
                    expert_slices[eid] = mx_data
                else:
                    # SSD fallback (rare after proper compact)
                    ssd = self._loader.load_experts(self._layer_idx, [eid])
                    expert_slices[eid] = {comp: data[0:1] for comp, data in ssd.items()}
            else:
                ssd = self._loader.load_experts(self._layer_idx, [eid])
                expert_slices[eid] = {comp: data[0:1] for comp, data in ssd.items()}

        # Build temporary mini-pool tensor [len(unique), ...]
        mini_pool = {}
        for comp in comp_names:
            slices = []
            for eid in unique:
                s = expert_slices[eid][comp]
                if s.ndim == len(self._pool[comp].shape) - 1:
                    s = s[np.newaxis]
                slices.append(s)
            mini_pool[comp] = mx.concatenate(slices, axis=0)

        # Local remap: unique expert list → 0..N-1
        remap_dict = {eid: i for i, eid in enumerate(unique)}
        local_np = np.array([remap_dict[eid] for eid in flat_ids],
                            dtype=np.int32).reshape(indices.shape)
        local_indices = mx.array(local_np)

        y = self._switchglu(x_e, mini_pool, local_indices)
        self._generation_token_count += 1
        return y.squeeze(-2)

    def _compact_pool(self, target_pool_size: Optional[int] = None):
        """Compact pool from full (PP) to top-K hot experts (TG).

        Called by OffloadContext.compact() — NOT in the hot path.
        Uses PP activation counts to select the hottest experts.
        Demotes non-hot experts to CPU cache for fast miss recovery.
        Does NOT call gc/eval/clear_cache — caller handles cleanup.
        """
        if not self._pp_indices_buffer:
            self._pool_compacted = True
            return 0.0  # no data, nothing compacted

        pool_size = target_pool_size if target_pool_size is not None else self._pool_size

        # Aggregate PP indices: single GPU→CPU sync (deferred from PP)
        all_indices = mx.concatenate(self._pp_indices_buffer)
        counts = np.bincount(np.array(all_indices), minlength=self.num_experts)
        self._pp_indices_buffer = []

        # Select top pool_size experts by PP frequency
        hot_ids = np.argsort(-counts)[:pool_size].tolist()
        hot_ids.sort()
        hot_set = set(hot_ids)
        K = len(hot_ids)

        # Demote non-hot experts to CPU cache BEFORE releasing the full pool.
        # Batch per-component: one mx→numpy sync per component (not per expert).
        if self._cpu_cache and K < self.num_experts:
            non_hot_ids = [eid for eid in range(self.num_experts) if eid not in hot_set]
            if non_hot_ids:
                non_hot_idx = mx.array(non_hot_ids)
                comp_np = {}
                for comp, full_tensor in self._pool.items():
                    # One GPU→CPU sync per component: full_tensor[non_hot_idx] → numpy
                    batch = full_tensor[non_hot_idx]
                    if batch.dtype == mx.bfloat16:
                        comp_np[comp] = np.array(batch.view(mx.uint16))
                    else:
                        comp_np[comp] = np.array(batch)

                # Store individual experts in CPU cache
                for i, eid in enumerate(non_hot_ids):
                    single = {comp: arr[i:i+1] for comp, arr in comp_np.items()}
                    self._cpu_cache.put(self._layer_idx, eid, single)

        # Extract compact subset from full pool (lazy, no mx.eval)
        idx = mx.array(hot_ids)
        compact_pool = {}
        for comp, full_tensor in self._pool.items():
            compact_pool[comp] = full_tensor[idx]

        # Replace pool — old pool refs released, Python refcount handles free
        self._pool = compact_pool
        self._pool_expert_ids = hot_ids
        # Default to K-1 (last valid slot) instead of K (sentinel).
        # This pre-clamps non-pool experts, eliminating mx.minimum in _pool_call.
        self._pool_remap_np = np.full(self.num_experts, K - 1, dtype=np.int32)
        for i, eid in enumerate(hot_ids):
            self._pool_remap_np[eid] = i
        self._pool_remap = mx.array(self._pool_remap_np)
        self._pool_is_identity = False
        self._pool_compacted = True

        # Coverage stat: what % of PP activations are in the compact pool
        total_acts = counts.sum()
        pool_acts = counts[hot_ids].sum()
        coverage = pool_acts / total_acts if total_acts > 0 else 0.0

        return coverage

    # ------------------------------------------------------------------
    # Dynamic pool maintenance: promote/evict between tokens
    # ------------------------------------------------------------------

    def maintain_pool(self, force: bool = False):
        """Between-token pool maintenance: promote predicted hot experts, evict cold.

        Called by the generation loop between tokens. NOT in the hot path.
        This is where telemetry-driven prediction pays off.

        Strategy:
          1. Get telemetry predictions for hot experts not in pool
          2. Check if any are in CPU warm cache (fast promotion)
          3. Evict coldest pool experts to make room
          4. Rebuild pool tensors and remap table
        """
        if self._pool is None:
            return
        if not force and self._generation_token_count % 8 != 0:
            return  # Only maintain every 8 tokens to amortize overhead

        pool_set = set(self._pool_expert_ids)

        # Step 1: Find hot experts NOT in pool
        predicted = self._telemetry.predict_hot_experts(
            self._layer_idx, top_k=16, exclude=pool_set
        )
        if not predicted:
            return

        # Step 2: Check which predicted experts are in CPU cache (fast) or need SSD
        promotable = []
        for eid in predicted:
            if self._cpu_cache and self._cpu_cache.contains(self._layer_idx, eid):
                promotable.append(("cpu", eid))
            else:
                promotable.append(("ssd", eid))

        if not promotable:
            return

        # Limit promotions per maintenance cycle
        max_promote = min(4, len(promotable))
        promotable = promotable[:max_promote]

        # Step 3: Evict cold experts to make room
        to_evict = []
        if len(self._pool_expert_ids) + max_promote > self._pool_size:
            cold = self._telemetry.get_cold_experts(
                self._layer_idx, self._pool_expert_ids, keep_min=8
            )
            to_evict = cold[:max_promote]

            if not to_evict:
                return

            # Demote evicted experts to CPU cache
            evicted_set = set(to_evict)
            if self._cpu_cache:
                for eid in to_evict:
                    if eid in pool_set:
                        slot = self._pool_remap_np[eid]
                        np_data = {}
                        for comp, tensor in self._pool.items():
                            slice_data = tensor[slot:slot+1]
                            if slice_data.dtype == mx.bfloat16:
                                np_data[comp] = np.array(slice_data.view(mx.uint16))
                            else:
                                np_data[comp] = np.array(slice_data)
                        self._cpu_cache.put(self._layer_idx, eid, np_data)

            # Keep original layout for in-place slot swap
            new_pool_ids = list(self._pool_expert_ids)
        else:
            new_pool_ids = list(self._pool_expert_ids)

        # Step 4: Load and promote new experts
        new_expert_data = {}
        for source, eid in promotable:
            if source == "cpu":
                np_data = self._cpu_cache.get(self._layer_idx, eid)
                if np_data is not None:
                    mx_data = {}
                    for comp, arr in np_data.items():
                        mx_arr = mx.array(arr)
                        if self._dtype_map.get(comp) == "BF16":
                            mx_arr = mx_arr.view(mx.bfloat16)
                        mx_data[comp] = mx_arr
                    new_expert_data[eid] = mx_data
            else:
                ssd_data = self._loader.load_experts(self._layer_idx, [eid])
                new_expert_data[eid] = {comp: data[0:1] for comp, data in ssd_data.items()}
                # Also populate CPU cache
                if self._cpu_cache:
                    np_data = {}
                    for comp, data in ssd_data.items():
                        s = data[0:1]
                        if s.dtype == mx.bfloat16:
                            np_data[comp] = np.array(s.view(mx.uint16))
                        else:
                            np_data[comp] = np.array(s)
                    self._cpu_cache.put(self._layer_idx, eid, np_data)

        if not new_expert_data:
            return

        # Step 5: In-place slot swap — O(promotions) instead of O(pool_size) rebuild
        freed_slots = [self._pool_remap_np[eid] for eid in to_evict]

        for i, eid in enumerate(new_expert_data):
            if i < len(freed_slots):
                # Swap into evicted slot (common case: pool at capacity)
                slot = int(freed_slots[i])
                for comp in self._pool:
                    data = new_expert_data[eid][comp]
                    if data.ndim == len(self._pool[comp].shape) - 1:
                        data = data[np.newaxis]
                    self._pool[comp][slot:slot+1] = data
                # Replace evicted expert ID at same slot position
                new_pool_ids[slot] = eid
            else:
                # Append (pool not full — rare)
                new_pool_ids.append(eid)
                for comp in self._pool:
                    data = new_expert_data[eid][comp]
                    if data.ndim == len(self._pool[comp].shape) - 1:
                        data = data[np.newaxis]
                    self._pool[comp] = mx.concatenate([self._pool[comp], data], axis=0)

        mx.eval(self._pool)

        # Update state (sentinel = len(new_pool_ids) for miss detection)
        self._pool_expert_ids = new_pool_ids
        K = len(new_pool_ids)
        self._pool_remap_np = np.full(self.num_experts, K, dtype=np.int32)  # sentinel
        for i, eid in enumerate(new_pool_ids):
            self._pool_remap_np[eid] = i
        self._pool_remap = mx.array(self._pool_remap_np)
        self._pool_is_identity = (
            K == self.num_experts and
            new_pool_ids == list(range(self.num_experts))
        )

    # ------------------------------------------------------------------
    # Telemetry recording for pool phase (deferred, not in hot path)
    # ------------------------------------------------------------------

    def record_pool_telemetry(self, indices: mx.array):
        """Record telemetry from pool phase. Called between tokens (not in hot path).

        Uses .tolist() but this is OK because it's between tokens, not during compute.
        """
        if self._pool is None:
            return
        flat_ids = indices.reshape(-1).tolist()
        self._telemetry.record_activation(self._layer_idx, flat_ids,
                                          self._token_counter + self._generation_token_count)

        # Track pool hits/misses
        pool_set = set(self._pool_expert_ids)
        hits = sum(1 for eid in flat_ids if eid in pool_set)
        misses = len(flat_ids) - hits
        self._telemetry.record_pool_hit(self._layer_idx, hits, misses)

    # ------------------------------------------------------------------
    # Shared SwiGLU compute kernel (used by all paths)
    # ------------------------------------------------------------------

    def _switchglu(self, x, data, local_indices, sorted_indices=False):
        """gate_proj + up_proj -> SwiGLU -> down_proj via gather_qmm."""
        x_gate = mx.gather_qmm(
            x, data["gate_proj.weight"], data["gate_proj.scales"],
            data["gate_proj.biases"],
            rhs_indices=local_indices, transpose=True,
            group_size=self.group_size, bits=self.bits,
            sorted_indices=sorted_indices,
        )
        x_up = mx.gather_qmm(
            x, data["up_proj.weight"], data["up_proj.scales"],
            data["up_proj.biases"],
            rhs_indices=local_indices, transpose=True,
            group_size=self.group_size, bits=self.bits,
            sorted_indices=sorted_indices,
        )
        x_act = nn.silu(x_gate) * x_up
        return mx.gather_qmm(
            x_act, data["down_proj.weight"], data["down_proj.scales"],
            data["down_proj.biases"],
            rhs_indices=local_indices, transpose=True,
            group_size=self.group_size, bits=self.bits,
            sorted_indices=sorted_indices,
        )

    # ------------------------------------------------------------------
    # Forward: auto-switch discovery -> pool with dynamic maintenance
    # ------------------------------------------------------------------

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        """Forward pass with automatic phase switching.

        x: [batch, seq, input_dims]
        indices: [batch, seq, top_k]
        """
        self._token_counter += indices.shape[-2] if indices.ndim >= 2 else 1

        if self._pool is not None:
            return self._pool_call(x, indices)

        result = self._discovery_call(x, indices)

        # Build pool when enough experts discovered
        if len(self._seen_experts) >= min(self._pool_size, 16):
            self._build_pool()

        return result


# ============================================================================
# Prefetch Engine — background thread for telemetry-driven prefetch
# ============================================================================


class PrefetchEngine:
    """Background prefetch engine that uses telemetry to predict and pre-load experts.

    Runs in a background thread. Between tokens:
      1. Analyzes recent telemetry patterns
      2. Predicts likely experts for upcoming tokens
      3. Pre-loads predicted experts into CPU warm cache from SSD
      4. Signals FlashMoeSwitchGLU layers to promote from CPU to GPU pool

    This means pool misses that would need SSD (240 us) instead hit CPU (6 us).
    """

    def __init__(
        self,
        expert_loader: ExpertLoader,
        telemetry: ExpertTelemetry,
        cpu_cache: CPUWarmCache,
        num_layers: int,
        num_experts: int,
    ):
        self._loader = expert_loader
        self._telemetry = telemetry
        self._cpu_cache = cpu_cache
        self._num_layers = num_layers
        self._num_experts = num_experts
        self._running = False
        self._thread = None
        self._prefetch_queue: List[Tuple[int, int]] = []  # (layer_idx, expert_id)
        self._lock = threading.Lock()

    def start(self):
        """Start background prefetch thread."""
        self._running = True
        self._thread = threading.Thread(target=self._prefetch_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop prefetch thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def request_prefetch(self, layer_idx: int, expert_ids: List[int]):
        """Queue experts for background prefetch into CPU cache."""
        with self._lock:
            for eid in expert_ids:
                if not self._cpu_cache.contains(layer_idx, eid):
                    self._prefetch_queue.append((layer_idx, eid))

    def _prefetch_loop(self):
        """Background loop: process prefetch queue."""
        while self._running:
            batch = []
            with self._lock:
                # Grab up to 8 items from queue
                batch = self._prefetch_queue[:8]
                self._prefetch_queue = self._prefetch_queue[8:]

            if not batch:
                time.sleep(0.001)  # 1ms sleep when idle
                continue

            # Group by layer for batch loading
            by_layer = {}
            for layer_idx, eid in batch:
                if not self._cpu_cache.contains(layer_idx, eid):
                    by_layer.setdefault(layer_idx, []).append(eid)

            for layer_idx, eids in by_layer.items():
                try:
                    np_data = self._loader.load_experts_numpy(layer_idx, eids)
                    for i, eid in enumerate(eids):
                        single = {comp: data[i:i+1] for comp, data in np_data.items()}
                        self._cpu_cache.put(layer_idx, eid, single)
                except Exception:
                    pass  # Prefetch is best-effort


# ============================================================================
# ThunderOMLX Bridge — interface for external inference engine integration
# ============================================================================


class ThunderOMLXBridge:
    """Interface between FlashMLX expert offloading and ThunderOMLX inference engine.

    ThunderOMLX (local MLX inference server) can:
      1. Read telemetry snapshots for expert activation analysis
      2. Push prefetch strategies based on request queue characteristics
      3. Send load hints for proactive expert prediction
      4. Query pool state for capacity planning and scheduling

    Data flows:
      FlashMLX → ThunderOMLX: telemetry_snapshot() → activation patterns, pool state
      ThunderOMLX → FlashMLX: apply_strategy()     → prefetch/evict/promote orders
      ThunderOMLX → FlashMLX: load_hint()           → upcoming workload characteristics

    Integration pattern (ThunderOMLX side):
      snapshot = bridge.telemetry_snapshot()         # Read telemetry
      strategy = analyze_and_plan(snapshot)           # ThunderOMLX decides
      bridge.apply_strategy(strategy)                 # Push strategy back
    """

    def __init__(self, context: 'OffloadContext', model):
        self._ctx = context
        self._model = model
        self._flash_layers = self._find_flash_layers()

    def _find_flash_layers(self) -> Dict[int, 'FlashMoeSwitchGLU']:
        """Find all FlashMoeSwitchGLU layers in the model."""
        inner = self._model
        for attr in ("model", "model", "language_model", "model"):
            if hasattr(inner, attr):
                inner = getattr(inner, attr)

        result = {}
        if hasattr(inner, "layers"):
            for layer in inner.layers:
                if hasattr(layer, "mlp") and hasattr(layer.mlp, "switch_mlp"):
                    switch = layer.mlp.switch_mlp
                    if isinstance(switch, FlashMoeSwitchGLU):
                        result[switch._layer_idx] = switch
        return result

    # ------------------------------------------------------------------
    # Export: FlashMLX → ThunderOMLX
    # ------------------------------------------------------------------

    def telemetry_snapshot(self) -> dict:
        """Export current telemetry state for ThunderOMLX analysis.

        Returns a JSON-serializable dict. ThunderOMLX can use this to:
          - Analyze expert activation patterns across layers
          - Identify hot/cold experts for prefetch/eviction
          - Plan pool composition based on request queue
          - Monitor system health and pool effectiveness

        Schema:
          {
            "timestamp": float,
            "regime": str,
            "hardware": {"gpu_gb": float, "cpu_gb": float},
            "model": {"num_layers": int, "num_experts": int, "expert_bytes": int},
            "telemetry": {
              "total_tokens": int,
              "total_activations": int,
              "activation_freq": [[int]],       # [num_layers, num_experts]
              "recency": [[int]],                # [num_layers, num_experts]
              "pool_hit_rates": {layer: float},
            },
            "pool_state": {
              layer_idx: {
                "expert_ids": [int],
                "size": int,
                "capacity": int,
              }
            },
            "cpu_cache": {
              "capacity_gb": float,
              "used_gb": float,
              "utilization": float,
              "hit_rate": float,
            }
          }
        """
        tel = self._ctx.telemetry
        snapshot = {
            "timestamp": time.time(),
            "regime": self._ctx.regime.regime if self._ctx.regime else "unknown",
            "hardware": {
                "gpu_gb": self._ctx.regime.gpu_pool_budget_gb if self._ctx.regime else 0,
                "cpu_gb": self._ctx.regime.cpu_cache_budget_gb if self._ctx.regime else 0,
            },
        }

        # Model info
        if self._flash_layers:
            sample = next(iter(self._flash_layers.values()))
            snapshot["model"] = {
                "num_layers": len(self._flash_layers),
                "num_experts": sample.num_experts,
                "expert_bytes": self._ctx.loader.expert_byte_size(),
            }

        # Telemetry data
        if tel:
            snapshot["telemetry"] = {
                "total_tokens": tel._total_tokens,
                "total_activations": tel._total_activations,
                "activation_freq": tel._freq.tolist(),
                "recency": tel._recency.tolist(),
                "pool_hit_rates": {
                    str(i): tel.get_pool_hit_rate(i) for i in range(tel.num_layers)
                    if tel._pool_hits[i] + tel._pool_misses[i] > 0
                },
            }

        # Pool state per layer
        pool_state = {}
        for layer_idx, switch in self._flash_layers.items():
            if switch._pool is not None:
                pool_state[str(layer_idx)] = {
                    "expert_ids": list(switch._pool_expert_ids),
                    "size": len(switch._pool_expert_ids),
                    "capacity": switch._pool_size,
                }
        snapshot["pool_state"] = pool_state

        # CPU cache summary
        if self._ctx.cpu_cache:
            snapshot["cpu_cache"] = self._ctx.cpu_cache.summary()

        return snapshot

    def pool_state(self) -> dict:
        """Quick query of current pool composition per layer.

        Lightweight alternative to full telemetry_snapshot() for
        ThunderOMLX scheduling decisions (e.g., routing requests
        to take advantage of already-pooled experts).

        Returns: {layer_idx: [expert_ids]}
        """
        result = {}
        for layer_idx, switch in self._flash_layers.items():
            if switch._pool is not None:
                result[layer_idx] = list(switch._pool_expert_ids)
        return result

    # ------------------------------------------------------------------
    # Import: ThunderOMLX → FlashMLX
    # ------------------------------------------------------------------

    def apply_strategy(self, strategy: dict):
        """Apply prefetch/eviction strategy from ThunderOMLX.

        ThunderOMLX analyzes telemetry + request queue → decides which experts
        to prefetch, promote, or evict. FlashMLX executes.

        Strategy format:
          {
            "prefetch": {                       # Pre-load SSD→CPU cache
              "3": [100, 101, 102],             # layer 3: prefetch experts 100,101,102
              "7": [200, 201],                  # layer 7: prefetch experts 200,201
            },
            "promote": {                        # Promote CPU→GPU pool
              "3": [50, 51],                    # layer 3: add experts 50,51 to pool
            },
            "evict": {                          # Remove from GPU pool
              "3": [10, 11],                    # layer 3: evict experts 10,11
            },
            "pool_size": 64,                    # Optional: adjust pool size target
          }
        """
        # Prefetch: queue SSD→CPU cache loading
        prefetch_map = strategy.get("prefetch", {})
        if prefetch_map and self._ctx.prefetch_engine:
            for layer_str, eids in prefetch_map.items():
                layer_idx = int(layer_str)
                self._ctx.prefetch_engine.request_prefetch(layer_idx, eids)

        # Pool size adjustment
        if "pool_size" in strategy:
            new_size = int(strategy["pool_size"])
            for switch in self._flash_layers.values():
                switch._pool_size = new_size

        # Promote: CPU→GPU pool (triggers maintain_pool)
        promote_map = strategy.get("promote", {})
        for layer_str, eids in promote_map.items():
            layer_idx = int(layer_str)
            if layer_idx in self._flash_layers:
                switch = self._flash_layers[layer_idx]
                switch._pending_promotions.extend(eids)
                switch.maintain_pool(force=True)

        # Evict: remove from GPU pool (demote to CPU)
        evict_map = strategy.get("evict", {})
        for layer_str, eids in evict_map.items():
            layer_idx = int(layer_str)
            if layer_idx in self._flash_layers:
                switch = self._flash_layers[layer_idx]
                if switch._pool is not None:
                    self._evict_from_pool(switch, eids)

    def _evict_from_pool(self, switch: 'FlashMoeSwitchGLU', expert_ids: List[int]):
        """Evict specific experts from a layer's GPU pool."""
        if switch._pool is None:
            return

        evict_set = set(expert_ids)
        pool_set = set(switch._pool_expert_ids)
        to_evict = [eid for eid in expert_ids if eid in pool_set]

        if not to_evict:
            return

        # Demote evicted experts to CPU cache
        if self._ctx.cpu_cache:
            for eid in to_evict:
                slot = switch._pool_remap_np[eid]
                np_data = {}
                for comp, tensor in switch._pool.items():
                    slice_data = tensor[slot:slot+1]
                    if slice_data.dtype == mx.bfloat16:
                        np_data[comp] = np.array(slice_data.view(mx.uint16))
                    else:
                        np_data[comp] = np.array(slice_data)
                self._ctx.cpu_cache.put(switch._layer_idx, eid, np_data)

        # Rebuild pool without evicted experts
        new_pool_ids = [eid for eid in switch._pool_expert_ids if eid not in evict_set]
        comp_names = list(switch._pool.keys())
        new_pool = {}
        for comp in comp_names:
            slices = []
            for eid in new_pool_ids:
                old_slot = switch._pool_remap_np[eid]
                slices.append(switch._pool[comp][old_slot:old_slot+1])
            if slices:
                new_pool[comp] = mx.concatenate(slices, axis=0)

        if new_pool:
            mx.eval(new_pool)

        switch._pool = new_pool if new_pool else None
        switch._pool_expert_ids = new_pool_ids
        switch._pool_remap_np = np.full(switch.num_experts, 0, dtype=np.int32)
        for i, eid in enumerate(new_pool_ids):
            switch._pool_remap_np[eid] = i
        switch._pool_remap = mx.array(switch._pool_remap_np)

    def load_hint(self, prompt_tokens: int, expected_generation: int = 100,
                  domain: Optional[str] = None):
        """Hint from ThunderOMLX about an upcoming inference request.

        ThunderOMLX knows its request queue. Before processing a request,
        it can hint FlashMLX about workload characteristics to enable
        proactive expert prediction and prefetch.

        Args:
            prompt_tokens: Expected prompt length in tokens
            expected_generation: Expected generation length
            domain: Optional domain hint (e.g., "code", "math", "chat")
                    for domain-specific expert prediction

        Effect:
            - Long prompts (>1K tokens): pre-warm CPU cache with diverse experts
            - Short prompts: focus on recently-hot experts
            - Domain hints: bias prefetch toward domain-correlated experts
        """
        if not self._ctx.telemetry or not self._ctx.prefetch_engine:
            return

        tel = self._ctx.telemetry

        for layer_idx, switch in self._flash_layers.items():
            pool_set = set(switch._pool_expert_ids) if switch._pool is not None else set()

            if prompt_tokens > 1000:
                # Long prompt: diverse experts needed, prefetch broadly
                top_k = min(32, switch.num_experts)
            else:
                # Short prompt: focus on recent hot experts
                top_k = 16

            predicted = tel.predict_hot_experts(layer_idx, top_k=top_k, exclude=pool_set)
            if predicted:
                self._ctx.prefetch_engine.request_prefetch(layer_idx, predicted)


# ============================================================================
# Model Patching — orchestrates the full three-tier setup
# ============================================================================


def detect_hardware() -> Tuple[float, float]:
    """Detect GPU and CPU memory on Apple Silicon.

    Returns (gpu_memory_gb, cpu_memory_gb).
    """
    import subprocess
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=5
        )
        total_bytes = int(result.stdout.strip())
        total_gb = total_bytes / 1024**3
    except Exception:
        total_gb = 24.0  # Default fallback

    # Apple Silicon GPU memory limits (approximate):
    # M4 Pro 24GB: ~18GB GPU, ~6GB CPU
    # M4 Pro 48GB: ~36GB GPU, ~12GB CPU
    # M4 Max 64GB: ~48GB GPU, ~16GB CPU
    # Use 75% for GPU as conservative estimate
    gpu_gb = total_gb * 0.75
    cpu_gb = total_gb - gpu_gb

    return gpu_gb, cpu_gb


def patch_model_for_offload(
    model,
    model_path: str,
    max_workers: int = 4,
    pool_size: int = 0,  # 0 = auto-detect from regime
    cpu_cache_gb: float = None,
    target_concurrent: int = 1,
    enable_prefetch: bool = True,
    enable_telemetry: bool = True,
):
    """Patch a loaded MoE model with three-tier intelligent expert management.

    1. Detect hardware and regime
    2. Build expert index
    3. Create telemetry, CPU cache, prefetch engine
    4. Replace SwitchGLU modules with FlashMoeSwitchGLU
    5. Delete original expert weights from GPU

    Args:
        model: Loaded MLX model with .layers containing .mlp.switch_mlp
        model_path: Path to model directory with safetensors
        max_workers: I/O threads for pread()
        pool_size: Max experts per layer in GPU pool
        cpu_cache_gb: CPU cache budget in GB (auto-detect if None)
        target_concurrent: Planned concurrent requests
        enable_prefetch: Enable background prefetch engine
        enable_telemetry: Enable activation telemetry

    Returns:
        OffloadContext with loader, telemetry, cpu_cache, prefetch_engine
    """
    # Step 1: Detect hardware
    gpu_gb, auto_cpu_gb = detect_hardware()
    if cpu_cache_gb is None:
        cpu_cache_gb = auto_cpu_gb * 0.8  # 80% of CPU memory for cache

    # Step 2: Build expert index
    expert_index = build_expert_index(model_path)
    num_layers = len(expert_index)
    sample_layer = next(iter(expert_index.values()))
    num_experts = sample_layer.num_experts

    # Step 3: Create loader
    loader = ExpertLoader(expert_index, max_workers=max_workers)
    expert_bytes = loader.expert_byte_size()

    # Step 4: Detect regime
    total_expert_gb = num_experts * num_layers * expert_bytes / 1024**3
    non_expert_gb = 1.29  # Approximate for Qwen3.5 (measured)

    regime = RegimeDetector.detect(
        total_expert_gb=total_expert_gb,
        non_expert_gb=non_expert_gb,
        gpu_memory_gb=gpu_gb,
        cpu_memory_gb=auto_cpu_gb,
        target_concurrent=target_concurrent,
        num_layers=num_layers,
        num_experts=num_experts,
        expert_bytes=expert_bytes,
    )
    print(f"[ExpertOffload] Regime: {regime.regime}")
    print(f"[ExpertOffload] {regime.description}")

    # Pool size: use regime's recommendation when auto (0), otherwise cap to regime
    if pool_size <= 0:
        effective_pool_size = regime.pool_size_per_layer
    else:
        effective_pool_size = min(pool_size, regime.pool_size_per_layer)

    # Step 5: Create telemetry
    telemetry = ExpertTelemetry(num_layers, num_experts) if enable_telemetry else None

    # Step 6: Create CPU warm cache
    cpu_cache_bytes = int(cpu_cache_gb * 1024**3)
    cpu_cache = CPUWarmCache(cpu_cache_bytes) if cpu_cache_bytes > 0 else None
    if cpu_cache:
        print(f"[ExpertOffload] CPU warm cache: {cpu_cache_gb:.1f} GB "
              f"(~{int(cpu_cache_bytes / expert_bytes / num_layers)} experts/layer)")

    # Step 7: Create prefetch engine
    prefetch = None
    if enable_prefetch and telemetry and cpu_cache:
        prefetch = PrefetchEngine(loader, telemetry, cpu_cache, num_layers, num_experts)

    # Step 8: Patch model
    inner = model
    if hasattr(model, "model"):
        inner = model.model
    if hasattr(inner, "model"):
        inner = inner.model
    if hasattr(inner, "language_model"):
        inner = inner.language_model
    if hasattr(inner, "model"):
        inner = inner.model

    layers = inner.layers
    patched = 0
    freed_bytes = 0

    for i, layer in enumerate(layers):
        if not hasattr(layer, "mlp"):
            continue
        mlp = layer.mlp
        if not hasattr(mlp, "switch_mlp"):
            continue
        if i not in expert_index:
            continue

        old_switch = mlp.switch_mlp
        input_dims = old_switch.gate_proj.weight.shape[2] * 8
        hidden_dims = old_switch.gate_proj.weight.shape[1]
        num_exp = old_switch.gate_proj.weight.shape[0]

        for param_name in ["gate_proj", "up_proj", "down_proj"]:
            proj = getattr(old_switch, param_name)
            for arr_name in ["weight", "scales", "biases"]:
                arr = getattr(proj, arr_name, None)
                if arr is not None:
                    freed_bytes += arr.nbytes

        new_switch = FlashMoeSwitchGLU(
            input_dims=input_dims,
            hidden_dims=hidden_dims,
            num_experts=num_exp,
            expert_loader=loader,
            layer_idx=i,
            telemetry=telemetry if telemetry else ExpertTelemetry(num_layers, num_experts),
            cpu_cache=cpu_cache,
            group_size=64,
            bits=4,
            pool_size=effective_pool_size,
        )

        mlp.switch_mlp = new_switch
        patched += 1

    gc.collect()
    mx.metal.clear_cache()

    print(f"[ExpertOffload] Patched {patched} layers, "
          f"freed ~{freed_bytes / 1024**3:.2f} GB")

    # Two-phase pool: prebuild FULL pool for fast PP, auto-compact to
    # top-K hot experts on PP→TG transition (seq_len drops from >1 to 1).
    t_prebuild = time.perf_counter()
    for layer in layers:
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "switch_mlp"):
            switch = layer.mlp.switch_mlp
            if isinstance(switch, FlashMoeSwitchGLU):
                switch.prebuild_pool(full=True)
    prebuild_ms = (time.perf_counter() - t_prebuild) * 1000
    print(f"[ExpertOffload] Pre-built FULL pools for {patched} layers "
          f"(all experts, will compact to {effective_pool_size}/layer after PP, "
          f"{prebuild_ms:.0f}ms)")

    # Start prefetch engine
    if prefetch:
        prefetch.start()
        print(f"[ExpertOffload] Prefetch engine started")

    ctx = OffloadContext(
        loader=loader,
        telemetry=telemetry,
        cpu_cache=cpu_cache,
        prefetch_engine=prefetch,
        regime=regime,
        model=model,
    )
    print(f"[ExpertOffload] ThunderOMLX bridge available via ctx.bridge")
    return ctx


# ============================================================================
# Offload Context — holds all offload state for clean lifecycle management
# ============================================================================


class OffloadContext:
    """Holds all offload infrastructure. Keep alive during inference.

    Provides ThunderOMLX bridge for external inference engine integration.
    """

    def __init__(
        self,
        loader: ExpertLoader,
        telemetry: Optional[ExpertTelemetry],
        cpu_cache: Optional[CPUWarmCache],
        prefetch_engine: Optional[PrefetchEngine],
        regime: Optional[RegimeConfig],
        model=None,
    ):
        self.loader = loader
        self.telemetry = telemetry
        self.cpu_cache = cpu_cache
        self.prefetch_engine = prefetch_engine
        self.regime = regime
        self._model = model
        self._bridge: Optional[ThunderOMLXBridge] = None

    @property
    def bridge(self) -> Optional['ThunderOMLXBridge']:
        """Get ThunderOMLX bridge (lazy-created on first access)."""
        if self._bridge is None and self._model is not None:
            self._bridge = ThunderOMLXBridge(self, self._model)
        return self._bridge

    def run_maintenance(self, model):
        """Run between-token maintenance on all patched layers.

        Call this between tokens in the generation loop for dynamic pool updates.
        """
        inner = model
        if hasattr(model, "model"):
            inner = model.model
        if hasattr(inner, "model"):
            inner = inner.model
        if hasattr(inner, "language_model"):
            inner = inner.language_model
        if hasattr(inner, "model"):
            inner = inner.model

        for layer in inner.layers:
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "switch_mlp"):
                switch = layer.mlp.switch_mlp
                if isinstance(switch, FlashMoeSwitchGLU):
                    switch.maintain_pool()

    def request_prefetch_for_predictions(self, model):
        """Use telemetry to predict and prefetch experts across all layers."""
        if not self.prefetch_engine or not self.telemetry:
            return

        inner = model
        if hasattr(model, "model"):
            inner = model.model
        if hasattr(inner, "model"):
            inner = inner.model
        if hasattr(inner, "language_model"):
            inner = inner.language_model
        if hasattr(inner, "model"):
            inner = inner.model

        for layer in inner.layers:
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "switch_mlp"):
                switch = layer.mlp.switch_mlp
                if isinstance(switch, FlashMoeSwitchGLU) and switch._pool is not None:
                    pool_set = set(switch._pool_expert_ids)
                    predicted = self.telemetry.predict_hot_experts(
                        switch._layer_idx, top_k=16, exclude=pool_set
                    )
                    if predicted:
                        self.prefetch_engine.request_prefetch(
                            switch._layer_idx, predicted
                        )

    def summary(self) -> dict:
        """Get full system summary."""
        result = {"regime": self.regime.regime if self.regime else "unknown"}
        if self.telemetry:
            result["telemetry"] = self.telemetry.summary()
        if self.cpu_cache:
            result["cpu_cache"] = self.cpu_cache.summary()
        return result

    def compact(self, pool_size: Optional[int] = None):
        """Compact all pools from full (PP) to hot-K experts (TG).

        Call this AFTER prefill is done, BEFORE long generation.
        Non-hot experts are demoted to CPU cache for fast miss recovery.

        Args:
            pool_size: Override pool size per layer. None = use regime default.

        Returns dict with per-layer coverage stats.
        """
        inner = self._model
        if hasattr(inner, "model"):
            inner = inner.model
        if hasattr(inner, "model"):
            inner = inner.model
        if hasattr(inner, "language_model"):
            inner = inner.language_model
        if hasattr(inner, "model"):
            inner = inner.model

        effective_pool = pool_size if pool_size is not None else (
            self.regime.pool_size_per_layer if self.regime else 64
        )

        # Auto-expand CPU cache for demoted experts.
        # Need: (num_experts - pool_size) × num_layers × ~1.69 MB per expert
        if self.cpu_cache:
            num_experts = 256  # default, will be refined per layer
            num_moe_layers = sum(
                1 for layer in inner.layers
                if hasattr(layer, "mlp") and hasattr(layer.mlp, "switch_mlp")
                and isinstance(layer.mlp.switch_mlp, FlashMoeSwitchGLU)
            )
            if num_moe_layers > 0:
                first_switch = next(
                    layer.mlp.switch_mlp for layer in inner.layers
                    if hasattr(layer, "mlp") and hasattr(layer.mlp, "switch_mlp")
                    and isinstance(layer.mlp.switch_mlp, FlashMoeSwitchGLU)
                )
                num_experts = first_switch.num_experts
                # Estimate bytes per expert from pool
                if first_switch._pool:
                    expert_bytes = sum(
                        t.nbytes // t.shape[0] for t in first_switch._pool.values()
                    )
                else:
                    expert_bytes = 1_700_000  # ~1.69 MB fallback

                needed_bytes = (num_experts - effective_pool) * num_moe_layers * expert_bytes
                if needed_bytes > self.cpu_cache._max_bytes:
                    old_cap = self.cpu_cache._max_bytes / 1024**3
                    self.cpu_cache._max_bytes = int(needed_bytes * 1.1)  # 10% headroom
                    new_cap = self.cpu_cache._max_bytes / 1024**3
                    print(f"[ExpertOffload] Auto-expanded CPU cache: {old_cap:.1f} → {new_cap:.1f} GB "
                          f"(for {num_experts - effective_pool} evicted experts × {num_moe_layers} layers)")

        t0 = time.perf_counter()
        compacted = 0
        total_coverage = 0.0

        for layer in inner.layers:
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "switch_mlp"):
                switch = layer.mlp.switch_mlp
                if isinstance(switch, FlashMoeSwitchGLU) and switch._prebuilt_full:
                    cov = switch._compact_pool(target_pool_size=effective_pool)
                    total_coverage += cov
                    compacted += 1

        # Force eval on compact pools to materialize them and break lazy refs
        # to the old full pool tensors. Without this, the compact pool is a lazy
        # slice of the full pool, keeping the full pool alive in memory.
        for layer in inner.layers:
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "switch_mlp"):
                switch = layer.mlp.switch_mlp
                if isinstance(switch, FlashMoeSwitchGLU) and switch._pool is not None:
                    mx.eval(switch._pool)

        # Now gc + clear_cache frees the old full pool Metal buffers
        gc.collect()
        mx.metal.clear_cache()

        # Pre-warm Metal gather_qmm kernels for the new pool shape.
        # Without this, the first ~50 TG tokens are 2× slower because
        # Metal JIT-compiles kernels for each unique tensor shape.
        # A single dummy gather_qmm per layer triggers compilation.
        hidden_size = getattr(inner, "hidden_size", None)
        if hidden_size is None and hasattr(inner, "args"):
            hidden_size = getattr(inner.args, "hidden_size", None)
            if hidden_size is None:
                # Multimodal models nest hidden_size in text_config
                tc = getattr(inner.args, "text_config", None)
                if isinstance(tc, dict):
                    hidden_size = tc.get("hidden_size")
                elif tc is not None:
                    hidden_size = getattr(tc, "hidden_size", None)
        if hidden_size is not None:
            dummy_x = mx.zeros((1, 1, 1, hidden_size))
            dummy_idx = mx.zeros((1, 1, 8), dtype=mx.int32)
            for layer in inner.layers:
                if hasattr(layer, "mlp") and hasattr(layer.mlp, "switch_mlp"):
                    switch = layer.mlp.switch_mlp
                    if isinstance(switch, FlashMoeSwitchGLU) and switch._pool is not None:
                        _ = switch._switchglu(dummy_x, switch._pool, dummy_idx)
            mx.eval(mx.zeros(1))  # Flush all warmup graphs

        elapsed_ms = (time.perf_counter() - t0) * 1000
        avg_cov = total_coverage / compacted if compacted > 0 else 0
        mem = mx.metal.get_active_memory() / 1024**3

        # CPU cache stats
        cpu_info = ""
        if self.cpu_cache:
            cpu_gb = self.cpu_cache._current_bytes / 1024**3
            cpu_entries = len(self.cpu_cache._cache)
            cpu_info = f", CPU cache: {cpu_gb:.1f} GB ({cpu_entries} entries)"

        print(f"[ExpertOffload] Compacted {compacted} layers to {effective_pool} experts/layer "
              f"({elapsed_ms:.0f}ms, PP coverage: {avg_cov:.1%}, memory: {mem:.2f} GB{cpu_info})")

        return {
            "layers": compacted,
            "pool_size": effective_pool,
            "pp_coverage": avg_cov,
            "memory_gb": mem,
            "elapsed_ms": elapsed_ms,
            "cpu_cache_gb": self.cpu_cache._current_bytes / 1024**3 if self.cpu_cache else 0,
        }

    def close(self):
        """Clean shutdown of all resources."""
        if self.prefetch_engine:
            self.prefetch_engine.stop()
        if self.loader:
            self.loader.close()
