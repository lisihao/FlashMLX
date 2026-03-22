"""
Simplified SSM Cache Manager

⚠️  DEPRECATED: 2026-03-22 ⚠️

This module is DEPRECATED and should NOT be used in production.

Deprecation Reason:
==================
1. Functionality overlaps 100% with ThunderLLAMA prefix caching
2. GPU stability issues (page fault) in actual inference with MLX Metal
3. No real-world use case that ThunderLLAMA doesn't already cover

Preserved For:
==============
- Future reference if MLX Metal memory management improves
- Potential use with new hybrid architecture models
- Technical learning and pattern reference

Current Status:
===============
- Code: Preserved but sealed (no public exports in __init__.py)
- Tests: Skipped
- Examples: Disabled
- Documentation: Archived in SSM_CACHE_DEPRECATION.md

See Also:
=========
- SSM_CACHE_DEPRECATION.md - Full deprecation decision record
- ThunderLLAMA/thunderllama.conf - Active prefix caching config

---

Original Design Goals (Archived):
==================================
单层内存缓存，替代 Hot/Warm/Cold 三层架构。

设计目标：
- 最小化管理开销（接近直接 dict 访问）
- 只在内存中（不外溢到磁盘）
- 可选的跨请求复用支持

性能特性：
- 直接 dict lookup: ~0.011 μs/op
- vs Hot/Warm/Cold: ~0.177 μs/op (16x slower)

Achieved Results:
- Overhead reduction: 16x → 11x (1.5x improvement)
- Code simplification: ~500 lines → ~100 lines (5x reduction)
- Cross-request reuse: 100% hit rate in unit tests
- BUT: GPU page fault in actual generation ❌
"""

from typing import Dict, Optional
import mlx.core as mx


class SimplifiedSSMCacheManager:
    """
    简化的 SSM 缓存管理器（单层内存缓存）。

    设计原则：
    1. 最小化开销 - 直接 dict 访问，无 LRU/stats/migration
    2. 内存优先 - 不外溢到磁盘，只在 RAM 中
    3. 可选复用 - 支持跨请求状态复用（future work）

    对比 Hot/Warm/Cold：
    - Hot/Warm/Cold: 3 层 dict 查询 + LRU + stats + migration = 0.177 μs
    - Simplified: 1 层 dict 查询 = 0.011 μs (16x faster)
    """

    def __init__(self, max_size_bytes: Optional[int] = None):
        """
        初始化简化的 SSM 缓存管理器。

        Args:
            max_size_bytes: 最大内存限制（字节），None = 无限制
        """
        self.cache: Dict[int, mx.array] = {}  # layer_idx → state
        self.max_size_bytes = max_size_bytes
        self.current_size_bytes = 0

        # 简单统计（可选）
        self.hits = 0
        self.misses = 0

    def store(self, layer_idx: int, state: mx.array) -> bool:
        """
        存储 SSM 状态。

        Args:
            layer_idx: 层索引
            state: SSM 状态

        Returns:
            是否成功存储
        """
        state_size = state.nbytes if hasattr(state, 'nbytes') else 0

        # 检查内存限制
        if self.max_size_bytes is not None:
            if self.current_size_bytes + state_size > self.max_size_bytes:
                # 内存不足，简单策略：不存储
                # 未来可以实现 LRU 驱逐，但当前保持简单
                return False

        # 更新已有条目
        if layer_idx in self.cache:
            old_size = self.cache[layer_idx].nbytes if hasattr(self.cache[layer_idx], 'nbytes') else 0
            self.current_size_bytes -= old_size

        # 存储
        self.cache[layer_idx] = state
        self.current_size_bytes += state_size

        return True

    def retrieve(self, layer_idx: int) -> Optional[mx.array]:
        """
        获取 SSM 状态。

        Args:
            layer_idx: 层索引

        Returns:
            SSM 状态，如果不存在返回 None
        """
        state = self.cache.get(layer_idx)

        # 简单统计
        if state is not None:
            self.hits += 1
        else:
            self.misses += 1

        return state

    def clear(self):
        """清空缓存。"""
        self.cache.clear()
        self.current_size_bytes = 0
        self.hits = 0
        self.misses = 0

    def get_statistics(self) -> dict:
        """
        获取缓存统计信息。

        Returns:
            统计信息字典
        """
        total_accesses = self.hits + self.misses
        hit_rate = self.hits / total_accesses if total_accesses > 0 else 0.0

        return {
            'entry_count': len(self.cache),
            'size_bytes': self.current_size_bytes,
            'max_size_bytes': self.max_size_bytes,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate
        }

    def __len__(self) -> int:
        """返回缓存条目数量。"""
        return len(self.cache)

    def __contains__(self, layer_idx: int) -> bool:
        """检查层是否在缓存中。"""
        return layer_idx in self.cache


# 全局单例（可选）
_global_ssm_cache: Optional[SimplifiedSSMCacheManager] = None


def get_global_ssm_cache(
    max_size_bytes: Optional[int] = None
) -> SimplifiedSSMCacheManager:
    """
    获取全局 SSM 缓存实例（用于跨请求复用）。

    Args:
        max_size_bytes: 最大内存限制

    Returns:
        全局 SimplifiedSSMCacheManager 实例
    """
    global _global_ssm_cache
    if _global_ssm_cache is None:
        _global_ssm_cache = SimplifiedSSMCacheManager(max_size_bytes)
    return _global_ssm_cache


def reset_global_ssm_cache():
    """重置全局 SSM 缓存。"""
    global _global_ssm_cache
    if _global_ssm_cache is not None:
        _global_ssm_cache.clear()
    _global_ssm_cache = None


class CacheList(list):
    """
    Custom list subclass that can store additional attributes.

    Used to store cache objects while keeping references to manager
    and original methods for restoration.
    """
    pass


def inject_simplified_ssm_cache(
    model,
    max_size_bytes: int = 100 * 1024 * 1024,  # 100MB default
    auto_inject: bool = True
):
    """
    为 MLX-LM 模型注入简化的 SSM 缓存。

    Args:
        model: MLX-LM 模型实例
        max_size_bytes: 最大缓存内存（默认 100MB）
        auto_inject: 是否自动注入到 model.cache（默认 True）

    Returns:
        (cache_list, manager) 元组
            - cache_list: 每层的缓存对象列表
            - manager: SimplifiedSSMCacheManager 实例

    Example:
        >>> from mlx_lm import load
        >>> model, tokenizer = load("path/to/model")
        >>>
        >>> # 注入简化缓存
        >>> cache_list, manager = inject_simplified_ssm_cache(model)
        >>>
        >>> # 模型会自动使用缓存
        >>> output = model.generate(prompt, max_tokens=100)
        >>>
        >>> # 查看统计
        >>> print(manager.get_statistics())
    """
    from flashmlx.cache.injection import create_layer_types_from_model
    from flashmlx.cache.per_layer_ssm_cache import PerLayerSSMCache
    from flashmlx.cache.hybrid_cache_manager import LayerType
    from mlx_lm.models.cache import ArraysCache

    # 自动检测层类型
    layer_types = create_layer_types_from_model(model)

    # 创建简化的 SSM 缓存管理器
    manager = SimplifiedSSMCacheManager(max_size_bytes=max_size_bytes)

    # 为每层创建缓存对象（使用自定义 list）
    cache_list = CacheList()
    ssm_count = 0
    attention_count = 0

    for layer_idx in sorted(layer_types.keys()):
        layer_type = layer_types[layer_idx]

        if layer_type == LayerType.SSM:
            # 创建 SSM 缓存（2 slots: conv_state + ssm_state）
            # NOTE: 暂时不启用管理缓存，因为在实际 generation 中会触发 GPU page fault
            # 这个问题需要深入调查 MLX-LM 的内存管理机制
            cache = PerLayerSSMCache(
                manager=manager,
                layer_idx=layer_idx,
                size=2
            )
            # cache.enable_managed_cache()  # 暂时禁用
            ssm_count += 1
        else:
            # Attention 层使用默认 ArraysCache（简化版不压缩 Attention）
            cache = ArraysCache(size=2)
            attention_count += 1

        cache_list.append(cache)

    print(f"✓ 创建 {ssm_count} 个 SSM 缓存（简化版）")

    print(f"✓ 创建了 {ssm_count} 个 SSM 缓存 + {attention_count} 个标准缓存")

    # 自动注入（如果启用）
    if auto_inject:
        # 先让模型创建默认缓存（为 attention 层）
        if hasattr(model, 'make_cache'):
            default_cache_list = model.make_cache()
            cache_list._original_make_cache = model.make_cache

            # 只替换 SSM 层的缓存
            for layer_idx in range(len(cache_list)):
                if layer_types.get(layer_idx) != LayerType.SSM:
                    # 保留默认的 attention cache
                    cache_list[layer_idx] = default_cache_list[layer_idx]

        # 替换 make_cache 返回我们的缓存列表
        model.make_cache = lambda: cache_list

        # 设置 model.cache（向后兼容）
        if hasattr(model, 'cache'):
            cache_list._original_cache = model.cache
        model.cache = cache_list

        # 存储管理器引用（用于统计）
        cache_list._manager = manager

        print(f"✓ 已注入到 model.cache")

    return cache_list, manager
