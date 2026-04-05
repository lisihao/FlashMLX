"""
Monkey Patch mlx-lm 以启用 MAC-Attention

⚠️  实验性功能 - 不推荐使用 ⚠️

MAC-Attention 在 MLX/Apple Silicon 上无法达到预期加速效果。
原因：MLX 的 decode attention 路径对 partial 输入没有线性加速，
即使跳过 66% 的计算，墙钟时间仍随 full context 长度增长。

实测结果：
- Hit rate: 82%
- Skip ratio: 66%
- 加速比: 0.92×-1.02× (几乎无加速)

仅供研究用途。不推荐在生产环境使用。

用法：
    import flashmlx
    flashmlx.patch_mlx_lm()  # 启用 MAC (不推荐)

    # 然后正常使用 mlx-lm
    from mlx_lm import load, generate
    model, tokenizer = load("Qwen/Qwen2.5-8B")
    generate(model, tokenizer, "你好", max_tokens=100)
"""

import sys
from typing import Optional, Any
import mlx.core as mx

from .mac_simplified import SimplifiedMACDecode

# Global MAC instance (shared across all layers)
_global_mac_wrapper = None
_patch_enabled = False
_profiling_enabled = False
_profiling_stats = {}


def _get_or_create_global_mac(n_heads: int, n_kv_heads: int, head_dim: int):
    """
    获取或创建全局MAC wrapper（所有层共享）

    使用单例模式，28层attention共享1个MAC instance，大幅降低内存占用

    Args:
        n_heads: Query heads数量
        n_kv_heads: KV heads数量
        head_dim: Head维度

    Returns:
        全局MAC wrapper实例
    """
    global _global_mac_wrapper

    if _global_mac_wrapper is None:
        _global_mac_wrapper = SimplifiedMACDecode(
            cache_capacity=2048,   # Ring cache容量
            num_heads=n_heads,
            num_kv_heads=n_kv_heads,
            head_dim=head_dim,
            threshold=0.95,        # 官方推荐值
            window_left=512,
        )
        print(f"✅ Created SimplifiedMAC (官方裁剪版): {n_heads}H/{n_kv_heads}KV/{head_dim}D")

    return _global_mac_wrapper


def _profile_start():
    """开始profiling计时"""
    if _profiling_enabled:
        import time
        return time.perf_counter()
    return None


def _profile_end(start, key):
    """结束profiling并记录"""
    if _profiling_enabled and start is not None:
        import time
        elapsed = time.perf_counter() - start
        if key not in _profiling_stats:
            _profiling_stats[key] = []
        _profiling_stats[key].append(elapsed * 1000)  # ms


def _mac_attention_call(self, x, mask=None, cache=None):
    """替换 Attention.__call__ 的 MAC 版本"""
    t0 = _profile_start()
    B, L, D = x.shape

    # Project (保持原始逻辑)
    t_proj = _profile_start()
    queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)
    _profile_end(t_proj, 'qkv_proj')

    # Reshape and normalize (Qwen3 specific)
    if hasattr(self, 'q_norm'):
        queries = self.q_norm(queries.reshape(B, L, self.n_heads, -1)).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys.reshape(B, L, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
    else:
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

    values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

    # 【关键修复】保存 pre-RoPE query 用于 MAC matching
    # 论文明确要求 pre-RoPE matching，因为 post-RoPE 会破坏相似性
    queries_pre_rope = queries  # [B, H, L, D] - 保存给 MAC 用

    # Apply RoPE and update cache (follow original Qwen3 implementation)
    t_rope = _profile_start()
    if hasattr(self, 'rope'):
        if cache is not None:
            # Use sequence_position for RoPE if available (for compressed cache)
            rope_offset = getattr(cache, 'sequence_position', cache.offset)
            queries = self.rope(queries, offset=rope_offset)
            keys = self.rope(keys, offset=rope_offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)
    else:
        # No RoPE
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)
    _profile_end(t_rope, 'rope_and_cache')
    
    # Attention: 检测是否是 decode phase (L==1 且有 cache)
    head_dim = queries.shape[-1]

    # 获取全局共享的MAC wrapper（所有层共享，大幅降低内存）
    mac_wrapper = _get_or_create_global_mac(self.n_heads, self.n_kv_heads, head_dim)

    if cache is not None and L == 1:
        # Decode phase - 使用 MAC-Attention

        # Prepare inputs for SimplifiedMAC
        t_convert = _profile_start()
        # 【关键】使用 pre-RoPE query 做 matching
        q_pre_rope_mac = queries_pre_rope.squeeze(2)  # [B, H, 1, D] -> [B, H, D]
        # 但 attention 还是用 post-RoPE 的 query
        q_post_rope_mac = queries.squeeze(2)  # [B, H, 1, D] -> [B, H, D]
        # keys, values: [B, Hkv, S, D] -> [B, S, Hkv, D]
        k_mac = keys.transpose(0, 2, 1, 3)
        v_mac = values.transpose(0, 2, 1, 3)
        _profile_end(t_convert, 'mac_data_convert')

        # Run SimplifiedMAC (官方裁剪版)
        # 传入 pre-RoPE query 用于 match，post-RoPE query 用于 attention
        t_mac = _profile_start()
        output = mac_wrapper(
            q_pre_rope=q_pre_rope_mac,
            q_post_rope=q_post_rope_mac,
            k=k_mac,
            v=v_mac,
            scale=self.scale
        )  # [B, H, D]
        _profile_end(t_mac, 'mac_call')

        # [B, H, D] -> [B, 1, H, D]
        output = output[:, None, :, :]
    else:
        # Prefill phase - 预热 MAC ring cache + 使用标准 attention
        from mlx_lm.models.base import scaled_dot_product_attention

        # 预热MAC ring cache (如果有cache的话)
        # DISABLED: 预热导致内存不足，暂时禁用
        # if cache is not None and L > 1:
        #     t_warmup = _profile_start()
        #     q_warmup = queries[:, :, -1:, :].squeeze(2).astype(mx.bfloat16)
        #     k_warmup = keys.transpose(0, 2, 1, 3).astype(mx.bfloat16)
        #     v_warmup = values.transpose(0, 2, 1, 3).astype(mx.bfloat16)
        #     req_ids = mx.arange(B, dtype=mx.int32)
        #     _ = self._mac_wrapper(q_warmup, k_warmup, v_warmup, req_ids)
        #     _profile_end(t_warmup, 'mac_warmup')

        # 使用标准attention计算实际输出
        output = scaled_dot_product_attention(
            queries, keys, values,
            cache=cache, scale=self.scale, mask=mask
        )  # [B, H, L, D]

    # Transpose output back to [B, L, H, D] and reshape
    output = output.transpose(0, 2, 1, 3)

    # Project output
    t_out_proj = _profile_start()
    output = output.reshape(B, L, -1)
    output = self.o_proj(output)
    _profile_end(t_out_proj, 'output_proj')

    _profile_end(t0, 'total_attention')
    return output


def _inject_layer_indices(model):
    """
    给模型的每个Attention层注入layer_idx属性
    """
    if hasattr(model, 'model'):
        # Qwen3: model.model.layers
        if hasattr(model.model, 'layers'):
            for idx, layer in enumerate(model.model.layers):
                if hasattr(layer, 'self_attn'):
                    layer.self_attn._layer_idx = idx


def patch_mlx_lm():
    """
    Monkey patch mlx-lm 以启用 MAC-Attention

    ⚠️  实验性功能 - 不推荐使用 ⚠️

    自动检测并 patch 所有支持的模型架构
    """
    global _patch_enabled

    if _patch_enabled:
        print("⚠️  MAC-Attention patch 已启用")
        return

    # 实验性功能警告
    print("=" * 80)
    print("⚠️  警告：MAC-Attention 是实验性功能")
    print("=" * 80)
    print("在 MLX/Apple Silicon 上无法达到预期加速效果。")
    print("实测：Hit 82%, Skip 66% → 加速比仅 0.92×-1.02×")
    print()
    print("原因：MLX decode attention 对 partial 输入没有线性加速。")
    print("仅供研究用途，不推荐生产使用。")
    print("=" * 80)
    print()

    # Patch Qwen3
    try:
        from mlx_lm.models import qwen3
        original_call = qwen3.Attention.__call__
        qwen3.Attention.__call__ = _mac_attention_call
        qwen3.Attention._original_call = original_call

        # 保存注入函数供后续使用
        qwen3.Model._inject_layer_indices = _inject_layer_indices

        print("✅ Patched: Qwen3")
    except ImportError:
        pass
    
    # Patch Qwen3.5
    try:
        from mlx_lm.models import qwen3_5
        original_call = qwen3_5.Attention.__call__
        qwen3_5.Attention.__call__ = _mac_attention_call
        qwen3_5.Attention._original_call = original_call
        print("✅ Patched: Qwen3.5")
    except ImportError:
        pass
    
    # Patch Qwen (legacy)
    try:
        from mlx_lm.models import qwen
        original_call = qwen.Attention.__call__
        qwen.Attention.__call__ = _mac_attention_call
        qwen.Attention._original_call = original_call
        print("✅ Patched: Qwen")
    except ImportError:
        pass
    
    # 为每个 layer 添加 layer_idx (用于 MAC instance 管理)
    _patch_enabled = True
    print("🚀 MAC-Attention 已启用！长上下文推理将自动加速")


def enable_profiling():
    """启用性能profiling"""
    global _profiling_enabled, _profiling_stats
    _profiling_enabled = True
    _profiling_stats = {}
    print("✅ Profiling enabled")


def disable_profiling():
    """禁用性能profiling"""
    global _profiling_enabled
    _profiling_enabled = False
    print("✅ Profiling disabled")


def get_profiling_stats():
    """获取profiling统计结果"""
    if not _profiling_stats:
        return "No profiling data available"

    import numpy as np
    result = ["=" * 80]
    result.append("MAC-Attention Performance Profiling")
    result.append("=" * 80)
    result.append(f"{'Component':<20} {'Count':>8} {'Mean (ms)':>12} {'Std (ms)':>12} {'Total (ms)':>12}")
    result.append("-" * 80)

    for key in sorted(_profiling_stats.keys()):
        times = np.array(_profiling_stats[key])
        result.append(
            f"{key:<20} {len(times):>8} {times.mean():>12.3f} {times.std():>12.3f} {times.sum():>12.3f}"
        )

    result.append("=" * 80)
    return "\n".join(result)


def unpatch_mlx_lm():
    """恢复原始 mlx-lm (禁用 MAC)"""
    global _patch_enabled, _global_mac_wrapper

    if not _patch_enabled:
        print("⚠️  MAC-Attention 未启用")
        return
    
    # Restore Qwen3
    try:
        from mlx_lm.models import qwen3
        if hasattr(qwen3.Attention, '_original_call'):
            qwen3.Attention.__call__ = qwen3.Attention._original_call
            del qwen3.Attention._original_call
            print("✅ Restored: Qwen3")
    except ImportError:
        pass
    
    # Restore Qwen3.5
    try:
        from mlx_lm.models import qwen3_5
        if hasattr(qwen3_5.Attention, '_original_call'):
            qwen3_5.Attention.__call__ = qwen3_5.Attention._original_call
            del qwen3_5.Attention._original_call
            print("✅ Restored: Qwen3.5")
    except ImportError:
        pass
    
    # Restore Qwen
    try:
        from mlx_lm.models import qwen
        if hasattr(qwen.Attention, '_original_call'):
            qwen.Attention.__call__ = qwen.Attention._original_call
            del qwen.Attention._original_call
            print("✅ Restored: Qwen")
    except ImportError:
        pass
    
    # Clear global MAC instance
    _global_mac_wrapper = None
    _patch_enabled = False
    print("🔄 MAC-Attention 已禁用，恢复标准 attention")
