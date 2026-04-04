"""
Monkey Patch mlx-lm 以启用 MAC-Attention

用法：
    import flashmlx
    flashmlx.patch_mlx_lm()  # 一行启用 MAC
    
    # 然后正常使用 mlx-lm
    from mlx_lm import load, generate
    model, tokenizer = load("Qwen/Qwen2.5-8B")
    generate(model, tokenizer, "你好", max_tokens=100)
"""

import sys
from typing import Optional, Any
import mlx.core as mx

from .mac import MACDecodeWrapper

# Global MAC instances (per layer)
_mac_instances = {}
_patch_enabled = False


def _create_mac_for_layer(layer_idx: int, n_heads: int, n_kv_heads: int, head_dim: int):
    """创建 MAC wrapper for specific layer"""
    key = (layer_idx, n_heads, n_kv_heads, head_dim)
    if key not in _mac_instances:
        _mac_instances[key] = MACDecodeWrapper(
            max_requests=64,
            capacity=1024,
            num_heads=n_heads,
            num_kv_heads=n_kv_heads,
            head_dim=head_dim,
            threshold=0.5,
            band_r=256,
            window_left=256,
            normalize_queries=True,
        )
    return _mac_instances[key]


def _mac_attention_call(self, x, mask=None, cache=None):
    """替换 Attention.__call__ 的 MAC 版本"""
    B, L, D = x.shape
    
    # Project (保持原始逻辑)
    queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)
    
    # Reshape and normalize (Qwen3 specific)
    if hasattr(self, 'q_norm'):
        queries = self.q_norm(queries.reshape(B, L, self.n_heads, -1))
        keys = self.k_norm(keys.reshape(B, L, self.n_kv_heads, -1))
    else:
        queries = queries.reshape(B, L, self.n_heads, -1)
        keys = keys.reshape(B, L, self.n_kv_heads, -1)
    
    values = values.reshape(B, L, self.n_kv_heads, -1)
    
    # Apply RoPE if exists
    if hasattr(self, 'rope'):
        if cache is not None:
            queries_rope = self.rope(queries.transpose(0, 2, 1, 3), offset=cache.offset)
            keys_rope = self.rope(keys.transpose(0, 2, 1, 3), offset=cache.offset)
            
            # Update KV cache
            keys_cached, values_cached = cache.update_and_fetch(
                keys_rope.transpose(0, 2, 1, 3), 
                values.transpose(0, 2, 1, 3)
            )
        else:
            queries_rope = self.rope(queries.transpose(0, 2, 1, 3))
            keys_rope = self.rope(keys.transpose(0, 2, 1, 3))
            keys_cached = keys_rope.transpose(0, 2, 1, 3)
            values_cached = values
    else:
        # No RoPE
        if cache is not None:
            keys_cached, values_cached = cache.update_and_fetch(keys, values)
        else:
            keys_cached, values_cached = keys, values
        queries_rope = queries.transpose(0, 2, 1, 3)
    
    # Attention: 检测是否是 decode phase (L==1 且有 cache)
    if cache is not None and L == 1:
        # Decode phase - 使用 MAC-Attention
        head_dim = queries.shape[-1]
        
        # 获取或创建 MAC instance
        if not hasattr(self, '_mac_wrapper'):
            layer_idx = getattr(self, '_layer_idx', 0)
            self._mac_wrapper = _create_mac_for_layer(
                layer_idx, self.n_heads, self.n_kv_heads, head_dim
            )
        
        # Prepare inputs for MAC
        q_mac = queries_rope.transpose(0, 2, 1, 3).squeeze(2)  # [B, H, D]
        k_mac = keys_cached.transpose(0, 2, 1, 3)  # [B, S, Hkv, D]
        v_mac = values_cached.transpose(0, 2, 1, 3)  # [B, S, Hkv, D]
        
        # 转换为 bf16 (MAC 需要)
        q_mac = q_mac.astype(mx.bfloat16)
        k_mac = k_mac.astype(mx.bfloat16)
        v_mac = v_mac.astype(mx.bfloat16)
        
        # Request IDs
        req_ids = mx.arange(B, dtype=mx.int32)
        
        # Run MAC
        output = self._mac_wrapper(q_mac, k_mac, v_mac, req_ids)  # [B, H, D]
        output = output[:, None, :, :]  # [B, 1, H, D]
    else:
        # Prefill phase - 使用标准 attention
        from mlx_lm.models.base import scaled_dot_product_attention
        
        queries_sdpa = queries_rope.transpose(0, 1, 2, 3) if queries_rope.ndim == 4 else queries.transpose(0, 2, 1, 3)
        keys_sdpa = keys_cached.transpose(0, 2, 1, 3)
        values_sdpa = values_cached.transpose(0, 2, 1, 3)
        
        output = scaled_dot_product_attention(
            queries_sdpa, keys_sdpa, values_sdpa,
            cache=cache, scale=self.scale, mask=mask
        )  # [B, H, L, D]
        output = output.transpose(0, 2, 1, 3)  # [B, L, H, D]
    
    # Project output
    output = output.reshape(B, L, -1)
    return self.o_proj(output)


def patch_mlx_lm():
    """
    Monkey patch mlx-lm 以启用 MAC-Attention
    
    自动检测并 patch 所有支持的模型架构
    """
    global _patch_enabled
    
    if _patch_enabled:
        print("⚠️  MAC-Attention patch 已启用")
        return
    
    # Patch Qwen3
    try:
        from mlx_lm.models import qwen3
        original_call = qwen3.Attention.__call__
        qwen3.Attention.__call__ = _mac_attention_call
        qwen3.Attention._original_call = original_call
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


def unpatch_mlx_lm():
    """恢复原始 mlx-lm (禁用 MAC)"""
    global _patch_enabled, _mac_instances
    
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
    
    # Clear MAC instances
    _mac_instances.clear()
    _patch_enabled = False
    print("🔄 MAC-Attention 已禁用，恢复标准 attention")
