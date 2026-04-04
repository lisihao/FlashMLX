"""
Qwen3 with MAC-Attention Integration

完整集成 MAC-Attention 的 Qwen3 模型，支持长上下文高速推理
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import mlx.core as mx
import mlx.nn as nn

# Import from mlx-lm
try:
    from mlx_lm.models.base import BaseModelArgs, create_attention_mask
    from mlx_lm.models.activations import swiglu
    from mlx_lm.models.rope_utils import initialize_rope
except ImportError:
    raise ImportError("需要安装 mlx-lm: pip install mlx-lm")

# Import MAC-Attention
from ..mac import MACDecodeWrapper


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int
    max_position_embeddings: int
    rope_theta: float
    head_dim: int
    tie_word_embeddings: bool
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    
    # MAC-Attention 配置
    use_mac: bool = True  # 是否启用 MAC-Attention
    mac_threshold: float = 0.5  # MAC 匹配阈值
    mac_capacity: int = 1024  # MAC ring cache 容量


class MACAttention(nn.Module):
    """MAC-Attention 集成的 Attention 层"""
    
    def __init__(self, args: ModelArgs):
        super().__init__()
        
        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads
        head_dim = args.head_dim
        self.scale = head_dim**-0.5
        
        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)
        
        self.q_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        self.rope = initialize_rope(
            head_dim,
            base=args.rope_theta,
            traditional=False,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )
        
        # MAC-Attention wrapper
        if args.use_mac:
            self.mac = MACDecodeWrapper(
                max_requests=64,  # 支持多请求
                capacity=args.mac_capacity,
                num_heads=n_heads,
                num_kv_heads=n_kv_heads,
                head_dim=head_dim,
                threshold=args.mac_threshold,
                band_r=256,
                window_left=256,
                normalize_queries=True,
            )
        else:
            self.mac = None
    
    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape
        
        # Project
        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        
        # Reshape and normalize
        queries = self.q_norm(queries.reshape(B, L, self.n_heads, -1))
        keys = self.k_norm(keys.reshape(B, L, self.n_kv_heads, -1))
        values = values.reshape(B, L, self.n_kv_heads, -1)
        
        # Apply RoPE
        if cache is not None:
            queries_rope = self.rope(queries.transpose(0, 2, 1, 3), offset=cache.offset)
            keys_rope = self.rope(keys.transpose(0, 2, 1, 3), offset=cache.offset)
            
            # Update KV cache
            keys_cached, values_cached = cache.update_and_fetch(
                keys_rope.transpose(0, 2, 1, 3), 
                values.transpose(0, 2, 1, 3)
            )
            keys_rope = keys_cached.transpose(0, 2, 1, 3)
            values_rope = values_cached.transpose(0, 2, 1, 3)
            queries_rope = queries_rope.transpose(0, 2, 1, 3)
        else:
            # Prefill phase - no MAC
            queries_rope = self.rope(queries.transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3)
            keys_rope = self.rope(keys.transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3)
            values_rope = values
        
        # Attention computation
        if self.mac is not None and cache is not None and L == 1:
            # Decode phase: use MAC-Attention
            # MAC expects: [N, H, D] queries, [N, S, Hkv, D] keys/values
            q_mac = queries_rope.squeeze(1)  # [B, H, D]
            k_mac = keys_rope.transpose(0, 2, 1, 3)  # [B, S, Hkv, D]
            v_mac = values_rope.transpose(0, 2, 1, 3)  # [B, S, Hkv, D]
            
            # Request IDs (assume batch processing)
            req_ids = mx.arange(B, dtype=mx.int32)
            
            # Run MAC-Attention
            output = self.mac(q_mac, k_mac, v_mac, req_ids)  # [B, H, D]
            output = output[:, None, :, :]  # [B, 1, H, D]
        else:
            # Prefill or standard attention
            # Use MLX's scaled_dot_product_attention
            from mlx_lm.models.base import scaled_dot_product_attention
            
            queries_sdpa = queries_rope.transpose(0, 2, 1, 3)  # [B, H, L, D]
            keys_sdpa = keys_rope.transpose(0, 2, 1, 3)  # [B, Hkv, S, D]
            values_sdpa = values_rope.transpose(0, 2, 1, 3)  # [B, Hkv, S, D]
            
            output = scaled_dot_product_attention(
                queries_sdpa, keys_sdpa, values_sdpa, 
                cache=cache, scale=self.scale, mask=mask
            )  # [B, H, L, D]
            output = output.transpose(0, 2, 1, 3)  # [B, L, H, D]
        
        # Reshape and project
        output = output.reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attn = MACAttention(args)
        self.mlp = MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask=mask, cache=cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [TransformerBlock(args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[list] = None,
    ):
        h = self.embed_tokens(inputs)

        mask = create_attention_mask(h, cache)

        if cache is None:
            cache = [None] * len(self.layers)

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, cache=c)

        return self.norm(h)


# 兼容 mlx-lm 的接口
def sanitize(weights):
    """MLX-LM 兼容的权重清理"""
    return weights
