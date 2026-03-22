"""
Attention Matching Injection V3 - 正确应用 Beta

完整移植作者的实现逻辑：
1. 压缩 KV cache 生成 (C1, beta, C2)
2. 替换 keys/values 为 C1/C2
3. Hook attention 函数，在 softmax 前应用 beta
"""

from typing import Optional
import types
import mlx.core as mx
import mlx.nn as nn

from .attention_matching_compressor_v2 import AttentionMatchingCompressorV2
from .attention_with_beta import scaled_dot_product_attention_with_beta


def inject_attention_matching_v3(
    model,
    compression_ratio: float = 2.0,
    num_queries: int = 100,
    verbose: bool = True
):
    """
    为模型注入 Attention Matching 压缩 V3（正确应用 Beta）

    Args:
        model: MLX-LM 模型实例
        compression_ratio: 压缩比例（默认 2.0x）
        num_queries: 查询数量（默认 100）
        verbose: 是否打印注入信息
    """
    # 创建 compressor
    compressor = AttentionMatchingCompressorV2(
        model=model,
        compression_ratio=compression_ratio,
        num_queries=num_queries,
    )

    # 获取模型层
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    elif hasattr(model, 'layers'):
        layers = model.layers
    else:
        raise ValueError("Cannot find model layers")

    num_layers = len(layers)

    if verbose:
        print(f"✓ Attention Matching V3 注入成功:")
        print(f"  - 层数: {num_layers}")
        print(f"  - 压缩比例: {compression_ratio}x")
        print(f"  - β 校准: 启用（正确应用）")
        print(f"  - 查询数量: {num_queries}")

    # Hook 每层的 attention
    for layer_idx, layer in enumerate(layers):
        # 保存原始 attention forward
        original_attention_call = layer.self_attn.__call__

        def make_hooked_attention(layer_idx, verbose_flag):
            def hooked_attention_call(self, x, mask=None, cache=None):
                """
                Hook attention __call__，替换 KV cache 并应用 beta
                注意：self 是 Attention 实例
                """
                # 无条件调试输出（所有层）
                print(f"🔥 HOOK CALLED! layer={layer_idx}, x.shape={x.shape}, cache={cache is not None}")

                B, L, D = x.shape

                # 投影 Q, K, V（使用 self 而不是 layer.self_attn）
                queries = self.q_proj(x)
                keys = self.k_proj(x)
                values = self.v_proj(x)

                # Reshape
                n_heads = self.n_heads
                n_kv_heads = self.n_kv_heads
                head_dim = D // n_heads

                queries = queries.reshape(B, L, n_heads, head_dim).transpose(0, 2, 1, 3)
                keys = keys.reshape(B, L, n_kv_heads, head_dim).transpose(0, 2, 1, 3)
                values = values.reshape(B, L, n_kv_heads, head_dim).transpose(0, 2, 1, 3)

                # Apply RoPE
                if cache is not None:
                    queries = self.rope(queries, offset=cache.offset)
                    keys = self.rope(keys, offset=cache.offset)
                    keys, values = cache.update_and_fetch(keys, values)
                else:
                    queries = self.rope(queries)
                    keys = self.rope(keys)

                # ✅ 压缩 KV cache
                # 调试输出（强制启用前3层）
                if layer_idx < 3 and cache is not None:
                    print(f"[Layer {layer_idx}] cache.offset={cache.offset}, keys.shape[2]={keys.shape[2]}, compression_ratio={compression_ratio}")

                if cache is not None and cache.offset > compression_ratio:
                    try:
                        print(f"[Layer {layer_idx}] 🔥 Triggering compression! offset={cache.offset}")
                        compressed_keys, compressed_values = compressor.compress_kv_cache(
                            layer_idx, (keys, values)
                        )

                        # 获取 beta
                        beta = None
                        for head_idx in range(n_kv_heads):
                            key = (layer_idx, head_idx)
                            if key in compressor.compressed_params:
                                C1, beta_head, C2 = compressor.compressed_params[key]
                                if beta is None:
                                    # 初始化 beta tensor
                                    beta = mx.zeros((B, n_kv_heads, beta_head.shape[0]))
                                beta[0, head_idx, :] = beta_head

                        # 替换 keys/values
                        keys = compressed_keys
                        values = compressed_values

                    except Exception as e:
                        # 压缩失败，使用原始 KV cache
                        print(f"⚠️ Warning: Compression failed for layer {layer_idx}: {e}")
                        beta = None
                else:
                    beta = None

                # Repeat KV heads for GQA
                if n_kv_heads != n_heads:
                    keys = mx.repeat(keys, n_heads // n_kv_heads, axis=1)
                    values = mx.repeat(values, n_heads // n_heads, axis=1)
                    if beta is not None:
                        beta = mx.repeat(beta, n_heads // n_kv_heads, axis=1)

                # ✅ 使用自定义 attention（应用 beta）
                output = scaled_dot_product_attention_with_beta(
                    queries,
                    keys,
                    values,
                    scale=self.scale,
                    mask=mask,
                    beta=beta,  # ✅ 传递 beta
                )

                # Reshape output
                output = output.transpose(0, 2, 1, 3).reshape(B, L, D)

                # Output projection
                return self.o_proj(output)

            return hooked_attention_call

        # 替换 attention __call__（使用 MethodType 正确绑定）
        hooked_fn = make_hooked_attention(layer_idx, verbose)
        layer.self_attn.__call__ = types.MethodType(hooked_fn, layer.self_attn)

    return compressor
