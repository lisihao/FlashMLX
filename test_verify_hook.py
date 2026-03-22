#!/usr/bin/env python3
"""
验证 hook 是否被正确设置
"""

from mlx_lm import load
from flashmlx.cache.simple_injection_v3 import inject_attention_matching_v3


def main():
    print("=" * 80)
    print("验证 Hook 设置")
    print("=" * 80)

    model_path = "/Volumes/toshiba/models/qwen3-8b-mlx"
    print(f"\n🔄 加载模型...")
    model, tokenizer = load(model_path)
    print(f"✓ 模型加载成功")

    # 记录原始 __call__
    layers = model.model.layers
    original_call = layers[0].self_attn.__call__
    print(f"\n原始 __call__: {original_call}")
    print(f"原始 __call__ 类型: {type(original_call)}")

    # 注入
    print(f"\n🔧 注入 Attention Matching V3...")
    compressor = inject_attention_matching_v3(
        model,
        compression_ratio=2.0,
        num_queries=100,
        verbose=True
    )

    # 检查 __call__ 是否被替换
    new_call = layers[0].self_attn.__call__
    print(f"\n新的 __call__: {new_call}")
    print(f"新的 __call__ 类型: {type(new_call)}")
    print(f"__call__ 是否被替换: {new_call is not original_call}")

    # 尝试手动调用一次
    print(f"\n🧪 手动调用 layer 0 attention...")
    import mlx.core as mx
    x = mx.random.normal((1, 10, 3584))  # (B, L, hidden_size)
    try:
        output = layers[0].self_attn(x, mask=None, cache=None)
        print(f"输出 shape: {output.shape}")
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
