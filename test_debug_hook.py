#!/usr/bin/env python3
"""
调试 V3 hook 是否被调用
"""

from mlx_lm import load
from flashmlx.cache.simple_injection_v3 import inject_attention_matching_v3
import mlx.core as mx


def main():
    print("=" * 80)
    print("调试 V3 Hook")
    print("=" * 80)

    model_path = "/Volumes/toshiba/models/qwen3-8b-mlx"
    print(f"\n🔄 加载模型...")
    model, tokenizer = load(model_path)
    print(f"✓ 模型加载成功")

    # 注入（verbose=True 应该打印注入信息）
    print(f"\n🔧 注入 Attention Matching V3...")
    compressor = inject_attention_matching_v3(
        model,
        compression_ratio=2.0,
        num_queries=100,
        verbose=True  # ← 确保 verbose=True
    )

    # 手动调用一次 forward 看看是否进入 hook
    print(f"\n🧪 测试前向传播...")

    # 准备输入
    prompt = "你好，请介绍一下自己。"
    tokens = tokenizer.encode(prompt)
    print(f"提示词: {prompt}")
    print(f"Token 数量: {len(tokens)}")

    # 转换为 MLX array
    input_ids = mx.array([tokens])
    print(f"input_ids shape: {input_ids.shape}")

    # 调用模型
    print(f"\n调用模型...")
    try:
        output = model(input_ids)
        print(f"输出 shape: {output.logits.shape if hasattr(output, 'logits') else output.shape}")
        print(f"✓ 前向传播成功")
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
