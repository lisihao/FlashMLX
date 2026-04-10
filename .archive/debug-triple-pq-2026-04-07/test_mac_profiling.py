#!/usr/bin/env python3
"""
MAC-Attention 性能Profiling - 找出瓶颈
"""

import sys
sys.path.insert(0, "src")
sys.path.insert(0, "mlx-lm-source")

from mlx_lm import load, generate
import flashmlx

print("=" * 80)
print("MAC-Attention Performance Profiling")
print("=" * 80)
print()

# 加载模型
print("加载模型...")
model, tokenizer = load("/Volumes/toshiba/models/qwen3-8b-mlx")
print("✅ 模型加载完成\n")

# 启用MAC + Profiling
flashmlx.patch_mlx_lm()
flashmlx.enable_profiling()

# 测试：500 tokens 上下文 (避免内存不足)
prompt = "你好" * 250  # ~250 tokens
actual_len = len(tokenizer.encode(prompt))
print(f"提示长度: {actual_len:,} tokens")
print("生成 10 tokens...\n")

response = generate(
    model, tokenizer,
    prompt=prompt,
    max_tokens=10,
    verbose=False
)

print("✅ 生成完成\n")

# 查看profiling结果
print(flashmlx.get_profiling_stats())

# 分析
print("\n" + "=" * 80)
print("分析")
print("=" * 80)
print("""
关键指标:
- mac_call: MAC本身的计算时间
- mac_data_convert: 数据转换开销 (transpose + astype)
- mac_warmup: Prefill阶段的预热开销
- qkv_proj: QKV projection
- rope_and_cache: RoPE + cache update
- output_proj: Output projection
- total_attention: 完整attention时间

如果 mac_data_convert 占比很大，说明数据转换是瓶颈
如果 mac_call 本身就很慢，说明MAC实现有问题
""")
