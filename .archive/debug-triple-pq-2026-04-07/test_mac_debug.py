#!/usr/bin/env python3
"""
Debug MAC - 检查ring cache状态
"""

import sys
sys.path.insert(0, "src")
sys.path.insert(0, "mlx-lm-source")

# Monkey patch MAC to add debug output
original_mac_call = None

def debug_mac_call(self, q, k, v, req_ids):
    """添加debug输出的MAC wrapper"""
    # 检查ring cache状态
    print(f"[MAC DEBUG] req_ids: {req_ids.tolist()}, k.shape: {k.shape}", flush=True)

    # 调用原始MAC
    return original_mac_call(q, k, v, req_ids)

# Patch before import
from flashmlx.mac import MACDecodeWrapper
original_mac_call = MACDecodeWrapper.__call__
MACDecodeWrapper.__call__ = debug_mac_call

from mlx_lm import load, generate
import flashmlx

print("加载模型...")
model, tokenizer = load("/Volumes/toshiba/models/qwen3-8b-mlx")
print("✅ 模型加载完成\n")

flashmlx.patch_mlx_lm()

prompt = "你好" * 100
print(f"提示: {len(tokenizer.encode(prompt))} tokens")
print("生成 3 tokens...\n")

response = generate(
    model, tokenizer,
    prompt=prompt,
    max_tokens=3,
    verbose=False
)

print(f"\n✅ 完成: {response[:50]}...")
