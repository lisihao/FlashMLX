#!/usr/bin/env python3
"""
找到MAC的crossover point
"""

import sys
sys.path.insert(0, "src")
sys.path.insert(0, "mlx-lm-source")

from mlx_lm import load, generate
import flashmlx
import time

model, tokenizer = load("/Volumes/toshiba/models/qwen3-8b-mlx")

contexts = [30000, 40000]

for repeat in contexts:
    prompt = "你好" * int(repeat/2)
    actual_len = len(tokenizer.encode(prompt))

    print(f"\n{'='*80}")
    print(f"{actual_len:,} tokens")
    print(f"{'='*80}")

    # Standard
    flashmlx.unpatch_mlx_lm()
    t0 = time.time()
    try:
        _ = generate(model, tokenizer, prompt=prompt, max_tokens=3, verbose=False)
        t_std = time.time() - t0
        print(f"Standard: {t_std:.1f}s")
    except Exception as e:
        print(f"Standard: 失败 - {e}")
        t_std = None

    # MAC
    flashmlx.patch_mlx_lm()
    t0 = time.time()
    try:
        _ = generate(model, tokenizer, prompt=prompt, max_tokens=3, verbose=False)
        t_mac = time.time() - t0
        print(f"MAC: {t_mac:.1f}s")
    except Exception as e:
        print(f"MAC: 失败 - {e}")
        t_mac = None

    if t_std and t_mac:
        speedup = t_std / t_mac
        status = "🎉" if speedup > 1.0 else "😞"
        print(f"加速: {speedup:.2f}× {status}")
