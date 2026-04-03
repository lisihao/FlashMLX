"""
Instance A — Prefill + h^(0) Capture + Shared Memory Write.

Loads Qwen3-8B, runs prefill on the prompt text, captures h^(0) via
apply_h0_capture_only, and writes the residual to POSIX shared memory
for Instance B to reconstruct.

Usage (standalone):
    python3 experiments/dual_instance_h0/instance_a_prefill.py \\
        --model /path/to/Qwen3-8B-MLX-4bit \\
        --prompt "Read this document carefully..." \\
        --shm-name flashmlx_h0_bridge \\
        --max-tokens 8192

Output (JSON to stdout):
    {"status": "ok", "n_tokens": 4096, "h0_bytes": 33554432,
     "prefill_ms": 234.5, "write_ms": 1.2, "d_hidden": 4096}
"""

from __future__ import annotations

import argparse
import json
import sys
import time

sys.path.insert(0, "/Users/lisihao/FlashMLX/mlx-lm-source")
sys.path.insert(0, "/Users/lisihao/FlashMLX/src")

import mlx.core as mx

from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.kv_direct_cache import (
    H0Store,
    apply_h0_capture_only,
    unpatch_model,
)

from shared_h0_transport import SharedH0Transport


def run_instance_a(
    model_path: str,
    prompt: str,
    shm_name: str = SharedH0Transport.DEFAULT_SHM_NAME,
    max_tokens: int = 8192,
) -> dict:
    """Run Instance A: prefill + h^(0) capture + shared memory write.

    Returns:
        dict with status and metrics.
    """
    result = {"status": "error"}

    # 1. Load model
    print("[A] Loading model...", file=sys.stderr)
    t0 = time.perf_counter()
    model, tokenizer = load(model_path)
    t_load = (time.perf_counter() - t0) * 1000
    print(f"[A] Model loaded in {t_load:.0f}ms", file=sys.stderr)

    # Warmup
    warm = mx.array(tokenizer.encode("Hello")).reshape(1, -1)
    model(warm)
    mx.eval(model.parameters())

    # 2. Create shared memory FIRST so Instance B can connect while we prefill
    # Get d_hidden from model config (not from quantized weights!)
    d_hidden = getattr(model, 'args', None)
    d_hidden = d_hidden.hidden_size if d_hidden is not None else 4096
    transport = SharedH0Transport(
        max_tokens=max_tokens, d_hidden=d_hidden, create=True, shm_name=shm_name,
    )
    print(f"[A] Shared memory created (d_hidden={d_hidden})", file=sys.stderr)

    # 3. Create standard cache + h^(0) capture
    unpatch_model(model)
    cache = make_prompt_cache(model)
    h0_store = H0Store()
    apply_h0_capture_only(model, h0_store)

    # 4. Tokenize
    tokens = mx.array(tokenizer.encode(prompt))
    n_tokens = tokens.shape[0]
    print(f"[A] Prompt: {n_tokens} tokens", file=sys.stderr)

    # 5. Prefill
    print("[A] Running prefill...", file=sys.stderr)
    t_prefill_start = time.perf_counter()
    model_out = model(tokens.reshape(1, -1), cache=cache)
    mx.eval(model_out)
    t_prefill = (time.perf_counter() - t_prefill_start) * 1000
    print(f"[A] Prefill done: {t_prefill:.0f}ms ({n_tokens / t_prefill * 1000:.0f} tok/s)", file=sys.stderr)

    # 6. Extract h^(0)
    assert h0_store.count == n_tokens, (
        f"h0_store.count={h0_store.count} != n_tokens={n_tokens}"
    )
    h0 = h0_store.get_range(0, h0_store.count)  # (1, N, d_hidden)
    mx.eval(h0)
    print(f"[A] h^(0): shape={h0.shape}, dtype={h0.dtype}, d_hidden={d_hidden}", file=sys.stderr)
    t_write_start = time.perf_counter()
    h0_bytes = transport.write_h0(h0, first_token_id=tokens[0].item())
    t_write = (time.perf_counter() - t_write_start) * 1000
    print(f"[A] Written: {h0_bytes} bytes ({h0_bytes/1024/1024:.1f} MB) in {t_write:.1f}ms", file=sys.stderr)

    # 7. Wait for B to read
    print("[A] Waiting for Instance B to read...", file=sys.stderr)
    if transport.wait_for_read(timeout_s=300.0):
        print("[A] Instance B confirmed read.", file=sys.stderr)
    else:
        print("[A] WARNING: Timeout waiting for Instance B", file=sys.stderr)

    # Cleanup
    transport.close()
    unpatch_model(model)

    result = {
        "status": "ok",
        "n_tokens": n_tokens,
        "d_hidden": d_hidden,
        "h0_bytes": h0_bytes,
        "prefill_ms": round(t_prefill, 2),
        "write_ms": round(t_write, 2),
        "load_ms": round(t_load, 2),
        "prefill_tok_per_s": round(n_tokens / t_prefill * 1000, 1),
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Instance A: Prefill + h^(0) capture")
    parser.add_argument("--model", required=True, help="Path to model directory")
    parser.add_argument("--prompt", required=True, help="Prompt text (or @file to read from file)")
    parser.add_argument("--shm-name", default=SharedH0Transport.DEFAULT_SHM_NAME)
    parser.add_argument("--max-tokens", type=int, default=8192)
    args = parser.parse_args()

    # Support @file syntax for long prompts
    prompt = args.prompt
    if prompt.startswith("@"):
        with open(prompt[1:], "r") as f:
            prompt = f.read()

    result = run_instance_a(
        model_path=args.model,
        prompt=prompt,
        shm_name=args.shm_name,
        max_tokens=args.max_tokens,
    )

    # Output JSON to stdout for orchestrator to parse
    print(json.dumps(result))


if __name__ == "__main__":
    main()
