"""
Instance A — Prefill + h^(0) Capture + Shared Memory Write.

Loads Qwen3-8B, runs prefill on the prompt text, captures h^(0) via
apply_h0_capture_only, and writes the residual to POSIX shared memory
for Instance B to reconstruct.

Modes:
  --embed-only   Use bare embed_tokens() instead of full prefill.
                 686× faster, 82% less memory. Verified bit-exact.

Usage (standalone):
    python3 experiments/dual_instance_h0/instance_a_prefill.py \\
        --model /path/to/Qwen3-8B-MLX-4bit \\
        --prompt "Read this document carefully..." \\
        --embed-only \\
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
    _find_inner_model,
    apply_h0_capture_only,
    unpatch_model,
)

from shared_h0_transport import SharedH0Transport


def run_instance_a(
    model_path: str,
    prompt: str,
    shm_name: str = SharedH0Transport.DEFAULT_SHM_NAME,
    max_tokens: int = 8192,
    embed_only: bool = False,
    h0_cache_dir: str | None = None,
    streaming: bool = False,
    chunk_size: int = 512,
) -> dict:
    """Run Instance A: prefill + h^(0) capture + shared memory write.

    Args:
        embed_only: If True, use bare embed_tokens() instead of full prefill.
            Verified bit-exact with full prefill in perf_analysis_v2.
        h0_cache_dir: If set, save h^(0) to disk for persistence cache.
        streaming: If True, write h^(0) in chunks for pipeline parallelism.
        chunk_size: Tokens per streaming chunk (default 512).

    Returns:
        dict with status and metrics.
    """
    result = {"status": "error"}
    mode = "embed_only" if embed_only else "full_prefill"

    # 1. Load model
    print(f"[A] Loading model (mode={mode})...", file=sys.stderr)
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

    # 3. Tokenize
    tokens = mx.array(tokenizer.encode(prompt))
    n_tokens = tokens.shape[0]
    print(f"[A] Prompt: {n_tokens} tokens", file=sys.stderr)

    if embed_only:
        # ---- Embed-Only Path ----
        # Only compute embed_tokens(). Verified bit-exact vs full prefill.
        inner_model = _find_inner_model(model)
        print("[A] Running embed_tokens only...", file=sys.stderr)
        t_prefill_start = time.perf_counter()
        h0 = inner_model.embed_tokens(tokens.reshape(1, -1))
        mx.eval(h0)
        t_prefill = (time.perf_counter() - t_prefill_start) * 1000
        print(f"[A] Embed done: {t_prefill:.2f}ms ({n_tokens / max(t_prefill, 0.001) * 1000:.0f} tok/s)", file=sys.stderr)
    else:
        # ---- Full Prefill Path (original) ----
        unpatch_model(model)
        cache = make_prompt_cache(model)
        h0_store = H0Store()
        apply_h0_capture_only(model, h0_store)

        print("[A] Running prefill...", file=sys.stderr)
        t_prefill_start = time.perf_counter()
        model_out = model(tokens.reshape(1, -1), cache=cache)
        mx.eval(model_out)
        t_prefill = (time.perf_counter() - t_prefill_start) * 1000
        print(f"[A] Prefill done: {t_prefill:.0f}ms ({n_tokens / t_prefill * 1000:.0f} tok/s)", file=sys.stderr)

        # Extract h^(0) from store
        assert h0_store.count == n_tokens, (
            f"h0_store.count={h0_store.count} != n_tokens={n_tokens}"
        )
        h0 = h0_store.get_range(0, h0_store.count)  # (1, N, d_hidden)
        mx.eval(h0)
        unpatch_model(model)

    print(f"[A] h^(0): shape={h0.shape}, dtype={h0.dtype}, d_hidden={d_hidden}", file=sys.stderr)

    # 4. Write h^(0) to shared memory
    from shared_h0_transport import DTYPE_BF16, DTYPE_F32, DTYPE_F16
    t_write_start = time.perf_counter()
    if streaming:
        # Streaming mode: write chunk by chunk
        dtype_code = DTYPE_BF16 if h0.dtype == mx.bfloat16 else (
            DTYPE_F32 if h0.dtype == mx.float32 else DTYPE_F16
        )
        transport.begin_streaming(n_tokens, d_hidden, dtype_code, chunk_size)
        n_chunks = (n_tokens + chunk_size - 1) // chunk_size
        h0_bytes = 0
        for ci in range(n_chunks):
            start_tok = ci * chunk_size
            end_tok = min(start_tok + chunk_size, n_tokens)
            h0_chunk = h0[:, start_tok:end_tok, :]
            cb = transport.write_h0_chunk(h0_chunk, ci, is_last=(ci == n_chunks - 1))
            h0_bytes += cb
        print(f"[A] Streamed {n_chunks} chunks", file=sys.stderr)
    else:
        h0_bytes = transport.write_h0(h0, first_token_id=tokens[0].item())
    t_write = (time.perf_counter() - t_write_start) * 1000
    print(f"[A] Written: {h0_bytes} bytes ({h0_bytes/1024/1024:.1f} MB) in {t_write:.1f}ms", file=sys.stderr)

    # 5. Optionally save h^(0) to disk cache
    prompt_hash = None
    if h0_cache_dir:
        import hashlib
        import json as _json
        import os
        os.makedirs(h0_cache_dir, exist_ok=True)
        prompt_hash = hashlib.sha256(
            _json.dumps(tokens.tolist()).encode()
        ).hexdigest()[:16]
        cache_path = os.path.join(h0_cache_dir, f"h0_{prompt_hash}")
        h0_store_for_save = H0Store()
        h0_store_for_save.append(h0)
        h0_store_for_save.save(cache_path, metadata={
            "prompt_hash": prompt_hash,
            "n_tokens": int(n_tokens),
            "d_hidden": int(d_hidden),
            "model_path": model_path,
        })
        print(f"[A] Cached h^(0) to {cache_path}.npz (hash={prompt_hash})", file=sys.stderr)

    # 6. Wait for B to read
    print("[A] Waiting for Instance B to read...", file=sys.stderr)
    if transport.wait_for_read(timeout_s=300.0):
        print("[A] Instance B confirmed read.", file=sys.stderr)
    else:
        print("[A] WARNING: Timeout waiting for Instance B", file=sys.stderr)

    # Cleanup
    transport.close()

    result = {
        "status": "ok",
        "mode": mode,
        "n_tokens": n_tokens,
        "d_hidden": d_hidden,
        "h0_bytes": h0_bytes,
        "prefill_ms": round(t_prefill, 2),
        "write_ms": round(t_write, 2),
        "load_ms": round(t_load, 2),
        "prefill_tok_per_s": round(n_tokens / max(t_prefill, 0.001) * 1000, 1),
    }
    if prompt_hash:
        result["prompt_hash"] = prompt_hash

    return result


def main():
    parser = argparse.ArgumentParser(description="Instance A: Prefill + h^(0) capture")
    parser.add_argument("--model", required=True, help="Path to model directory")
    parser.add_argument("--prompt", required=True, help="Prompt text (or @file to read from file)")
    parser.add_argument("--shm-name", default=SharedH0Transport.DEFAULT_SHM_NAME)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--embed-only", action="store_true",
                        help="Use bare embed_tokens() instead of full prefill (686x faster)")
    parser.add_argument("--h0-cache-dir", default=None,
                        help="Directory to save h^(0) for persistence cache")
    parser.add_argument("--streaming", action="store_true",
                        help="Write h^(0) in chunks for pipeline parallelism")
    parser.add_argument("--chunk-size", type=int, default=512,
                        help="Tokens per streaming chunk (default: 512)")
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
        embed_only=args.embed_only,
        h0_cache_dir=args.h0_cache_dir,
        streaming=args.streaming,
        chunk_size=args.chunk_size,
    )

    # Output JSON to stdout for orchestrator to parse
    print(json.dumps(result))


if __name__ == "__main__":
    main()
