"""
Instance B — Read h^(0) + Reconstruct KV + Decode.

Reads h^(0) from POSIX shared memory (written by Instance A), reconstructs
K/V cache via reconstruct_prefix_kv(), injects into standard KVCache, and
generates text.

Usage (standalone):
    python3 experiments/dual_instance_h0/instance_b_decode.py \\
        --model /path/to/Qwen3-8B-MLX-4bit \\
        --question "What is the secret project code name?" \\
        --shm-name flashmlx_h0_bridge \\
        --max-tg-tokens 200

Output (JSON to stdout):
    {"status": "ok", "answer": "...", "n_prefix_tokens": 4096,
     "read_ms": 0.5, "recon_ms": 180.0, "tg_ms": 1200.0, "tg_tok_per_s": 45.2}
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
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.kv_direct_cache import (
    H0Store,
    _find_inner_model,
    extract_kv_from_temp_caches,
    reconstruct_prefix_kv,
    reconstruct_prefix_kv_stateful,
)
from mlx_lm.sample_utils import make_sampler

from shared_h0_transport import SharedH0Transport

GREEDY = make_sampler(temp=0.0)


def run_instance_b(
    model_path: str,
    question: str,
    shm_name: str = SharedH0Transport.DEFAULT_SHM_NAME,
    max_tg_tokens: int = 200,
    timeout_s: float = 120.0,
    recon_chunk_size: int = 512,
    h0_cache_dir: str | None = None,
    prompt_hash: str | None = None,
    fan_out_reader: bool = False,
    streaming: bool = False,
) -> dict:
    """Run Instance B: read h^(0) + reconstruct + decode.

    Args:
        h0_cache_dir: If set, check for cached h^(0) on disk before SHM.
        prompt_hash: Cache key for h^(0) lookup (from orchestrator).
        fan_out_reader: If True, read without ACK (for multi-reader fan-out).

    Returns:
        dict with status, answer, and metrics.
    """
    result = {"status": "error"}

    # 1. Load model
    print("[B] Loading model...", file=sys.stderr)
    t0 = time.perf_counter()
    model, tokenizer = load(model_path)
    t_load = (time.perf_counter() - t0) * 1000
    print(f"[B] Model loaded in {t_load:.0f}ms", file=sys.stderr)

    # Warmup
    warm = mx.array(tokenizer.encode("Hello")).reshape(1, -1)
    model(warm)
    mx.eval(model.parameters())

    # 2. Read h^(0) — try disk cache first, then fall back to SHM
    h0 = None
    t_read = 0.0
    cache_hit = False

    if h0_cache_dir and prompt_hash:
        import os
        cache_path = os.path.join(h0_cache_dir, f"h0_{prompt_hash}")
        npz_path = cache_path if cache_path.endswith(".npz") else cache_path + ".npz"
        if os.path.exists(npz_path):
            print(f"[B] Cache HIT: loading h^(0) from {npz_path}", file=sys.stderr)
            t_read_start = time.perf_counter()
            loaded_store, meta = H0Store.load(cache_path)
            h0 = loaded_store.get_range(0, loaded_store.count)
            mx.eval(h0)
            t_read = (time.perf_counter() - t_read_start) * 1000
            cache_hit = True
            print(f"[B] Loaded from cache in {t_read:.1f}ms ({loaded_store.count} tokens)", file=sys.stderr)
        else:
            print(f"[B] Cache MISS: {npz_path} not found", file=sys.stderr)

    # Connect to SHM (needed for both normal and streaming modes if no cache hit)
    transport = None
    if h0 is None:
        print("[B] Connecting to shared memory...", file=sys.stderr)
        for attempt in range(int(timeout_s)):
            try:
                transport = SharedH0Transport(create=False, shm_name=shm_name)
                break
            except FileNotFoundError:
                if attempt % 10 == 0:
                    print(f"[B] Shared memory not yet available, retrying... ({attempt}s)", file=sys.stderr)
                time.sleep(1.0)
        if transport is None:
            raise TimeoutError(f"Could not connect to shared memory '{shm_name}' after {timeout_s}s")

    inner_model = _find_inner_model(model)

    if streaming and transport is not None:
        # ---- Streaming Path: read + reconstruct chunk by chunk ----
        print("[B] Streaming mode: reading and reconstructing chunks...", file=sys.stderr)
        from mlx_lm.models.cache import KVCache
        num_layers = len(inner_model.layers)
        temp_caches = [KVCache() for _ in range(num_layers)]

        t_read_start = time.perf_counter()
        n_prefix_tokens = 0
        n_chunks = 0
        for h0_chunk in transport.read_h0_streaming(timeout_s=timeout_s):
            chunk_tokens = h0_chunk.shape[1]
            reconstruct_prefix_kv_stateful(inner_model, h0_chunk, temp_caches)
            mx.eval(temp_caches[-1].keys)
            n_prefix_tokens += chunk_tokens
            n_chunks += 1
        t_read = (time.perf_counter() - t_read_start) * 1000
        transport.close()

        d_hidden = h0_chunk.shape[2]
        t_recon = t_read  # read + recon are interleaved
        n_layers = num_layers
        print(f"[B] Streamed {n_chunks} chunks, {n_prefix_tokens} tokens in {t_read:.0f}ms", file=sys.stderr)

        # Extract KV from temp_caches
        kv_pairs = extract_kv_from_temp_caches(temp_caches)
    else:
        # ---- Normal Path: read all h^(0), then reconstruct ----
        if h0 is None:
            print("[B] Waiting for h^(0) from Instance A...", file=sys.stderr)
            t_read_start = time.perf_counter()
            if fan_out_reader:
                h0 = transport.read_h0_no_ack(timeout_s=timeout_s)
            else:
                h0 = transport.read_h0(timeout_s=timeout_s)
            t_read = (time.perf_counter() - t_read_start) * 1000
            transport.close()

        n_prefix_tokens = h0.shape[1]
        d_hidden = h0.shape[2]
        print(f"[B] Received h^(0): shape={h0.shape}, dtype={h0.dtype} in {t_read:.1f}ms", file=sys.stderr)

        # Create H0Store and populate
        h0_store = H0Store()
        h0_store.append(h0)
        assert h0_store.count == n_prefix_tokens

        # Reconstruct K/V from h^(0)
        print(f"[B] Reconstructing K/V for {n_prefix_tokens} tokens...", file=sys.stderr)
        t_recon_start = time.perf_counter()
        kv_pairs = reconstruct_prefix_kv(
            inner_model, h0_store, 0, h0_store.count,
            chunk_size=recon_chunk_size, eval_every=8,
        )
        mx.eval(*[k for k, v in kv_pairs] + [v for k, v in kv_pairs])
        t_recon = (time.perf_counter() - t_recon_start) * 1000
        n_layers = len(kv_pairs)
        print(f"[B] Reconstructed {n_layers} layers in {t_recon:.0f}ms "
              f"({n_prefix_tokens / t_recon * 1000:.0f} tok/s)", file=sys.stderr)

    # Inject K/V into standard cache
    print("[B] Injecting K/V into cache...", file=sys.stderr)
    cache = make_prompt_cache(model)
    for layer_idx, (keys, values) in enumerate(kv_pairs):
        cache[layer_idx].state = (keys, values)
    mx.eval([c.keys for c in cache] + [c.values for c in cache])
    print(f"[B] Cache injected: {n_layers} layers, offset={cache[0].offset}", file=sys.stderr)

    # 6. Generate answer
    print("[B] Generating...", file=sys.stderr)
    question_tokens = mx.array(tokenizer.encode(question))
    n_question_tokens = question_tokens.shape[0]

    gen_tokens = []
    t_tg_start = time.perf_counter()
    for token_id, logprobs in generate_step(
        question_tokens, model,
        max_tokens=max_tg_tokens,
        sampler=GREEDY,
        prompt_cache=cache,
    ):
        gen_tokens.append(token_id)
    t_tg = (time.perf_counter() - t_tg_start) * 1000

    answer = tokenizer.decode(gen_tokens).strip()
    n_gen = len(gen_tokens)
    tg_tok_per_s = n_gen / t_tg * 1000 if t_tg > 0 else 0

    print(f"[B] Generated {n_gen} tokens in {t_tg:.0f}ms ({tg_tok_per_s:.1f} tok/s)", file=sys.stderr)
    print(f"[B] Answer: {answer[:200]}", file=sys.stderr)

    result = {
        "status": "ok",
        "answer": answer,
        "n_prefix_tokens": n_prefix_tokens,
        "n_question_tokens": n_question_tokens,
        "n_gen_tokens": n_gen,
        "d_hidden": d_hidden,
        "n_layers": n_layers,
        "read_ms": round(t_read, 2),
        "recon_ms": round(t_recon, 2),
        "tg_ms": round(t_tg, 2),
        "tg_tok_per_s": round(tg_tok_per_s, 1),
        "load_ms": round(t_load, 2),
        "cache_hit": cache_hit,
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Instance B: Reconstruct + Decode")
    parser.add_argument("--model", required=True, help="Path to model directory")
    parser.add_argument("--question", required=True, help="Question text to generate answer for")
    parser.add_argument("--shm-name", default=SharedH0Transport.DEFAULT_SHM_NAME)
    parser.add_argument("--max-tg-tokens", type=int, default=200)
    parser.add_argument("--timeout", type=float, default=120.0, help="Timeout waiting for h^(0)")
    parser.add_argument("--recon-chunk-size", type=int, default=512)
    parser.add_argument("--h0-cache-dir", default=None,
                        help="Directory to check for cached h^(0)")
    parser.add_argument("--prompt-hash", default=None,
                        help="Prompt hash for cache lookup (from orchestrator)")
    parser.add_argument("--fan-out-reader", action="store_true",
                        help="Read without ACK (for multi-reader fan-out)")
    parser.add_argument("--streaming", action="store_true",
                        help="Read h^(0) in streaming chunks (pipeline with A)")
    args = parser.parse_args()

    result = run_instance_b(
        model_path=args.model,
        question=args.question,
        shm_name=args.shm_name,
        max_tg_tokens=args.max_tg_tokens,
        timeout_s=args.timeout,
        recon_chunk_size=args.recon_chunk_size,
        h0_cache_dir=args.h0_cache_dir,
        prompt_hash=args.prompt_hash,
        fan_out_reader=args.fan_out_reader,
        streaming=args.streaming,
    )

    # Output JSON to stdout for orchestrator to parse
    print(json.dumps(result))


if __name__ == "__main__":
    main()
