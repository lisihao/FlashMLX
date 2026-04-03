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
    reconstruct_prefix_kv,
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
) -> dict:
    """Run Instance B: read h^(0) + reconstruct + decode.

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

    # 2. Read h^(0) from shared memory (retry connection if A hasn't created it yet)
    print("[B] Connecting to shared memory...", file=sys.stderr)
    transport = None
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

    print("[B] Waiting for h^(0) from Instance A...", file=sys.stderr)
    t_read_start = time.perf_counter()
    h0 = transport.read_h0(timeout_s=timeout_s)
    t_read = (time.perf_counter() - t_read_start) * 1000
    transport.close()

    n_prefix_tokens = h0.shape[1]
    d_hidden = h0.shape[2]
    print(f"[B] Received h^(0): shape={h0.shape}, dtype={h0.dtype} in {t_read:.1f}ms", file=sys.stderr)

    # 3. Create H0Store and populate
    h0_store = H0Store()
    h0_store.append(h0)
    assert h0_store.count == n_prefix_tokens

    # 4. Reconstruct K/V from h^(0)
    print(f"[B] Reconstructing K/V for {n_prefix_tokens} tokens...", file=sys.stderr)
    inner_model = _find_inner_model(model)
    t_recon_start = time.perf_counter()
    kv_pairs = reconstruct_prefix_kv(
        inner_model, h0_store, 0, h0_store.count,
        chunk_size=recon_chunk_size, eval_every=8,
    )
    # kv_pairs: List[Tuple[mx.array, mx.array]], per-layer (keys, values)
    mx.eval(*[k for k, v in kv_pairs] + [v for k, v in kv_pairs])
    t_recon = (time.perf_counter() - t_recon_start) * 1000
    n_layers = len(kv_pairs)
    print(f"[B] Reconstructed {n_layers} layers in {t_recon:.0f}ms "
          f"({n_prefix_tokens / t_recon * 1000:.0f} tok/s)", file=sys.stderr)

    # 5. Create standard cache and inject K/V
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
    args = parser.parse_args()

    result = run_instance_b(
        model_path=args.model,
        question=args.question,
        shm_name=args.shm_name,
        max_tg_tokens=args.max_tg_tokens,
        timeout_s=args.timeout,
        recon_chunk_size=args.recon_chunk_size,
    )

    # Output JSON to stdout for orchestrator to parse
    print(json.dumps(result))


if __name__ == "__main__":
    main()
