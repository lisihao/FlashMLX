"""
Baseline — Single-instance gold reference.

Standard single-process flow: load model → prefill full text → generate.
Used as ground truth to verify the dual-instance h^(0) experiment.

Usage:
    python3 experiments/dual_instance_h0/baseline_single.py \\
        --model /path/to/Qwen3-8B-MLX-4bit \\
        --prompt "Read this document carefully..." \\
        --question "What is the secret code name?" \\
        --max-tg-tokens 200

Output (JSON to stdout):
    {"status": "ok", "answer": "...", "n_prompt_tokens": 4200,
     "prefill_ms": 300.0, "tg_ms": 1100.0, "tg_tok_per_s": 48.5}
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
from mlx_lm.sample_utils import make_sampler

GREEDY = make_sampler(temp=0.0)


def run_baseline(
    model_path: str,
    prompt: str,
    question: str,
    max_tg_tokens: int = 200,
) -> dict:
    """Run single-instance baseline.

    Prefills prompt + question as one continuous sequence, then generates.

    Returns:
        dict with status, answer, and metrics.
    """
    result = {"status": "error"}

    # 1. Load model
    print("[Baseline] Loading model...", file=sys.stderr)
    t0 = time.perf_counter()
    model, tokenizer = load(model_path)
    t_load = (time.perf_counter() - t0) * 1000
    print(f"[Baseline] Model loaded in {t_load:.0f}ms", file=sys.stderr)

    # Warmup
    warm = mx.array(tokenizer.encode("Hello")).reshape(1, -1)
    model(warm)
    mx.eval(model.parameters())

    # 2. Tokenize full sequence (prompt + question)
    full_text = prompt + question
    tokens = mx.array(tokenizer.encode(full_text))
    n_prompt_tokens = tokens.shape[0]
    print(f"[Baseline] Full prompt: {n_prompt_tokens} tokens", file=sys.stderr)

    # 3. Create standard cache
    cache = make_prompt_cache(model)

    # 4. Generate
    print("[Baseline] Generating...", file=sys.stderr)
    gen_tokens = []
    t_gen_start = time.perf_counter()
    for token_id, logprobs in generate_step(
        tokens, model,
        max_tokens=max_tg_tokens,
        sampler=GREEDY,
        prompt_cache=cache,
    ):
        gen_tokens.append(token_id)
    t_gen = (time.perf_counter() - t_gen_start) * 1000

    answer = tokenizer.decode(gen_tokens).strip()
    n_gen = len(gen_tokens)

    # Split timing: first token is prefill-dominated
    # (approximate: generate_step includes both prefill and TG)
    tg_tok_per_s = n_gen / t_gen * 1000 if t_gen > 0 else 0

    print(f"[Baseline] Generated {n_gen} tokens in {t_gen:.0f}ms ({tg_tok_per_s:.1f} tok/s)", file=sys.stderr)
    print(f"[Baseline] Answer: {answer[:200]}", file=sys.stderr)

    result = {
        "status": "ok",
        "answer": answer,
        "n_prompt_tokens": n_prompt_tokens,
        "n_gen_tokens": n_gen,
        "gen_ms": round(t_gen, 2),
        "tg_tok_per_s": round(tg_tok_per_s, 1),
        "load_ms": round(t_load, 2),
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Baseline: Single-instance reference")
    parser.add_argument("--model", required=True, help="Path to model directory")
    parser.add_argument("--prompt", required=True, help="Prompt text (or @file)")
    parser.add_argument("--question", required=True, help="Question text")
    parser.add_argument("--max-tg-tokens", type=int, default=200)
    args = parser.parse_args()

    # Support @file syntax
    prompt = args.prompt
    if prompt.startswith("@"):
        with open(prompt[1:], "r") as f:
            prompt = f.read()

    result = run_baseline(
        model_path=args.model,
        prompt=prompt,
        question=args.question,
        max_tg_tokens=args.max_tg_tokens,
    )

    print(json.dumps(result))


if __name__ == "__main__":
    main()
