#!/usr/bin/env python3
"""Quality benchmark: MATH-500 accuracy for expert offload configs.

Tests whether expert offloading actually hurts model *capability*,
not just char-match. Runs N MATH problems, extracts boxed answers,
compares to ground truth.

Configs:
  A: standard (no offload, full 6-bit)
  B: pool=32 + zero_out + rerank (5 GB, fastest)
  C: pool=64 + zero_out + rerank (8 GB, highest HR)
"""

import argparse
import gc
import json
import re
import sys
import time

sys.path.insert(0, "mlx-lm-source")

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import stream_generate
from mlx_lm.models.expert_offload import patch_model_for_offload


def extract_boxed(text: str) -> str:
    """Extract \\boxed{...} answer from model output."""
    # Find last \boxed{...}
    matches = re.findall(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', text)
    if matches:
        return matches[-1].strip()
    # Fallback: look for "the answer is X" patterns
    m = re.search(r'(?:answer|result)\s*(?:is|=)\s*[\\$]*([^\s,.$]+)', text, re.I)
    if m:
        return m.group(1).strip()
    return ""


def normalize_answer(ans: str) -> str:
    """Normalize math answer for comparison."""
    ans = ans.strip()
    # Remove LaTeX formatting
    ans = ans.replace('\\$', '').replace('$', '')
    ans = ans.replace('\\text{', '').replace('}', '')
    ans = ans.replace('\\mathrm{', '').replace('\\', '')
    ans = ans.replace(' ', '')
    # Remove trailing period
    ans = ans.rstrip('.')
    # Try to normalize fractions
    m = re.match(r'^(\d+)/(\d+)$', ans)
    if m:
        try:
            from fractions import Fraction
            f = Fraction(int(m.group(1)), int(m.group(2)))
            ans = f"{f.numerator}/{f.denominator}" if f.denominator != 1 else str(f.numerator)
        except:
            pass
    return ans


def run_problem(model, tokenizer, problem: dict, max_tokens: int = 1024) -> dict:
    """Run a single MATH problem and check answer."""
    try:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content":
              f"Solve this math problem step by step, then put your final answer "
              f"in \\boxed{{}}.\n\n{problem['problem']}"}],
            add_generation_prompt=True, tokenize=False,
            enable_thinking=False,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content":
              f"Solve this math problem step by step, then put your final answer "
              f"in \\boxed{{}}.\n\n{problem['problem']}"}],
            add_generation_prompt=True, tokenize=False,
        )

    parts = []
    for r in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens):
        parts.append(r.text)
    text = "".join(parts)

    predicted = normalize_answer(extract_boxed(text))
    expected = normalize_answer(problem["answer"])
    correct = predicted == expected

    return {
        "problem_id": problem["problem_id"],
        "subject": problem.get("subject", ""),
        "correct": correct,
        "predicted": predicted,
        "expected": expected,
    }


def run_config(model_path, problems, max_tokens, label, setup_fn):
    """Run all problems with a specific config."""
    print(f"\n  [{label}] Loading model...")
    model, tokenizer = load(model_path)
    mx.eval(model.parameters())
    gc.collect()

    ctx = setup_fn(model, model_path, tokenizer)

    correct = 0
    total = len(problems)
    results = []

    for i, prob in enumerate(problems):
        t0 = time.perf_counter()
        result = run_problem(model, tokenizer, prob, max_tokens)
        elapsed = time.perf_counter() - t0
        results.append(result)

        if result["correct"]:
            correct += 1
        status = "✓" if result["correct"] else "✗"
        print(f"    [{i+1}/{total}] {status} {result['subject'][:12]:12s} "
              f"pred={result['predicted'][:20]:20s} "
              f"exp={result['expected'][:20]:20s} "
              f"({elapsed:.1f}s)")
        # Prevent metal buffer accumulation across problems
        gc.collect()
        mx.metal.clear_cache()

    accuracy = correct / total if total > 0 else 0
    print(f"  [{label}] Accuracy: {correct}/{total} = {accuracy:.1%}")

    if ctx:
        ctx.close()
    del model, tokenizer
    gc.collect()
    mx.metal.clear_cache()

    return {"label": label, "correct": correct, "total": total,
            "accuracy": accuracy, "results": results}


def setup_standard(model, model_path, tokenizer):
    """No offload."""
    return None


def _setup_offload(model, model_path, tokenizer, pool_size):
    """Common offload setup for pool=N + zero_out + rerank."""
    ctx = patch_model_for_offload(
        model, model_path, pool_size=256,
        max_workers=4, cpu_cache_gb=0.0,
        enable_prefetch=False, enable_telemetry=True,
    )
    gc.collect()

    # PP warmup
    dummy = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hi"}],
        add_generation_prompt=True, tokenize=False
    )
    for r in stream_generate(model, tokenizer, dummy, max_tokens=5):
        pass

    ctx.compact(pool_size=pool_size, disable_coverage_gate=True,
                auto_expand_cpu_cache=False)

    # Set k1_clamp for TG warmup
    inner = model
    for attr in ("model", "model", "language_model", "model"):
        if hasattr(inner, attr):
            inner = getattr(inner, attr)
    for layer in inner.layers:
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "switch_mlp"):
            sw = layer.mlp.switch_mlp
            if hasattr(sw, "_miss_policy"):
                sw._pool_is_identity = False
                sw._pool_compacted = True
                sw._miss_policy = "k1_clamp"

    # TG warmup
    warmup = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Count from 1 to 20."}],
        add_generation_prompt=True, tokenize=False
    )
    for r in stream_generate(model, tokenizer, warmup, max_tokens=15):
        pass

    ctx.decode_recompact(pool_size=pool_size)

    for layer in inner.layers:
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "switch_mlp"):
            sw = layer.mlp.switch_mlp
            if hasattr(sw, "_miss_policy"):
                sw._miss_policy = "zero_out"

    ctx.enable_reranking(bonus=0.01)
    return ctx


def setup_pool32_rr(model, model_path, tokenizer):
    """pool=32 + zero_out + rerank."""
    return _setup_offload(model, model_path, tokenizer, pool_size=32)


def setup_pool64_rr(model, model_path, tokenizer):
    """pool=64 + zero_out + rerank."""
    return _setup_offload(model, model_path, tokenizer, pool_size=64)


def main():
    parser = argparse.ArgumentParser(description="Quality benchmark")
    parser.add_argument("--model", default="/Users/lisihao/models/Qwen3.5-35B-A3B-6bit")
    parser.add_argument("--data", default="/Users/lisihao/ThunderOMLX/data/memcollab/math500_subset_50.jsonl")
    parser.add_argument("--n", type=int, default=20, help="Number of problems")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--configs", default="A,B,C", help="Configs to run")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    # Load problems
    with open(args.data) as f:
        problems = [json.loads(line) for line in f][:args.n]

    print("=" * 70)
    print("  MATH-500 Quality Benchmark for Expert Offload")
    print(f"  Model: {args.model}")
    print(f"  Problems: {len(problems)}")
    print(f"  Max tokens: {args.max_tokens}")
    print("=" * 70)

    configs = {
        "A": ("A_standard", setup_standard),
        "B": ("B_pool32_rr", setup_pool32_rr),
        "C": ("C_pool64_rr", setup_pool64_rr),
    }

    all_results = []
    for cfg_key in args.configs.split(","):
        cfg_key = cfg_key.strip()
        if cfg_key in configs:
            label, setup_fn = configs[cfg_key]
            result = run_config(args.model, problems, args.max_tokens,
                                label, setup_fn)
            all_results.append(result)

    # Summary
    print("\n" + "=" * 70)
    print("  QUALITY RESULTS")
    print("=" * 70)
    print(f"  {'Config':<16s} {'Correct':>8s} {'Total':>6s} {'Accuracy':>9s}")
    print(f"  {'-'*16} {'-'*8} {'-'*6} {'-'*9}")
    for r in all_results:
        print(f"  {r['label']:<16s} {r['correct']:>8d} {r['total']:>6d} "
              f"{r['accuracy']:>8.1%}")

    # Per-subject breakdown if enough data
    if len(all_results) > 1:
        print(f"\n  PER-SUBJECT BREAKDOWN:")
        subjects = set()
        for r in all_results:
            for p in r["results"]:
                subjects.add(p["subject"])

        for subj in sorted(subjects):
            line = f"    {subj[:20]:<20s}"
            for r in all_results:
                subj_results = [p for p in r["results"] if p["subject"] == subj]
                if subj_results:
                    correct = sum(1 for p in subj_results if p["correct"])
                    line += f"  {correct}/{len(subj_results)}"
                else:
                    line += "  -"
            print(line)

    # Save
    out_path = args.output or ".solar/tep-quality.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Report saved to {out_path}")


if __name__ == "__main__":
    main()
