"""
KV-Direct vs Standard Benchmark — Qwen3-8B

Measures: PP (prefill tok/s), TG (generation tok/s), TTFT (ms), Memory (MB)
Compares: standard vs kv_direct at various budgets
"""

import sys
import time
import gc

import mlx.core as mx

sys.path.insert(0, "/Users/lisihao/FlashMLX/mlx-lm-source")
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.cache_factory import get_cache_info
from mlx_lm.sample_utils import make_sampler

MODEL = "/Volumes/toshiba/models/qwen3-8b-mlx"
GREEDY = make_sampler(temp=0.0)

# Prompts of varying length
SHORT_PROMPT = "Explain quantum computing in simple terms."
MEDIUM_PROMPT = (
    "Write a detailed analysis of the economic impact of artificial intelligence "
    "on the global job market over the next decade. Consider factors such as "
    "automation of routine tasks, creation of new job categories, changes in "
    "required skill sets, geographic distribution of impacts, and policy "
    "recommendations for governments and educational institutions to prepare "
    "their workforce for this transition. Include specific examples from "
    "manufacturing, healthcare, finance, and education sectors."
)
LONG_PROMPT = MEDIUM_PROMPT + " " + (
    "Furthermore, analyze the role of large language models in transforming "
    "knowledge work. How will tools like AI assistants change the nature of "
    "writing, programming, research, and creative work? What are the ethical "
    "implications of widespread AI adoption in these fields? Discuss both the "
    "potential benefits and risks, including issues of bias, privacy, and "
    "the concentration of power among a few technology companies. "
    "Consider historical parallels with previous technological revolutions "
    "such as the Industrial Revolution and the advent of the internet."
) * 2


def warmup(model, tokenizer):
    """Warmup to stabilize measurements."""
    tokens = mx.array(tokenizer.encode("Hello")).reshape(1, -1)
    model(tokens)
    mx.eval(model.parameters())


def bench_config(model, tokenizer, prompt_text, strategy, budget=512,
                 max_tokens=100, label=""):
    """Benchmark a single configuration. Returns dict of metrics."""
    tokens_list = tokenizer.encode(prompt_text)
    prompt_tokens = mx.array(tokens_list)
    prompt_len = len(tokens_list)

    # Force GC + clear MLX cache
    gc.collect()
    mx.clear_cache()

    # Measure memory before
    mem_before = mx.metal.get_active_memory() / (1024 * 1024)  # MB

    # Create cache
    cache_kwargs = {"kv_cache": strategy}
    if strategy == "kv_direct":
        cache_kwargs["kv_direct_budget"] = budget
    cache = make_prompt_cache(model, **cache_kwargs)

    # --- Prefill (PP) ---
    t0 = time.perf_counter()

    # First forward pass = prefill
    logits = model(prompt_tokens[None], cache=cache)
    mx.eval(logits)

    t_prefill = time.perf_counter() - t0
    ttft_ms = t_prefill * 1000
    pp_toks = prompt_len / t_prefill

    mem_after_prefill = mx.metal.get_active_memory() / (1024 * 1024)

    # --- Token Generation (TG) ---
    # Use the sampler to get the first token from prefill logits
    first_token = GREEDY(logits[:, -1, :])
    mx.eval(first_token)
    y = first_token.squeeze()

    gen_tokens = [y.item()]
    t_gen_start = time.perf_counter()

    for i in range(max_tokens - 1):
        logits = model(y.reshape(1, 1), cache=cache)
        y = GREEDY(logits[:, -1, :]).squeeze()
        mx.eval(y)
        gen_tokens.append(y.item())
        # Stop on EOS
        if y.item() in (151643, 151645):  # Qwen3 EOS tokens
            break

    t_gen = time.perf_counter() - t_gen_start
    n_gen = len(gen_tokens) - 1  # exclude first token (already counted in prefill)
    tg_toks = n_gen / t_gen if t_gen > 0 else 0

    mem_after_gen = mx.metal.get_active_memory() / (1024 * 1024)

    # Cache memory
    cache_bytes = sum(c.nbytes for c in cache)
    cache_mb = cache_bytes / (1024 * 1024)

    # Cache info
    info = get_cache_info(cache)

    output_text = tokenizer.decode(gen_tokens)

    result = {
        "strategy": strategy,
        "budget": budget if strategy == "kv_direct" else "-",
        "prompt_len": prompt_len,
        "gen_tokens": n_gen + 1,
        "pp_toks": pp_toks,
        "tg_toks": tg_toks,
        "ttft_ms": ttft_ms,
        "cache_mb": cache_mb,
        "mem_before": mem_before,
        "mem_after_prefill": mem_after_prefill,
        "mem_after_gen": mem_after_gen,
        "mem_delta": mem_after_gen - mem_before,
        "info": info,
        "output": output_text[:80],
    }
    return result


def print_results(results, title=""):
    """Pretty-print benchmark results."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")
    print(f"  {'Strategy':<20} {'PP tok/s':>10} {'TG tok/s':>10} {'TTFT ms':>10} "
          f"{'Cache MB':>10} {'Mem Δ MB':>10}")
    print(f"  {'-' * 70}")
    for r in results:
        strat = f"{r['strategy']}"
        if r['budget'] != '-':
            strat += f"(B={r['budget']})"
        print(f"  {strat:<20} {r['pp_toks']:>10.1f} {r['tg_toks']:>10.1f} "
              f"{r['ttft_ms']:>10.1f} {r['cache_mb']:>10.2f} {r['mem_delta']:>10.1f}")

    # Output comparison
    print(f"\n  Output comparison:")
    for r in results:
        strat = f"{r['strategy']}"
        if r['budget'] != '-':
            strat += f"(B={r['budget']})"
        print(f"    {strat}: {r['output']!r}...")


def main():
    print("Loading model...")
    model, tokenizer = load(MODEL)
    print(f"Model loaded: Qwen3-8B-4bit")

    warmup(model, tokenizer)
    print("Warmup done.\n")

    # Benchmark matrix
    prompts = [
        ("Short (~20 tok)", SHORT_PROMPT, 50),
        ("Medium (~100 tok)", MEDIUM_PROMPT, 100),
        ("Long (~300 tok)", LONG_PROMPT, 100),
    ]

    strategies = [
        ("standard", {}),
        ("kv_direct", {"budget": 512}),
        ("kv_direct", {"budget": 256}),
        ("kv_direct", {"budget": 128}),
        ("kv_direct", {"budget": 64}),
    ]

    for prompt_name, prompt_text, max_tokens in prompts:
        prompt_len = len(tokenizer.encode(prompt_text))
        results = []

        for strategy, kwargs in strategies:
            budget = kwargs.get("budget", 512)
            label = f"{strategy}(B={budget})" if strategy == "kv_direct" else strategy

            # Skip budgets larger than prompt for cleaner results
            r = bench_config(
                model, tokenizer, prompt_text, strategy,
                budget=budget, max_tokens=max_tokens, label=label,
            )
            results.append(r)

        print_results(results, f"{prompt_name} (prompt={prompt_len} tok, gen={max_tokens})")

    # Quality verification: do outputs match?
    print(f"\n{'=' * 80}")
    print(f"  Quality Check: Standard vs KV-Direct outputs")
    print(f"{'=' * 80}")
    for prompt_name, prompt_text, max_tokens in prompts:
        std_result = bench_config(model, tokenizer, prompt_text, "standard",
                                  max_tokens=max_tokens)
        kvd_result = bench_config(model, tokenizer, prompt_text, "kv_direct",
                                  budget=64, max_tokens=max_tokens)
        match = std_result["output"] == kvd_result["output"]
        print(f"  {prompt_name}: match={match}")
        if not match:
            print(f"    STD: {std_result['output']!r}")
            print(f"    KVD: {kvd_result['output']!r}")


if __name__ == "__main__":
    main()
