"""
Route 5: Scored KV-Direct Benchmark — Qwen3-1.7B

Compares: standard vs scored_pq vs scored_kv_direct (bf16/Q8/Q4 h^(0))
Measures: PP tok/s, TG tok/s, TTFT ms, Cache MB, h^(0) MB
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

MODEL = "/Users/lisihao/models/Qwen3-1.7B-MLX-4bit"
GREEDY = make_sampler(temp=0.0)

SHORT_PROMPT = "Explain quantum computing in simple terms."
MEDIUM_PROMPT = (
    "Write a detailed analysis of the economic impact of artificial intelligence "
    "on the global job market over the next decade. Consider factors such as "
    "automation of routine tasks, creation of new job categories, changes in "
    "required skill sets, geographic distribution of impacts, and policy "
    "recommendations for governments and educational institutions."
)
LONG_PROMPT = MEDIUM_PROMPT + " " + (
    "Furthermore, analyze the role of large language models in transforming "
    "knowledge work. How will tools like AI assistants change the nature of "
    "writing, programming, research, and creative work? What are the ethical "
    "implications of widespread AI adoption in these fields? Discuss both the "
    "potential benefits and risks, including issues of bias and privacy."
) * 3


def warmup(model, tokenizer):
    tokens = mx.array(tokenizer.encode("Hello")).reshape(1, -1)
    model(tokens)
    mx.eval(model.parameters())


def bench_config(model, tokenizer, prompt_text, strategy, max_tokens=50,
                 h0_quant=None):
    """Benchmark a single configuration."""
    tokens_list = tokenizer.encode(prompt_text)
    prompt_tokens = mx.array(tokens_list)
    prompt_len = len(tokens_list)

    gc.collect()
    mx.clear_cache()

    mem_before = mx.get_active_memory() / (1024 * 1024)

    # Create cache
    cache_kwargs = {"kv_cache": strategy}
    if h0_quant is not None:
        cache_kwargs["h0_quant"] = h0_quant
    cache = make_prompt_cache(model, **cache_kwargs)

    # --- Prefill ---
    t0 = time.perf_counter()
    logits = model(prompt_tokens[None], cache=cache)
    mx.eval(logits)
    t_prefill = time.perf_counter() - t0
    ttft_ms = t_prefill * 1000
    pp_toks = prompt_len / t_prefill

    mem_after_prefill = mx.get_active_memory() / (1024 * 1024)

    # --- TG ---
    first_token = GREEDY(logits[:, -1, :])
    mx.eval(first_token)
    y = first_token.squeeze()
    gen_tokens = [y.item()]

    t_gen_start = time.perf_counter()
    for _ in range(max_tokens - 1):
        logits = model(y.reshape(1, 1), cache=cache)
        y = GREEDY(logits[:, -1, :]).squeeze()
        mx.eval(y)
        gen_tokens.append(y.item())
        if y.item() in (151643, 151645):
            break
    t_gen = time.perf_counter() - t_gen_start
    n_gen = len(gen_tokens) - 1
    tg_toks = n_gen / t_gen if t_gen > 0 else 0

    mem_after_gen = mx.get_active_memory() / (1024 * 1024)

    # Cache info
    try:
        cache_bytes = sum(c.nbytes for c in cache)
    except (NotImplementedError, AttributeError):
        # Standard KVCache may not implement nbytes
        cache_bytes = 0
        for c in cache:
            s = c.state
            if s and len(s) == 2:
                cache_bytes += s[0].nbytes + s[1].nbytes
    cache_mb = cache_bytes / (1024 * 1024)
    info = get_cache_info(cache)

    h0_mb = info.get("h0_bytes", 0) / (1024 * 1024)
    h0_count = info.get("h0_count", 0)

    output_text = tokenizer.decode(gen_tokens)

    return {
        "label": strategy + (f"(h0={h0_quant})" if h0_quant else ""),
        "prompt_len": prompt_len,
        "gen_tokens": n_gen + 1,
        "pp_toks": pp_toks,
        "tg_toks": tg_toks,
        "ttft_ms": ttft_ms,
        "cache_mb": cache_mb,
        "h0_mb": h0_mb,
        "h0_count": h0_count,
        "mem_delta": mem_after_gen - mem_before,
        "output": output_text[:80],
        "info": info,
    }


def print_results(results, title=""):
    print(f"\n{'=' * 90}")
    print(f"  {title}")
    print(f"{'=' * 90}")
    print(f"  {'Strategy':<28} {'PP tok/s':>9} {'TG tok/s':>9} {'TTFT ms':>9} "
          f"{'KV MB':>8} {'h0 MB':>7} {'Mem Δ':>8}")
    print(f"  {'-' * 78}")

    std = results[0] if results else None
    for r in results:
        pp_delta = f" ({r['pp_toks']/std['pp_toks']*100-100:+.0f}%)" if std and r != std else ""
        tg_delta = f" ({r['tg_toks']/std['tg_toks']*100-100:+.0f}%)" if std and r != std else ""
        print(f"  {r['label']:<28} {r['pp_toks']:>9.1f} {r['tg_toks']:>9.1f} "
              f"{r['ttft_ms']:>9.1f} {r['cache_mb']:>8.2f} {r['h0_mb']:>7.2f} "
              f"{r['mem_delta']:>8.1f}")
        if pp_delta:
            print(f"  {'':28} {pp_delta:>9} {tg_delta:>9}")

    # Output
    print(f"\n  Output comparison:")
    for r in results:
        print(f"    {r['label']}: {r['output']!r}...")


def main():
    print("Loading model...")
    model, tokenizer = load(MODEL)
    print(f"Model loaded: Qwen3-1.7B-MLX-4bit")
    warmup(model, tokenizer)
    print("Warmup done.\n")

    prompts = [
        ("Short (~20 tok)", SHORT_PROMPT, 50),
        ("Medium (~100 tok)", MEDIUM_PROMPT, 80),
        ("Long (~400 tok)", LONG_PROMPT, 80),
    ]

    strategies = [
        ("standard", {}),
        ("scored_pq", {}),
        ("scored_kv_direct", {}),
        ("scored_kv_direct", {"h0_quant": "q8"}),
        ("scored_kv_direct", {"h0_quant": "q4"}),
    ]

    for prompt_name, prompt_text, max_tokens in prompts:
        prompt_len = len(tokenizer.encode(prompt_text))
        results = []

        for strategy, kwargs in strategies:
            r = bench_config(
                model, tokenizer, prompt_text, strategy,
                max_tokens=max_tokens, **kwargs,
            )
            results.append(r)

        print_results(results,
                      f"{prompt_name} (prompt={prompt_len} tok, gen={max_tokens})")

    # Memory summary for Route 5 h^(0) overhead
    print(f"\n{'=' * 90}")
    print(f"  h^(0) Memory Summary")
    print(f"{'=' * 90}")
    for prompt_name, prompt_text, max_tokens in prompts:
        prompt_len = len(tokenizer.encode(prompt_text))
        total_tokens = prompt_len + max_tokens

        bf16_h0 = total_tokens * 2048 * 2 / (1024 * 1024)
        q8_h0 = total_tokens * (2048 + 2) / (1024 * 1024)  # int8 + scales
        q4_h0 = total_tokens * (1024 + 2) / (1024 * 1024)  # packed + scales

        print(f"\n  {prompt_name} ({total_tokens} total tokens):")
        print(f"    bf16 h^(0): {bf16_h0:.2f} MB")
        print(f"    Q8   h^(0): {q8_h0:.2f} MB ({bf16_h0/q8_h0:.1f}x smaller)")
        print(f"    Q4   h^(0): {q4_h0:.2f} MB ({bf16_h0/q4_h0:.1f}x smaller)")

        # Projected for 32K tokens (Qwen3-8B scale)
    print(f"\n  --- Projected for Qwen3-8B at 32K tokens ---")
    d_hidden_8b = 4096
    tokens_32k = 32768
    bf16_32k = tokens_32k * d_hidden_8b * 2 / (1024**2)
    q8_32k = tokens_32k * (d_hidden_8b + 2) / (1024**2)
    q4_32k = tokens_32k * (d_hidden_8b // 2 + 2) / (1024**2)
    print(f"    bf16 h^(0): {bf16_32k:.0f} MB")
    print(f"    Q8   h^(0): {q8_32k:.0f} MB ({bf16_32k/q8_32k:.1f}x smaller)")
    print(f"    Q4   h^(0): {q4_32k:.0f} MB ({bf16_32k/q4_32k:.1f}x smaller)")


if __name__ == "__main__":
    main()
