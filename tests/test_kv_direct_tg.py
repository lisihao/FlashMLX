"""
KV-Direct TG-phase eviction test.

Uses generate_step with pre-built cache to force TG eviction.
"""

import sys
import mlx.core as mx

sys.path.insert(0, "/Users/lisihao/FlashMLX/mlx-lm-source")
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler

MODEL = "/Users/lisihao/models/Qwen3-1.7B-MLX-4bit"
GREEDY = make_sampler(temp=0.0)


def test_tg_eviction():
    """Force TG-phase eviction by using a small budget with pre-built cache."""
    print("=== TG Phase Eviction Test (manual cache) ===")
    model, tokenizer = load(MODEL)

    prompt = "The quick brown fox jumps over the lazy dog."
    prompt_tokens = mx.array(tokenizer.encode(prompt))
    prompt_len = len(prompt_tokens)
    budget = 8  # Very small — will evict heavily during TG
    max_tokens = 30

    print(f"  Prompt: {prompt_len} tokens, budget: {budget}, generating: {max_tokens}")

    # --- Standard generation ---
    std_tokens = []
    for token_id, logprobs in generate_step(
        prompt_tokens, model, max_tokens=max_tokens, sampler=GREEDY,
    ):
        std_tokens.append(token_id)
    std_text = tokenizer.decode(std_tokens)
    print(f"  Standard: {std_text!r}")

    # --- KV-Direct with tiny budget ---
    kvd_cache = make_prompt_cache(
        model, kv_cache="kv_direct", kv_direct_budget=budget,
    )
    kvd_tokens = []
    for token_id, logprobs in generate_step(
        prompt_tokens, model, max_tokens=max_tokens, sampler=GREEDY,
        prompt_cache=kvd_cache,
    ):
        kvd_tokens.append(token_id)
    kvd_text = tokenizer.decode(kvd_tokens)
    print(f"  KV-Direct(budget={budget}): {kvd_text!r}")

    # Report eviction stats
    c0 = kvd_cache[0]
    n_evicted = c0.offset - c0._recent_count
    h0 = c0._h0_store
    h0_info = f"h0={h0.count} tokens, {h0.nbytes/1024:.0f}KB" if h0 else "no h0"
    print(f"  Final state: offset={c0.offset}, recent={c0._recent_count}, "
          f"evicted={n_evicted}, {h0_info}")

    # Compare
    match = std_tokens == kvd_tokens
    common = sum(1 for a, b in zip(std_tokens, kvd_tokens) if a == b)
    total = max(len(std_tokens), len(kvd_tokens))
    overlap = common / total if total > 0 else 1.0

    print(f"  Exact match: {match}")
    print(f"  Token overlap: {common}/{total} ({overlap:.1%})")

    # Show first divergence point
    if not match:
        for i, (a, b) in enumerate(zip(std_tokens, kvd_tokens)):
            if a != b:
                print(f"  First divergence at token {i}: "
                      f"std={a}({tokenizer.decode([a])!r}) vs "
                      f"kvd={b}({tokenizer.decode([b])!r})")
                break

    return match, overlap


if __name__ == "__main__":
    match, overlap = test_tg_eviction()
    print(f"\nResult: {'PASS (exact)' if match else f'PARTIAL ({overlap:.0%} overlap)'}")
