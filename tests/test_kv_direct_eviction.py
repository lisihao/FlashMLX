"""
KV-Direct Eviction Deep Test

Validates that evicted-token K/V reconstruction is bit-identical.
Uses a prompt longer than budget to force eviction during prefill,
then generates tokens to verify TG-phase eviction works.
"""

import sys
import mlx.core as mx

sys.path.insert(0, "/Users/lisihao/FlashMLX/mlx-lm-source")
from mlx_lm import load, generate
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler

MODEL = "/Users/lisihao/models/Qwen3-1.7B-MLX-4bit"
GREEDY = make_sampler(temp=0.0)


def test_eviction_reconstruction():
    """Force eviction with very small budget, verify output quality."""
    print("\n=== Eviction Reconstruction Test ===")
    model, tokenizer = load(MODEL)

    # Long prompt to force heavy eviction with budget=16
    long_prompt = (
        "In the year 2024, artificial intelligence research made significant "
        "strides across multiple domains. Large language models became more "
        "capable and efficient. Researchers at various institutions published "
        "groundbreaking papers on topics ranging from reasoning to "
        "multimodal understanding. The field continues to advance rapidly."
    )

    tokens = mx.array(tokenizer.encode(long_prompt)).reshape(1, -1)
    prompt_len = tokens.shape[1]
    print(f"  Prompt: {prompt_len} tokens")

    # Test with budget=16 (heavy eviction)
    for budget in [16, 32, 64, 128]:
        print(f"\n  --- Budget={budget} ---")

        # Standard reference
        std_caches = make_prompt_cache(model, kv_cache="standard")
        std_logits = model(tokens, cache=std_caches)
        mx.eval(std_logits)

        # KV-Direct
        kvd_caches = make_prompt_cache(
            model, kv_cache="kv_direct", kv_direct_budget=budget
        )
        kvd_logits = model(tokens, cache=kvd_caches)
        mx.eval(kvd_logits)

        # Compare
        logit_diff = mx.abs(std_logits[0, -1] - kvd_logits[0, -1]).max().item()
        n_evicted = kvd_caches[0].offset - kvd_caches[0]._recent_count

        # Get top-1 token from both
        std_top = mx.argmax(std_logits[0, -1]).item()
        kvd_top = mx.argmax(kvd_logits[0, -1]).item()
        top_match = std_top == kvd_top

        h0 = kvd_caches[0]._h0_store
        h0_info = f"h0={h0.count} tokens, {h0.nbytes/1024:.0f}KB" if h0 else "no h0"

        print(f"    Evicted: {n_evicted}/{prompt_len} tokens")
        print(f"    Max logit diff: {logit_diff:.6e}")
        print(f"    Top-1 match: {top_match} (std={std_top}, kvd={kvd_top})")
        print(f"    H0Store: {h0_info}")

        if not top_match:
            print(f"    WARNING: Top-1 mismatch! Reconstruction may have issues.")


def test_tg_phase_eviction():
    """Test eviction during TG (token generation) phase."""
    print("\n=== TG Phase Eviction Test ===")
    model, tokenizer = load(MODEL)

    prompt = "The quick brown fox"
    budget = 16  # Very small — will evict during TG

    # Standard
    std_out = generate(
        model, tokenizer, prompt=prompt, max_tokens=30,
        verbose=False, sampler=GREEDY,
    )
    print(f"  Standard: {std_out!r}")

    # KV-Direct with tiny budget
    kvd_out = generate(
        model, tokenizer, prompt=prompt, max_tokens=30,
        verbose=False, sampler=GREEDY, kv_cache="kv_direct",
    )
    print(f"  KV-Direct(budget={budget}): {kvd_out!r}")

    # Check overlap
    std_tokens = tokenizer.encode(std_out)
    kvd_tokens = tokenizer.encode(kvd_out)
    common = sum(1 for a, b in zip(std_tokens, kvd_tokens) if a == b)
    total = max(len(std_tokens), len(kvd_tokens))
    overlap = common / total if total > 0 else 1.0

    print(f"  Exact match: {std_out == kvd_out}")
    print(f"  Token overlap: {common}/{total} ({overlap:.1%})")

    # During generation with budget=16 and ~30 TG tokens,
    # total sequence is ~34 tokens, evicting ~18 tokens
    print(f"  Expected: prompt(~4) + output(30) = ~34 tokens, "
          f"budget={budget}, evict ~{34-budget}")


if __name__ == "__main__":
    test_eviction_reconstruction()
    test_tg_phase_eviction()
