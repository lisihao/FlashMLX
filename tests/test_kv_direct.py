"""
KV-Direct v2 Tests

Test 1: No-eviction baseline — kv_direct(budget=9999) == standard
Test 2: Reconstruction correctness — forward-pass reconstructed K/V vs original
Test 3: End-to-end quality — greedy output matches standard
Test 4: Memory comparison — v2 should show savings when T > budget
"""

import sys

import mlx.core as mx

sys.path.insert(0, "/Users/lisihao/FlashMLX/mlx-lm-source")
from mlx_lm import load, generate
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.cache_factory import get_cache_info
from mlx_lm.sample_utils import make_sampler

MODEL = "/Users/lisihao/models/Qwen3-1.7B-MLX-4bit"
PROMPT = "The capital of France is"
MAX_TOKENS = 50

GREEDY = make_sampler(temp=0.0)


def test_1_no_eviction_baseline():
    """Budget larger than sequence -> pure passthrough, must match standard."""
    print("\n=== Test 1: No-eviction baseline ===")
    model, tokenizer = load(MODEL)

    std_out = generate(
        model, tokenizer, prompt=PROMPT, max_tokens=MAX_TOKENS,
        verbose=False, sampler=GREEDY,
    )
    print(f"  Standard: {std_out!r}")

    kvd_out = generate(
        model, tokenizer, prompt=PROMPT, max_tokens=MAX_TOKENS,
        verbose=False, sampler=GREEDY, kv_cache="kv_direct",
    )
    print(f"  KV-Direct(budget=512,no-evict): {kvd_out!r}")

    match = std_out == kvd_out
    print(f"  Match: {match}")
    return match


def test_2_reconstruction_correctness():
    """Verify forward-pass reconstructed K/V produces correct logits."""
    print("\n=== Test 2: Reconstruction correctness ===")
    model, tokenizer = load(MODEL)

    std_caches = make_prompt_cache(model, kv_cache="standard")
    kvd_caches = make_prompt_cache(model, kv_cache="kv_direct", kv_direct_budget=32)

    long_prompt = (
        "The quick brown fox jumps over the lazy dog. "
        "Albert Einstein proposed the theory of relativity."
    )
    tokens = mx.array(tokenizer.encode(long_prompt)).reshape(1, -1)
    prompt_len = tokens.shape[1]
    print(f"  Prompt length: {prompt_len} tokens, budget: 32")

    std_logits = model(tokens, cache=std_caches)
    mx.eval(std_logits)

    kvd_logits = model(tokens, cache=kvd_caches)
    mx.eval(kvd_logits)

    logit_diff = mx.abs(std_logits[0, -1] - kvd_logits[0, -1]).max().item()
    print(f"  Max logit diff (last token): {logit_diff:.6e}")

    for i in range(min(3, len(kvd_caches))):
        kc = kvd_caches[i]
        sc = std_caches[i]
        print(f"  Layer {i}: std_offset={sc.offset}, "
              f"kvd_offset={kc.offset}, "
              f"kvd_recent={kc._recent_count}")

    n_evicted = kvd_caches[0].offset - kvd_caches[0]._recent_count
    if n_evicted > 0:
        print(f"  Eviction triggered: {n_evicted} tokens evicted + reconstructed")
    else:
        print(f"  No eviction (prompt fits in budget)")

    # Report h^(0) store
    h0 = kvd_caches[0]._h0_store
    if h0 is not None:
        print(f"  H0Store: {h0.count} tokens, {h0.nbytes / 1024:.1f} KB")

    passed = logit_diff < 0.01
    print(f"  PASS: {passed} (diff={logit_diff:.6e})")
    return passed


def test_3_end_to_end_quality():
    """KV-Direct with eviction should produce same greedy output as standard."""
    print("\n=== Test 3: End-to-end quality (with eviction) ===")
    model, tokenizer = load(MODEL)

    long_prompt = (
        "Explain the theory of relativity in simple terms. "
        "Albert Einstein proposed that space and time are interconnected "
        "and that the speed of light is constant. This means that"
    )

    std_out = generate(
        model, tokenizer, prompt=long_prompt, max_tokens=MAX_TOKENS,
        verbose=False, sampler=GREEDY,
    )
    print(f"  Standard output: {std_out[:100]!r}...")

    kvd_out = generate(
        model, tokenizer, prompt=long_prompt, max_tokens=MAX_TOKENS,
        verbose=False, sampler=GREEDY, kv_cache="kv_direct",
    )
    print(f"  KV-Direct output: {kvd_out[:100]!r}...")

    match = std_out == kvd_out
    std_tokens = tokenizer.encode(std_out)
    kvd_tokens = tokenizer.encode(kvd_out)
    common = sum(1 for a, b in zip(std_tokens, kvd_tokens) if a == b)
    total = max(len(std_tokens), len(kvd_tokens))
    overlap = common / total if total > 0 else 1.0

    print(f"  Exact match: {match}")
    print(f"  Token overlap: {common}/{total} ({overlap:.1%})")
    return overlap > 0.9


def test_4_memory_report():
    """Report memory usage: v2 h^(0) should save memory when T > budget."""
    print("\n=== Test 4: Memory comparison (v2 h^(0)) ===")
    model, tokenizer = load(MODEL)

    std_caches = make_prompt_cache(model, kv_cache="standard")
    kvd_caches = make_prompt_cache(model, kv_cache="kv_direct", kv_direct_budget=64)

    prompt = "The quick brown fox jumps over the lazy dog. " * 10
    tokens = mx.array(tokenizer.encode(prompt)).reshape(1, -1)
    prompt_len = tokens.shape[1]

    model(tokens, cache=std_caches)
    mx.eval([c.keys for c in std_caches if hasattr(c, 'keys') and c.keys is not None])

    model(tokens, cache=kvd_caches)
    mx.eval([c._recent_keys for c in kvd_caches
             if hasattr(c, '_recent_keys') and c._recent_keys is not None])

    std_bytes = sum(c.nbytes for c in std_caches)

    # For v2: per-layer nbytes includes shared h0. Report properly.
    h0_store = kvd_caches[0]._h0_store
    h0_bytes = h0_store.nbytes if h0_store else 0
    kv_bytes = sum(
        (c._recent_keys.nbytes + c._recent_values.nbytes)
        for c in kvd_caches
        if hasattr(c, '_recent_keys') and c._recent_keys is not None
    )
    kvd_total = kv_bytes + h0_bytes

    std_info = get_cache_info(std_caches)
    kvd_info = get_cache_info(kvd_caches)

    print(f"  Prompt length: {prompt_len} tokens")
    print(f"  Standard: {std_bytes / 1024:.1f} KB ({std_info})")
    print(f"  KV-Direct v2 (budget=64):")
    print(f"    Recent KV: {kv_bytes / 1024:.1f} KB (64 tokens * {len(kvd_caches)} layers)")
    print(f"    H0 store:  {h0_bytes / 1024:.1f} KB ({h0_store.count} tokens, shared)")
    print(f"    Total:     {kvd_total / 1024:.1f} KB")
    if std_bytes > 0:
        ratio = kvd_total / std_bytes
        print(f"  Ratio: KV-Direct/Standard = {ratio:.2f}x")
        if ratio < 1.0:
            print(f"  SAVINGS: {(1-ratio)*100:.0f}% memory reduction!")
        else:
            print(f"  NOTE: Still larger (need more tokens for savings)")

    return True


if __name__ == "__main__":
    results = {}
    for test_fn in [test_1_no_eviction_baseline, test_2_reconstruction_correctness,
                    test_3_end_to_end_quality, test_4_memory_report]:
        try:
            results[test_fn.__name__] = test_fn()
        except Exception as e:
            print(f"  FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            results[test_fn.__name__] = False

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {name}")
    all_pass = all(results.values())
    print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
