"""
Route 0 Benchmark D: Recall ability verification.

Tests the "前端敢压，后端能救" (aggressive compress + backend rescue) thesis.

Embeds specific facts ("needles") at known positions in a haystack prompt,
then asks questions that require recalling those facts. Compares:
  - standard (no compression, gold reference)
  - scored_pq baseline (3x compression)
  - ultra_long (10x compression, aggressive eviction)
  - recall_first (10x + h^(0) reconstruction, should recover details)

Usage:
    python3 benchmarks/bench_recall_d.py /path/to/model
    python3 benchmarks/bench_recall_d.py /path/to/model --prompt-tokens 8192
"""

import argparse
import gc
import sys
import time

sys.path.insert(0, "/Users/lisihao/FlashMLX/mlx-lm-source")
sys.path.insert(0, "/Users/lisihao/FlashMLX/src")

import mlx.core as mx

from flashmlx.model_cards import load_card_or_detect
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.cache_factory import get_cache_info
from mlx_lm.models.kv_direct_cache import (
    _find_inner_model,
    _run_reconstruction,
    unpatch_model,
)
from mlx_lm.sample_utils import make_sampler

GREEDY = make_sampler(temp=0.0)

# --- Haystack + Needle Construction ---

FILLER_PARA = (
    "The development of artificial intelligence has progressed rapidly in recent years. "
    "Machine learning algorithms continue to improve across various benchmarks. Research "
    "teams around the world are exploring new architectures for language understanding. "
    "The computational requirements for training large models have grown exponentially. "
    "Transfer learning has enabled smaller teams to build on pre-trained foundations. "
    "Ethical considerations remain central to AI development discussions globally. "
)

# Facts to embed at different positions in the context
NEEDLES = [
    {
        "id": "needle_early",
        "fact": "The secret project code name is 'AURORA-7732' and it was started on March 15th, 2024.",
        "question": "What is the secret project code name and when was it started?",
        "expected_keywords": ["AURORA-7732", "March 15"],
    },
    {
        "id": "needle_mid",
        "fact": "Dr. Emily Zhang discovered that the optimal learning rate for the NEXUS model is exactly 0.00037.",
        "question": "What is the optimal learning rate for the NEXUS model, and who discovered it?",
        "expected_keywords": ["0.00037", "Emily Zhang"],
    },
    {
        "id": "needle_late",
        "fact": "The quarterly revenue for Q3 was $847.2 million, a 23% increase over the previous quarter.",
        "question": "What was the Q3 quarterly revenue and what was the growth percentage?",
        "expected_keywords": ["847.2", "23%"],
    },
]


def build_haystack_prompt(tokenizer, target_tokens, needles):
    """Build prompt with needles embedded at 10%, 50%, 90% positions."""
    # First estimate tokens per paragraph
    filler_tokens = len(tokenizer.encode(FILLER_PARA))
    n_paras_needed = (target_tokens // filler_tokens) + 10

    # Calculate needle positions (paragraph indices)
    needle_positions = [
        max(1, int(n_paras_needed * 0.10)),  # early: 10%
        max(2, int(n_paras_needed * 0.50)),  # mid: 50%
        max(3, int(n_paras_needed * 0.90)),  # late: 90%
    ]

    # Build the text
    prefix = "Read the following document carefully. You will be asked questions about it.\n\n"
    paragraphs = [prefix]

    needle_idx = 0
    for i in range(n_paras_needed):
        if needle_idx < len(needles) and i == needle_positions[needle_idx]:
            paragraphs.append(f"\n[Important Note] {needles[needle_idx]['fact']}\n\n")
            needle_idx += 1
        paragraphs.append(FILLER_PARA)

    full_text = "".join(paragraphs)
    tokens = tokenizer.encode(full_text)

    if len(tokens) > target_tokens:
        tokens = tokens[:target_tokens]
        full_text = tokenizer.decode(tokens)

    return full_text, len(tokens)


def get_mem_mb():
    try:
        return mx.get_active_memory() / (1024 * 1024)
    except AttributeError:
        return mx.metal.get_active_memory() / (1024 * 1024)


def ask_question(model, tokenizer, haystack, question, cache_kwargs, label,
                 tg_tokens=200, trigger_recon=False, targeted_recon=False):
    """Ask a question about the haystack and return the answer.

    When trigger_recon=True, uses a two-phase approach:
    1. Prefill haystack only (compression happens here)
    2. Reconstruct evicted tokens from h^(0)
    3. Continue prefill with question (now with reconstructed context)
    4. Generate answer
    """
    gc.collect()
    mx.clear_cache()
    time.sleep(0.2)

    # Unpatch before creating new cache
    unpatch_model(model)

    mem_before = get_mem_mb()
    cache = make_prompt_cache(model, **cache_kwargs)

    if not trigger_recon:
        # Single-shot: process everything in one go
        full_prompt = haystack + f"\n\nQuestion: {question}\nAnswer:"
        prompt_tokens = mx.array(tokenizer.encode(full_prompt))
        prompt_len = prompt_tokens.shape[0]

        gen_tokens = []
        t_start = time.perf_counter()
        recon_ms = 0.0

        for i, (token_id, logprobs) in enumerate(generate_step(
            prompt_tokens, model, max_tokens=tg_tokens, sampler=GREEDY,
            prompt_cache=cache,
        )):
            if i == 0:
                ttft_ms = (time.perf_counter() - t_start) * 1000
            gen_tokens.append(token_id)
    else:
        # Two-phase: haystack → reconstruct → question → generate
        haystack_tokens = mx.array(tokenizer.encode(haystack))
        question_text = f"\n\nQuestion: {question}\nAnswer:"
        question_tokens = mx.array(tokenizer.encode(question_text))
        prompt_len = haystack_tokens.shape[0] + question_tokens.shape[0]

        t_start = time.perf_counter()

        # Phase 1: Prefill haystack (fills and compresses cache)
        # Use generate_step with max_tokens=0 equivalent: just prefill
        # Process haystack through model manually
        model_out = model(haystack_tokens.reshape(1, -1), cache=cache)
        mx.eval(model_out)
        t_prefill = time.perf_counter()

        # Phase 2: Reconstruct from h^(0)
        recon_start = time.perf_counter()
        _trigger_reconstruction(model, cache, targeted=targeted_recon)
        recon_ms = (time.perf_counter() - recon_start) * 1000

        # Phase 3: Process question with reconstructed context + generate
        gen_tokens = []
        for i, (token_id, logprobs) in enumerate(generate_step(
            question_tokens, model, max_tokens=tg_tokens, sampler=GREEDY,
            prompt_cache=cache,
        )):
            if i == 0:
                ttft_ms = (time.perf_counter() - t_start) * 1000
            gen_tokens.append(token_id)

    t_end = time.perf_counter()
    mem_after = get_mem_mb()

    info = get_cache_info(cache)
    answer = tokenizer.decode(gen_tokens)

    return {
        "label": label,
        "prompt_len": prompt_len,
        "answer": answer.strip(),
        "ttft_ms": ttft_ms,
        "recon_ms": recon_ms,
        "tg_mem_mb": mem_after - mem_before,
        "h0_mb": info.get("h0_bytes", 0) / (1024 * 1024),
        "strategy": info.get("strategy", "?"),
    }


def _trigger_reconstruction(model, cache, targeted=False):
    """Trigger h^(0) reconstruction for evicted tokens.

    Args:
        targeted: If True, use probe importance scores for depth reduction.
    """
    # Find h0_store from cache objects (stored by cache_factory on each layer)
    h0_store = None
    for c in cache:
        h0_store = getattr(c, "_h0_store", None)
        if h0_store is not None:
            break

    if h0_store is None:
        print("    [recon] No h0_store found on any cache layer")
        return

    try:
        inner_model = _find_inner_model(model)
    except ValueError:
        print("    [recon] Cannot find inner model")
        return

    # All layers can receive reconstruction
    caches = list(cache)
    kv_direct_indices = list(range(len(caches)))
    n_evicted = h0_store.count

    if n_evicted <= 0:
        print("    [recon] No h0 tokens to reconstruct")
        return

    # Get probe importance scores for targeted reconstruction
    importance_scores = None
    if targeted:
        from mlx_lm.models.triple_layer_cache import TripleLayerKVCache
        probe = TripleLayerKVCache._shared_probe
        if probe is not None:
            print(f"    [recon] Running probe for targeted reconstruction...")
            importance_scores = probe.score_tokens(h0_store)
            print(f"    [recon] Probe done, {len(importance_scores)} token scores")
        else:
            print("    [recon] No probe available, falling back to full reconstruction")

    first_cache = cache[0]
    flat_prefix = getattr(first_cache, '_flat_prefix_token_count', 0)
    mode = "targeted" if importance_scores is not None else "full"
    print(f"    [recon] Reconstructing {n_evicted} tokens ({mode}) from h^(0) "
          f"(h0 has {h0_store.count}, flat_prefix={flat_prefix})...")

    _run_reconstruction(inner_model, caches, h0_store, n_evicted, kv_direct_indices,
                        importance_scores=importance_scores)
    recon_arrays = [c._recon_keys for c in caches if getattr(c, '_recon_keys', None) is not None]
    if recon_arrays:
        mx.eval(*recon_arrays)
    for c in caches:
        if getattr(c, '_recon_keys', None) is not None:
            c._flat_prefix_token_count = max(c._flat_prefix_token_count, n_evicted)
    print(f"    [recon] Done — injected into {len(recon_arrays)} layers, dedup updated to {n_evicted}")


def score_answer(answer, expected_keywords):
    """Score how many expected keywords appear in the answer."""
    answer_lower = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return hits, len(expected_keywords)


def main():
    parser = argparse.ArgumentParser(description="Route 0 Benchmark D: Recall Verification")
    parser.add_argument("model_path", help="Path to model directory")
    parser.add_argument("--prompt-tokens", type=int, default=8192,
                        help="Target number of prompt tokens")
    parser.add_argument("--tg-tokens", type=int, default=200,
                        help="Max tokens to generate for answer")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    model, tokenizer = load(args.model_path)

    card = load_card_or_detect(model, args.model_path)
    print(f"Card: {card.model_name}")

    # Build configs
    # recall_first mode now has auto_reconstruct=true in model card,
    # but for non-AUTO configs we disable it to test manual paths.
    recall_kwargs = card.to_cache_kwargs(mode="recall_first")
    recall_manual = {k: v for k, v in recall_kwargs.items() if k != "auto_reconstruct"}
    configs = [
        ("standard (no compress)", {"kv_cache": "standard"}),
        ("baseline (scored_pq)", card.to_cache_kwargs()),
        ("ultra_long (10x)", card.to_cache_kwargs(mode="ultra_long")),
        ("recall_first (10x+h0)", recall_manual),
        ("recall_first+RECON", recall_manual),
        ("recall_first+TARGETED", recall_manual),
        ("recall_first+AUTO", recall_kwargs),
    ]

    # Build haystack
    print(f"\nBuilding {args.prompt_tokens:,}-token haystack with {len(NEEDLES)} needles...")
    haystack, actual_tokens = build_haystack_prompt(tokenizer, args.prompt_tokens, NEEDLES)
    print(f"  Actual: {actual_tokens:,} tokens")

    # Warmup
    print("Warming up...")
    warm_tokens = mx.array(tokenizer.encode("Hello")).reshape(1, -1)
    model(warm_tokens)
    mx.eval(model.parameters())

    # Test each needle with each config
    print(f"\n{'='*120}")
    print(f"  Route 0 Recall Benchmark | {card.model_name} | {actual_tokens:,} tokens | {len(NEEDLES)} needles")
    print(f"{'='*120}")

    all_results = {}  # needle_id -> list of (label, answer, score)

    for needle in NEEDLES:
        print(f"\n--- Needle: {needle['id']} ---")
        print(f"  Fact: {needle['fact'][:80]}...")
        print(f"  Question: {needle['question']}")
        print()

        results = []
        for label, kwargs in configs:
            trigger_recon = "+RECON" in label or "+TARGETED" in label
            targeted_recon = "+TARGETED" in label
            print(f"  Testing {label}...")
            try:
                r = ask_question(
                    model, tokenizer, haystack, needle["question"],
                    cache_kwargs=kwargs, label=label,
                    tg_tokens=args.tg_tokens,
                    trigger_recon=trigger_recon,
                    targeted_recon=targeted_recon,
                )
                hits, total = score_answer(r["answer"], needle["expected_keywords"])
                r["score"] = f"{hits}/{total}"
                r["hits"] = hits
                r["total"] = total
                results.append(r)
            except Exception as e:
                print(f"    FAILED: {e}")
                import traceback
                traceback.print_exc()

        all_results[needle["id"]] = results

        # Print per-needle results
        print(f"\n  {'Mode':<30} {'Score':>6} {'TTFT':>10} {'Recon':>8} "
              f"{'Mem':>8} {'h0':>6} {'Answer (first 100 chars)'}")
        print(f"  {'-'*110}")
        for r in results:
            answer_preview = r["answer"][:100].replace("\n", " ")
            recon_str = f"{r['recon_ms']:.0f}ms" if r["recon_ms"] > 0 else "-"
            print(f"  {r['label']:<30} {r['score']:>6} {r['ttft_ms']:>9.0f}ms {recon_str:>8} "
                  f"{r['tg_mem_mb']:>7.0f}M {r['h0_mb']:>5.1f} {answer_preview}")

    # Summary scorecard
    print(f"\n\n{'='*120}")
    print(f"  RECALL SCORECARD")
    print(f"{'='*120}")
    print(f"  {'Mode':<30}", end="")
    for needle in NEEDLES:
        print(f" {needle['id']:>15}", end="")
    print(f" {'TOTAL':>8}")
    print(f"  {'-'*95}")

    for cfg_idx in range(len(configs)):
        label = configs[cfg_idx][0]
        total_hits = 0
        total_possible = 0
        print(f"  {label:<30}", end="")
        for needle in NEEDLES:
            results = all_results.get(needle["id"], [])
            if cfg_idx < len(results):
                r = results[cfg_idx]
                total_hits += r["hits"]
                total_possible += r["total"]
                mark = "PASS" if r["hits"] == r["total"] else f"{r['hits']}/{r['total']}"
                print(f" {mark:>15}", end="")
            else:
                print(f" {'FAIL':>15}", end="")
        overall = f"{total_hits}/{total_possible}"
        print(f" {overall:>8}")


if __name__ == "__main__":
    main()
