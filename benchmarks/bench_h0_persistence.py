"""
H0 Persistence Benchmark: Write → Clear → Load → Reconstruct → Verify.

Tests that h^(0) can be saved to disk, loaded back, and used for
reconstruction that produces correct K/V (identical generation output).

Usage:
    python3 benchmarks/bench_h0_persistence.py /path/to/model
    python3 benchmarks/bench_h0_persistence.py /path/to/model --h0-quant q8
"""

import argparse
import gc
import os
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
    H0Store,
    _find_inner_model,
    _run_reconstruction,
    unpatch_model,
)
from mlx_lm.sample_utils import make_sampler

GREEDY = make_sampler(temp=0.0)

# Toshiba primary, /tmp fallback
PERSIST_DIR = "/Volumes/toshiba/flashmlx/h0_cache"
FALLBACK_DIR = "/tmp/flashmlx_h0_test"

# Reuse bench_recall_d's haystack infrastructure
FILLER_PARA = (
    "The development of artificial intelligence has progressed rapidly in recent years. "
    "Machine learning algorithms continue to improve across various benchmarks. Research "
    "teams around the world are exploring new architectures for language understanding. "
    "The computational requirements for training large models have grown exponentially. "
    "Transfer learning has enabled smaller teams to build on pre-trained foundations. "
    "Ethical considerations remain central to AI development discussions globally. "
)

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
    filler_tokens = len(tokenizer.encode(FILLER_PARA))
    n_paras_needed = (target_tokens // filler_tokens) + 10
    needle_positions = [
        max(1, int(n_paras_needed * 0.10)),
        max(2, int(n_paras_needed * 0.50)),
        max(3, int(n_paras_needed * 0.90)),
    ]
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


def score_answer(answer, expected_keywords):
    answer_lower = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return hits, len(expected_keywords)


def generate_answer(model, tokenizer, prompt_tokens, cache, tg_tokens=200):
    gen_tokens = []
    for i, (token_id, logprobs) in enumerate(generate_step(
        prompt_tokens, model, max_tokens=tg_tokens, sampler=GREEDY,
        prompt_cache=cache,
    )):
        gen_tokens.append(token_id)
    return tokenizer.decode(gen_tokens).strip()


def main():
    parser = argparse.ArgumentParser(description="H0 Persistence Benchmark")
    parser.add_argument("model_path", help="Path to model directory")
    parser.add_argument("--prompt-tokens", type=int, default=8192)
    parser.add_argument("--tg-tokens", type=int, default=150)
    parser.add_argument("--h0-quant", default=None, choices=[None, "q8", "q4"],
                        help="h^(0) quantization mode")
    args = parser.parse_args()

    # Determine persist directory
    persist_dir = PERSIST_DIR if os.path.isdir("/Volumes/toshiba") else FALLBACK_DIR
    os.makedirs(persist_dir, exist_ok=True)
    print(f"Persist directory: {persist_dir}")

    print(f"Loading model from {args.model_path}...")
    model, tokenizer = load(args.model_path)
    card = load_card_or_detect(model, args.model_path)
    print(f"Card: {card.model_name}")

    # Build haystack
    print(f"\nBuilding {args.prompt_tokens:,}-token haystack with {len(NEEDLES)} needles...")
    haystack, actual_tokens = build_haystack_prompt(tokenizer, args.prompt_tokens, NEEDLES)
    print(f"  Actual: {actual_tokens:,} tokens")

    # Warmup
    warm_tokens = mx.array(tokenizer.encode("Hello")).reshape(1, -1)
    model(warm_tokens)
    mx.eval(model.parameters())

    quant_modes = [args.h0_quant] if args.h0_quant else [None, "q8", "q4"]

    print(f"\n{'='*100}")
    print(f"  H0 PERSISTENCE BENCHMARK | {card.model_name} | {actual_tokens:,} tokens")
    print(f"{'='*100}")

    for h0_quant in quant_modes:
        quant_label = h0_quant or "bf16"
        print(f"\n--- Quant mode: {quant_label} ---")

        # Phase 1: Gold reference (standard, no compression)
        print("  Phase 1: Gold reference (standard)...")
        unpatch_model(model)
        gc.collect(); mx.clear_cache()

        gold_answers = {}
        for needle in NEEDLES:
            cache = make_prompt_cache(model, kv_cache="standard")
            full_prompt = haystack + f"\n\nQuestion: {needle['question']}\nAnswer:"
            prompt_tokens = mx.array(tokenizer.encode(full_prompt))
            answer = generate_answer(model, tokenizer, prompt_tokens, cache, args.tg_tokens)
            hits, total = score_answer(answer, needle["expected_keywords"])
            gold_answers[needle["id"]] = {
                "answer": answer,
                "hits": hits,
                "total": total,
            }
            print(f"    {needle['id']}: {hits}/{total} — {answer[:80]}")

        # Phase 2: Scored_kv_direct run — captures h^(0)
        print(f"  Phase 2: scored_kv_direct run (h0_quant={quant_label})...")
        unpatch_model(model)
        gc.collect(); mx.clear_cache()

        cache_kwargs = card.to_cache_kwargs(mode="recall_first")
        if h0_quant:
            cache_kwargs["h0_quant"] = h0_quant
        cache = make_prompt_cache(model, **cache_kwargs)

        # Process haystack to fill h^(0)
        haystack_tokens = mx.array(tokenizer.encode(haystack))
        model_out = model(haystack_tokens.reshape(1, -1), cache=cache)
        mx.eval(model_out)

        # Find h0_store
        h0_store = None
        for c in cache:
            h0_store = getattr(c, "_h0_store", None)
            if h0_store is not None:
                break

        if h0_store is None:
            print("    FAILED: No h0_store found")
            continue

        print(f"    h0_store: {h0_store.count} tokens, {h0_store.nbytes / (1024*1024):.1f} MB in GPU")

        # Phase 3: Save to disk
        ts = int(time.time())
        save_path = os.path.join(persist_dir, f"qwen3-8b_{quant_label}_{ts}")
        print(f"  Phase 3: Saving to {save_path}...")
        t_save_start = time.perf_counter()
        h0_store.save(save_path, metadata={
            "model_name": card.model_name,
            "model_path": args.model_path,
            "n_prompt_tokens": actual_tokens,
        })
        t_save = (time.perf_counter() - t_save_start) * 1000
        file_path = save_path + ".npz" if not save_path.endswith(".npz") else save_path
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"    Saved: {file_size_mb:.1f} MB, {t_save:.0f} ms")

        # Phase 4: Clear GPU
        print("  Phase 4: Clearing GPU...")
        unpatch_model(model)
        mem_before_clear = get_mem_mb()
        del cache, h0_store, model_out
        gc.collect()
        mx.clear_cache()
        time.sleep(0.5)
        mem_after_clear = get_mem_mb()
        print(f"    Memory: {mem_before_clear:.0f} MB → {mem_after_clear:.0f} MB")

        # Phase 5: Load from disk
        print(f"  Phase 5: Loading from {file_path}...")
        t_load_start = time.perf_counter()
        h0_loaded, meta = H0Store.load(save_path)
        t_load = (time.perf_counter() - t_load_start) * 1000
        print(f"    Loaded: {h0_loaded.count} tokens, {t_load:.0f} ms")
        print(f"    Metadata: quant={meta.get('quant')}, model={meta.get('model_name')}")

        # Phase 6: Reconstruct + generate
        # Use scored_pq (no h0 capture/probe) — we already have h0 from disk
        print("  Phase 6: Reconstruct + generate...")
        unpatch_model(model)
        recon_answers = {}

        # Build scored_pq kwargs (same as recall_first but without scored_kv_direct)
        base_kwargs = card.to_cache_kwargs()  # scored_pq baseline
        if h0_quant:
            base_kwargs["h0_quant"] = h0_quant

        for needle in NEEDLES:
            gc.collect(); mx.clear_cache()
            unpatch_model(model)

            # Create scored_pq cache (no h0 capture, no probe)
            cache = make_prompt_cache(model, **base_kwargs)

            # Process haystack through model (fills and compresses cache normally)
            model_out = model(haystack_tokens.reshape(1, -1), cache=cache)
            mx.eval(model_out)

            # Reconstruct from loaded h^(0)
            try:
                inner_model = _find_inner_model(model)
            except ValueError:
                print("    FAILED: Cannot find inner model")
                break

            caches = list(cache)
            t_recon_start = time.perf_counter()
            _run_reconstruction(
                inner_model, caches, h0_loaded, h0_loaded.count,
                kv_direct_indices=list(range(len(caches))),
            )
            recon_arrays = [c._recon_keys for c in caches if getattr(c, '_recon_keys', None) is not None]
            if recon_arrays:
                mx.eval(*recon_arrays)
            for c in caches:
                if getattr(c, '_recon_keys', None) is not None:
                    c._flat_prefix_token_count = max(
                        getattr(c, '_flat_prefix_token_count', 0),
                        h0_loaded.count,
                    )
            t_recon = (time.perf_counter() - t_recon_start) * 1000

            # Generate answer
            question_text = f"\n\nQuestion: {needle['question']}\nAnswer:"
            question_tokens = mx.array(tokenizer.encode(question_text))
            answer = generate_answer(model, tokenizer, question_tokens, cache, args.tg_tokens)
            hits, total = score_answer(answer, needle["expected_keywords"])
            recon_answers[needle["id"]] = {
                "answer": answer,
                "hits": hits,
                "total": total,
                "recon_ms": t_recon,
            }
            print(f"    {needle['id']}: {hits}/{total} (recon={t_recon:.0f}ms) — {answer[:80]}")

        # Phase 7: Compare
        print(f"\n  {'='*90}")
        print(f"  COMPARISON | quant={quant_label}")
        print(f"  {'='*90}")
        print(f"  {'Needle':<20} {'Gold':>6} {'Restored':>10} {'Recon ms':>10} {'Match':>8}")
        print(f"  {'-'*60}")

        total_gold_hits = 0
        total_recon_hits = 0
        total_possible = 0
        for needle in NEEDLES:
            g = gold_answers.get(needle["id"], {})
            r = recon_answers.get(needle["id"], {})
            g_score = f"{g.get('hits', 0)}/{g.get('total', 0)}"
            r_score = f"{r.get('hits', 0)}/{r.get('total', 0)}"
            recon_ms = f"{r.get('recon_ms', 0):.0f}ms"
            match = "PASS" if r.get("hits", 0) == g.get("total", 0) else "DEGRADED"
            total_gold_hits += g.get("hits", 0)
            total_recon_hits += r.get("hits", 0)
            total_possible += g.get("total", 0)
            print(f"  {needle['id']:<20} {g_score:>6} {r_score:>10} {recon_ms:>10} {match:>8}")

        print(f"  {'-'*60}")
        print(f"  {'TOTAL':<20} {total_gold_hits}/{total_possible}     "
              f"{total_recon_hits}/{total_possible}")
        print(f"\n  File: {file_path} ({file_size_mb:.1f} MB)")
        print(f"  Save: {t_save:.0f} ms | Load: {t_load:.0f} ms")

        # Cleanup test file
        if persist_dir == FALLBACK_DIR:
            os.remove(file_path)
            print(f"  (Cleaned up test file)")

    print(f"\n{'='*100}")
    print(f"  PERSISTENCE BENCHMARK COMPLETE")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
