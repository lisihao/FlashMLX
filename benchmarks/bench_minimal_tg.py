#!/usr/bin/env python3
"""Minimal TG speed test — single request + batch=4, generate_step only."""
import gc, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mlx-lm-source"))

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step, BatchGenerator

MODEL = "mlx-community/Qwen3.5-35B-A3B-4bit"
CTX = 16384
GEN = 100

def resolve_model(path):
    if not os.path.isdir(path):
        from huggingface_hub import snapshot_download
        return snapshot_download(path, local_files_only=True)
    return path

def build_prompt(tokenizer, target):
    block = (
        "Section {n}: Performance metrics show steady improvement in Q3. "
        "Budget allocations reviewed. Infrastructure upgrades scheduled. "
        "Training programs expanded. Resource allocation optimized. "
        "Cross-departmental collaboration continues positive results. "
    )
    blocks = []
    n = 1
    while True:
        blocks.append(block.format(n=n))
        text = "".join(blocks) + "\nWhat is the annual budget for Project Alpha? Answer: $12,500."
        if len(tokenizer.encode(text)) >= target - 100:
            break
        n += 1
    return text

def main():
    path = resolve_model(MODEL)
    print(f"Loading {MODEL}...")
    model, tokenizer = load(path)
    mx.eval(model.parameters())
    gc.collect()
    mem = mx.get_active_memory() / 1024**3
    print(f"Model loaded: {mem:.2f} GB")

    prompt_text = build_prompt(tokenizer, CTX)
    msgs = [{"role": "user", "content": prompt_text}]
    try:
        fmt = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    except TypeError:
        fmt = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    prompt = mx.array(tokenizer.encode(fmt))
    print(f"Prompt: {prompt.shape[0]} tokens")

    # === Test 1: Single request, generate_step ===
    print(f"\n{'='*60}")
    print(f"  Test 1: Single request, generate_step, gen={GEN}")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    gen = generate_step(prompt, model, max_tokens=GEN)
    first_tok, _ = next(gen)
    ttft = time.perf_counter() - t0

    tokens = [first_tok.item() if hasattr(first_tok, 'item') else first_tok]
    t_decode_start = time.perf_counter()

    for tok, _ in gen:
        t = tok.item() if hasattr(tok, 'item') else tok
        tokens.append(t)
        if len(tokens) >= GEN:
            break

    t_decode_end = time.perf_counter()
    decode_time = t_decode_end - t_decode_start
    tg = (len(tokens) - 1) / decode_time
    print(f"  TTFT:  {ttft:.2f}s")
    print(f"  TG:    {tg:.1f} tok/s  ({len(tokens)-1} tokens / {decode_time:.3f}s)")
    print(f"  Per-step: {decode_time/(len(tokens)-1)*1000:.1f}ms")
    text = tokenizer.decode(tokens)
    print(f"  Output: {text[:80]}")

    # Clean up for batch test
    del gen, tokens
    gc.collect()
    mx.clear_cache()

    # === Test 2: Batch=4, BatchGenerator, non-interleaved ===
    print(f"\n{'='*60}")
    print(f"  Test 2: Batch=4, BatchGenerator (non-interleaved), gen={GEN}")
    print(f"{'='*60}")

    # Build 4 prompts
    prompts_enc = []
    for i in range(4):
        prompts_enc.append(tokenizer.encode(fmt))

    bg = BatchGenerator(model, max_tokens=GEN,
                        completion_batch_size=4, prefill_batch_size=4)
    uids = bg.insert(prompts_enc, max_tokens=[GEN]*4)

    all_tokens = {uid: [] for uid in uids}
    finished = set()
    total_gen = 0
    first_token_time = None

    t0 = time.perf_counter()
    consecutive_empty = 0

    while True:
        responses = bg.next()
        if not responses:
            consecutive_empty += 1
            if consecutive_empty > 500:
                break
            continue
        consecutive_empty = 0
        for resp in responses:
            if first_token_time is None:
                first_token_time = time.perf_counter()
            all_tokens[resp.uid].append(resp.token)
            total_gen += 1
            if resp.finish_reason is not None:
                finished.add(resp.uid)
        if len(finished) == 4:
            break

    elapsed = time.perf_counter() - t0
    ttft = first_token_time - t0 if first_token_time else elapsed
    decode_time = elapsed - ttft
    tg = total_gen / decode_time if decode_time > 0 else 0
    n_steps = max(len(v) for v in all_tokens.values())

    bg.close()

    print(f"  TTFT:  {ttft:.2f}s")
    print(f"  TG:    {tg:.1f} tok/s  ({total_gen} tokens / {decode_time:.3f}s)")
    print(f"  Steps: {n_steps}  →  {decode_time/n_steps*1000:.1f}ms/step")
    print(f"  Total: {elapsed:.2f}s")

    # Print per-request token counts
    for uid in uids:
        toks = all_tokens[uid]
        text = tokenizer.decode(toks)
        print(f"  UID {uid}: {len(toks)} tokens → {text[:60]}")

    print(f"\nDone!")

if __name__ == "__main__":
    main()
