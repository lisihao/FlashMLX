#!/usr/bin/env python3
"""Minimal batch=4 only test — NO single-request warmup."""
import gc, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mlx-lm-source"))

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import BatchGenerator

MODEL = "mlx-community/Qwen3.5-35B-A3B-4bit"
CTX = 16384
GEN = 128

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
    enc = tokenizer.encode(fmt)
    print(f"Prompt: {len(enc)} tokens")

    # Batch=4, NO warmup
    print(f"\n{'='*60}")
    print(f"  Batch=4, gen={GEN}, NO warmup")
    print(f"{'='*60}")

    prompts_enc = [enc[:] for _ in range(4)]

    bg = BatchGenerator(model, max_tokens=GEN,
                        completion_batch_size=4, prefill_batch_size=4)
    uids = bg.insert(prompts_enc, max_tokens=[GEN]*4)

    all_tokens = {uid: [] for uid in uids}
    finished = set()
    total_gen = 0
    first_token_time = None
    step_times = []

    t0 = time.perf_counter()
    consecutive_empty = 0

    while True:
        t_step = time.perf_counter()
        responses = bg.next()
        dt_step = time.perf_counter() - t_step
        if not responses:
            consecutive_empty += 1
            if consecutive_empty > 500:
                break
            continue
        consecutive_empty = 0
        step_times.append(dt_step)
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

    # Step timing
    if step_times:
        avg = sum(step_times) / len(step_times) * 1000
        first5 = [f"{t*1000:.1f}" for t in step_times[:5]]
        last5 = [f"{t*1000:.1f}" for t in step_times[-5:]]
        decode_steps = step_times[1:]  # exclude first (includes prefill)
        if decode_steps:
            avg_decode = sum(decode_steps) / len(decode_steps) * 1000
            min_d = min(decode_steps) * 1000
            max_d = max(decode_steps) * 1000
            # Count fast vs slow steps
            fast = sum(1 for t in decode_steps if t < 0.05)
            slow = sum(1 for t in decode_steps if t >= 0.05)
            print(f"  Decode steps: {len(decode_steps)}, avg={avg_decode:.1f}ms, "
                  f"min={min_d:.1f}ms, max={max_d:.1f}ms")
            print(f"  Fast (<50ms): {fast}, Slow (>=50ms): {slow}")

        print(f"  First 5 steps: {first5}")
        print(f"  Last  5 steps: {last5}")

    print(f"\n  TTFT:  {ttft:.2f}s")
    print(f"  TG:    {tg:.1f} tok/s  ({total_gen} tokens / {decode_time:.3f}s)")
    print(f"  Steps: {n_steps}  →  {decode_time/n_steps*1000:.1f}ms/step (wall)")
    print(f"  Total: {elapsed:.2f}s")

    for uid in uids:
        text = tokenizer.decode(all_tokens[uid])
        print(f"  UID {uid}: {len(all_tokens[uid])} tokens → {text[:60]}")

    print(f"\nDone!")

if __name__ == "__main__":
    main()
