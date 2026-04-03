#!/usr/bin/env python3
"""
Dual-Instance h^(0) Experiment Orchestrator.

Runs the complete experiment:
1. Baseline: single-instance gold reference
2. Dual-instance: A (prefill) → shared memory → B (reconstruct + decode)
3. Comparison: output quality + performance metrics

Usage:
    python3 experiments/dual_instance_h0/run_experiment.py \\
        --model /path/to/Qwen3-8B-MLX-4bit \\
        --prompt-tokens 4096 \\
        --max-tg-tokens 200

    # Skip baseline (faster iteration):
    python3 experiments/dual_instance_h0/run_experiment.py \\
        --model /path/to/Qwen3-8B-MLX-4bit \\
        --skip-baseline
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, "/Users/lisihao/FlashMLX/mlx-lm-source")
sys.path.insert(0, "/Users/lisihao/FlashMLX/src")

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Filler paragraph for building haystack (same as bench_h0_persistence)
FILLER_PARA = (
    "The development of artificial intelligence has progressed rapidly in recent years. "
    "Machine learning algorithms continue to improve across various benchmarks. Research "
    "teams around the world are exploring new architectures for language understanding. "
    "The computational requirements for training large models have grown exponentially. "
    "Transfer learning has enabled smaller teams to build on pre-trained foundations. "
    "Ethical considerations remain central to AI development discussions globally. "
)

NEEDLE = {
    "fact": "The secret project code name is 'AURORA-7732' and it was started on March 15th, 2024.",
    "question": "\n\nQuestion: What is the secret project code name and when was it started?\nAnswer:",
    "expected_keywords": ["AURORA-7732", "March 15"],
}


def build_haystack(target_tokens: int, tokenizer_encode) -> str:
    """Build a haystack prompt with an embedded needle."""
    filler_tokens = len(tokenizer_encode(FILLER_PARA))
    n_paras = (target_tokens // filler_tokens) + 5

    # Place needle at ~30% position
    needle_pos = max(1, int(n_paras * 0.30))

    prefix = "Read the following document carefully. You will be asked questions about it.\n\n"
    parts = [prefix]
    for i in range(n_paras):
        if i == needle_pos:
            parts.append(f"\n[Important Note] {NEEDLE['fact']}\n\n")
        parts.append(FILLER_PARA)

    full_text = "".join(parts)
    tokens = tokenizer_encode(full_text)
    if len(tokens) > target_tokens:
        from mlx_lm import load as _load
        # Simple truncation by re-encoding
        tokens = tokens[:target_tokens]

    return full_text, len(tokens)


def score_answer(answer: str, keywords: list[str]) -> tuple[int, int]:
    answer_lower = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return hits, len(keywords)


def run_subprocess(script: str, args: list[str], timeout: float = 600) -> dict:
    """Run a subprocess and parse JSON from its stdout."""
    cmd = [sys.executable, os.path.join(EXPERIMENT_DIR, script)] + args
    print(f"  CMD: {' '.join(cmd[:3])} ...", file=sys.stderr)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=EXPERIMENT_DIR,
    )

    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        return {"status": "timeout", "stderr": stderr.decode(errors="replace")[-2000:]}

    stderr_text = stderr.decode(errors="replace")
    # Print stderr in real-time style
    for line in stderr_text.strip().split("\n"):
        if line:
            print(f"  {line}", file=sys.stderr)

    if proc.returncode != 0:
        return {
            "status": "error",
            "returncode": proc.returncode,
            "stderr": stderr_text[-2000:],
        }

    # Parse JSON from last line of stdout
    stdout_text = stdout.decode(errors="replace").strip()
    for line in reversed(stdout_text.split("\n")):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                pass

    return {"status": "error", "stdout": stdout_text[-1000:], "stderr": stderr_text[-1000:]}


def main():
    parser = argparse.ArgumentParser(description="Dual-Instance h^(0) Experiment")
    parser.add_argument("--model", required=True, help="Path to model directory")
    parser.add_argument("--prompt-tokens", type=int, default=4096,
                        help="Target prompt length in tokens")
    parser.add_argument("--max-tg-tokens", type=int, default=200)
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip baseline (use for faster iteration)")
    parser.add_argument("--embed-only", action="store_true",
                        help="Instance A uses bare embed_tokens() instead of full prefill")
    parser.add_argument("--h0-cache-dir", default=None,
                        help="Directory for h^(0) persistence cache")
    parser.add_argument("--fan-out", type=int, default=1,
                        help="Number of Instance B's to launch (1=normal, >1=fan-out)")
    parser.add_argument("--shm-name", default="flashmlx_h0_bridge")
    args = parser.parse_args()

    print("=" * 90)
    print("  DUAL-INSTANCE h^(0) EXPERIMENT")
    print(f"  Model: {args.model}")
    print(f"  Prompt tokens: {args.prompt_tokens:,}")
    print(f"  Max TG tokens: {args.max_tg_tokens}")
    print("=" * 90)

    # Build haystack prompt — need tokenizer just for this
    print("\n[Orchestrator] Building haystack prompt...", file=sys.stderr)
    from mlx_lm import load
    _, tokenizer = load(args.model)
    prompt_text, actual_tokens = build_haystack(args.prompt_tokens, tokenizer.encode)
    print(f"[Orchestrator] Haystack: {actual_tokens:,} tokens", file=sys.stderr)

    # Save prompt to temp file for subprocess access
    prompt_file = os.path.join(tempfile.gettempdir(), "h0_experiment_prompt.txt")
    with open(prompt_file, "w") as f:
        f.write(prompt_text)

    question = NEEDLE["question"]

    # ===== Phase 1: Baseline =====
    baseline_result = None
    if not args.skip_baseline:
        print("\n" + "-" * 90)
        print("  PHASE 1: BASELINE (single-instance)")
        print("-" * 90)

        t_baseline_start = time.perf_counter()
        baseline_result = run_subprocess("baseline_single.py", [
            "--model", args.model,
            "--prompt", f"@{prompt_file}",
            "--question", question,
            "--max-tg-tokens", str(args.max_tg_tokens),
        ], timeout=600)
        t_baseline = time.perf_counter() - t_baseline_start

        if baseline_result.get("status") == "ok":
            answer = baseline_result["answer"]
            hits, total = score_answer(answer, NEEDLE["expected_keywords"])
            baseline_result["hits"] = hits
            baseline_result["total"] = total
            print(f"\n  Baseline answer ({hits}/{total}): {answer[:150]}")
            print(f"  Total time: {t_baseline:.1f}s")
        else:
            print(f"\n  BASELINE FAILED: {baseline_result}")

    # ===== Phase 2: Dual-Instance =====
    print("\n" + "-" * 90)
    print("  PHASE 2: DUAL-INSTANCE (A: prefill → shm → B: reconstruct + decode)")
    print("-" * 90)

    # Clean up any stale shared memory
    try:
        from multiprocessing import shared_memory
        stale = shared_memory.SharedMemory(name=args.shm_name)
        stale.close()
        stale.unlink()
        print("  (Cleaned stale shared memory)", file=sys.stderr)
    except FileNotFoundError:
        pass

    # Compute prompt hash for cache (if caching enabled)
    prompt_hash = None
    if args.h0_cache_dir:
        import hashlib
        # Tokenize to get token IDs for hashing
        haystack_tokens = tokenizer.encode(prompt_text)
        prompt_hash = hashlib.sha256(
            json.dumps(haystack_tokens).encode()
        ).hexdigest()[:16]
        os.makedirs(args.h0_cache_dir, exist_ok=True)
        print(f"[Orchestrator] h^(0) cache dir: {args.h0_cache_dir}, hash: {prompt_hash}", file=sys.stderr)

    # Launch Instance A (in background)
    a_mode = "embed-only" if args.embed_only else "full-prefill"
    print(f"\n  Starting Instance A ({a_mode})...", file=sys.stderr)
    a_cmd = [
        sys.executable, os.path.join(EXPERIMENT_DIR, "instance_a_prefill.py"),
        "--model", args.model,
        "--prompt", f"@{prompt_file}",
        "--shm-name", args.shm_name,
        "--max-tokens", str(args.prompt_tokens + 1024),
    ]
    if args.embed_only:
        a_cmd.append("--embed-only")
    if args.h0_cache_dir:
        a_cmd.extend(["--h0-cache-dir", args.h0_cache_dir])
    a_proc = subprocess.Popen(
        a_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=EXPERIMENT_DIR,
    )

    # Give A a moment to load model and start writing
    # Then launch Instance B(s) (which will poll for h^(0))
    time.sleep(2.0)

    # Build Instance B base command
    b_base_cmd = [
        sys.executable, os.path.join(EXPERIMENT_DIR, "instance_b_decode.py"),
        "--model", args.model,
        "--question", question,
        "--shm-name", args.shm_name,
        "--max-tg-tokens", str(args.max_tg_tokens),
        "--timeout", "300",
    ]
    if args.h0_cache_dir and prompt_hash:
        b_base_cmd.extend(["--h0-cache-dir", args.h0_cache_dir, "--prompt-hash", prompt_hash])

    n_b_instances = args.fan_out
    b_procs = []
    t_dual_start = time.perf_counter()

    for b_idx in range(n_b_instances):
        b_cmd = list(b_base_cmd)
        if n_b_instances > 1:
            b_cmd.append("--fan-out-reader")
        label = f"B{b_idx}" if n_b_instances > 1 else "B"
        print(f"  Starting Instance {label} (reconstruct + decode)...", file=sys.stderr)
        b_proc = subprocess.Popen(
            b_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=EXPERIMENT_DIR,
        )
        b_procs.append(b_proc)

    # Wait for all B instances to complete
    print(f"  Waiting for A + {n_b_instances} B instance(s)...", file=sys.stderr)

    b_results_raw = []
    for b_proc in b_procs:
        try:
            b_stdout, b_stderr = b_proc.communicate(timeout=600)
        except subprocess.TimeoutExpired:
            b_proc.kill()
            b_stdout, b_stderr = b_proc.communicate()
        b_results_raw.append((b_stdout, b_stderr))

    try:
        a_stdout, a_stderr = a_proc.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        a_proc.kill()
        a_stdout, a_stderr = a_proc.communicate()

    t_dual = time.perf_counter() - t_dual_start

    # Print stderr from A and all B's
    a_stderr_text = a_stderr.decode(errors="replace").strip()
    for line in a_stderr_text.split("\n"):
        if line:
            print(f"  {line}", file=sys.stderr)
    for b_idx, (b_stdout_raw, b_stderr_raw) in enumerate(b_results_raw):
        b_label = f"B{b_idx}" if n_b_instances > 1 else "B"
        b_stderr_text = b_stderr_raw.decode(errors="replace").strip()
        for line in b_stderr_text.split("\n"):
            if line:
                print(f"  [{b_label}] {line}", file=sys.stderr)

    # Parse results
    def parse_json_output(stdout_bytes):
        text = stdout_bytes.decode(errors="replace").strip()
        for line in reversed(text.split("\n")):
            line = line.strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    pass
        return {"status": "parse_error", "raw": text[-500:]}

    a_result = parse_json_output(a_stdout)
    b_results = [parse_json_output(raw[0]) for raw in b_results_raw]
    b_result = b_results[0]  # primary B for backward compat

    if a_result.get("status") != "ok":
        print(f"\n  INSTANCE A FAILED: {a_result}")
    for b_idx, br in enumerate(b_results):
        if br.get("status") != "ok":
            label = f"B{b_idx}" if n_b_instances > 1 else "B"
            print(f"\n  INSTANCE {label} FAILED: {br}")

    dual_answer = b_result.get("answer", "")
    dual_hits, dual_total = score_answer(dual_answer, NEEDLE["expected_keywords"])

    print(f"\n  Dual-instance answer ({dual_hits}/{dual_total}): {dual_answer[:150]}")
    print(f"  Total dual time: {t_dual:.1f}s")

    # Fan-out consistency check
    if n_b_instances > 1:
        all_answers = [br.get("answer", "").strip() for br in b_results]
        all_match = all(a == all_answers[0] for a in all_answers)
        print(f"  Fan-out ({n_b_instances} B's): {'ALL MATCH' if all_match else 'MISMATCH!'}")

    # ===== Phase 3: Comparison =====
    print("\n" + "=" * 90)
    print("  RESULTS COMPARISON")
    print("=" * 90)

    # Size analysis
    n_tokens = a_result.get("n_tokens", 0)
    d_hidden = a_result.get("d_hidden", 4096)
    h0_bytes = a_result.get("h0_bytes", 0)
    # Qwen3-8B: 36 layers, 4 KV heads, 128 head_dim
    n_layers = b_result.get("n_layers", 36)
    n_kv_heads = 4
    head_dim = 128
    kv_bytes = 2 * n_layers * n_kv_heads * head_dim * 2 * n_tokens

    print(f"\n  State Transfer:")
    print(f"    h^(0) transferred:  {h0_bytes / (1024*1024):.2f} MB ({n_tokens:,} tokens × {d_hidden} × 2B)")
    print(f"    Equivalent KV:      {kv_bytes / (1024*1024):.2f} MB ({n_tokens:,} tokens × {n_layers}L × {n_kv_heads}H × {head_dim}D × 2 × 2B)")
    if h0_bytes > 0:
        print(f"    Compression ratio:  {kv_bytes / h0_bytes:.1f}×")
    print(f"    Bandwidth saved:    {(kv_bytes - h0_bytes) / (1024*1024):.2f} MB ({(1 - h0_bytes/kv_bytes)*100:.1f}%)" if kv_bytes > 0 else "")

    print(f"\n  Timing (Instance A — {a_result.get('mode', 'full_prefill')}):")
    if a_result.get("status") == "ok":
        print(f"    Model load:   {a_result.get('load_ms', 0):.0f} ms")
        label = "Embed" if a_result.get("mode") == "embed_only" else "Prefill"
        print(f"    {label}:      {a_result.get('prefill_ms', 0):.2f} ms ({a_result.get('prefill_tok_per_s', 0):.0f} tok/s)")
        print(f"    SHM write:    {a_result.get('write_ms', 0):.1f} ms")
        if a_result.get("prompt_hash"):
            print(f"    Cache hash:   {a_result['prompt_hash']}")

    print(f"\n  Timing (Instance B):")
    if b_result.get("status") == "ok":
        print(f"    Model load:   {b_result.get('load_ms', 0):.0f} ms")
        print(f"    SHM read:     {b_result.get('read_ms', 0):.1f} ms")
        print(f"    Reconstruct:  {b_result.get('recon_ms', 0):.0f} ms")
        print(f"    TG decode:    {b_result.get('tg_ms', 0):.0f} ms ({b_result.get('tg_tok_per_s', 0):.1f} tok/s)")

    print(f"\n  Quality:")
    if baseline_result and baseline_result.get("status") == "ok":
        bl_answer = baseline_result["answer"]
        bl_hits = baseline_result.get("hits", 0)
        bl_total = baseline_result.get("total", 0)
        print(f"    Baseline:       {bl_hits}/{bl_total} keywords — {bl_answer[:100]}")
        print(f"    Dual-instance:  {dual_hits}/{dual_total} keywords — {dual_answer[:100]}")
        exact_match = bl_answer.strip() == dual_answer.strip()
        print(f"    Exact match:    {'YES' if exact_match else 'NO'}")
    else:
        print(f"    Dual-instance:  {dual_hits}/{dual_total} keywords — {dual_answer[:100]}")
        print(f"    (Baseline skipped)")

    print(f"\n  Theoretical Significance:")
    print(f"    KV state:     2 × L × n_kv × d_h × b = 2 × {n_layers} × {n_kv_heads} × {head_dim} × 2 = {2 * n_layers * n_kv_heads * head_dim * 2:,} bytes/token")
    print(f"    Residual:     d_model × b = {d_hidden} × 2 = {d_hidden * 2:,} bytes/token")
    print(f"    Ratio:        {2 * n_layers * n_kv_heads * head_dim * 2 / (d_hidden * 2):.1f}× — layer-independent compression")

    print("\n" + "=" * 90)
    verdict = "PASS" if dual_hits == dual_total else "PARTIAL"
    if b_result.get("status") != "ok":
        verdict = "FAIL"
    print(f"  VERDICT: {verdict}")
    print("=" * 90)

    # Cleanup
    try:
        os.remove(prompt_file)
    except OSError:
        pass

    # Return exit code
    sys.exit(0 if verdict in ("PASS", "PARTIAL") else 1)


if __name__ == "__main__":
    main()
