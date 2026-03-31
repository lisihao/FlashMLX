#!/usr/bin/env python3
"""
AM Offline Calibration

一次性拟合 AM 压缩参数，存储到文件供后续使用。

Usage:
    python calibrate_am_offline.py --model qwen3-8b --ratio 2.0 --queries 12288
"""

import sys
sys.path.insert(0, '/Users/lisihao/FlashMLX/mlx-lm-source')

import mlx.core as mx
from mlx_lm import load
import numpy as np
import pickle
from pathlib import Path
import argparse
from datetime import datetime

def log(msg, end='\n'):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True, end=end)

# Calibration corpus
CALIBRATION_CORPUS = """
Dr. Sarah Chen founded the Quantum Dynamics Research Lab at Stanford University in 2019 with $5 million from the National Science Foundation. Her team aimed to develop room-temperature quantum computers.

The initial phase involved assembling a diverse team. Chen recruited Dr. Robert Kim from MIT, Dr. Elena Rodriguez from Caltech, and Dr. Yuki Tanaka from Tokyo. They started with theoretical modeling.

In 2020, they built their first prototype in the basement laboratory. Early tests were disappointing - quantum coherence lasted only milliseconds at room temperature.

The team persevered through 2021, making incremental improvements. They experimented with different materials and by December 2021 had extended coherence to 3 seconds at 280 Kelvin.

The breakthrough came on July 15, 2022, at 3:47 AM. The quantum processor achieved stable coherence at 294 Kelvin (room temperature) for 47 seconds. They ran 127 experiments with 89% success rate.

Professor Marcus Blackwell from Oxford criticized the results. However, five teams (Tokyo, Cambridge, Zurich, Tsinghua, Caltech) replicated them with 84% success rate. Blackwell later admitted his calibration error.

Dr. Chen received the Nobel Prize in 2024, shared with Dr. Robert Kim and Dr. Elena Rodriguez. She announced open-sourcing the basic designs.
"""

CALIBRATION_QUESTIONS = [
    # Factual questions
    "When was the Quantum Dynamics Research Lab founded?",
    "Who founded the research lab?",
    "How much funding did the lab receive initially?",
    "What was the lab's main research goal?",

    # Detail questions
    "Who did Dr. Chen recruit for her team?",
    "Where did Dr. Robert Kim come from?",
    "What happened in 2020?",
    "How long did quantum coherence last in early tests?",

    # Timeline questions
    "What improvements were made in 2021?",
    "When did the breakthrough occur?",
    "What time did the breakthrough happen?",
    "How many seconds of coherence was achieved in the breakthrough?",

    # Numeric questions
    "How many experiments were run during the breakthrough?",
    "What was the success rate of the breakthrough experiments?",
    "How many teams replicated the results?",
    "What was the replication success rate?",

    # People questions
    "Who criticized the results?",
    "Where was Professor Blackwell from?",
    "What did Blackwell later admit?",
    "Who received the Nobel Prize?",
    "When did Dr. Chen receive the Nobel Prize?",

    # Summary questions
    "Summarize the timeline of the quantum computing research.",
    "Describe the breakthrough that occurred in July 2022.",
    "What was the controversy surrounding the results?",
]

def generate_queries_repeat_prefill(model, tokenizer, num_repeats=5):
    """
    Generate queries using repeat-prefill method (stable, conservative).

    Repeats the same prompt multiple times to get stable query distribution.
    """
    from mlx_lm.models.cache import KVCache

    log(f"Generating queries using repeat-prefill (n={num_repeats})...")

    # Use a representative prompt (medium length ~300 tokens)
    SHORT_CORPUS = CALIBRATION_CORPUS[:1500]  # ~300 tokens
    prompt = f"{SHORT_CORPUS}\n\nQuestion: When was the lab founded?\nAnswer:"
    prompt_tokens = tokenizer.encode(prompt)

    log(f"  Prompt length: {len(prompt_tokens)} tokens")

    num_layers = len(model.model.layers)
    all_queries_per_layer = [[] for _ in range(num_layers)]

    for rep in range(num_repeats):
        log(f"  [{rep+1}/{num_repeats}] Repeat prefill...")

        # Create cache
        cache = [KVCache() for _ in range(num_layers)]

        # Prefill only (no decode)
        y = mx.array([prompt_tokens])
        logits = model(y[:, :-1], cache=cache)

        # Extract queries from KV cache
        for layer_idx in range(num_layers):
            keys = cache[layer_idx].keys
            all_queries_per_layer[layer_idx].append(keys)

        log(f"      Collected {cache[0].offset} queries")

    # Merge
    merged_queries_per_layer = []
    for layer_idx in range(num_layers):
        merged = mx.concatenate(all_queries_per_layer[layer_idx], axis=2)
        merged_queries_per_layer.append(merged)

    total = merged_queries_per_layer[0].shape[2]
    log(f"✓ Repeat-prefill: {total} queries")

    return merged_queries_per_layer, total

def generate_queries_self_study(model, tokenizer, num_queries_target):
    """
    Generate queries using self-study method (diverse).

    Uses shorter corpus (256-512 tokens) for better length matching.
    """
    from mlx_lm.models.cache import KVCache

    log(f"Generating queries using self-study (target: {num_queries_target})...")

    # Use shorter corpus for better length control (256-512 tokens)
    SHORT_CORPUS = CALIBRATION_CORPUS[:2000]  # ~400 tokens
    log(f"  Corpus: {len(SHORT_CORPUS)} chars (~400 tokens)")
    log(f"  Questions: {len(CALIBRATION_QUESTIONS)}")

    num_layers = len(model.model.layers)
    all_queries_per_layer = [[] for _ in range(num_layers)]

    total_queries_collected = 0

    for i, question in enumerate(CALIBRATION_QUESTIONS):
        log(f"  [{i+1}/{len(CALIBRATION_QUESTIONS)}] {question[:50]}...")

        # Build prompt with SHORT corpus
        prompt = f"{SHORT_CORPUS}\n\nQuestion: {question}\nAnswer:"
        prompt_tokens = tokenizer.encode(prompt)

        # Create cache
        cache = [KVCache() for _ in range(num_layers)]

        # Prefill
        y = mx.array([prompt_tokens])
        logits = model(y[:, :-1], cache=cache)

        # Decode (generate complete answer, max 50 tokens to keep total ~512)
        y = mx.array([[prompt_tokens[-1]]])
        for _ in range(50):
            logits = model(y, cache=cache)
            y = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
            token_id = y[0, 0].item()
            if token_id == tokenizer.eos_token_id:
                break

        # Extract queries from KV cache
        for layer_idx in range(num_layers):
            # cache[layer_idx].keys: (B=1, num_heads, seq_len, head_dim)
            keys = cache[layer_idx].keys
            all_queries_per_layer[layer_idx].append(keys)

        total_queries_collected += cache[0].offset
        log(f"      Collected {cache[0].offset} queries (total: {total_queries_collected})")

        # Stop if we have enough queries
        if total_queries_collected >= num_queries_target:
            log(f"  ✓ Reached target {num_queries_target} queries")
            break

    # Merge all queries
    log("Merging self-study queries from all questions...")
    merged_queries_per_layer = []
    for layer_idx in range(num_layers):
        # Concatenate along seq_len dimension
        merged = mx.concatenate(all_queries_per_layer[layer_idx], axis=2)
        merged_queries_per_layer.append(merged)

    total_queries = merged_queries_per_layer[0].shape[2]
    log(f"✓ Self-study: {total_queries} queries")

    return merged_queries_per_layer, total_queries

def generate_queries(model, tokenizer, num_queries_target):
    """
    Generate queries using repeat-prefill + self-study combination.

    Strategy:
    - Repeat-prefill: 20% (stable, conservative)
    - Self-study: 80% (diverse, comprehensive)
    """
    log(f"Generating {num_queries_target} queries using hybrid method...")
    log(f"  Strategy: 20% repeat-prefill + 80% self-study")

    # 1. Repeat-prefill (20%)
    num_repeat_queries = int(num_queries_target * 0.2)
    num_repeats = max(5, num_repeat_queries // 300)  # ~300 tokens per repeat
    repeat_queries, repeat_total = generate_queries_repeat_prefill(
        model, tokenizer, num_repeats=num_repeats
    )

    # 2. Self-study (80%)
    num_self_study_queries = num_queries_target - repeat_total
    self_study_queries, self_study_total = generate_queries_self_study(
        model, tokenizer, num_self_study_queries
    )

    # 3. Merge both
    log("Merging repeat-prefill + self-study queries...")
    merged_queries_per_layer = []
    num_layers = len(model.model.layers)

    for layer_idx in range(num_layers):
        merged = mx.concatenate([
            repeat_queries[layer_idx],
            self_study_queries[layer_idx]
        ], axis=2)
        merged_queries_per_layer.append(merged)

    total_queries = merged_queries_per_layer[0].shape[2]
    log(f"✓ Total queries: {total_queries}")
    log(f"  - Repeat-prefill: {repeat_total} ({repeat_total/total_queries*100:.1f}%)")
    log(f"  - Self-study: {self_study_total} ({self_study_total/total_queries*100:.1f}%)")

    return merged_queries_per_layer, total_queries

def fit_am_layer(layer_idx, queries, keys, compression_ratio):
    """
    Fit AM compression parameters for a single layer.

    Returns: {
        'Ck': selected keys,
        'beta': weights,
        'selected_indices': indices of selected keys
    }
    """
    import scipy.optimize
    from scipy.linalg import lstsq

    # Convert to numpy for scipy (handle bfloat16 -> float32 conversion)
    queries_mlx = queries[0, 0, :, :]
    keys_mlx = keys[0, 0, :, :]

    # Convert to float32 first to avoid bfloat16 conversion issues
    if queries_mlx.dtype == mx.bfloat16:
        queries_mlx = queries_mlx.astype(mx.float32)
    if keys_mlx.dtype == mx.bfloat16:
        keys_mlx = keys_mlx.astype(mx.float32)

    queries_np = np.array(queries_mlx)  # (num_queries, head_dim)
    keys_np = np.array(keys_mlx)        # (seq_len, head_dim)

    num_queries, head_dim = queries_np.shape
    seq_len = keys_np.shape[0]

    # Budget
    budget = int(seq_len / compression_ratio)

    # OMP: Select top-k keys
    # Compute attention scores: Q @ K^T
    raw_scores = queries_np @ keys_np.T  # (num_queries, seq_len)

    # Apply softmax to get attention weights (AM works on softmax-ed weights)
    # Note: We don't apply scaling (1/sqrt(d_k)) here to keep scores in reasonable range
    scores = scipy.special.softmax(raw_scores, axis=1)  # (num_queries, seq_len)

    avg_scores = np.mean(scores, axis=0)  # (seq_len,) - use softmax-ed scores for selection

    # Select top budget keys
    selected_indices = np.argsort(avg_scores)[-budget:]
    selected_indices = np.sort(selected_indices)  # Keep order

    Ck = keys_np[selected_indices]  # (budget, head_dim)

    # Fit beta using bounded least-squares
    # Target: Q @ K^T
    # Approximation: Q @ (Ck @ diag(beta))^T = Q @ diag(beta) @ Ck^T
    # We want: Q @ K^T ≈ Q @ diag(beta) @ Ck^T
    # → K^T ≈ diag(beta) @ Ck^T
    # → K ≈ Ck @ diag(beta)^T = Ck * beta (element-wise)

    # Fit beta using bounded least-squares
    R_S = scores[:, selected_indices]  # (num_queries, budget)

    # ✅ FIX: Target should be row-wise SUM (= 1.0 after softmax), not mean
    # AM goal: S @ 1 ≈ S[:, selected] @ beta
    # where S @ 1 = row-wise sum = 1.0 (softmax property)
    target = np.sum(scores, axis=1)   # (num_queries,) - should be all 1.0

    # Verify target is close to 1.0
    assert np.abs(target.mean() - 1.0) < 0.01, f"Target mean {target.mean()} != 1.0, softmax issue?"

    # Solve: R_S @ beta ≈ target
    # With bounds: 0 ≤ beta_i ≤ 2
    res = scipy.optimize.lsq_linear(
        R_S,
        target,
        bounds=(0, 2),
        method='bvls'
    )
    beta = res.x

    return {
        'Ck': mx.array(Ck),
        'beta': mx.array(beta),
        'selected_indices': mx.array(selected_indices, dtype=mx.int32),
        'compression_ratio': compression_ratio,
        'budget': budget
    }

def calibrate_offline(model, tokenizer, compression_ratio, num_queries_target):
    """
    Offline calibration: fit AM parameters for all layers.

    Returns: calibration dict
    """
    num_layers = len(model.model.layers)

    # Step 1: Generate queries
    log("=" * 70)
    log("Step 1: Generate Queries")
    log("=" * 70)
    queries_per_layer, total_queries = generate_queries(model, tokenizer, num_queries_target)

    # Step 2: Generate full KV cache for calibration corpus
    log("\n" + "=" * 70)
    log("Step 2: Generate Full KV Cache")
    log("=" * 70)

    from mlx_lm.models.cache import KVCache

    # Use SHORT corpus to match query length (~400 tokens)
    SHORT_CORPUS = CALIBRATION_CORPUS[:2000]  # ~400 tokens
    full_text = SHORT_CORPUS + "\n\nQuestion: When was the lab founded?\nAnswer: The lab was founded in 2019."
    tokens = tokenizer.encode(full_text)
    full_cache = [KVCache() for _ in range(num_layers)]
    y = mx.array([tokens])
    _ = model(y[:, :-1], cache=full_cache)

    log(f"✓ Full cache generated: {full_cache[0].offset} tokens (target: 307-400)")

    # Step 3: Fit AM parameters for each layer
    log("\n" + "=" * 70)
    log("Step 3: Fit AM Parameters (Offline)")
    log("=" * 70)

    calibration = {}

    for layer_idx in range(num_layers):
        log(f"Layer {layer_idx:2d}/{num_layers}: ", end='')

        params = fit_am_layer(
            layer_idx,
            queries=queries_per_layer[layer_idx],
            keys=full_cache[layer_idx].keys,
            compression_ratio=compression_ratio
        )

        calibration[layer_idx] = params

        log(f"budget={params['budget']}, "
            f"beta ∈ [{params['beta'].min():.2f}, {params['beta'].max():.2f}], "
            f"mean={params['beta'].mean():.2f}")

    return calibration, total_queries

def save_calibration(calibration, total_queries, model_name, compression_ratio, output_path):
    """Save calibration to file."""
    calibration_data = {
        'model_name': model_name,
        'compression_ratio': compression_ratio,
        'num_queries': total_queries,
        'num_layers': len(calibration),
        'calibration': calibration,
        'created_at': datetime.now().isoformat(),
        'version': '1.0'
    }

    # Convert MLX arrays to numpy for pickle
    calibration_np = {}
    for layer_idx, params in calibration.items():
        calibration_np[layer_idx] = {
            'Ck': np.array(params['Ck']),
            'beta': np.array(params['beta']),
            'selected_indices': np.array(params['selected_indices']),
            'compression_ratio': params['compression_ratio'],
            'budget': params['budget']
        }

    calibration_data['calibration'] = calibration_np

    with open(output_path, 'wb') as f:
        pickle.dump(calibration_data, f)

    log(f"✓ Calibration saved to: {output_path}")
    log(f"  File size: {Path(output_path).stat().st_size / 1024:.1f} KB")

def main():
    parser = argparse.ArgumentParser(description='AM Offline Calibration')
    parser.add_argument('--model', default='qwen3-8b', help='Model name')
    parser.add_argument('--model-path', default='/Volumes/toshiba/models/qwen3-8b-mlx',
                        help='Path to model')
    parser.add_argument('--ratio', type=float, default=2.0,
                        help='Compression ratio')
    parser.add_argument('--queries', type=int, default=12288,
                        help='Target number of queries')
    parser.add_argument('--output', default=None,
                        help='Output calibration file')
    args = parser.parse_args()

    log("=" * 70)
    log("🔬 AM Offline Calibration")
    log("=" * 70)
    log(f"Model: {args.model}")
    log(f"Compression ratio: {args.ratio}x")
    log(f"Target queries: {args.queries}")

    # Load model
    log("\nLoading model...")
    model, tokenizer = load(args.model_path)
    log(f"✓ Model loaded: {len(model.model.layers)} layers")

    # Calibrate
    calibration, total_queries = calibrate_offline(
        model, tokenizer, args.ratio, args.queries
    )

    # Save
    output_path = args.output or f"calibrations/am_calibration_{args.model}_{args.ratio}x.pkl"
    log("\n" + "=" * 70)
    log("Saving Calibration")
    log("=" * 70)
    save_calibration(calibration, total_queries, args.model, args.ratio, output_path)

    log("\n" + "=" * 70)
    log("✅ Offline Calibration Complete!")
    log("=" * 70)
    log(f"Model: {args.model}")
    log(f"Queries: {total_queries}")
    log(f"Layers: {len(calibration)}")
    log(f"Compression: {args.ratio}x")
    log(f"Output: {output_path}")
    log("\nNext step: Use this calibration file in online inference")
    log("=" * 70)

if __name__ == '__main__':
    main()
