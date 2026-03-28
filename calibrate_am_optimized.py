#!/usr/bin/env python3
"""
Optimized AM Calibration - Fast Version
- Uses diverse corpus (no repetition)
- Optimized attention computation
- Fewer queries for speed
"""

import sys
sys.path.insert(0, '/Users/lisihao/FlashMLX/mlx-lm-source')

import mlx.core as mx
from mlx_lm import load
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime

# Load diverse corpus
with open('/tmp/diverse_corpus.txt', 'r') as f:
    CALIBRATION_CORPUS = f.read()

print(f"[INFO] Corpus: {len(CALIBRATION_CORPUS)} chars (~{len(CALIBRATION_CORPUS)//4} tokens)")

# Use fewer questions for speed (10 instead of 36)
CALIBRATION_QUESTIONS = [
    "When was the Quantum Dynamics Research Lab founded?",
    "What was the breakthrough temperature achieved?",
    "How many teams replicated the results?",
    "What is the Thwaites Glacier nicknamed?",
    "When was the Transformer architecture introduced?",
    "When did James Webb launch?",
    "Who discovered CRISPR-Cas9?",
    "What was peak inflation in 2022?",
    "How many neurons are in the human brain?",
    "What is solar panel efficiency?",
]

def safe_beta(weights, layer_idx=None):
    """Safe beta computation"""
    weights_clipped = np.clip(weights, np.exp(-3), np.exp(3))
    beta = np.log(weights_clipped)
    beta = np.clip(beta, -3.0, 3.0)
    if layer_idx is not None and layer_idx >= 27:
        beta = np.zeros_like(beta)
    return beta

def generate_queries_fast(model, tokenizer, target_length):
    """Fast query generation - use FULL corpus directly"""
    from mlx_lm.models.cache import KVCache

    print(f"  Generating queries...")
    num_layers = len(model.model.layers)

    # Use FULL corpus for each question (no truncation/repetition)
    all_queries = []

    for i, question in enumerate(CALIBRATION_QUESTIONS):
        # Use full corpus + question
        prompt = f"{CALIBRATION_CORPUS}\n\nQuestion: {question}\nAnswer:"
        tokens = tokenizer.encode(prompt)

        # Truncate to target if needed
        if len(tokens) > target_length:
            tokens = tokens[:target_length]

        print(f"    Q{i+1}: {len(tokens)} tokens - {question[:40]}...")

        # Prefill
        cache = [KVCache() for _ in range(num_layers)]
        y = mx.array([tokens])
        logits = model(y[:, :-1], cache=cache)

        # Collect queries (use keys as proxy)
        layer_queries = []
        for layer_idx in range(num_layers):
            keys = cache[layer_idx].keys  # (B, n_heads, seq_len, head_dim)
            layer_queries.append(keys)

        all_queries.append(layer_queries)

    # Merge across questions
    merged_per_layer = []
    for layer_idx in range(num_layers):
        layer_data = [q[layer_idx] for q in all_queries]
        merged = mx.concatenate(layer_data, axis=2)  # Concat along seq_len
        merged_per_layer.append(merged)

    total = merged_per_layer[0].shape[2]
    print(f"  ✓ Total queries: {total}")
    return merged_per_layer

def generate_kv(model, tokenizer, target_length):
    """Generate KV cache"""
    from mlx_lm.models.cache import KVCache

    print(f"  Generating KV cache...")

    # Use full corpus
    corpus = CALIBRATION_CORPUS
    full_text = corpus + "\n\nQuestion: Summarize the key points.\nAnswer:"
    tokens = tokenizer.encode(full_text)

    if len(tokens) > target_length:
        tokens = tokens[:target_length]

    print(f"    Using {len(tokens)} tokens")

    cache = [KVCache() for _ in range(len(model.model.layers))]
    y = mx.array([tokens])
    logits = model(y[:, :-1], cache=cache)

    return cache, cache[0].offset

def fit_layer_optimized(layer_idx, queries, keys, compression_ratio):
    """Optimized layer fitting - average heads FIRST"""
    # queries: (B, n_heads, num_q, head_dim)
    # keys: (B, n_heads, seq_len, head_dim)

    B, n_heads, seq_len, head_dim = keys.shape
    _, _, num_q, _ = queries.shape

    budget = int(seq_len / compression_ratio)

    print(f"    Layer {layer_idx:2d}: {seq_len} → {budget} ({compression_ratio}x)", end='', flush=True)

    # Average across heads FIRST (faster)
    q_avg = mx.mean(queries, axis=1)[0]  # (num_q, head_dim)
    k_avg = mx.mean(keys, axis=1)[0]      # (seq_len, head_dim)

    # Compute attention
    scale = head_dim ** 0.5
    scores = (q_avg @ k_avg.T) / scale  # (num_q, seq_len)

    # Softmax
    max_s = mx.max(scores, axis=1, keepdims=True)
    exp_s = mx.exp(scores - max_s)
    sum_s = mx.sum(exp_s, axis=1, keepdims=True)
    attn = exp_s / sum_s

    # Average across queries
    key_scores = mx.mean(attn, axis=0)  # (seq_len,)

    # Select top-k
    sorted_idx = mx.argsort(-key_scores)
    selected_idx = sorted_idx[:budget]
    selected_idx = mx.sort(selected_idx)  # Chronological order

    # Compute beta (use attention sum as weight)
    weights = mx.sum(attn[:, selected_idx], axis=0)  # (budget,)
    weights_np = np.array(weights.tolist())
    beta = safe_beta(weights_np, layer_idx)

    print(" ✓")

    return {
        'selected_indices': selected_idx,
        'beta': beta,
        'compression_ratio': compression_ratio,
        'budget': budget
    }

def calibrate(model, tokenizer, target_length, compression_ratio):
    """Main calibration"""
    print(f"\n{'='*70}")
    print(f"Calibrating L{target_length}, R{compression_ratio}")
    print(f"{'='*70}")

    # Step 1: Queries
    print("Step 1: Generate queries...")
    queries_per_layer = generate_queries_fast(model, tokenizer, target_length)

    # Step 2: KV cache
    print("Step 2: Generate KV cache...")
    kv_cache, actual_len = generate_kv(model, tokenizer, target_length)
    print(f"  ✓ KV cache: {actual_len} tokens")

    # Step 3: Fit layers
    print("Step 3: Fit AM compression...")
    num_layers = len(model.model.layers)
    calibration_data = []

    for layer_idx in range(num_layers):
        params = fit_layer_optimized(
            layer_idx,
            queries_per_layer[layer_idx],
            kv_cache[layer_idx].keys,
            compression_ratio
        )
        calibration_data.append(params)

    # Build file
    calib_file = {
        'metadata': {
            'calibration_length': actual_len,
            'target_length': target_length,
            'compression_ratio': compression_ratio,
            'budget': int(actual_len / compression_ratio),
            'num_queries': queries_per_layer[0].shape[2],
            'compression_scope': 'prefix_only',
            'compatible_runtime_mode': 'triple_layer',
            'recent_window_size': 512,
            'beta_safe_guard': True,
            'deep_layer_threshold': 27,
            'created_at': datetime.now().isoformat(),
            'version': '2.1-optimized'
        },
        'calibration': calibration_data
    }

    return calib_file

# Main
if __name__ == '__main__':
    print("="*80)
    print("🔧 Optimized AM Calibration (Diverse Corpus)")
    print("="*80)

    print("\nLoading model...")
    model, tokenizer = load('/Volumes/toshiba/models/qwen3-8b-mlx')
    print(f"✓ Loaded: {len(model.model.layers)} layers")

    # Calibrate
    target_length = 2729
    compression_ratio = 2.0

    calib_file = calibrate(model, tokenizer, target_length, compression_ratio)

    # Save
    output_dir = Path('/tmp/am_calibrations_fixed')
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f'am_calibration_L{target_length}_R{compression_ratio}.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(calib_file, f)

    print(f"\n✓ Saved: {output_file}")
    print("="*80)
