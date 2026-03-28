#!/usr/bin/env python3
"""
AM Calibration - FIXED VERSION
- Uses diverse long corpus (NO repetition)
- Better query generation
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

print(f"Corpus length: {len(CALIBRATION_CORPUS)} chars (~{len(CALIBRATION_CORPUS)//4} tokens)")

CALIBRATION_QUESTIONS = [
    # Quantum Computing
    "When was the Quantum Dynamics Research Lab founded?",
    "What was the breakthrough temperature achieved?",
    "How many teams replicated the results?",

    # Climate Science
    "What is the Thwaites Glacier nicknamed?",
    "How much ice does it lose annually?",
    "What are the three climate scenarios?",

    # Neural Networks
    "When was the Transformer architecture introduced?",
    "How many parameters does GPT-3 have?",
    "What is Constitutional AI?",

    # Space
    "When did James Webb launch?",
    "How far is the L2 Lagrange point?",
    "What exoplanet was analyzed?",

    # Biotech
    "Who discovered CRISPR-Cas9?",
    "When was the first CRISPR therapy approved?",
    "How much does genome sequencing cost in 2023?",

    # Economics
    "What was peak inflation in 2022?",
    "What is Modern Monetary Theory?",
    "What was Bitcoin's peak market cap?",

    # Neuroscience
    "How many neurons are in the human brain?",
    "What does fMRI reveal?",
    "How does deep brain stimulation work?",

    # Energy
    "What is solar panel efficiency?",
    "How tall are modern wind turbines?",
    "What is the green hydrogen cost target?",

    # History
    "When was the Library of Alexandria founded?",
    "Who calculated Earth's circumference?",
    "When did Mount Vesuvius erupt?",

    # Philosophy
    "What is Sartre's famous phrase?",
    "What is the Chinese Room experiment?",
    "What is the simulation hypothesis?",

    # Cybersecurity
    "What is RSA encryption based on?",
    "What did Stuxnet attack?",
    "What was the average ransomware payment in 2022?",

    # Linguistics
    "How many living languages exist?",
    "How many Mandarin characters for basic literacy?",
    "Who proposed Universal Grammar?",
]

def generate_kv_for_length(model, tokenizer, target_length):
    """Generate KV cache WITHOUT repetition"""
    from mlx_lm.models.cache import KVCache

    # Use corpus directly (no repetition!)
    chars_needed = target_length * 4

    if chars_needed <= len(CALIBRATION_CORPUS):
        corpus = CALIBRATION_CORPUS[:chars_needed]
    else:
        # If still not enough, use full corpus + padding
        corpus = CALIBRATION_CORPUS
        print(f"  Warning: corpus too short ({len(CALIBRATION_CORPUS)} < {chars_needed}), using full corpus")

    full_text = corpus + "\n\nQuestion: Summarize the key breakthroughs.\nAnswer:"
    tokens = tokenizer.encode(full_text)

    if len(tokens) > target_length:
        tokens = tokens[:target_length]

    print(f"  Using {len(tokens)} tokens (target: {target_length})")

    # Generate KV cache
    cache = [mx.zeros((1, 0, 0, 0))] * len(model.model.layers)  # Dummy init
    from mlx_lm.models.cache import KVCache
    cache = [KVCache() for _ in range(len(model.model.layers))]

    y = mx.array([tokens])
    logits = model(y[:, :-1], cache=cache)

    return cache, cache[0].offset

def generate_queries_for_length(model, tokenizer, target_length):
    """
    Generate queries using DIVERSE corpus (no repetition!)
    """
    from mlx_lm.models.cache import KVCache

    print(f"  Generating queries for length {target_length}...")

    num_layers = len(model.model.layers)
    all_queries_per_layer = [[] for _ in range(num_layers)]

    # Use ALL questions (not just 10)
    questions_to_use = CALIBRATION_QUESTIONS

    for i, question in enumerate(questions_to_use):
        # Build prompt with diverse sections
        chars_needed = target_length * 4
        if chars_needed <= len(CALIBRATION_CORPUS):
            corpus = CALIBRATION_CORPUS[:chars_needed]
        else:
            corpus = CALIBRATION_CORPUS  # Use full corpus

        prompt = f"{corpus}\n\nQuestion: {question}\nAnswer:"
        prompt_tokens = tokenizer.encode(prompt)

        # Truncate if too long
        if len(prompt_tokens) > target_length:
            prompt_tokens = prompt_tokens[:target_length]

        # Create cache
        cache = [KVCache() for _ in range(num_layers)]

        # Prefill only
        y = mx.array([prompt_tokens])
        logits = model(y[:, :-1], cache=cache)

        # Extract queries (keys)
        for layer_idx in range(num_layers):
            keys = cache[layer_idx].keys
            all_queries_per_layer[layer_idx].append(keys)

    # Merge all queries
    merged_queries_per_layer = []
    for layer_idx in range(num_layers):
        merged = mx.concatenate(all_queries_per_layer[layer_idx], axis=2)
        merged_queries_per_layer.append(merged)

    total_queries = merged_queries_per_layer[0].shape[2]
    print(f"    ✓ Collected {total_queries} queries from {len(questions_to_use)} questions")

    return merged_queries_per_layer, total_queries

def safe_beta(weights, layer_idx=None):
    """Runtime-safe beta computation"""
    weights_clipped = np.clip(weights, np.exp(-3), np.exp(3))
    beta = np.log(weights_clipped)
    beta = np.clip(beta, -3.0, 3.0)

    # Deep layer fallback
    if layer_idx is not None and layer_idx >= 27:
        beta = np.zeros_like(beta)

    return beta

def fit_am_layer(layer_idx, queries, keys, compression_ratio):
    """Fit AM compression for single layer"""
    seq_len = keys.shape[2]
    budget = int(seq_len / compression_ratio)

    print(f"    Layer {layer_idx}: {seq_len} → {budget} tokens ({compression_ratio}x)")

    # Reshape for attention computation
    # queries: (B, n_heads, num_queries, head_dim)
    # keys: (B, n_heads, seq_len, head_dim)
    B, n_heads, num_queries, head_dim = queries.shape
    _, _, seq_len, _ = keys.shape

    # Compute attention scores per head, then average
    all_key_scores = []
    for h in range(n_heads):
        q_h = queries[:, h, :, :]  # (B, num_queries, head_dim)
        k_h = keys[:, h, :, :]      # (B, seq_len, head_dim)

        # Remove batch dim (B=1)
        q_h = q_h[0]  # (num_queries, head_dim)
        k_h = k_h[0]  # (seq_len, head_dim)

        # Compute attention: (num_queries, seq_len)
        scale = head_dim ** 0.5
        scores = (q_h @ k_h.T) / scale

        # Softmax normalization
        max_scores = mx.max(scores, axis=1, keepdims=True)
        exp_scores = mx.exp(scores - max_scores)
        sum_exp = mx.sum(exp_scores, axis=1, keepdims=True)
        attention_weights = exp_scores / sum_exp

        # Average across queries
        key_scores_h = mx.mean(attention_weights, axis=0)  # (seq_len,)
        all_key_scores.append(key_scores_h)

    # Average across heads
    key_scores = mx.mean(mx.stack(all_key_scores, axis=0), axis=0)  # (seq_len,)

    # Select top-k
    sorted_indices = mx.argsort(-key_scores)
    selected_indices = sorted_indices[:budget]

    # Keep chronological order (for RoPE)
    selected_indices = mx.sort(selected_indices)

    # Fit beta (use attention scores as weights)
    # Average attention across all heads
    selected_scores = []
    for h in range(n_heads):
        q_h = queries[:, h, :, :][0]  # (num_queries, head_dim)
        k_h = keys[:, h, :, :][0]      # (seq_len, head_dim)
        k_selected = k_h[selected_indices, :]  # (budget, head_dim)

        scale = head_dim ** 0.5
        scores_h = (q_h @ k_selected.T) / scale  # (num_queries, budget)

        # Softmax
        max_s = mx.max(scores_h, axis=1, keepdims=True)
        exp_s = mx.exp(scores_h - max_s)
        sum_s = mx.sum(exp_s, axis=1, keepdims=True)
        attn_h = exp_s / sum_s

        # Sum across queries (importance score)
        weights_h = mx.sum(attn_h, axis=0)  # (budget,)
        selected_scores.append(weights_h)

    # Average across heads
    weights_raw = mx.mean(mx.stack(selected_scores, axis=0), axis=0)  # (budget,)
    weights_np = np.array(weights_raw.tolist())  # Convert via tolist() to avoid dtype issues

    beta_values = safe_beta(weights_np, layer_idx=layer_idx)

    return {
        'selected_indices': selected_indices,
        'beta': beta_values,
        'compression_ratio': compression_ratio,
        'budget': budget
    }

def calibrate_for_length(model, tokenizer, target_length, compression_ratio):
    """Main calibration function"""
    print(f"\nCalibrating for L{target_length}, R{compression_ratio}")
    print("=" * 70)

    # Step 1: Generate queries
    print("Step 1: Generate queries...")
    queries_per_layer, num_queries = generate_queries_for_length(
        model, tokenizer, target_length
    )

    # Step 2: Generate KV cache
    print("Step 2: Generate KV cache...")
    kv_cache, actual_len = generate_kv_for_length(model, tokenizer, target_length)
    print(f"  ✓ Generated KV cache: {actual_len} tokens")

    # Step 3: Fit AM for each layer
    print("Step 3: Fit AM compression...")
    num_layers = len(model.model.layers)
    calibration_data = []

    for layer_idx in range(num_layers):
        params = fit_am_layer(
            layer_idx,
            queries=queries_per_layer[layer_idx],
            keys=kv_cache[layer_idx].keys,
            compression_ratio=compression_ratio
        )
        calibration_data.append(params)

    # Build calibration file
    calibration_file = {
        'metadata': {
            'calibration_length': actual_len,
            'target_length': target_length,
            'compression_ratio': compression_ratio,
            'budget': int(actual_len / compression_ratio),
            'num_queries': num_queries,
            'compression_scope': 'prefix_only',
            'compatible_runtime_mode': 'double_layer',
            'recent_window_size': 256,
            'beta_safe_guard': True,
            'deep_layer_threshold': 27,
            'created_at': datetime.now().isoformat(),
            'version': '2.0'
        },
        'calibration': calibration_data
    }

    return calibration_file

# Main
if __name__ == '__main__':
    print("=" * 80)
    print("🔧 AM Calibration - FIXED (No Repetition)")
    print("=" * 80)

    # Load model
    print("\nLoading model...")
    model_path = '/Volumes/toshiba/models/qwen3-8b-mlx'
    model, tokenizer = load(model_path)
    print(f"✓ Loaded: {len(model.model.layers)} layers")

    # Generate R2.0 calibration for L2729
    target_length = 2729
    compression_ratio = 2.0

    calib_file = calibrate_for_length(model, tokenizer, target_length, compression_ratio)

    # Save
    output_dir = Path('/tmp/am_calibrations_fixed')
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f'am_calibration_L{target_length}_R{compression_ratio}.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(calib_file, f)

    print(f"\n✓ Saved: {output_file}")
    print("=" * 80)
