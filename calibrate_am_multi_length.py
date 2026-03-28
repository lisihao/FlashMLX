#!/usr/bin/env python3
"""
AM Multi-Length Calibration Generator

Generates calibration files for multiple prefix lengths (256, 512, 768, 1K, 1.5K, 2K, etc.)
with Beta Safe Guard and metadata versioning.

Usage:
    python calibrate_am_multi_length.py --ratio 2.0 --lengths 256,512,768,1024,1536,2048
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

# Inline safe_beta function (to avoid import conflicts)
def safe_beta(
    weights,
    layer_idx=None,
    w_min=np.exp(-3),
    w_max=np.exp(3),
    beta_min=-3.0,
    beta_max=3.0,
    deep_layer_threshold=27,
    enable_deep_layer_fallback=True
):
    """Runtime-safe beta computation for AM compression."""
    # 1. Clip weights to safe range
    weights_clipped = np.clip(weights, w_min, w_max)

    # 2. Safe log computation
    beta = np.log(weights_clipped)

    # 3. Hard bounds
    beta = np.clip(beta, beta_min, beta_max)

    # 4. Deep layer fallback
    if (enable_deep_layer_fallback and
        layer_idx is not None and
        layer_idx >= deep_layer_threshold):
        beta = np.zeros_like(beta)

    # 5. NaN/Inf check
    if np.any(np.isnan(beta)) or np.any(np.isinf(beta)):
        beta = np.zeros_like(beta)

    return beta

def validate_beta(beta):
    """Validate beta values are safe."""
    # Check for NaN/Inf
    if np.any(np.isnan(beta)) or np.any(np.isinf(beta)):
        return False

    # Check range
    if np.any(beta < -10) or np.any(beta > 10):
        return False

    return True

def log(msg, end='\n'):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True, end=end)

# Calibration corpus (same as original)
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

def generate_kv_for_length(model, tokenizer, target_length):
    """
    Generate KV cache of target length by truncating/padding corpus.

    Returns: (cache, actual_length)
    """
    from mlx_lm.models.cache import KVCache

    # Estimate corpus length needed (rough: 4 chars per token)
    chars_needed = target_length * 4

    # Truncate or repeat corpus to match target
    if chars_needed <= len(CALIBRATION_CORPUS):
        corpus = CALIBRATION_CORPUS[:chars_needed]
    else:
        # Repeat corpus if needed
        repeat_times = (chars_needed // len(CALIBRATION_CORPUS)) + 1
        corpus = (CALIBRATION_CORPUS * repeat_times)[:chars_needed]

    # Add a question to make it realistic
    full_text = corpus + "\n\nQuestion: When was the lab founded?\nAnswer: The lab was founded in 2019."
    tokens = tokenizer.encode(full_text)

    # Truncate to exact target length
    if len(tokens) > target_length:
        tokens = tokens[:target_length]

    # Generate KV cache
    num_layers = len(model.model.layers)
    cache = [KVCache() for _ in range(num_layers)]
    y = mx.array([tokens])
    _ = model(y[:, :-1], cache=cache)

    actual_length = cache[0].offset

    return cache, actual_length

def generate_queries_for_length(model, tokenizer, target_length):
    """
    Generate queries using same corpus length as target.

    Returns: (queries_per_layer, num_queries)
    """
    from mlx_lm.models.cache import KVCache

    log(f"  Generating queries for length {target_length}...")

    num_layers = len(model.model.layers)
    all_queries_per_layer = [[] for _ in range(num_layers)]

    # Use first 10 questions for calibration
    questions_to_use = CALIBRATION_QUESTIONS[:10]

    total_queries_collected = 0

    for i, question in enumerate(questions_to_use):
        # Build prompt
        chars_needed = target_length * 4
        if chars_needed <= len(CALIBRATION_CORPUS):
            corpus = CALIBRATION_CORPUS[:chars_needed]
        else:
            repeat_times = (chars_needed // len(CALIBRATION_CORPUS)) + 1
            corpus = (CALIBRATION_CORPUS * repeat_times)[:chars_needed]

        prompt = f"{corpus}\n\nQuestion: {question}\nAnswer:"
        prompt_tokens = tokenizer.encode(prompt)

        # Truncate if too long
        if len(prompt_tokens) > target_length:
            prompt_tokens = prompt_tokens[:target_length]

        # Create cache
        cache = [KVCache() for _ in range(num_layers)]

        # Prefill only (no decode to keep length consistent)
        y = mx.array([prompt_tokens])
        logits = model(y[:, :-1], cache=cache)

        # Extract queries
        for layer_idx in range(num_layers):
            keys = cache[layer_idx].keys
            all_queries_per_layer[layer_idx].append(keys)

        total_queries_collected += cache[0].offset

    # Merge all queries
    merged_queries_per_layer = []
    for layer_idx in range(num_layers):
        merged = mx.concatenate(all_queries_per_layer[layer_idx], axis=2)
        merged_queries_per_layer.append(merged)

    total_queries = merged_queries_per_layer[0].shape[2]
    log(f"    ✓ Collected {total_queries} queries")

    return merged_queries_per_layer, total_queries

def fit_am_layer_with_safe_beta(layer_idx, queries, keys, compression_ratio):
    """
    Fit AM compression parameters with Beta Safe Guard.

    Returns: {
        'selected_indices': indices of selected keys,
        'beta': safe beta weights,
        'Cv': coverage metrics,
        'C2': quality metrics
    }
    """
    import scipy.optimize

    # Convert to numpy for scipy
    queries_mlx = queries[0, 0, :, :]
    keys_mlx = keys[0, 0, :, :]

    # Convert to float32 first to avoid bfloat16 issues
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

    # Apply softmax
    scores = scipy.special.softmax(raw_scores, axis=1)  # (num_queries, seq_len)

    avg_scores = np.mean(scores, axis=0)  # (seq_len,)

    # Select top budget keys
    selected_indices = np.argsort(avg_scores)[-budget:]
    selected_indices = np.sort(selected_indices)  # Keep order

    # Fit beta using bounded least-squares
    R_S = scores[:, selected_indices]  # (num_queries, budget)
    target = np.sum(scores, axis=1)   # (num_queries,) - should be all 1.0

    # Verify target
    assert np.abs(target.mean() - 1.0) < 0.01, f"Target mean {target.mean()} != 1.0"

    # Solve: R_S @ beta ≈ target with bounds [0, 2]
    res = scipy.optimize.lsq_linear(
        R_S,
        target,
        bounds=(0, 2),
        method='bvls'
    )
    beta_raw = res.x

    # ✅ Apply Beta Safe Guard
    beta_safe = safe_beta(
        weights=beta_raw,
        layer_idx=layer_idx,
        w_min=np.exp(-3),
        w_max=np.exp(3),
        beta_min=-3.0,
        beta_max=3.0,
        deep_layer_threshold=27,
        enable_deep_layer_fallback=True
    )

    # Validate beta
    if not validate_beta(beta_safe):
        log(f"    ⚠️ Layer {layer_idx}: Beta validation failed, using zeros")
        beta_safe = np.zeros_like(beta_raw)

    # Compute quality metrics
    Cv = np.sum(avg_scores[selected_indices]) / np.sum(avg_scores)  # Coverage

    # C2: Reconstruction quality (L2 norm)
    # Approximate: scores ≈ scores[:, selected] @ diag(beta)
    reconstructed = R_S @ beta_safe
    C2 = 1.0 - np.mean((target - reconstructed) ** 2) / np.var(target)

    return {
        'layer_idx': layer_idx,
        'selected_indices': selected_indices.astype(np.int32),  # numpy array
        'beta': beta_safe,  # numpy array
        'Cv': float(Cv),
        'C2': float(C2)
    }

def calibrate_single_length(model, tokenizer, target_length, compression_ratio):
    """
    Calibrate AM for a single target length.

    Returns: (calibration_data, actual_length, metadata)
    """
    num_layers = len(model.model.layers)

    log(f"\n{'='*70}")
    log(f"Calibrating for length: {target_length}")
    log(f"{'='*70}")

    # Step 1: Generate queries
    log("Step 1: Generate queries...")
    queries_per_layer, num_queries = generate_queries_for_length(
        model, tokenizer, target_length
    )
    log(f"  ✓ Generated {num_queries} queries")

    # Step 2: Generate KV cache
    log("Step 2: Generate KV cache...")
    kv_cache, actual_length = generate_kv_for_length(
        model, tokenizer, target_length
    )
    log(f"  ✓ KV cache length: {actual_length} (target: {target_length})")

    # Step 3: Fit AM parameters for each layer
    log("Step 3: Fit AM parameters with Beta Safe Guard...")
    calibration_data = []

    for layer_idx in range(num_layers):
        params = fit_am_layer_with_safe_beta(
            layer_idx,
            queries=queries_per_layer[layer_idx],
            keys=kv_cache[layer_idx].keys,
            compression_ratio=compression_ratio
        )

        calibration_data.append(params)

        log(f"  Layer {layer_idx:2d}/{num_layers}: "
            f"budget={len(params['selected_indices'])}, "
            f"beta ∈ [{params['beta'].min():.2f}, {params['beta'].max():.2f}], "
            f"Cv={params['Cv']:.3f}, C2={params['C2']:.3f}")

    # Create metadata
    metadata = {
        'calibration_length': actual_length,
        'target_length': target_length,
        'compression_ratio': compression_ratio,
        'budget': int(actual_length / compression_ratio),
        'num_queries': num_queries,
        'compression_scope': 'prefix_only',
        'compatible_runtime_mode': 'double_layer',
        'recent_window_size': 256,
        'beta_safe_guard': True,
        'deep_layer_threshold': 27,
        'created_at': datetime.now().isoformat(),
        'version': '2.0'
    }

    return calibration_data, actual_length, metadata

def save_calibration(calibration_data, metadata, output_dir):
    """
    Save calibration to file with standardized naming.

    Filename format: am_calibration_L{length}_R{ratio}.pkl
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    length = metadata['calibration_length']
    ratio = metadata['compression_ratio']

    filename = f"am_calibration_L{length}_R{ratio:.1f}.pkl"
    filepath = output_path / filename

    # Create final structure
    calibration_file = {
        'metadata': metadata,
        'calibration': calibration_data  # Already numpy arrays
    }

    with open(filepath, 'wb') as f:
        pickle.dump(calibration_file, f)

    file_size_kb = filepath.stat().st_size / 1024
    log(f"  ✓ Saved: {filename} ({file_size_kb:.1f} KB)")

    return filepath

def main():
    parser = argparse.ArgumentParser(description='AM Multi-Length Calibration')
    parser.add_argument('--model-path', default='/Volumes/toshiba/models/qwen3-8b-mlx',
                        help='Path to model')
    parser.add_argument('--ratio', type=float, default=2.0,
                        help='Compression ratio')
    parser.add_argument('--lengths', default='256,512,768,1024,1536,2048',
                        help='Comma-separated target lengths')
    parser.add_argument('--output-dir', default='/tmp/am_calibrations_multi_length',
                        help='Output directory')
    args = parser.parse_args()

    # Parse lengths
    lengths = [int(x.strip()) for x in args.lengths.split(',')]

    log("=" * 70)
    log("🔬 AM Multi-Length Calibration Generator")
    log("=" * 70)
    log(f"Model: {args.model_path}")
    log(f"Compression ratio: {args.ratio}x")
    log(f"Target lengths: {lengths}")
    log(f"Output directory: {args.output_dir}")

    # Load model
    log("\nLoading model...")
    model, tokenizer = load(args.model_path)
    num_layers = len(model.model.layers)
    log(f"✓ Model loaded: {num_layers} layers")

    # Calibrate for each length
    results = []

    for length in lengths:
        calibration_data, actual_length, metadata = calibrate_single_length(
            model, tokenizer, length, args.ratio
        )

        filepath = save_calibration(calibration_data, metadata, args.output_dir)

        results.append({
            'target': length,
            'actual': actual_length,
            'filepath': filepath
        })

    # Summary
    log("\n" + "=" * 70)
    log("✅ Multi-Length Calibration Complete!")
    log("=" * 70)
    log(f"Generated {len(results)} calibration files:")
    for r in results:
        log(f"  - Target {r['target']:5d} → Actual {r['actual']:5d}: {r['filepath'].name}")
    log(f"\nOutput directory: {args.output_dir}")
    log("\nNext step: Use CalibrationRegistry in DoubleLayerKVCache")
    log("=" * 70)

if __name__ == '__main__':
    main()
