#!/usr/bin/env python3
"""
AM On-policy Calibration（增量式）

不重复压前层，逐步推进：
  Phase 1: 复用 15.8K queries → 前 18 层（已验证）
  Phase 2: 从已压缩模型提取 queries → 第 18-26 层
  Phase 3: 继续提取 queries → 第 27-35 层

Usage:
    python calibrate_am_onpolicy.py --model qwen3-8b --ratio 2.0
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

# Calibration corpus (same as offline)
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
    "When was the Quantum Dynamics Research Lab founded?",
    "Who founded the research lab?",
    "How much funding did the lab receive initially?",
    "What was the lab's main research goal?",
    "Who did Dr. Chen recruit for her team?",
    "Where did Dr. Robert Kim come from?",
    "What happened in 2020?",
    "How long did quantum coherence last in early tests?",
    "What improvements were made in 2021?",
    "When did the breakthrough occur?",
    "What time did the breakthrough happen?",
    "How many seconds of coherence was achieved in the breakthrough?",
    "How many experiments were run during the breakthrough?",
    "What was the success rate of the breakthrough experiments?",
    "How many teams replicated the results?",
    "What was the replication success rate?",
    "Who criticized the results?",
    "Where was Professor Blackwell from?",
    "What did Blackwell later admit?",
    "Who received the Nobel Prize?",
    "When did Dr. Chen receive the Nobel Prize?",
    "Summarize the timeline of the quantum computing research.",
    "Describe the breakthrough that occurred in July 2022.",
    "Explain the significance of room-temperature quantum computing.",
]

def extract_queries_from_model(model, tokenizer, target_layers, compressed_layers_calibration, num_queries_target=8192):
    """
    从已部分压缩的模型提取 queries

    关键：
    1. 前面已压缩的层使用 CompactedKVCache（加载 calibration）
    2. 目标层使用普通 KVCache（收集 queries）
    3. 后面未处理的层使用普通 KVCache
    """
    from mlx_lm.models.cache import ArraysCache, KVCache
    from mlx_lm.models.compacted_cache import CompactedKVCache

    log(f"Extracting queries from partially compressed model (target: {num_queries_target})...")
    log(f"  Compressed layers: 0-{len(compressed_layers_calibration)-1}")
    log(f"  Target layers: {target_layers[0]}-{target_layers[-1]}")

    # Use SHORT corpus for query generation
    SHORT_CORPUS = CALIBRATION_CORPUS[:2000]  # ~400 tokens

    # Create temp calibration file with proper format
    # CompactedKVCache expects the NEW format with metadata wrapper
    import tempfile
    temp_calib_data = {
        'model_name': 'temp',
        'compression_ratio': 2.0,
        'num_queries': 0,
        'num_layers': len(compressed_layers_calibration),
        'calibration': compressed_layers_calibration,  # The actual per-layer data
        'created_at': 'temp',
        'version': '1.0'
    }
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
        pickle.dump(temp_calib_data, f)
        temp_calib_file = f.name

    all_queries = []
    total_queries_collected = 0

    for q_idx, question in enumerate(CALIBRATION_QUESTIONS):
        if total_queries_collected >= num_queries_target:
            break

        prompt = f"{SHORT_CORPUS}\n\nQuestion: {question}\nAnswer:"
        prompt_tokens = tokenizer.encode(prompt)

        log(f"  [{q_idx+1}/{len(CALIBRATION_QUESTIONS)}] {question[:50]}...")

        # Create cache with mixed compression state
        cache = ArraysCache(size=len(model.model.layers))

        # Already compressed layers: use CompactedKVCache with calibration
        for layer_idx in range(len(compressed_layers_calibration)):
            cache[layer_idx] = CompactedKVCache(
                max_size=1000,  # Large enough to not trigger runtime compression
                enable_compression=False,  # We'll manually trigger compression
                compression_ratio=2.0,
                use_quality_path=True,
                quality_fit_beta=True,
                quality_fit_c2=True,
                calibration_file=temp_calib_file,
                layer_idx=layer_idx
            )

        # Target layers and beyond: use regular KVCache
        for layer_idx in range(len(compressed_layers_calibration), len(model.model.layers)):
            cache[layer_idx] = KVCache()

        # Prefill to generate KV cache
        y = mx.array([prompt_tokens])
        logits = model(y[:, :-1], cache=cache)

        # Manually trigger compression for already-compressed layers
        for layer_idx in range(len(compressed_layers_calibration)):
            if cache[layer_idx].offset > 100:  # If there's enough content
                cache[layer_idx].compact()  # Apply pre-fitted compression

        # Decode tokens
        y = mx.array([[prompt_tokens[-1]]])
        for _ in range(50):  # Max 50 tokens
            logits = model(y, cache=cache)
            y = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
            token_id = y[0, 0].item()
            if token_id == tokenizer.eos_token_id:
                break

        # Extract queries from target layers
        layer_queries = []
        for layer_idx in target_layers:
            if cache[layer_idx] is None:
                continue

            # Extract keys as queries (Q ≈ K assumption)
            keys = cache[layer_idx].keys[..., :cache[layer_idx].offset, :]
            layer_queries.append(keys)

        if layer_queries:
            # Merge across layers and concatenate
            merged = mx.concatenate(layer_queries, axis=2)  # (B, n_heads, queries, head_dim)
            all_queries.append(merged)
            total_queries_collected += merged.shape[2]
            log(f"      Collected {merged.shape[2]} queries (total: {total_queries_collected})")

    # Clean up temp file
    Path(temp_calib_file).unlink()

    # Merge all questions
    if not all_queries:
        raise RuntimeError("No queries collected!")

    queries = mx.concatenate(all_queries, axis=2)
    log(f"✓ Extracted {queries.shape[2]} queries from partially compressed model")

    return queries

def fit_am_layer(layer_idx, keys, values, queries, compression_ratio=2.0):
    """拟合单层 AM 参数 (完整实现，参考 calibrate_am_offline.py)"""
    import scipy.optimize

    B, n_heads, seq_len, head_dim = keys.shape
    budget = int(seq_len / compression_ratio)

    # Convert MLX arrays to numpy using astype(float32) first to avoid dtype issues
    keys_f32 = keys.astype(mx.float32)
    queries_f32 = queries.astype(mx.float32)

    # Convert to numpy for processing
    keys_np = np.array(keys_f32).reshape(-1, seq_len, head_dim)  # (B*n_heads, seq_len, head_dim)
    queries_np = np.array(queries_f32).reshape(-1, queries.shape[2], head_dim)  # (B*n_heads, num_queries, head_dim)

    # Average across batch and heads
    keys_avg = np.mean(keys_np, axis=0)  # (seq_len, head_dim)
    queries_avg = np.mean(queries_np, axis=0)  # (num_queries, head_dim)

    # Compute attention scores: Q @ K^T
    scores = queries_avg @ keys_avg.T  # (num_queries, seq_len)
    scores = scores / np.sqrt(head_dim)  # Scale

    # Apply softmax
    scores_exp = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    scores = scores_exp / np.sum(scores_exp, axis=1, keepdims=True)

    # Average attention scores across queries
    avg_scores = np.mean(scores, axis=0)  # (seq_len,)

    # Select top budget keys
    selected_indices = np.argsort(avg_scores)[-budget:]
    selected_indices = np.sort(selected_indices)  # Keep order

    Ck = keys_avg[selected_indices]  # (budget, head_dim)

    # Fit beta using bounded least-squares
    R_S = scores[:, selected_indices]  # (num_queries, budget)
    target = np.mean(scores, axis=1)   # (num_queries,) - mean attention

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

def apply_layer_compression(model, layer_idx, calibration):
    """
    应用单层压缩（增量式）
    
    Note: 这是概念性的 - 实际需要在 model 的 cache 中设置 calibration
    """
    log(f"  Applying compression to layer {layer_idx}")
    # In practice, this would set the calibration in the model's cache
    # For now, we just return the calibration to be saved
    return calibration

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='qwen3-8b')
    parser.add_argument('--ratio', type=float, default=2.0)
    parser.add_argument('--phase', type=str, default='all', choices=['all', '1', '2', '3'])
    args = parser.parse_args()
    
    log("=" * 70)
    log("🔬 AM On-policy Calibration (Incremental)")
    log("=" * 70)
    log(f"Model: {args.model}")
    log(f"Compression ratio: {args.ratio}x")
    log(f"Phase: {args.phase}")
    log("")
    
    # Load model
    model_path = f"/Volumes/toshiba/models/{args.model}-mlx"
    log(f"Loading model: {model_path}")
    model, tokenizer = load(model_path)
    num_layers = len(model.model.layers)
    log(f"✓ Model loaded: {num_layers} layers")
    
    # Load existing calibration for Phase 1 (layers 0-17)
    # Priority: on-policy file (if running Phase 3) > offline file
    onpolicy_calibration_file = f"am_calibration_{args.model}_{args.ratio}x_onpolicy.pkl"
    offline_calibration_file = f"/tmp/am_calibration_{args.model}_{args.ratio}x.pkl"

    # Choose calibration file based on phase and availability
    if args.phase == '3' and Path(onpolicy_calibration_file).exists():
        # Phase 3: prioritize on-policy file (contains Phase 2 results)
        existing_calibration_file = onpolicy_calibration_file
        log(f"✓ Found on-policy calibration: {existing_calibration_file}")
    elif Path(offline_calibration_file).exists():
        # Phase 2 or Phase 1: use offline file
        existing_calibration_file = offline_calibration_file
        log(f"✓ Found existing calibration: {existing_calibration_file}")
    else:
        log(f"✗ No calibration file found")
        log("  Please run calibrate_am_offline.py first")
        return

    with open(existing_calibration_file, 'rb') as f:
        calib_data = pickle.load(f)

    # Extract actual per-layer calibration (handle both old and new format)
    if isinstance(calib_data, dict) and 'calibration' in calib_data:
        phase1_calibration = calib_data['calibration']
        log(f"  Loaded calibration for {len(phase1_calibration)} layers (new format)")
    else:
        phase1_calibration = calib_data
        log(f"  Loaded calibration for {len(phase1_calibration)} layers (old format)")
    
    # Initialize full calibration dict
    full_calibration = {}

    # ================================================================
    # Phase 1: Load existing calibration (0-17 or 0-26 depending on file)
    # ================================================================
    # Note: Always load existing layers, as Phase 2/3 depend on it
    num_existing_layers = len(phase1_calibration)

    # Determine how many layers to load
    if args.phase == '3':
        # Phase 3: load all available layers (should be 27 from Phase 2)
        layers_to_load = min(num_existing_layers, 27)
        log("\n" + "=" * 70)
        log(f"Loading existing calibration: Layers 0-{layers_to_load-1}")
        log("=" * 70)
    else:
        # Phase 1/2: only load Phase 1 (0-17)
        layers_to_load = 18
        log("\n" + "=" * 70)
        log("Phase 1: Layers 0-17 (Existing Calibration)")
        log("=" * 70)

    for layer_idx in range(layers_to_load):
        full_calibration[layer_idx] = phase1_calibration[layer_idx]
        if args.phase in ['all', '1'] or (args.phase == '3' and layer_idx < 5):
            # Show first few layers when loading
            log(f"Layer {layer_idx:2d}/36: Using existing calibration (budget={phase1_calibration[layer_idx]['budget']})")
        elif args.phase == '3' and layer_idx == layers_to_load - 1:
            # Show last layer for Phase 3
            log(f"... (loaded {layers_to_load} layers total)")
        else:
            # Silent load for other cases
            pass
    
    # ================================================================
    # Phase 2: Layers 18-26 (Extract from compressed model)
    # ================================================================
    if args.phase in ['all', '2']:
        log("\n" + "=" * 70)
        log("Phase 2: Layers 18-26 (On-policy)")
        log("=" * 70)

        # Extract queries from model with layers 0-17 compressed
        target_layers = list(range(18, 27))
        compressed_layers = {i: full_calibration[i] for i in range(18)}
        queries = extract_queries_from_model(
            model, tokenizer, target_layers,
            compressed_layers_calibration=compressed_layers,
            num_queries_target=8192
        )
        
        # Generate full KV cache for fitting
        SHORT_CORPUS = CALIBRATION_CORPUS[:2000]
        prompt = f"{SHORT_CORPUS}\n\nQuestion: When was the lab founded?\nAnswer: The lab was founded in 2019."
        prompt_tokens = tokenizer.encode(prompt)
        
        from mlx_lm.models.cache import ArraysCache, KVCache
        cache = ArraysCache(size=num_layers)
        # Initialize all layers with regular KVCache (no compression for fitting)
        for layer_idx in range(num_layers):
            cache[layer_idx] = KVCache()

        y = mx.array([prompt_tokens])
        logits = model(y[:, :-1], cache=cache)

        log(f"✓ Full cache generated: {cache[0].offset} tokens")
        
        # Fit each layer
        for layer_idx in range(18, 27):
            log(f"Layer {layer_idx:2d}/36: ", end='')
            
            keys = cache[layer_idx].keys[..., :cache[layer_idx].offset, :]
            values = cache[layer_idx].values[..., :cache[layer_idx].offset, :]
            
            calibration = fit_am_layer(layer_idx, keys, values, queries, args.ratio)
            full_calibration[layer_idx] = calibration

            # Convert beta to numpy for printing
            beta_np = np.array(calibration['beta'])
            log(f"budget={calibration['budget']}, beta ∈ [{np.min(beta_np):.2f}, {np.max(beta_np):.2f}]")
    
    # ================================================================
    # Phase 3: Layers 27-35 (Extract from more compressed model)
    # ================================================================
    if args.phase in ['all', '3']:
        log("\n" + "=" * 70)
        log("Phase 3: Layers 27-35 (On-policy)")
        log("=" * 70)

        # Check if Phase 2 was completed
        if 26 not in full_calibration:
            log("✗ Phase 3 requires Phase 2 to be completed first (layers 18-26)")
            log("  Please run --phase all or --phase 2 first")
            return

        # Extract queries from model with layers 0-26 compressed
        target_layers = list(range(27, 36))
        compressed_layers = {i: full_calibration[i] for i in range(27)}
        queries = extract_queries_from_model(
            model, tokenizer, target_layers,
            compressed_layers_calibration=compressed_layers,
            num_queries_target=8192
        )
        
        # Generate full KV cache
        SHORT_CORPUS = CALIBRATION_CORPUS[:2000]
        prompt = f"{SHORT_CORPUS}\n\nQuestion: When was the lab founded?\nAnswer: The lab was founded in 2019."
        prompt_tokens = tokenizer.encode(prompt)
        
        from mlx_lm.models.cache import ArraysCache, KVCache
        cache = ArraysCache(size=num_layers)
        # Initialize all layers with regular KVCache (no compression for fitting)
        for layer_idx in range(num_layers):
            cache[layer_idx] = KVCache()

        y = mx.array([prompt_tokens])
        logits = model(y[:, :-1], cache=cache)

        log(f"✓ Full cache generated: {cache[0].offset} tokens")
        
        # Fit each layer
        for layer_idx in range(27, 36):
            log(f"Layer {layer_idx:2d}/36: ", end='')
            
            keys = cache[layer_idx].keys[..., :cache[layer_idx].offset, :]
            values = cache[layer_idx].values[..., :cache[layer_idx].offset, :]
            
            calibration = fit_am_layer(layer_idx, keys, values, queries, args.ratio)
            full_calibration[layer_idx] = calibration

            # Convert beta to numpy for printing
            beta_np = np.array(calibration['beta'])
            log(f"budget={calibration['budget']}, beta ∈ [{np.min(beta_np):.2f}, {np.max(beta_np):.2f}]")
    
    # ================================================================
    # Save calibration
    # ================================================================
    log("\n" + "=" * 70)
    log("Saving Calibration")
    log("=" * 70)
    
    output_file = f"am_calibration_{args.model}_{args.ratio}x_onpolicy.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(full_calibration, f)
    
    file_size = Path(output_file).stat().st_size / 1024
    log(f"✓ Calibration saved to: {output_file}")
    log(f"  File size: {file_size:.1f} KB")
    
    log("\n" + "=" * 70)
    log("✅ On-policy Calibration Complete!")
    log("=" * 70)
    log(f"Model: {args.model}")
    log(f"Layers: {len(full_calibration)}")
    log(f"Compression: {args.ratio}x")
    log(f"Output: {output_file}")
    log("")
    log("Next step: Test with test_calibrated_inference.py")
    log("=" * 70)

if __name__ == '__main__':
    main()
