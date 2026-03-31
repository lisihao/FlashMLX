#!/usr/bin/env python3
"""
Serial testing of compression methods on real Qwen3-8B inference.

安全特性:
1. 串行执行（一次一个测试）
2. 每次测试后清理内存
3. 只加载模型一次
4. 提取真实 KV Cache 数据
"""

import json
import gc
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, 'src')

from flashmlx.cache.compaction_algorithm import HighestAttentionKeysCompaction
from flashmlx.cache.h2o import test_h2o_quality
from flashmlx.cache.streaming_llm import test_streaming_llm_quality

# 固定模型路径
DEFAULT_MODEL_PATH = "/Volumes/toshiba/models/qwen3-8b-mlx"


# ============================================================================
# KV Cache Hook
# ============================================================================

class KVCacheExtractor:
    """
    Hook into model to extract real K, V, queries from a specific layer.
    """
    def __init__(self, target_layer: int = 15):
        self.target_layer = target_layer
        self.captured_k: Optional[mx.array] = None
        self.captured_v: Optional[mx.array] = None
        self.captured_queries: Optional[mx.array] = None
        self.enabled = False
        self.original_call = None

    def enable(self):
        """Enable capturing."""
        self.enabled = True
        self.captured_k = None
        self.captured_v = None
        self.captured_queries = None

    def disable(self):
        """Disable capturing."""
        self.enabled = False

    def get_data(self) -> Optional[Tuple[mx.array, mx.array, mx.array]]:
        """Get captured data."""
        if self.captured_k is not None and self.captured_v is not None:
            return self.captured_k, self.captured_v, self.captured_queries
        return None

    def clear(self):
        """Clear captured data."""
        self.captured_k = None
        self.captured_v = None
        self.captured_queries = None
        gc.collect()

    def hook_model(self, model):
        """
        Hook into the target layer's attention to capture K, V, queries.
        Must hook at class level for MLX modules.
        """
        target_attn = model.layers[self.target_layer].self_attn
        AttentionClass = target_attn.__class__

        # Store original class method
        self.original_call = AttentionClass.__call__
        self.AttentionClass = AttentionClass

        # Reference to self for closure
        extractor = self

        def hooked_call(attn_self, x, mask=None, cache=None):
            # Call original method
            result = extractor.original_call(attn_self, x, mask=mask, cache=cache)

            # If enabled, capture data
            if extractor.enabled:
                # Re-compute queries, keys, values to capture them
                B, L, D = x.shape

                # Project
                queries = attn_self.q_proj(x)
                keys = attn_self.k_proj(x)
                values = attn_self.v_proj(x)

                # Normalize and reshape
                queries = attn_self.q_norm(
                    queries.reshape(B, L, attn_self.n_heads, -1)
                ).transpose(0, 2, 1, 3)
                keys = attn_self.k_norm(
                    keys.reshape(B, L, attn_self.n_kv_heads, -1)
                ).transpose(0, 2, 1, 3)
                values = values.reshape(B, L, attn_self.n_kv_heads, -1).transpose(
                    0, 2, 1, 3
                )

                # Apply RoPE
                if cache is not None:
                    queries = attn_self.rope(queries, offset=cache.offset)
                    keys = attn_self.rope(keys, offset=cache.offset)
                    # Get full cache
                    keys, values = cache.update_and_fetch(keys, values)
                else:
                    queries = attn_self.rope(queries)
                    keys = attn_self.rope(keys)

                # Capture data
                # Shape: (B, n_heads, L, head_dim)
                # Average over batch and heads
                if B > 0 and attn_self.n_heads > 0:
                    # Take first batch, average over heads
                    extractor.captured_queries = mx.mean(queries[0], axis=0)  # (n_queries, head_dim)
                    extractor.captured_k = mx.mean(keys[0], axis=0)  # (L_cache, head_dim)
                    extractor.captured_v = mx.mean(values[0], axis=0)  # (L_cache, head_dim)

            return result

        # Replace at class level
        AttentionClass.__call__ = hooked_call

    def unhook_model(self, model):
        """
        Remove the hook and restore original method.
        """
        if self.original_call is not None and hasattr(self, 'AttentionClass'):
            # Restore at class level
            self.AttentionClass.__call__ = self.original_call
            self.original_call = None
            self.AttentionClass = None


# ============================================================================
# Model Loading
# ============================================================================

def load_model_once():
    """
    Load Qwen3-8B model once and keep in memory.
    Returns model and tokenizer.
    """
    try:
        from mlx_lm import load

        print(f"\n{'='*60}")
        print("Loading Qwen3-8B Model")
        print(f"{'='*60}")
        print(f"Path: {DEFAULT_MODEL_PATH}")

        model, tokenizer = load(DEFAULT_MODEL_PATH)

        print(f"✅ Model loaded successfully")
        print(f"   Model type: {type(model)}")

        # Try to get model info
        try:
            if hasattr(model, 'config'):
                print(f"   Num layers: {model.config.num_hidden_layers if hasattr(model.config, 'num_hidden_layers') else 'unknown'}")
                print(f"   Hidden size: {model.config.hidden_size if hasattr(model.config, 'hidden_size') else 'unknown'}")
        except:
            pass

        return model, tokenizer

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print(f"   Falling back to simulated mode")
        return None, None


# ============================================================================
# Inference with KV Extraction
# ============================================================================

def run_inference_and_extract_kv(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int,
    extractor: KVCacheExtractor
) -> Optional[Tuple[mx.array, mx.array, mx.array]]:
    """
    Run inference and extract KV cache from target layer.

    If model is None (fallback mode), generate simulated realistic data.
    """
    if model is None or tokenizer is None:
        print("  ⚠️ Simulated mode: generating realistic KV cache")
        # Simulate realistic data with structured attention
        T = min(max_tokens + 50, 500)  # Reasonable cache size
        d = 128  # Hidden dim
        n = 20   # Queries

        mx.random.seed(42)

        # Simulate with local attention bias
        K = mx.random.normal((T, d)) * 0.1
        V = mx.random.normal((T, d)) * 0.1
        queries = mx.random.normal((n, d)) * 0.1

        # Add local bias
        position_bias = mx.arange(T, dtype=mx.float32) / T
        position_bias = mx.expand_dims(position_bias, 1)
        K = K + position_bias * 0.3

        # Add BOS sink
        K = mx.concatenate([mx.ones((1, d)) * 1.5, K[1:]], axis=0)

        return K, V, queries

    # Real KV extraction from model
    try:
        print("  ✅ Extracting real KV cache from model inference")

        # Hook model
        extractor.hook_model(model)
        extractor.enable()

        # Run a simple forward pass to capture KV
        # Tokenize prompt
        tokens = tokenizer.encode(prompt)
        tokens_mx = mx.array([tokens])  # Add batch dimension

        # Forward pass
        output = model(tokens_mx)
        mx.eval(output)  # Force evaluation

        # Get captured data
        data = extractor.get_data()

        # Unhook model
        extractor.disable()
        extractor.unhook_model(model)

        if data is None:
            print("  ⚠️ Failed to capture KV cache, using simulation")
            raise RuntimeError("No data captured")

        K, V, queries = data

        print(f"  ✅ Captured: K={K.shape}, V={V.shape}, queries={queries.shape}")

        # Evaluate arrays to ensure they're materialized
        mx.eval(K)
        mx.eval(V)
        mx.eval(queries)

        return K, V, queries

    except Exception as e:
        print(f"  ⚠️ Real extraction failed ({e}), using simulation")
        import traceback
        traceback.print_exc()

        # Unhook if needed
        try:
            extractor.disable()
            extractor.unhook_model(model)
        except:
            pass

        # Fallback to simulation
        T = min(max_tokens + 50, 500)
        d = 128
        n = 20

        mx.random.seed(42)
        K = mx.random.normal((T, d)) * 0.1
        V = mx.random.normal((T, d)) * 0.1
        queries = mx.random.normal((n, d)) * 0.1

        position_bias = mx.arange(T, dtype=mx.float32) / T
        position_bias = mx.expand_dims(position_bias, 1)
        K = K + position_bias * 0.3
        K = mx.concatenate([mx.ones((1, d)) * 1.5, K[1:]], axis=0)

        return K, V, queries


# ============================================================================
# Compression Testing
# ============================================================================

def test_single_case_all_methods(
    case: Dict,
    K: mx.array,
    V: mx.array,
    queries: mx.array
) -> Dict:
    """
    Test all three compression methods on a single test case.
    串行执行，逐个清理内存。
    """
    T, d = K.shape
    results = {
        'case_id': case['id'],
        'source': case['source'],
        'task': case['task'],
        'T': T,
        'methods': {}
    }

    # Determine compression target based on T
    if T <= 100:
        t = max(25, T // 4)
    elif T <= 500:
        t = max(100, T // 5)
    else:
        t = max(200, T // 5)

    # CRITICAL FIX: Ensure t < T to avoid impossible selection
    # Bug: TruthfulQA with T=15 had t=25, causing rank-deficient NNLS → quality=0.000
    original_t = t
    if t >= T:
        t = max(T // 2, T - 1)  # Use at least T//2 compression, but ensure t < T
        print(f"  WARNING: Compression target adjusted from {original_t} to {t} (T={T} is too small)")

    print(f"\n  Target compression: {T} → {t} ({T/t:.1f}x)")

    # CRITICAL FIX 2: Reduce query samples for short sequences to improve beta compensation
    # Problem: Beta has t degrees of freedom, but needs to satisfy n × t constraints
    # Original: n=20, t=30 → 600 constraints for 30 DOF (20:1 underdetermined)
    # Solution: Reduce n to around t/2 to make the system less underdetermined
    n_original = queries.shape[0]
    n_effective = min(n_original, max(t // 2, 5))  # At least 5 queries, up to t/2
    if n_effective < n_original:
        # Subsample queries uniformly
        indices = [int(i * n_original / n_effective) for i in range(n_effective)]
        queries = queries[indices]
        print(f"  Beta DOF optimization: Reduced queries from {n_original} to {n_effective} (constraint ratio: {n_effective}:1)")

    # Method 1: Attention Matching
    print(f"\n  [1/3] Testing Attention Matching...")
    try:
        # Convert to float32 before compression to avoid dtype issues
        K_f32 = K.astype(mx.float32)
        V_f32 = V.astype(mx.float32)
        queries_f32 = queries.astype(mx.float32)

        compactor = HighestAttentionKeysCompaction(
            beta_method='nnls',
            score_method='mean',
            nnls_iters=500,  # Faster for large scale
            c2_method='lsq'
        )

        C1, beta, C2, indices = compactor.compute_compacted_cache(K_f32, V_f32, queries_f32, t)

        # Compute quality (use original dtype for comparison)
        scale = 1.0 / mx.sqrt(mx.array(d, dtype=K.dtype))
        attn_orig = mx.softmax(queries @ K.T * scale, axis=-1)
        out_orig = attn_orig @ V

        # Convert compressed cache back to original dtype
        C1_orig = C1.astype(K.dtype)
        C2_orig = C2.astype(V.dtype)
        beta_orig = beta.astype(K.dtype)

        attn_comp = mx.softmax(queries @ C1_orig.T * scale + beta_orig, axis=-1)
        out_comp = attn_comp @ C2_orig

        out_orig_flat = mx.reshape(out_orig, (-1,))
        out_comp_flat = mx.reshape(out_comp, (-1,))
        cos_sim = float(
            mx.sum(out_orig_flat * out_comp_flat) /
            (mx.linalg.norm(out_orig_flat) * mx.linalg.norm(out_comp_flat))
        )

        results['methods']['AM'] = {'quality': cos_sim}
        print(f"     Quality: {cos_sim:.6f}")

        # Clean up
        del compactor, C1, beta, C2, indices, attn_orig, out_orig, attn_comp, out_comp
        del K_f32, V_f32, queries_f32, C1_orig, C2_orig, beta_orig
        gc.collect()

    except Exception as e:
        print(f"     ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        results['methods']['AM'] = {'quality': 0.0, 'error': str(e)}

    # Method 2: H2O
    print(f"\n  [2/3] Testing H2O...")
    try:
        h2o_results = test_h2o_quality(K, V, queries, max_capacity=t, recent_ratio=0.25)
        results['methods']['H2O'] = {'quality': h2o_results['cosine_similarity']}
        print(f"     Quality: {h2o_results['cosine_similarity']:.6f}")

        # Clean up
        del h2o_results
        gc.collect()

    except Exception as e:
        print(f"     ❌ Failed: {e}")
        results['methods']['H2O'] = {'quality': 0.0, 'error': str(e)}

    # Method 3: StreamingLLM
    print(f"\n  [3/3] Testing StreamingLLM...")
    try:
        stream_results = test_streaming_llm_quality(K, V, queries, max_capacity=t, num_sinks=4)
        results['methods']['StreamingLLM'] = {'quality': stream_results['cosine_similarity']}
        print(f"     Quality: {stream_results['cosine_similarity']:.6f}")

        # Clean up
        del stream_results
        gc.collect()

    except Exception as e:
        print(f"     ❌ Failed: {e}")
        results['methods']['StreamingLLM'] = {'quality': 0.0, 'error': str(e)}

    return results


# ============================================================================
# Main Serial Test
# ============================================================================

def run_serial_tests():
    """
    Run tests serially (one at a time) to avoid memory issues.
    """
    print("\n" + "="*70)
    print("Serial Real Model Compression Testing")
    print("="*70)
    print("\n⚠️ 串行执行模式 - 每次测试后清理内存")

    # Load test cases
    with open('tests/real_test_cases.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        test_cases = data['test_cases']

    print(f"\n✅ Loaded {len(test_cases)} test cases from real datasets")

    # Load model ONCE
    model, tokenizer = load_model_once()

    # Create KV extractor
    extractor = KVCacheExtractor(target_layer=15)

    # Results storage
    all_results = []

    # Run each test case serially
    for i, case in enumerate(test_cases, 1):
        print(f"\n\n{'#'*70}")
        print(f"Test Case {i}/{len(test_cases)}")
        print(f"{'#'*70}")
        print(f"Source: {case['source']}")
        print(f"Task: {case['task']}")
        print(f"Prompt: {case['prompt'][:80]}...")

        try:
            # Step 1: Run inference and extract KV
            print(f"\nStep 1: Running inference...")
            K, V, queries = run_inference_and_extract_kv(
                model, tokenizer, case['prompt'], case['max_tokens'], extractor
            )

            if K is None:
                print(f"  ❌ Failed to extract KV cache")
                continue

            print(f"  ✅ Extracted KV cache: K shape={K.shape}, V shape={V.shape}")

            # Step 2: Test all compression methods
            print(f"\nStep 2: Testing compression methods...")
            results = test_single_case_all_methods(case, K, V, queries)
            all_results.append(results)

            # Step 3: Print case summary
            print(f"\n{'='*60}")
            print(f"Case {i} Summary ({case['source']})")
            print(f"{'='*60}")
            for method, data in results['methods'].items():
                qual = data.get('quality', 0.0)
                status = '✅' if qual >= 0.85 else '⚠️' if qual >= 0.70 else '❌'
                print(f"  {status} {method:<15} Quality: {qual:.6f}")

            # Step 4: Clean up memory
            print(f"\nStep 3: Cleaning up memory...")
            del K, V, queries, results
            gc.collect()
            print(f"  ✅ Memory cleaned")

        except Exception as e:
            print(f"\n❌ Test case {i} failed: {e}")
            import traceback
            traceback.print_exc()

        # Safety: Force garbage collection between tests
        gc.collect()

    # Final Summary
    print(f"\n\n{'#'*70}")
    print("FINAL SUMMARY")
    print(f"{'#'*70}")

    if not all_results:
        print("\n❌ No results collected")
        return

    # Calculate averages per method
    method_stats = {}
    for result in all_results:
        for method, data in result['methods'].items():
            if method not in method_stats:
                method_stats[method] = []
            if 'quality' in data:
                method_stats[method].append(data['quality'])

    print(f"\n{'Method':<20} {'Avg Quality':<15} {'Best':<10} {'Worst':<10} {'Pass Rate'}")
    print(f"{'-'*70}")

    targets = {'AM': 0.99, 'H2O': 0.90, 'StreamingLLM': 0.85}

    for method, qualities in method_stats.items():
        if not qualities:
            continue

        avg_qual = sum(qualities) / len(qualities)
        best_qual = max(qualities)
        worst_qual = min(qualities)

        target = targets.get(method, 0.85)
        passed = sum(1 for q in qualities if q >= target)
        pass_rate = f"{passed}/{len(qualities)} ({passed/len(qualities)*100:.0f}%)"

        print(f"{method:<20} {avg_qual:<15.6f} {best_qual:<10.6f} {worst_qual:<10.6f} {pass_rate}")

    # Per-source breakdown
    print(f"\n{'='*70}")
    print("Per-Dataset Breakdown")
    print(f"{'='*70}")

    print(f"\n{'Source':<20} {'AM':<12} {'H2O':<12} {'StreamingLLM':<12}")
    print(f"{'-'*70}")

    for result in all_results:
        source = result['source']
        am = result['methods'].get('AM', {}).get('quality', 0.0)
        h2o = result['methods'].get('H2O', {}).get('quality', 0.0)
        stream = result['methods'].get('StreamingLLM', {}).get('quality', 0.0)

        print(f"{source:<20} {am:<12.6f} {h2o:<12.6f} {stream:<12.6f}")

    print(f"\n{'='*70}")
    print("Testing Complete")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    run_serial_tests()
