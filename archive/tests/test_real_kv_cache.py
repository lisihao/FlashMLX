"""
Real KV Cache Compression Quality Test

基于用户当机前的测试，重建版本
测试配置来自 Exp 2.2（最佳配置）

测试步骤：
1. 加载 Qwen3-8B 模型
2. 生成真实 KV cache（文章 + forward pass）
3. 应用 Attention Matching 压缩
4. 验证压缩质量（目标 ≥ 0.950）
"""
import mlx.core as mx
import numpy as np
from pathlib import Path
import time

# MLX-LM
from mlx_lm import load

# FlashMLX
from flashmlx.compaction.offline_compressor import offline_compress_kv_cache


# 全局变量：提取的 KV cache
_extracted_kv_cache = []


def patch_cache_for_kv_extraction():
    """
    Monkey patch KVCache.update_and_fetch 来提取 KV
    """
    from mlx_lm.models.cache import KVCache

    original_update_and_fetch = KVCache.update_and_fetch

    def update_and_fetch_with_extraction(self, keys, values):
        # 调用原始方法
        updated_keys, updated_values = original_update_and_fetch(self, keys, values)

        # 提取 KV（在更新后）
        _extracted_kv_cache.append((updated_keys, updated_values))

        return updated_keys, updated_values

    KVCache.update_and_fetch = update_and_fetch_with_extraction
    print("   ✓ Patched KVCache for KV extraction")


def generate_real_kv_cache(model, tokenizer, article: str):
    """
    用真实模型生成 KV cache

    方法：创建 cache 对象，forward pass 后提取

    Returns:
        past_key_values: list of (key, value) tuples per layer
    """
    print("🔧 Generating real KV cache...")
    print(f"   Article length: {len(article)} chars")

    # 1. Patch KVCache
    patch_cache_for_kv_extraction()

    # 2. Tokenize
    _extracted_kv_cache.clear()
    tokens = mx.array(tokenizer.encode(article))
    print(f"   Tokens: {tokens.shape[0]}")

    # 3. Create cache for the model
    from mlx_lm.models.cache import make_prompt_cache
    cache = make_prompt_cache(model)

    # 4. Forward pass with cache
    _ = model(tokens[None, :], cache=cache)

    print(f"   Extracted {len(_extracted_kv_cache)} layer caches")

    # 5. Return extracted KV cache
    if len(_extracted_kv_cache) == 0:
        print("   ⚠️ No KV cache extracted, trying alternative method...")

        # Alternative: access cache directly
        kv_cache = []
        if hasattr(cache, 'cache'):
            for layer_cache in cache.cache:
                if hasattr(layer_cache, 'keys') and hasattr(layer_cache, 'values'):
                    kv_cache.append((layer_cache.keys, layer_cache.values))

            print(f"   ✓ Extracted {len(kv_cache)} layer KV caches (alternative)")
            return kv_cache

    print(f"   ✓ Extracted {len(_extracted_kv_cache)} layer KV caches")
    return _extracted_kv_cache


def compute_attention_with_real_kv(queries, keys, values):
    """
    标准 attention 计算（真实 KV cache）

    Args:
        queries: (num_queries, head_dim)
        keys: (seq_len, head_dim)
        values: (seq_len, head_dim)

    Returns:
        output: (num_queries, head_dim)
    """
    # Attention scores
    scores = queries @ keys.T  # (num_queries, seq_len)

    # Scale
    scale = 1.0 / mx.sqrt(mx.array(queries.shape[-1], dtype=mx.float32))
    scores = scores * scale

    # Softmax
    weights = mx.softmax(scores, axis=-1)

    # Weighted sum
    output = weights @ values

    return output


def compute_compressed_attention(queries, C1, beta, C2):
    """
    压缩后的 attention 计算

    Args:
        queries: (num_queries, head_dim)
        C1: (budget, head_dim)
        beta: (budget,)
        C2: (budget, head_dim)

    Returns:
        output: (num_queries, head_dim)
    """
    # Attention scores
    scores = queries @ C1.T  # (num_queries, budget)

    # Scale
    scale = 1.0 / mx.sqrt(mx.array(queries.shape[-1], dtype=mx.float32))
    scores = scores * scale + beta[None, :]

    # Softmax
    weights = mx.softmax(scores, axis=-1)

    # Weighted sum
    output = weights @ C2

    return output


def test_real_kv_cache_compression():
    """
    真实 KV cache 压缩质量测试

    配置来自 Exp 2.2（最佳配置）：
    - Question type: summarize
    - QA pairs: 9
    - Queries per head: 50
    - Baseline: 0.931600
    - Target: 0.950000
    """
    print("=" * 70)
    print("REAL KV CACHE TEST")
    print("=" * 70)
    print()

    # === Configuration ===
    MODEL_PATH = "/Volumes/Toshiba/models/qwen3-8b-mlx"
    COMPRESSION_RATIO = 4
    NUM_QUERIES = 50  # 来自 Exp 2.2
    TARGET_QUALITY = 0.950000
    BASELINE_QUALITY = 0.931600

    print(f"📋 Configuration (best from Exp 2.2):")
    print(f"   Question type: summarize")
    print(f"   QA pairs: 9")
    print(f"   Queries per head: {NUM_QUERIES}")
    print(f"   Exp 2.2 baseline (synthetic cache): {BASELINE_QUALITY:.6f}")
    print(f"   Target: {TARGET_QUALITY:.6f}")
    print()

    # === Test Article ===
    ARTICLE = """
Artificial Intelligence (AI) has become one of the most transformative technologies of the 21st century.
From healthcare to finance, AI is revolutionizing how we work and live. Machine learning algorithms can
now diagnose diseases, predict market trends, and even create art. However, with great power comes great
responsibility. Ethical concerns about bias, privacy, and job displacement must be addressed as AI continues
to evolve and integrate into society.
"""

    # === Load Model ===
    print(f"📥 Loading model from {MODEL_PATH}...")
    try:
        model, tokenizer = load(MODEL_PATH)
        print(f"   ✓ Model loaded")
        print()
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
        return

    # === Generate Real KV Cache ===
    try:
        kv_cache = generate_real_kv_cache(model, tokenizer, ARTICLE)
        print()
    except Exception as e:
        print(f"   ❌ Failed to generate KV cache: {e}")
        import traceback
        traceback.print_exc()
        return

    # === Test First Layer ===
    print("🧪 Testing compression on first layer...")

    if len(kv_cache) == 0:
        print("   ❌ No KV cache extracted")
        return

    # Get first layer KV
    k, v = kv_cache[0]

    print(f"   K shape: {k.shape}")
    print(f"   V shape: {v.shape}")

    # Extract single head for testing
    # Shape: (batch, num_heads, seq, head_dim)
    if len(k.shape) == 4:
        # Qwen3: (batch=1, num_heads=8, seq_len=92, head_dim=128)
        batch_size, num_heads, seq_len, head_dim = k.shape
        print(f"   Batch: {batch_size}, Heads: {num_heads}, Seq: {seq_len}, Dim: {head_dim}")

        # Extract first head: (seq_len, head_dim)
        keys = k[0, 0, :, :]  # First head
        values = v[0, 0, :, :]
    else:
        print(f"   ⚠️ Unexpected K shape: {k.shape}")
        return

    print(f"   Keys (single head): {keys.shape}")
    print(f"   Values (single head): {values.shape}")

    seq_len, head_dim = keys.shape
    print(f"   Sequence length: {seq_len}")
    print(f"   Head dimension: {head_dim}")
    print()

    # === Compress KV Cache (generates queries internally) ===
    print(f"🗜️  Compressing KV cache ({COMPRESSION_RATIO}x)...")
    print(f"   Using self-study query generation ({NUM_QUERIES} queries)...")
    t0 = time.time()

    # Reshape for offline_compress_kv_cache
    # Expected: (batch, num_heads, seq, head_dim)
    K = keys[None, None, :, :]
    V = values[None, None, :, :]

    try:
        # Get compressed cache AND the queries used for compression
        C1, beta, C2, queries_4d = offline_compress_kv_cache(
            K, V,
            compression_ratio=COMPRESSION_RATIO,
            num_queries=NUM_QUERIES,
            use_omp=False,
            verbose=False,
            return_queries=True  # ← Get the queries used internally!
        )

        # Extract single head
        C1 = C1[0, 0]
        beta = beta[0, 0]
        C2 = C2[0, 0]
        queries = queries_4d[0, 0]  # Extract queries for first head

        t_comp = time.time() - t0
        budget = seq_len // COMPRESSION_RATIO

        print(f"   ✓ Compressed: {seq_len} -> {budget} tokens")
        print(f"   ✓ C1 shape: {C1.shape}")
        print(f"   ✓ beta shape: {beta.shape}")
        print(f"   ✓ C2 shape: {C2.shape}")
        print(f"   ✓ queries shape: {queries.shape}")
        print(f"   ✓ Time: {t_comp:.4f}s")

        # Debug: check for NaN/Inf
        print(f"\n   🔍 Debug checks:")
        print(f"      C1 - has NaN: {bool(mx.any(mx.isnan(C1)))}, has Inf: {bool(mx.any(mx.isinf(C1)))}")
        print(f"      beta - has NaN: {bool(mx.any(mx.isnan(beta)))}, has Inf: {bool(mx.any(mx.isinf(beta)))}")
        print(f"      beta - min: {float(mx.min(beta)):.6f}, max: {float(mx.max(beta)):.6f}, mean: {float(mx.mean(beta)):.6f}")
        print(f"      C2 - has NaN: {bool(mx.any(mx.isnan(C2)))}, has Inf: {bool(mx.any(mx.isinf(C2)))}")
        print()
    except Exception as e:
        print(f"   ❌ Compression failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # === Original Attention (using self-study queries) ===
    print("🔍 Computing original attention...")
    t0 = time.time()
    original_output = compute_attention_with_real_kv(queries, keys, values)
    t_orig = time.time() - t0
    print(f"   ✓ Output shape: {original_output.shape}")
    print(f"   ✓ Time: {t_orig:.4f}s")
    print()

    # === Compressed Attention ===
    print("🔍 Computing compressed attention...")
    t0 = time.time()
    compressed_output = compute_compressed_attention(queries, C1, beta, C2)
    t_comp_attn = time.time() - t0
    print(f"   ✓ Output shape: {compressed_output.shape}")
    print(f"   ✓ Time: {t_comp_attn:.4f}s")
    print()

    # === Quality Evaluation ===
    print("📈 Quality evaluation...")

    # MSE
    mse = mx.mean((original_output - compressed_output) ** 2)
    print(f"   MSE: {float(mse):.6f}")

    # Cosine similarity (per query)
    def cosine_similarity(a, b):
        dot = mx.sum(a * b, axis=-1)
        norm_a = mx.sqrt(mx.sum(a ** 2, axis=-1))
        norm_b = mx.sqrt(mx.sum(b ** 2, axis=-1))
        return dot / (norm_a * norm_b)

    cos_sim = cosine_similarity(original_output, compressed_output)
    avg_cos_sim = float(mx.mean(cos_sim))
    min_cos_sim = float(mx.min(cos_sim))

    print(f"   Cosine similarity (avg): {avg_cos_sim:.6f}")
    print(f"   Cosine similarity (min): {min_cos_sim:.6f}")
    print()

    # === Verdict ===
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)

    if avg_cos_sim >= TARGET_QUALITY:
        print(f"✅ PASS: {avg_cos_sim:.6f} >= {TARGET_QUALITY:.6f}")
    elif avg_cos_sim >= BASELINE_QUALITY:
        print(f"⚠️  MARGINAL: {avg_cos_sim:.6f} >= {BASELINE_QUALITY:.6f} (baseline)")
        print(f"   但未达到目标 {TARGET_QUALITY:.6f}")
    else:
        print(f"❌ FAIL: {avg_cos_sim:.6f} < {BASELINE_QUALITY:.6f} (baseline)")

    print()

    return {
        "mse": float(mse),
        "cosine_similarity_avg": avg_cos_sim,
        "cosine_similarity_min": min_cos_sim,
        "target": TARGET_QUALITY,
        "baseline": BASELINE_QUALITY,
        "passed": avg_cos_sim >= TARGET_QUALITY
    }


if __name__ == "__main__":
    print()
    results = test_real_kv_cache_compression()
    print()

    if results:
        print("📊 Final Results:")
