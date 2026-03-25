#!/usr/bin/env python3
"""
Phase 3 诊断 - Beta 值分析

检查 AM 压缩时的 β 值分布，诊断 NNLS 求解状态
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "mlx-lm-source"))

import mlx.core as mx
import numpy as np
from mlx_lm import load


def safe_softmax(x, axis=-1):
    """数值稳定的 softmax"""
    x_max = mx.max(x, axis=axis, keepdims=True)
    exp_x = mx.exp(x - x_max)
    return exp_x / mx.sum(exp_x, axis=axis, keepdims=True)


def compute_attention_output(queries, keys, values, beta=None, scale=None):
    """计算 attention 输出"""
    if scale is None:
        scale = keys.shape[-1] ** 0.5

    scores = (queries @ keys.T) / scale

    if beta is not None:
        scores = scores + beta[None, :]

    weights = safe_softmax(scores, axis=1)
    output = weights @ values
    return output


def select_keys_attention_aware(queries, keys, budget, scale=None):
    """基于 attention 权重选择 keys"""
    query_len, head_dim = queries.shape
    seq_len = keys.shape[0]

    if scale is None:
        scale = head_dim ** 0.5

    scores = (queries @ keys.T) / scale
    total_scores = mx.sum(scores, axis=0)

    if budget >= seq_len:
        return mx.arange(seq_len)

    sorted_indices = mx.argsort(-total_scores)
    selected_indices = sorted_indices[:budget]
    selected_indices = mx.sort(selected_indices)

    return selected_indices


def nnls_clamped(M, y, lower_bound=1e-12, max_iters=100):
    """NNLS 求解器（带诊断）"""
    m, n = M.shape

    # 使用最小二乘作为初始解
    MtM = M.T @ M
    Mty = M.T @ y

    try:
        x = mx.linalg.solve(MtM, Mty)
    except:
        # 如果求解失败，使用伪逆
        x = mx.linalg.pinv(M) @ y

    # Clamp to positive
    x = mx.maximum(x, lower_bound)

    # 诊断信息
    residual = mx.linalg.norm(M @ x - y)
    x_mean = float(mx.mean(x))
    x_std = float(mx.std(x))
    x_min = float(mx.min(x))
    x_max = float(mx.max(x))

    diagnostics = {
        'residual': float(residual),
        'mean': x_mean,
        'std': x_std,
        'min': x_min,
        'max': x_max,
        'has_nan': bool(mx.any(mx.isnan(x))),
        'has_inf': bool(mx.any(mx.isinf(x)))
    }

    return x, diagnostics


def compact_single_head_with_diagnosis(queries, keys, values, budget, scale=None):
    """
    带诊断信息的单头压缩
    """
    query_len, head_dim = queries.shape
    seq_len = keys.shape[0]

    if scale is None:
        scale = head_dim ** 0.5

    diagnostics = {}

    # Fix #1: 确保 budget < seq_len
    original_budget = budget
    if budget >= seq_len:
        budget = max(seq_len // 2, seq_len - 1)
        diagnostics['budget_adjusted'] = True
        diagnostics['original_budget'] = original_budget
        diagnostics['adjusted_budget'] = budget
    else:
        diagnostics['budget_adjusted'] = False

    # Fix #2: 减少 query 样本
    n_original = query_len
    n_effective = min(n_original, max(budget // 2, 5))
    diagnostics['query_subsampling'] = {
        'original': n_original,
        'effective': n_effective,
        'ratio': n_effective / n_original
    }

    if n_effective < n_original:
        indices = [int(i * n_original / n_effective) for i in range(n_effective)]
        queries_subsampled = queries[indices]
    else:
        queries_subsampled = queries

    # Step 1: 选择 keys
    indices = select_keys_attention_aware(queries_subsampled, keys, budget, scale)
    C1 = keys[indices]

    # Step 2: 拟合 beta
    compressed_scores = (queries_subsampled @ C1.T) / scale
    original_scores = (queries_subsampled @ keys.T) / scale

    # Compute exp scores
    exp_scores_original = mx.exp(original_scores - mx.max(original_scores, axis=1, keepdims=True))
    exp_scores_compressed = mx.exp(compressed_scores - mx.max(compressed_scores, axis=1, keepdims=True))

    target_mass = mx.sum(exp_scores_original, axis=1)

    M = exp_scores_compressed
    y = target_mass

    # 诊断 NNLS 输入
    diagnostics['nnls_input'] = {
        'M_shape': M.shape,
        'y_shape': y.shape,
        'M_mean': float(mx.mean(M)),
        'M_std': float(mx.std(M)),
        'y_mean': float(mx.mean(y)),
        'y_std': float(mx.std(y)),
        'M_cond': float(mx.linalg.cond(M.T @ M))
    }

    B, nnls_diag = nnls_clamped(M, y, lower_bound=1e-12)

    diagnostics['nnls_solution'] = nnls_diag

    # Convert B to beta
    beta = mx.log(B + 1e-12)
    beta = mx.clip(beta, -10.0, 10.0)

    # 诊断 beta
    diagnostics['beta'] = {
        'mean': float(mx.mean(beta)),
        'std': float(mx.std(beta)),
        'min': float(mx.min(beta)),
        'max': float(mx.max(beta)),
        'has_nan': bool(mx.any(mx.isnan(beta))),
        'has_inf': bool(mx.any(mx.isinf(beta))),
        'clipped': float(mx.sum((beta == -10.0) | (beta == 10.0)))
    }

    # Step 3: 直接拷贝 C2（简化测试）
    C2 = values[indices]

    # 验证压缩后的 attention 权重
    compressed_scores_with_beta = (queries_subsampled @ C1.T) / scale + beta[None, :]
    compressed_attn_weights = safe_softmax(compressed_scores_with_beta, axis=1)

    diagnostics['attention_weights'] = {
        'original_mean': float(mx.mean(mx.sum(exp_scores_original, axis=1))),
        'compressed_mean': float(mx.mean(mx.sum(compressed_attn_weights, axis=1))),
        'weights_sum_mean': float(mx.mean(mx.sum(compressed_attn_weights, axis=1)))
    }

    return C1, beta, C2, diagnostics


def diagnose_compression(model_name, prompt, target_layer=39, compression_ratio=2.0):
    """诊断压缩过程"""
    print("="*70)
    print("Beta 值诊断")
    print("="*70)
    print(f"\n模型: {model_name}")
    print(f"Prompt: '{prompt}'")
    print(f"Target layer: {target_layer}")
    print(f"Compression ratio: {compression_ratio}\n")

    # 加载模型
    print("Loading model...")
    model, tokenizer = load(model_name)
    print(f"Model loaded: {len(model.layers)} layers\n")

    # Tokenize prompt
    tokens = mx.array(tokenizer.encode(prompt))
    print(f"Prompt tokens: {tokens.shape[0]}\n")

    # Forward pass to get KV cache
    print("Running forward pass...")
    output = model(tokens[None, :])
    print("Forward pass complete\n")

    # 获取目标层的 cache
    if hasattr(model, '_cache') and model._cache is not None:
        cache = model._cache
        if target_layer < len(cache):
            layer_cache = cache[target_layer]

            if layer_cache is not None and len(layer_cache) == 2:
                keys, values = layer_cache

                print(f"Layer {target_layer} cache:")
                print(f"  Keys shape: {keys.shape}")
                print(f"  Values shape: {values.shape}\n")

                # 对每个 head 进行诊断
                n_heads = keys.shape[0]
                seq_len = keys.shape[1]
                head_dim = keys.shape[2]

                budget = int(seq_len / compression_ratio)

                print(f"Compression parameters:")
                print(f"  Sequence length: {seq_len}")
                print(f"  Compression ratio: {compression_ratio}")
                print(f"  Budget: {budget}")
                print(f"  Num heads: {n_heads}\n")

                # 只诊断第一个 head
                head_idx = 0
                K_h = keys[head_idx]
                V_h = values[head_idx]
                Q_h = K_h  # 使用 keys 作为 queries（自注意力近似）

                print(f"Diagnosing head {head_idx}...")
                print(f"  K shape: {K_h.shape}")
                print(f"  V shape: {V_h.shape}")
                print(f"  Q shape: {Q_h.shape}\n")

                C1, beta, C2, diagnostics = compact_single_head_with_diagnosis(
                    Q_h, K_h, V_h, budget, scale=head_dim ** 0.5
                )

                print("="*70)
                print("诊断结果")
                print("="*70 + "\n")

                # Budget adjustment
                if diagnostics.get('budget_adjusted', False):
                    print(f"⚠️ Budget adjusted: {diagnostics['original_budget']} → {diagnostics['adjusted_budget']}")

                # Query subsampling
                qs = diagnostics['query_subsampling']
                print(f"\n📊 Query subsampling:")
                print(f"  Original queries: {qs['original']}")
                print(f"  Effective queries: {qs['effective']}")
                print(f"  Ratio: {qs['ratio']:.2f}")

                # NNLS input
                nnls_in = diagnostics['nnls_input']
                print(f"\n📐 NNLS input:")
                print(f"  M shape: {nnls_in['M_shape']}")
                print(f"  y shape: {nnls_in['y_shape']}")
                print(f"  M mean: {nnls_in['M_mean']:.6f}")
                print(f"  M std: {nnls_in['M_std']:.6f}")
                print(f"  y mean: {nnls_in['y_mean']:.6f}")
                print(f"  y std: {nnls_in['y_std']:.6f}")
                print(f"  M condition number: {nnls_in['M_cond']:.2e}")

                # NNLS solution
                nnls_sol = diagnostics['nnls_solution']
                print(f"\n🔧 NNLS solution (B):")
                print(f"  Residual: {nnls_sol['residual']:.6f}")
                print(f"  Mean: {nnls_sol['mean']:.6f}")
                print(f"  Std: {nnls_sol['std']:.6f}")
                print(f"  Min: {nnls_sol['min']:.6e}")
                print(f"  Max: {nnls_sol['max']:.6e}")
                print(f"  Has NaN: {nnls_sol['has_nan']}")
                print(f"  Has Inf: {nnls_sol['has_inf']}")

                # Beta values
                beta_diag = diagnostics['beta']
                print(f"\n⚡ Beta values (log(B)):")
                print(f"  Mean: {beta_diag['mean']:.6f}")
                print(f"  Std: {beta_diag['std']:.6f}")
                print(f"  Min: {beta_diag['min']:.6f}")
                print(f"  Max: {beta_diag['max']:.6f}")
                print(f"  Has NaN: {beta_diag['has_nan']}")
                print(f"  Has Inf: {beta_diag['has_inf']}")
                print(f"  Clipped values: {beta_diag['clipped']}")

                # Attention weights
                attn_w = diagnostics['attention_weights']
                print(f"\n🎯 Attention weights:")
                print(f"  Original mean mass: {attn_w['original_mean']:.6f}")
                print(f"  Compressed mean mass: {attn_w['compressed_mean']:.6f}")
                print(f"  Weights sum (should be ~1.0): {attn_w['weights_sum_mean']:.6f}")

                # Analysis
                print(f"\n" + "="*70)
                print("分析结论")
                print("="*70 + "\n")

                issues = []

                if beta_diag['has_nan'] or beta_diag['has_inf']:
                    issues.append("❌ Beta 包含 NaN/Inf，NNLS 求解失败")

                if beta_diag['clipped'] > 0:
                    issues.append(f"⚠️ {beta_diag['clipped']} 个 beta 值被裁剪到 [-10, 10]")

                if abs(beta_diag['mean']) > 5.0:
                    issues.append(f"⚠️ Beta 均值过大 ({beta_diag['mean']:.2f})，可能导致数值不稳定")

                if beta_diag['std'] > 5.0:
                    issues.append(f"⚠️ Beta 方差过大 ({beta_diag['std']:.2f})，分布不均匀")

                if nnls_in['M_cond'] > 1e10:
                    issues.append(f"⚠️ M 矩阵条件数过大 ({nnls_in['M_cond']:.2e})，NNLS 不稳定")

                if abs(attn_w['weights_sum_mean'] - 1.0) > 0.1:
                    issues.append(f"⚠️ Attention 权重和不为 1 ({attn_w['weights_sum_mean']:.2f})，归一化失败")

                if issues:
                    print("检测到问题：\n")
                    for issue in issues:
                        print(f"  {issue}")
                else:
                    print("✅ 未检测到明显问题，但压缩仍然失败")
                    print("   可能原因：")
                    print("   - Query 生成不合理（使用 keys 作为 queries）")
                    print("   - Qwen3.5 Attention 层实现特殊")
                    print("   - AM 假设在此架构上被打破")

                return diagnostics

            else:
                print(f"❌ Layer {target_layer} cache 格式异常")
        else:
            print(f"❌ Target layer {target_layer} 超出范围")
    else:
        print("❌ 模型没有 cache")

    return None


def main():
    model_name = "/Volumes/toshiba/models/qwen3.5-35b-mlx"
    prompt = "介绍机器学习的基本概念和应用场景"

    diagnostics = diagnose_compression(
        model_name=model_name,
        prompt=prompt,
        target_layer=39,
        compression_ratio=2.0
    )

    if diagnostics:
        # 保存诊断报告
        report_path = Path(__file__).parent.parent / ".solar" / "beta-diagnosis-report.md"
        with open(report_path, "w") as f:
            f.write("# Beta 值诊断报告\n\n")
            f.write(f"**日期**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**模型**: {model_name}\n")
            f.write(f"**Layer**: 39\n")
            f.write(f"**Compression ratio**: 2.0\n\n")

            f.write("## 诊断数据\n\n")
            f.write("```python\n")
            f.write(str(diagnostics))
            f.write("\n```\n\n")

        print(f"\n详细报告已保存到: {report_path}")


if __name__ == "__main__":
    main()
