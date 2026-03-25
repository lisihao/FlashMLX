"""
AM Query Sampling Strategies (按照论文实现)

论文使用三种策略收集大量 diverse queries：
1. Repeat-prefill: 重复跑 prefill，收集所有 layer 的 queries
2. Self-study: 让模型生成文本，收集 generation 阶段的 queries
3. On-policy: 实际任务中的 queries

关键：数量要大（数千到数万），要 diverse（不是连续的）
"""

import mlx.core as mx
from typing import Optional, List, Tuple
import numpy as np


def sample_queries_repeat_prefill(
    model,
    tokenizer,
    prompt: str,
    target_queries: int = 5000,
    layer_idx: int = 0,
) -> mx.array:
    """
    Repeat-prefill strategy: 重复跑 prefill 阶段，收集 queries

    策略：
    1. 将 prompt 分成多个 chunks
    2. 每个 chunk 单独 prefill，收集该 layer 的 queries
    3. 拼接所有 queries

    Parameters
    ----------
    model : 模型
    tokenizer : tokenizer
    prompt : str
        完整 prompt
    target_queries : int
        目标 query 数量
    layer_idx : int
        要收集的 layer index

    Returns
    -------
    queries : mx.array, shape (1, n_heads, num_queries, head_dim)
        收集到的 queries
    """
    from .cache import ArraysCache

    prompt_tokens = tokenizer.encode(prompt)
    total_tokens = len(prompt_tokens)

    # 计算需要的 repeat 次数
    # 假设每次 prefill 可以收集 total_tokens 个 queries
    num_repeats = max(1, target_queries // total_tokens)

    print(f"[Repeat-prefill] Collecting queries from layer {layer_idx}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Target queries: {target_queries}")
    print(f"  Num repeats: {num_repeats}")

    collected_queries = []

    for repeat_idx in range(num_repeats):
        # 创建临时 cache，hook 住指定 layer 的 queries
        cache = ArraysCache(size=len(model.model.layers))

        # Hack: 保存 queries 的容器
        saved_queries = []

        # Hook attention layer 的 forward
        original_call = model.model.layers[layer_idx].self_attn.__call__

        def hooked_call(x, mask=None, cache=None):
            print(f"[DEBUG] Hook called! x.shape={x.shape}", flush=True)

            # 提取 queries (在 forward 之前)
            # 直接计算 queries，避免依赖 forward 内部状态
            B, L, D = x.shape

            # 计算 queries: q_proj → reshape → q_norm → transpose
            q = model.model.layers[layer_idx].self_attn.q_proj(x)
            n_heads = model.model.layers[layer_idx].self_attn.n_heads
            head_dim = q.shape[-1] // n_heads
            q = q.reshape(B, L, n_heads, head_dim)

            # Apply q_norm if exists
            if hasattr(model.model.layers[layer_idx].self_attn, 'q_norm'):
                q = model.model.layers[layer_idx].self_attn.q_norm(q)

            queries = q.transpose(0, 2, 1, 3)  # (B, n_heads, L, head_dim)

            # 保存 queries
            saved_queries.append(queries)
            print(f"[DEBUG] Saved queries, shape={queries.shape}, total saved={len(saved_queries)}", flush=True)

            # 调用原始 forward
            output = original_call(x, mask=mask, cache=cache)

            return output

        # 替换 forward
        model.model.layers[layer_idx].self_attn.__call__ = hooked_call

        # 运行 prefill
        y = mx.array([prompt_tokens])
        try:
            _ = model(y[:, :-1], cache=cache)
        finally:
            # 恢复原始 forward
            model.model.layers[layer_idx].self_attn.__call__ = original_call

        # 收集这次的 queries
        if saved_queries:
            # 拼接所有 token 的 queries
            # saved_queries: List[array(B, n_heads, L, head_dim)]
            batch_queries = mx.concatenate(saved_queries, axis=2)  # (B, n_heads, total_L, head_dim)
            collected_queries.append(batch_queries)

        if len(collected_queries) * total_tokens >= target_queries:
            break

    if not collected_queries:
        raise ValueError("Failed to collect any queries")

    # 拼接所有 repeat 的 queries
    all_queries = mx.concatenate(collected_queries, axis=2)  # (B, n_heads, num_queries, head_dim)

    # 截取到目标数量
    if all_queries.shape[2] > target_queries:
        all_queries = all_queries[:, :, :target_queries, :]

    print(f"  Collected {all_queries.shape[2]} queries")

    return all_queries


def sample_queries_self_study(
    model,
    tokenizer,
    prompt: str,
    num_generate: int = 2000,
    layer_idx: int = 0,
) -> mx.array:
    """
    Self-study strategy: 让模型生成文本，收集 generation 阶段的 queries

    策略：
    1. 从 prompt 开始生成 num_generate 个 tokens
    2. 每个 generation step 收集该 layer 的 query
    3. 拼接所有 queries

    Parameters
    ----------
    model : 模型
    tokenizer : tokenizer
    prompt : str
        初始 prompt
    num_generate : int
        生成的 token 数量
    layer_idx : int
        要收集的 layer index

    Returns
    -------
    queries : mx.array, shape (1, n_heads, num_generate, head_dim)
        收集到的 queries
    """
    from .cache import ArraysCache

    prompt_tokens = tokenizer.encode(prompt)

    print(f"[Self-study] Generating {num_generate} tokens from layer {layer_idx}")

    # 创建 cache
    cache = ArraysCache(size=len(model.model.layers))

    # Prefill
    y = mx.array([prompt_tokens])
    _ = model(y[:, :-1], cache=cache)

    # Generation loop，收集 queries
    collected_queries = []

    # Hook attention layer
    original_call = model.model.layers[layer_idx].self_attn.__call__

    def hooked_call(x, mask=None, cache=None):
        # 提取 queries (在 forward 之前)
        B, L, D = x.shape

        # 计算 queries: q_proj → reshape → q_norm → transpose
        q = model.model.layers[layer_idx].self_attn.q_proj(x)
        n_heads = model.model.layers[layer_idx].self_attn.n_heads
        head_dim = q.shape[-1] // n_heads
        q = q.reshape(B, L, n_heads, head_dim)

        # Apply q_norm if exists
        if hasattr(model.model.layers[layer_idx].self_attn, 'q_norm'):
            q = model.model.layers[layer_idx].self_attn.q_norm(q)

        queries = q.transpose(0, 2, 1, 3)  # (B, n_heads, L, head_dim)

        # 保存 queries
        collected_queries.append(queries)

        # 调用原始 forward
        output = original_call(x, mask=mask, cache=cache)

        return output

    model.model.layers[layer_idx].self_attn.__call__ = hooked_call

    # Generate
    y = mx.array([[prompt_tokens[-1]]])
    try:
        for i in range(num_generate):
            logits = model(y, cache=cache)
            y = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)

            # Early stop if EOS
            if y[0, 0].item() == tokenizer.eos_token_id:
                break
    finally:
        model.model.layers[layer_idx].self_attn.__call__ = original_call

    if not collected_queries:
        raise ValueError("Failed to collect any queries during generation")

    # 拼接
    all_queries = mx.concatenate(collected_queries, axis=2)  # (B, n_heads, num_queries, head_dim)

    print(f"  Collected {all_queries.shape[2]} generation queries")

    return all_queries


def sample_queries_hybrid(
    model,
    tokenizer,
    prompt: str,
    target_queries: int = 7000,
    layer_idx: int = 0,
    repeat_prefill_ratio: float = 0.7,
) -> mx.array:
    """
    Hybrid strategy: 结合 repeat-prefill 和 self-study

    按照论文，QuALITY 上平均：
    - 5k self-study tokens
    - 7k repeat-prefill tokens

    Parameters
    ----------
    model : 模型
    tokenizer : tokenizer
    prompt : str
        完整 prompt
    target_queries : int
        目标 query 总数
    layer_idx : int
        要收集的 layer index
    repeat_prefill_ratio : float
        repeat-prefill 的比例

    Returns
    -------
    queries : mx.array, shape (1, n_heads, total_queries, head_dim)
        混合采样的 queries
    """
    num_repeat = int(target_queries * repeat_prefill_ratio)
    num_self = target_queries - num_repeat

    print(f"[Hybrid sampling] Target: {target_queries} queries")
    print(f"  Repeat-prefill: {num_repeat}")
    print(f"  Self-study: {num_self}")

    # Repeat-prefill
    queries_repeat = sample_queries_repeat_prefill(
        model, tokenizer, prompt,
        target_queries=num_repeat,
        layer_idx=layer_idx
    )

    # Self-study
    queries_self = sample_queries_self_study(
        model, tokenizer, prompt,
        num_generate=num_self,
        layer_idx=layer_idx
    )

    # 拼接
    all_queries = mx.concatenate([queries_repeat, queries_self], axis=2)

    print(f"[Hybrid sampling] Total collected: {all_queries.shape[2]} queries")

    return all_queries


def sample_queries_simple_diverse(
    cache,
    num_queries: int = 1000,
    stride: int = 3,
) -> mx.array:
    """
    简化版：从 cache 中采样 diverse queries

    策略：
    1. 不只取最后 10 个连续 token
    2. 从整个 cache 中均匀采样，stride 跳跃
    3. 增加 diversity

    Parameters
    ----------
    cache : KV cache
    num_queries : int
        目标 query 数量
    stride : int
        采样步长

    Returns
    -------
    queries : mx.array, shape (1, n_heads, num_queries, head_dim)
        采样的 queries（用 keys 近似）
    """
    keys = cache.keys
    if len(keys.shape) == 4:
        keys = keys[0]  # (n_heads, seq_len, head_dim)

    offset = cache.offset if hasattr(cache, 'offset') else keys.shape[1]

    # 从整个 sequence 中采样，使用 stride
    indices = list(range(0, offset, stride))

    # 如果不够，补充随机采样
    if len(indices) < num_queries:
        remaining = num_queries - len(indices)
        random_indices = np.random.choice(offset, size=remaining, replace=False).tolist()
        indices.extend(random_indices)

    # 截取
    indices = indices[:num_queries]
    indices = sorted(indices)  # 保持顺序

    # 采样
    # keys: (n_heads, seq_len, head_dim)
    # 我们用 keys 作为 queries 的近似
    queries = mx.take(keys, mx.array(indices), axis=1)  # (n_heads, num_queries, head_dim)
    queries = queries[None, ...]  # (1, n_heads, num_queries, head_dim)

    print(f"[Simple diverse sampling] Sampled {len(indices)} queries with stride={stride}")

    return queries
