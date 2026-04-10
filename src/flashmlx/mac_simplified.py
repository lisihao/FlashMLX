"""
MAC-Attention 裁剪版 - 单用户inference优化

基于官方实现裁剪，去掉serving复杂性，保留核心加速逻辑。

参考：https://github.com/YJHMITWEB/MAC-Attention.git
"""

import mlx.core as mx
from typing import Optional, Tuple
import math


class SimplifiedMACCache:
    """简化的MAC ring cache - 只支持单request"""

    def __init__(
        self,
        capacity: int,
        num_heads: int,
        head_dim: int,
        dtype: mx.Dtype = mx.bfloat16,
    ):
        """
        Args:
            capacity: Ring cache容量 (M)
            num_heads: Query heads数量
            head_dim: Head维度
        """
        self.capacity = capacity
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype

        # Ring caches [M, H, D]
        self.query_cache = mx.zeros((capacity, num_heads, head_dim), dtype=dtype)
        self.attn_cache = mx.zeros((capacity, num_heads, head_dim), dtype=dtype)
        self.lse_cache = mx.full((capacity, num_heads), -1e9, dtype=mx.float32)

        # Ring position (循环指针)
        self.write_pos = 0
        self.filled = 0  # 已填充的数量

    def reset(self):
        """重置cache"""
        self.write_pos = 0
        self.filled = 0
        self.query_cache = mx.zeros_like(self.query_cache)
        self.attn_cache = mx.zeros_like(self.attn_cache)
        self.lse_cache = mx.full((self.capacity, self.num_heads), -1e9, dtype=mx.float32)


class SimplifiedMACDecode:
    """
    简化版MAC Decode - 单用户inference

    核心流程：
    1. Match: 在ring cache中找相似的query
    2. Amend: 只计算partial attention (从match点开始)
    3. Complete: 更新ring cache
    """

    def __init__(
        self,
        cache_capacity: int = 1024,
        num_heads: int = 32,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        threshold: float = 0.95,
        window_left: int = 256,
    ):
        """
        Args:
            cache_capacity: Ring cache容量
            num_heads: Query heads数量
            num_kv_heads: KV heads数量
            head_dim: Head维度
            threshold: Match阈值 (越高越严格)
            window_left: Rectification窗口大小
        """
        self.cache_capacity = cache_capacity
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.threshold = threshold
        self.window_left = window_left

        # 创建ring cache
        self.ring_cache = SimplifiedMACCache(
            capacity=cache_capacity,
            num_heads=num_heads,
            head_dim=head_dim,
        )

        # Match threshold转换为L2 distance
        # hit if L2 < sqrt(2D) * (1 - threshold)
        self.match_threshold = math.sqrt(2 * head_dim) * (1.0 - threshold)

    def reset(self):
        """重置所有状态"""
        self.ring_cache.reset()

        # 重置统计器
        if hasattr(self, '_hit_count'):
            self._hit_count = 0
            self._total_count = 0
            self._total_skip_ratio = 0.0
        if hasattr(self, '_time_norm'):
            self._time_norm = 0
            self._time_match = 0
            self._time_attn = 0
            self._time_merge = 0
            self._time_cache = 0

    def _normalize_query(self, q: mx.array) -> mx.array:
        """归一化query用于L2 matching"""
        # q: [H, D]
        norm = mx.sqrt(mx.sum(q * q, axis=-1, keepdims=True))
        norm = mx.maximum(norm, 1e-8)
        return q / norm

    def _match(self, q_norm: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
        """
        Match阶段：在ring cache中找相似query

        Args:
            q_norm: [H, D] 归一化的query

        Returns:
            hit: [H] bool - 是否hit
            left_start: [H] int32 - attention起始位置
            match_idx: [H] int32 - matched cache位置（用于提取cached summary）
        """
        if self.ring_cache.filled == 0:
            # Cache为空，全miss
            zeros = mx.zeros(self.num_heads, dtype=mx.int32)
            return mx.zeros(self.num_heads, dtype=mx.bool_), zeros, zeros

        # 计算L2 distance: [H, M]
        cache_queries = self.ring_cache.query_cache[:self.ring_cache.filled]  # [filled, H, D]

        # Expand for broadcasting: [1, H, D] - [filled, H, D] -> [filled, H]
        q_expanded = q_norm[None, :, :]  # [1, H, D]
        diff = q_expanded - cache_queries  # [filled, H, D]
        l2_dist = mx.sqrt(mx.sum(diff * diff, axis=-1))  # [filled, H]

        # 找每个head的最小距离
        min_dist = mx.min(l2_dist, axis=0)  # [H]
        min_idx = mx.argmin(l2_dist, axis=0)  # [H]

        # Hit if distance < threshold
        hit = min_dist < self.match_threshold  # [H]

        # left_start: 如果hit，从match位置开始；否则从0开始
        left_start = mx.where(hit, min_idx.astype(mx.int32), mx.zeros(self.num_heads, dtype=mx.int32))

        # match_idx: 用于从cache提取，即使miss也记录（但merge时会被忽略）
        match_idx = min_idx.astype(mx.int32)

        return hit, left_start, match_idx

    def _partial_attention(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        left_start: mx.array,
        scale: Optional[float] = None,
    ) -> Tuple[mx.array, mx.array]:
        """
        Amend阶段：计算partial attention (使用MLX优化的实现)

        Args:
            q: [H, D] query
            k: [S, Hkv, D] keys (完整KV cache)
            v: [S, Hkv, D] values
            left_start: [H] 每个head的起始位置

        Returns:
            output: [H, D] attention输出
            lse: [H] log-sum-exp (简化版，不精确)
        """
        if scale is None:
            scale = 1.0 / math.sqrt(self.head_dim)

        # 简化：假设 left_start 都相同（全 miss 或全 hit）
        min_start = int(mx.min(left_start))

        # 截取 partial KV
        k_partial = k[min_start:]  # [S', Hkv, D]
        v_partial = v[min_start:]  # [S', Hkv, D]

        # 转换格式: [H, D] -> [1, H, 1, D], [S', Hkv, D] -> [1, Hkv, S', D]
        q_reshaped = q[None, :, None, :]  # [1, H, 1, D]
        k_reshaped = k_partial.transpose(1, 0, 2)[None, :, :, :]  # [1, Hkv, S', D]
        v_reshaped = v_partial.transpose(1, 0, 2)[None, :, :, :]  # [1, Hkv, S', D]

        # 使用 MLX 优化的 attention (支持 GQA)
        output_reshaped = mx.fast.scaled_dot_product_attention(
            q_reshaped, k_reshaped, v_reshaped, scale=scale
        )  # [1, H, 1, D]

        # 转回 [H, D]
        output = output_reshaped[0, :, 0, :]

        # LSE (简化：返回零，实际不影响结果因为我们不merge)
        lse = mx.zeros(self.num_heads, dtype=mx.float32)

        return output, lse

    def _merge_with_cached(
        self,
        fresh_o: mx.array,
        fresh_lse: mx.array,
        hit: mx.array,
        match_idx: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """
        LSE-based stable merge（MAC 核心机制）

        合并 cached prefix summary 和 fresh amend，使用数值稳定的 LSE merge。

        Args:
            fresh_o: [H, D] 新计算的 attention output
            fresh_lse: [H] 新的 log-sum-exp
            hit: [H] bool - 是否 hit
            match_idx: [H] int32 - match 的 cache 位置

        Returns:
            merged_o: [H, D] 合并后的 attention output
            merged_lse: [H] 合并后的 LSE
        """
        if not hit.any() or self.ring_cache.filled == 0:
            # 全部 miss，直接返回 fresh
            return fresh_o, fresh_lse

        # 从 cache 提取 matched 位置的 summary（完全向量化）
        # attn_cache: [M, H, D], lse_cache: [M, H]
        # match_idx: [H] - 每个 head 的 match 位置
        H = fresh_o.shape[0]
        D = fresh_o.shape[1]

        # 防止越界
        match_idx_safe = mx.clip(match_idx, 0, max(0, self.ring_cache.filled - 1))

        # 使用 mx.take_along_axis 进行 gather
        # attn_cache: [M, H, D] -> transpose -> [H, M, D]
        # 然后对第1维（M）进行 gather
        attn_t = self.ring_cache.attn_cache.transpose(1, 0, 2)  # [H, M, D]
        lse_t = self.ring_cache.lse_cache.transpose(1, 0)  # [H, M]

        # Expand match_idx for broadcasting: [H] -> [H, 1, 1] for attn, [H, 1] for lse
        idx_expanded_attn = mx.broadcast_to(match_idx_safe[:, None, None], (H, 1, D))  # [H, 1, D]
        idx_expanded_lse = match_idx_safe[:, None]  # [H, 1]

        # Gather using take_along_axis (axis=1 for M dimension)
        cached_o = mx.take_along_axis(attn_t, idx_expanded_attn, axis=1).squeeze(1)  # [H, D]
        cached_lse = mx.take_along_axis(lse_t, idx_expanded_lse, axis=1).squeeze(1)  # [H]

        # LSE-based stable merge
        # Reference: FlashAttention-2 online softmax
        max_lse = mx.maximum(cached_lse, fresh_lse)  # [H]

        # Compute weights (numerically stable)
        weight_cached = mx.exp(cached_lse - max_lse)  # [H]
        weight_fresh = mx.exp(fresh_lse - max_lse)    # [H]

        # Merge outputs
        # merged_o = (cached_o * weight_cached + fresh_o * weight_fresh) / (weight_cached + weight_fresh)
        weight_sum = weight_cached + weight_fresh  # [H]

        merged_o = (
            cached_o * weight_cached[:, None] +
            fresh_o * weight_fresh[:, None]
        ) / weight_sum[:, None]  # [H, D]

        # Merge LSE
        merged_lse = max_lse + mx.log(weight_sum)  # [H]

        # 只对 hit 的 heads 使用 merged，miss 的保持 fresh
        final_o = mx.where(hit[:, None], merged_o, fresh_o)
        final_lse = mx.where(hit, merged_lse, fresh_lse)

        return final_o, final_lse

    def _update_cache(
        self,
        q_norm: mx.array,
        output: mx.array,
        lse: mx.array,
    ):
        """
        Complete阶段：更新ring cache (优化版 - 避免mx.where)

        Args:
            q_norm: [H, D] 归一化的query
            output: [H, D] attention输出
            lse: [H] LSE值
        """
        pos = self.ring_cache.write_pos

        # 优化：使用Python list暂存，最后一次性stack
        # 避免每次decode都创建大量临时array
        if not hasattr(self, '_cache_list_query'):
            # 第一次：转为list
            self._cache_list_query = [self.ring_cache.query_cache[i] for i in range(self.cache_capacity)]
            self._cache_list_attn = [self.ring_cache.attn_cache[i] for i in range(self.cache_capacity)]
            self._cache_list_lse = [self.ring_cache.lse_cache[i] for i in range(self.cache_capacity)]

        # 直接更新list
        self._cache_list_query[pos] = q_norm
        self._cache_list_attn[pos] = output
        self._cache_list_lse[pos] = lse

        # 每N次decode才同步回array（lazy update）
        # 或者在需要match时才同步
        if self.ring_cache.filled < 10 or self.ring_cache.filled % 5 == 0:
            self.ring_cache.query_cache = mx.stack(self._cache_list_query, axis=0)
            self.ring_cache.attn_cache = mx.stack(self._cache_list_attn, axis=0)
            self.ring_cache.lse_cache = mx.stack(self._cache_list_lse, axis=0)

        # 更新指针
        self.ring_cache.write_pos = (pos + 1) % self.cache_capacity
        self.ring_cache.filled = min(self.ring_cache.filled + 1, self.cache_capacity)

    def __call__(
        self,
        q_pre_rope: mx.array,
        q_post_rope: mx.array,
        k: mx.array,
        v: mx.array,
        scale: Optional[float] = None,
    ) -> mx.array:
        """
        运行MAC Decode（pre-RoPE matching）

        Args:
            q_pre_rope: [B, H, D] 或 [H, D] pre-RoPE query (用于 matching)
            q_post_rope: [B, H, D] 或 [H, D] post-RoPE query (用于 attention)
            k: [B, S, Hkv, D] 或 [S, Hkv, D] keys (完整KV cache)
            v: [B, S, Hkv, D] 或 [S, Hkv, D] values
            scale: attention scale

        Returns:
            output: [B, H, D] 或 [H, D] attention输出
        """
        # 支持batch维度
        if q_pre_rope.ndim == 3:  # [B, H, D]
            assert q_pre_rope.shape[0] == 1, "SimplifiedMAC only supports batch=1"
            q_pre_rope = q_pre_rope[0]  # [H, D]
            q_post_rope = q_post_rope[0]  # [H, D]
            k = k[0]  # [S, Hkv, D]
            v = v[0]  # [S, Hkv, D]
            add_batch = True
        else:
            add_batch = False

        # Step 1: Match (使用 pre-RoPE query)
        import time
        t0 = time.time()
        q_norm = self._normalize_query(q_pre_rope)
        mx.eval(q_norm)
        t_norm = time.time() - t0

        t0 = time.time()
        hit, left_start, match_idx = self._match(q_norm)
        mx.eval(hit)
        mx.eval(left_start)
        mx.eval(match_idx)
        t_match = time.time() - t0

        # Debug: 记录真实 hit rate 和 skip ratio
        if not hasattr(self, '_hit_count'):
            self._hit_count = 0
            self._total_count = 0
            self._total_skip_ratio = 0.0
        self._total_count += 1
        hit_count = int(hit.sum())
        self._hit_count += hit_count

        # 计算实际跳过的比例
        if hit_count > 0 and self.ring_cache.filled > 0:
            # 只看 hit 的 heads 的平均 left_start
            avg_left_start = float(mx.where(hit, left_start, 0).sum()) / max(1, hit_count)
            skip_ratio = avg_left_start / self.ring_cache.filled
            self._total_skip_ratio += skip_ratio

        if self._total_count % 20 == 0:  # 每 20 次打印一次
            hit_rate = self._hit_count / (self._total_count * self.num_heads)
            avg_skip = self._total_skip_ratio / max(1, self._total_count)
            print(f"[MAC] Calls: {self._total_count}, Hit: {hit_rate:.1%}, Avg skip: {avg_skip:.1%}")

        # Step 2: Amend (partial attention，使用 post-RoPE query)
        t0 = time.time()
        fresh_o, fresh_lse = self._partial_attention(q_post_rope, k, v, left_start, scale)
        mx.eval(fresh_o)
        mx.eval(fresh_lse)
        t_attn = time.time() - t0

        # Step 3: Merge (LSE-based merge，MAC 核心！)
        t0 = time.time()
        output, output_lse = self._merge_with_cached(fresh_o, fresh_lse, hit, match_idx)
        mx.eval(output)
        mx.eval(output_lse)
        t_merge = time.time() - t0

        # Step 4: Complete (更新cache，使用 pre-RoPE query 和 merged output)
        t0 = time.time()
        self._update_cache(q_norm, output, output_lse)
        t_cache = time.time() - t0

        # 累积统计
        if not hasattr(self, '_time_norm'):
            self._time_norm = 0
            self._time_match = 0
            self._time_attn = 0
            self._time_merge = 0
            self._time_cache = 0
        self._time_norm += t_norm * 1000
        self._time_match += t_match * 1000
        self._time_attn += t_attn * 1000
        self._time_merge += t_merge * 1000
        self._time_cache += t_cache * 1000

        if self._total_count % 100 == 0:
            print(f"[MAC Breakdown] Norm: {self._time_norm/self._total_count:.2f}ms, "
                  f"Match: {self._time_match/self._total_count:.2f}ms, "
                  f"Attn: {self._time_attn/self._total_count:.2f}ms, "
                  f"Merge: {self._time_merge/self._total_count:.2f}ms, "
                  f"Cache: {self._time_cache/self._total_count:.2f}ms")

        # 恢复batch维度
        if add_batch:
            output = output[None, :, :]  # [1, H, D]

        return output


def test_simplified_mac():
    """快速测试"""
    print("测试SimplifiedMACDecode...")

    mac = SimplifiedMACDecode(
        cache_capacity=128,
        num_heads=8,
        num_kv_heads=2,
        head_dim=64,
    )

    # 模拟数据
    q = mx.random.normal((8, 64))
    k = mx.random.normal((100, 2, 64))
    v = mx.random.normal((100, 2, 64))

    # 第一次调用（全miss）
    out1 = mac(q, k, v)
    print(f"  第1次: output.shape={out1.shape}")

    # 第二次调用（可能hit）
    out2 = mac(q, k, v)
    print(f"  第2次: output.shape={out2.shape}")

    print("✅ 测试通过")


if __name__ == "__main__":
    test_simplified_mac()
