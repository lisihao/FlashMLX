#!/usr/bin/env python3
"""
Qwen3-8B 测试 - 使用 Random Queries
绕过 cache keys 采样，直接用随机 queries 验证算法本身
"""

import json
from mlx_lm import load, generate
from flashmlx.cache.simple_injection import inject_attention_matching


TEST_SCENARIOS = {
    "简单数学": "问题：3 + 5 = ?\n回答：",
    "中文": "请用一句话描述春天。\n回答：",
    "推理": "小明有5个苹果，吃了2个，还剩几个？\n回答：",
}

def main():
    print("=" * 80)
    print("Qwen3-8B 测试 - Random Queries（跳过 cache keys 采样）")
    print("=" * 80)

    model_path = "/Volumes/toshiba/models/qwen3-8b-mlx"
    print(f"\n🔄 加载模型...")
    model, tokenizer = load(model_path)
    print(f"✓ 模型加载成功（36 层）")

    compression_ratio = 2.0
    print(f"\n🔧 注入 Attention Matching (ratio={compression_ratio}x, random queries)...")

    # ✅ 修改 AttentionMatchingCompressorV2，强制使用 random queries
    import flashmlx.cache.attention_matching_compressor_v2 as am_module

    # Monkey patch: 强制使用 random queries
    original_compress = am_module.AttentionMatchingCompressorV2.compress_kv_cache

    def compress_with_random_queries(self, layer_idx, kv_cache):
        keys, values = kv_cache
        batch_size, num_heads, seq_len, head_dim = keys.shape

        if seq_len < self.compression_ratio:
            return keys, values

        target_seq_len = max(1, int(seq_len / self.compression_ratio))

        wrapper = self._get_wrapper(layer_idx)

        if batch_size > 1:
            raise NotImplementedError("Batch size > 1 not yet supported")

        # 批量处理
        import mlx.core as mx
        keys_3d = mx.squeeze(keys, axis=0)
        values_3d = mx.squeeze(values, axis=0)

        keys_torch = wrapper.mlx_to_torch(keys_3d)
        values_torch = wrapper.mlx_to_torch(values_3d)

        import torch
        C1_list = []
        beta_list = []
        C2_list = []

        for head_idx in range(num_heads):
            head_keys_torch = keys_torch[head_idx]
            head_values_torch = values_torch[head_idx]

            # ✅ 使用 random queries（绕过 cache keys 采样）
            num_queries_actual = min(self.num_queries, seq_len)
            sampled_queries_torch = torch.randn(
                num_queries_actual, head_dim,
                dtype=head_keys_torch.dtype,
                device=head_keys_torch.device
            )

            C1_torch, beta_torch, C2_torch, indices = wrapper.algorithm.compute_compacted_cache(
                K=head_keys_torch,
                V=head_values_torch,
                queries=sampled_queries_torch,
                t=target_seq_len,
            )

            C1_list.append(C1_torch)
            beta_list.append(beta_torch)
            C2_list.append(C2_torch)

        C1_torch_stacked = torch.stack(C1_list, dim=0)
        beta_torch_stacked = torch.stack(beta_list, dim=0)
        C2_torch_stacked = torch.stack(C2_list, dim=0)

        C1_mlx = wrapper.torch_to_mlx(C1_torch_stacked)
        beta_mlx = wrapper.torch_to_mlx(beta_torch_stacked)
        C2_mlx = wrapper.torch_to_mlx(C2_torch_stacked)

        for head_idx in range(num_heads):
            self.compressed_params[(layer_idx, head_idx)] = (
                C1_mlx[head_idx],
                beta_mlx[head_idx],
                C2_mlx[head_idx]
            )

        compressed_keys = mx.expand_dims(C1_mlx, axis=0)
        compressed_values = mx.expand_dims(C2_mlx, axis=0)

        self.compression_stats["total_compressions"] += 1
        self.compression_stats["total_keys_before"] += seq_len
        self.compression_stats["total_keys_after"] += compressed_keys.shape[2]

        return compressed_keys, compressed_values

    # 应用 monkey patch
    am_module.AttentionMatchingCompressorV2.compress_kv_cache = compress_with_random_queries

    inject_attention_matching(model, compression_ratio=compression_ratio, num_queries=100)
    print(f"✓ 注入成功（使用 random queries）")

    # 测试
    results = {}
    for name, prompt in TEST_SCENARIOS.items():
        print(f"\n📝 {name}")
        output = generate(model, tokenizer, prompt=prompt, max_tokens=50, verbose=False)
        print(f"  输出: {output[:100]}...")

        # 检查是否乱码
        is_garbage = (
            output.count('S') > len(output) * 0.5 or
            output.count(')') > len(output) * 0.3 or
            len(set(output[:50])) < 5  # 重复字符太多
        )

        status = "❌ 乱码" if is_garbage else "✓ 正常"
        print(f"  状态: {status}")
        results[name] = {"output": output, "is_garbage": is_garbage}

    # 总结
    print(f"\n{'=' * 80}")
    successful = sum(1 for r in results.values() if not r["is_garbage"])
    print(f"结果: {successful}/{len(results)} 成功")

    with open("test_random_queries_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
