#!/usr/bin/env python3
"""
Attention Matching Profiling Script

目标：
1. 识别 Attention Matching 的性能瓶颈
2. 量化各个环节的时间开销
3. 找出优化方向

测量维度：
- 压缩各阶段时间（attention score, matching, selection, compression）
- 解压时间
- 内存使用
- 对 PP/TG 的影响
"""

import time
import gc
import numpy as np
import mlx.core as mx
from mlx_lm import load, generate
from flashmlx.cache import (
    AttentionMatchingCompressor,
    BudgetManager,
    BudgetConfig,
    TierType
)


class DetailedProfiler:
    """详细的性能分析器"""

    def __init__(self):
        self.timings = {}
        self.counters = {}
        self.memory_snapshots = {}

    def start(self, name: str):
        """开始计时"""
        self.timings[name] = {'start': time.perf_counter()}

    def end(self, name: str):
        """结束计时"""
        if name in self.timings:
            elapsed = time.perf_counter() - self.timings[name]['start']
            self.timings[name]['elapsed'] = elapsed
            self.timings[name]['count'] = self.timings[name].get('count', 0) + 1

    def record_memory(self, name: str):
        """记录内存使用"""
        self.memory_snapshots[name] = mx.get_active_memory() / (1024 ** 2)

    def increment(self, name: str, value: int = 1):
        """计数器"""
        self.counters[name] = self.counters.get(name, 0) + value

    def get_timing(self, name: str) -> float:
        """获取计时"""
        return self.timings.get(name, {}).get('elapsed', 0.0)

    def get_avg_timing(self, name: str) -> float:
        """获取平均计时"""
        timing = self.timings.get(name, {})
        elapsed = timing.get('elapsed', 0.0)
        count = timing.get('count', 1)
        return elapsed / count

    def report(self):
        """生成报告"""
        print("\n" + "="*70)
        print("Profiling Report")
        print("="*70)

        # 时间分析
        print("\n### 时间分析 ###\n")

        # 按时间排序
        sorted_timings = sorted(
            [(name, data.get('elapsed', 0), data.get('count', 1))
             for name, data in self.timings.items()],
            key=lambda x: x[1],
            reverse=True
        )

        total_time = sum(t[1] for t in sorted_timings)

        print(f"{'操作':<40} {'总时间 (ms)':<15} {'占比':<10} {'次数':<8} {'平均 (ms)':<12}")
        print("-"*85)

        for name, elapsed, count in sorted_timings:
            elapsed_ms = elapsed * 1000
            avg_ms = elapsed_ms / count if count > 0 else 0
            percentage = (elapsed / total_time * 100) if total_time > 0 else 0
            print(f"{name:<40} {elapsed_ms:>12.3f}    {percentage:>6.1f}%    {count:>6}    {avg_ms:>9.3f}")

        print("-"*85)
        print(f"{'总计':<40} {total_time * 1000:>12.3f}    {'100.0%':>7}    {'':<6}    {'':<9}")

        # 内存分析
        if self.memory_snapshots:
            print("\n### 内存分析 ###\n")
            print(f"{'阶段':<40} {'内存 (MB)':<15}")
            print("-"*55)

            for name, mem_mb in self.memory_snapshots.items():
                print(f"{name:<40} {mem_mb:>12.1f}")

        # 计数器
        if self.counters:
            print("\n### 统计计数 ###\n")
            print(f"{'指标':<40} {'数值':<15}")
            print("-"*55)

            for name, count in sorted(self.counters.items()):
                print(f"{name:<40} {count:>12}")


# 创建全局 profiler
profiler = DetailedProfiler()


class InstrumentedAttentionMatchingCompressor(AttentionMatchingCompressor):
    """带 profiling 的 AttentionMatchingCompressor"""

    def compress_kv_cache(
        self,
        layer_idx: int,
        kv_cache: tuple[mx.array, mx.array]
    ) -> tuple[mx.array, mx.array]:
        """压缩 KV cache（带详细 profiling）"""

        keys, values = kv_cache
        batch_size, num_heads, seq_len, head_dim = keys.shape

        profiler.start("compress_total")
        profiler.record_memory("compress_start")

        # 计算目标长度
        target_seq_len = max(1, int(seq_len / self.compression_ratio))

        # 如果序列太短，不压缩
        if seq_len < self.compression_ratio or seq_len <= target_seq_len:
            profiler.end("compress_total")
            profiler.record_memory("compress_end")
            return keys, values

        # Step 1: 计算平均 attention 权重
        profiler.start("compute_attention_weights")
        avg_attention_weights = self._compute_avg_attention_weights(
            layer_idx=layer_idx,
            keys=keys
        )
        profiler.end("compute_attention_weights")

        # Step 2: 选择要保留的 keys
        profiler.start("token_selection")
        keep_indices = self._select_keys_to_keep(
            attention_weights=avg_attention_weights,
            target_count=target_seq_len,
            eviction_policy=self.eviction_policy
        )
        profiler.end("token_selection")
        profiler.increment("tokens_selected", len(keep_indices))
        profiler.increment("tokens_dropped", seq_len - len(keep_indices))

        # Step 3: 压缩 keys 和 values（数据拷贝）
        profiler.start("actual_compression")
        keep_indices_mx = mx.array(keep_indices, dtype=mx.int32)
        compressed_keys = mx.take(keys, keep_indices_mx, axis=2)
        compressed_values = mx.take(values, keep_indices_mx, axis=2)
        mx.eval(compressed_keys)
        mx.eval(compressed_values)
        profiler.end("actual_compression")

        # Step 4: β calibration
        if self.beta_calibration:
            profiler.start("beta_calibration")
            beta = self._calibrate_beta(
                layer_idx=layer_idx,
                original_weights=avg_attention_weights,
                keep_indices=keep_indices
            )
            self.beta_params[layer_idx] = beta
            profiler.end("beta_calibration")

        profiler.record_memory("compress_end")
        profiler.end("compress_total")

        # 更新统计
        self.compression_stats["total_compressions"] += 1
        self.compression_stats["total_keys_before"] += seq_len
        self.compression_stats["total_keys_after"] += target_seq_len

        return compressed_keys, compressed_values


def profile_compression_stages():
    """详细 profiling 压缩各阶段"""

    print("\n" + "="*70)
    print("阶段 1: 压缩各环节 Profiling")
    print("="*70)

    # 创建测试数据
    batch_size = 1
    seq_len = 1000
    num_heads = 32
    head_dim = 128

    print(f"\n测试配置:")
    print(f"  Sequence length: {seq_len}")
    print(f"  Num heads: {num_heads}")
    print(f"  Head dim: {head_dim}")
    print(f"  Compression ratio: 3.0x")

    # 创建 instrumented compressor（不需要 budget_manager）
    compressor = InstrumentedAttentionMatchingCompressor(
        compression_ratio=3.0,
        beta_calibration=True
    )

    # 生成测试数据 (需要 4D 形状: batch, num_heads, seq_len, head_dim)
    profiler.start("generate_test_data")
    keys = mx.random.normal(shape=(batch_size, num_heads, seq_len, head_dim))
    values = mx.random.normal(shape=(batch_size, num_heads, seq_len, head_dim))
    mx.eval(keys)
    mx.eval(values)
    profiler.end("generate_test_data")

    profiler.record_memory("before_compression")

    # 运行多次压缩以获得稳定的测量
    num_iterations = 10
    print(f"\n运行 {num_iterations} 次压缩测试...")

    layer_idx = 0
    for i in range(num_iterations):
        compressed_keys, compressed_values = compressor.compress_kv_cache(
            layer_idx=layer_idx,
            kv_cache=(keys, values)
        )

        if i == 0:
            compressed_seq_len = compressed_keys.shape[2]
            print(f"  压缩结果: {seq_len} → {compressed_seq_len} tokens")

    profiler.record_memory("after_compression")

    # 生成报告
    profiler.report()

    # 计算压缩率（使用最后一次压缩的结果）
    original_size = keys.nbytes + values.nbytes
    compressed_size = compressed_keys.nbytes + compressed_values.nbytes
    if compressed_size > 0:
        actual_ratio = original_size / compressed_size
    else:
        actual_ratio = 0.0

    print(f"\n### 压缩效果 ###\n")
    print(f"原始大小:   {original_size / (1024**2):.2f} MB")
    print(f"压缩后:     {compressed_size / (1024**2):.2f} MB")
    print(f"实际压缩率: {actual_ratio:.2f}x")
    print(f"目标压缩率: 3.0x")
    print(f"\n原始 shape: {keys.shape}")
    print(f"压缩后 shape: {compressed_keys.shape}")

    return compressed_keys, compressed_values


def profile_end_to_end_generation():
    """端到端 generation profiling"""

    print("\n" + "="*70)
    print("阶段 2: 端到端 Generation Profiling")
    print("="*70)

    model_path = "/Volumes/toshiba/models/qwen3.5-35b-mlx/"

    print(f"\n加载模型: {model_path}")

    profiler.start("model_loading")
    model, tokenizer = load(model_path)
    profiler.end("model_loading")

    profiler.record_memory("model_loaded")

    # 测试 prompt
    prompt = "Explain the concept of attention mechanism in neural networks."

    print(f"\nPrompt: {prompt}")
    print(f"Max tokens: 100")

    # Baseline: 无压缩
    print("\n--- Baseline (无压缩) ---")

    profiler.start("baseline_generation")
    profiler.record_memory("baseline_start")

    response_baseline = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=100,
        verbose=False
    )

    profiler.end("baseline_generation")
    profiler.record_memory("baseline_end")

    baseline_tokens = len(tokenizer.encode(response_baseline))
    print(f"Generated {baseline_tokens} tokens")
    print(f"Time: {profiler.get_timing('baseline_generation')*1000:.1f} ms")

    # 清理
    gc.collect()
    mx.clear_cache()
    time.sleep(1)

    # With compression (模拟)
    # 注意：这里需要实际注入 compressed cache，暂时跳过
    print("\n⚠️  压缩版本需要实际注入 CompressedKVCache")
    print("   当前只能 profile 压缩算法本身的开销")

    # 生成最终报告
    profiler.report()


def main():
    print("="*70)
    print("Attention Matching Profiling")
    print("="*70)

    # 阶段 1: 压缩各环节详细 profiling
    profile_compression_stages()

    # 阶段 2: 端到端 generation (可选，较慢)
    run_e2e = input("\n是否运行端到端 generation profiling? (y/N): ").strip().lower()

    if run_e2e == 'y':
        profile_end_to_end_generation()
    else:
        print("\n跳过端到端测试")

    print("\n" + "="*70)
    print("✓ Profiling 完成")
    print("="*70)

    # 分析建议
    print("\n### 优化建议 ###\n")

    compress_time = profiler.get_timing("compress_total")
    attention_time = profiler.get_timing("compute_attention_scores")
    selection_time = profiler.get_timing("token_selection")
    compression_time = profiler.get_timing("actual_compression")

    if compress_time > 0:
        print(f"压缩总时间: {compress_time*1000:.2f} ms")
        print(f"  - Attention scores: {attention_time/compress_time*100:.1f}%")
        print(f"  - Token selection:  {selection_time/compress_time*100:.1f}%")
        print(f"  - 实际压缩:         {compression_time/compress_time*100:.1f}%")

        print("\n瓶颈分析:")

        if attention_time / compress_time > 0.5:
            print("  ⚠️  Attention score 计算是主要瓶颈")
            print("      建议: 使用近似方法或缓存 attention patterns")

        if selection_time / compress_time > 0.3:
            print("  ⚠️  Token selection 开销较大")
            print("      建议: 优化排序算法或使用采样")

        if compression_time / compress_time > 0.3:
            print("  ⚠️  数据拷贝开销较大")
            print("      建议: 使用 view 或 in-place 操作")


if __name__ == "__main__":
    main()
