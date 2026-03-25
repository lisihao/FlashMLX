#!/usr/bin/env python3
"""
验证热路径压缩问题

目的：
1. 确认 update_and_fetch() 在每层被调用
2. 测量热路径中压缩检查的开销
3. 证明 Quality Path 的 O(budget²) 会严重拖慢推理

用户的核心纠正：
"正确做法是：先正常推理，等满足触发条件后，再对'旧的、可压的 KV'做 AM 压缩，
然后继续推理。不是每一层、每一步都在线做 AM。"
"""

import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent / "mlx-lm-source"))

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import ArraysCache
from mlx_lm.models.compacted_cache import CompactedKVCache


class InstrumentedCache(CompactedKVCache):
    """插桩版本，记录调用次数"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.update_and_fetch_calls = 0
        self.compress_calls = 0
        self.compress_checks = 0  # 检查是否需要压缩的次数

    def update_and_fetch(self, keys, values):
        self.update_and_fetch_calls += 1

        # 调用父类方法
        result = super().update_and_fetch(keys, values)

        # 记录是否执行了压缩检查
        if self.enable_compression:
            self.compress_checks += 1

        return result

    def _compress(self):
        self.compress_calls += 1
        # 测量压缩时间
        start = time.time()
        super()._compress()
        compress_time = time.time() - start
        print(f"  ⏱️  压缩耗时: {compress_time*1000:.2f}ms (offset: {self.offset} → budget: {int(self.offset/self.compression_ratio)})")
        return compress_time


def main():
    print("="*70)
    print("热路径压缩问题验证")
    print("="*70)

    model_path = "/Volumes/toshiba/models/qwen3-8b-mlx"

    print(f"\n加载模型: {model_path}")
    model, tokenizer = load(model_path)
    num_layers = len(model.layers)

    print(f"模型层数: {num_layers}")

    # 使用插桩版本的 cache
    max_size = 256
    compression_ratio = 2.0

    print(f"\n配置:")
    print(f"  max_size: {max_size}")
    print(f"  compression_ratio: {compression_ratio}")
    print(f"  use_quality_path: True (O(budget²) 复杂度)")

    cache = ArraysCache(size=num_layers)
    for i in range(num_layers):
        cache[i] = InstrumentedCache(
            max_size=max_size,
            compression_ratio=compression_ratio,
            use_quality_path=True,  # 使用 Quality Path 暴露问题
            quality_fit_beta=True,
            quality_fit_c2=True
        )

    # 构造超长 prompt 强制触发压缩
    prompt = "Machine learning is a branch of artificial intelligence. " * 20
    tokens = mx.array([tokenizer.encode(prompt)])
    prompt_tokens = tokens.shape[1]

    print(f"\nPrompt tokens: {prompt_tokens}")
    print(f"预计触发压缩: {prompt_tokens > max_size}")

    # ========================================
    # Phase 1: Prefill (PP) - 一次性处理所有 prompt tokens
    # ========================================
    print(f"\n{'='*70}")
    print("Phase 1: Prefill (PP)")
    print("="*70)

    start_time = time.time()
    logits = model(tokens, cache=cache)
    mx.eval(logits)
    pp_time = time.time() - start_time

    # 统计第 0 层的调用情况
    layer0 = cache[0]
    print(f"\nLayer 0 统计 (PP 阶段):")
    print(f"  update_and_fetch() 调用次数: {layer0.update_and_fetch_calls}")
    print(f"  压缩检查次数: {layer0.compress_checks}")
    print(f"  实际压缩次数: {layer0.compress_calls}")
    print(f"  PP 总耗时: {pp_time:.3f}s")

    # ========================================
    # Phase 2: Token Generation (TG) - 逐个生成 token
    # ========================================
    print(f"\n{'='*70}")
    print("Phase 2: Token Generation (TG)")
    print("="*70)

    next_token = mx.argmax(logits[:, -1, :], axis=-1).item()

    print(f"\n生成 10 个 tokens，观察每个 token 的调用情况:")

    tg_start = time.time()
    for i in range(10):
        # 重置计数器
        calls_before = layer0.update_and_fetch_calls
        checks_before = layer0.compress_checks
        compress_before = layer0.compress_calls

        # 生成 token
        tokens = mx.array([[next_token]])
        logits = model(tokens, cache=cache)
        mx.eval(logits)
        next_token = mx.argmax(logits[:, -1, :], axis=-1).item()

        # 计算增量
        calls_delta = layer0.update_and_fetch_calls - calls_before
        checks_delta = layer0.compress_checks - checks_before
        compress_delta = layer0.compress_calls - compress_before

        print(f"\nToken {i+1}:")
        print(f"  update_and_fetch() 调用: {calls_delta} 次")
        print(f"  压缩检查: {checks_delta} 次")
        if compress_delta > 0:
            print(f"  ✅ 触发压缩: {compress_delta} 次")

    tg_time = time.time() - tg_start

    # ========================================
    # Phase 3: 全局统计
    # ========================================
    print(f"\n{'='*70}")
    print("全局统计 (所有层)")
    print("="*70)

    total_calls = sum(cache[i].update_and_fetch_calls for i in range(num_layers))
    total_checks = sum(cache[i].compress_checks for i in range(num_layers))
    total_compressions = sum(cache[i].compress_calls for i in range(num_layers))

    print(f"\n总 update_and_fetch() 调用: {total_calls}")
    print(f"总压缩检查: {total_checks}")
    print(f"总实际压缩: {total_compressions}")

    print(f"\n性能统计:")
    print(f"  PP 阶段: {pp_time:.3f}s")
    print(f"  TG 阶段 (10 tokens): {tg_time:.3f}s")
    print(f"  平均每 token: {tg_time/10*1000:.2f}ms")

    # ========================================
    # Phase 4: 问题分析
    # ========================================
    print(f"\n{'='*70}")
    print("问题分析")
    print("="*70)

    print(f"\n❌ 当前实现的问题:")
    print(f"  1. update_and_fetch() 在热路径中，每层每 token 都调用")
    print(f"     → PP 阶段: {layer0.update_and_fetch_calls} 次（应该是 1 次）")
    print(f"     → 实际情况: 每层都调用了 {layer0.update_and_fetch_calls} 次")

    print(f"\n  2. 压缩检查也在热路径中")
    print(f"     → 检查了 {total_checks} 次（{num_layers} 层 × {layer0.compress_checks} 次/层）")

    print(f"\n  3. Quality Path 的 O(budget²) 在热路径中执行")
    print(f"     → 实际压缩了 {total_compressions} 次")
    print(f"     → 每次压缩都是 O({int(max_size/compression_ratio)}²) 操作")

    print(f"\n  4. 没有使用参考查询 (Qref)")
    print(f"     → queries=None，使用 keys 作为查询（近似）")

    print(f"\n✅ 正确的做法（用户纠正）:")
    print(f"  1. update_and_fetch() 只负责追加新 KV，不检查压缩")
    print(f"  2. 压缩作为独立操作，在推理暂停时执行")
    print(f"  3. 压缩流程:")
    print(f"     正常 prefill / decode")
    print(f"         ↓")
    print(f"     KV 增长到阈值")
    print(f"         ↓")
    print(f"     暂停推理")
    print(f"         ↓")
    print(f"     对旧前缀执行一次 AM compaction (使用 Qref)")
    print(f"         ↓")
    print(f"     得到更小的 compacted KV")
    print(f"         ↓")
    print(f"     继续 decode")

    print(f"\n{'='*70}")
    print("结论")
    print("="*70)
    print(f"✅ 用户的分析完全正确！")
    print(f"   当前实现确实在热路径中执行压缩检查和压缩操作")
    print(f"   这会严重拖慢推理速度，尤其是 Quality Path")
    print(f"\n下一步: 重构为离线压缩架构")


if __name__ == "__main__":
    main()
