#!/usr/bin/env python3
"""
UMA 微基准测试 - Task #64
测量 Mac UMA 架构下 CPU/GPU 传输延迟
"""

import mlx.core as mx
import time

def test_cpu_gpu_transfer():
    """测试 1: 纯传输延迟"""
    print("=== 测试 1: 纯传输延迟 ===")
    sizes = [512 * 1024, 1024 * 1024, 2048 * 1024]  # 512KB, 1MB, 2MB

    results = []
    for size_bytes in sizes:
        # 在 CPU 创建数据
        num_elements = size_bytes // 4  # float32
        cpu_data = mx.random.normal((num_elements,), dtype=mx.float32)

        # 测量传输到 GPU 的时间（MLX 默认使用 GPU）
        times = []
        for _ in range(100):  # 重复 100 次取平均
            start = time.perf_counter()
            # MLX 默认使用 GPU，这里测量的是数据评估延迟
            gpu_data = cpu_data + 0  # 触发计算，强制数据传输
            mx.eval(gpu_data)  # 强制同步
            end = time.perf_counter()
            times.append((end - start) * 1e6)  # 转换为 μs

        avg_latency = sum(times) / len(times)
        size_kb = size_bytes // 1024
        print(f"{size_kb}KB: {avg_latency:.2f} μs")
        results.append((size_kb, avg_latency))

    return results

def test_inference_with_fetch():
    """测试 2: 模拟推理场景"""
    print("\n=== 测试 2: 模拟推理场景 ===")

    # 模拟：在推理循环中从 CPU 拉取 KV Cache
    baseline_times = []
    with_fetch_times = []

    print("运行基准测试...")
    for _ in range(50):
        # Baseline: 纯 GPU 计算
        start = time.perf_counter()
        x = mx.random.normal((4096, 4096))
        y = mx.matmul(x, x)
        mx.eval(y)
        end = time.perf_counter()
        baseline_times.append((end - start) * 1000)  # ms

        # With CPU fetch: 模拟 512KB KV Cache 拉取
        start = time.perf_counter()
        cpu_kv = mx.random.normal((512 * 256,))  # 512KB
        gpu_kv = cpu_kv + 0  # 触发评估
        mx.eval(gpu_kv)
        x = mx.random.normal((4096, 4096))
        y = mx.matmul(x, x)
        mx.eval(y)
        end = time.perf_counter()
        with_fetch_times.append((end - start) * 1000)  # ms

    baseline_avg = sum(baseline_times) / len(baseline_times)
    with_fetch_avg = sum(with_fetch_times) / len(with_fetch_times)
    overhead = (with_fetch_avg - baseline_avg) / baseline_avg * 100

    print(f"Baseline: {baseline_avg:.2f} ms")
    print(f"With CPU fetch: {with_fetch_avg:.2f} ms")
    print(f"Overhead: {overhead:.2f}%")

    return baseline_avg, with_fetch_avg, overhead

def make_decision(transfer_results, overhead):
    """Go/No-Go 决策"""
    print("\n=== Go/No-Go 决策 ===")

    # 检查 512KB 传输延迟
    latency_512kb = transfer_results[0][1]

    # 决策标准
    go_criteria = latency_512kb < 10 and overhead < 5
    no_go_criteria = latency_512kb > 50 or overhead > 10

    if go_criteria:
        decision = "✅ **GO**"
        reason = f"512KB 传输延迟 {latency_512kb:.2f} μs < 10 μs，推理开销 {overhead:.2f}% < 5%"
        recommendation = "UMA-Dynamic Paging 可行，立即启动 Task #65"
    elif no_go_criteria:
        decision = "❌ **NO-GO**"
        reason = f"512KB 传输延迟 {latency_512kb:.2f} μs 或推理开销 {overhead:.2f}% 超出阈值"
        recommendation = "StreamingLLM (Task #61) 作为生产方案，重新评估 v3 架构"
    else:
        decision = "⚠️ **CONDITIONAL**"
        reason = f"512KB 传输延迟 {latency_512kb:.2f} μs，推理开销 {overhead:.2f}%"
        recommendation = "需要监护人判断是否可接受"

    print(f"决策: {decision}")
    print(f"理由: {reason}")
    print(f"建议: {recommendation}")

    return decision, reason, recommendation

if __name__ == "__main__":
    print("UMA 微基准测试 - Task #64")
    print("=" * 60)

    # 测试 1: 纯传输延迟
    transfer_results = test_cpu_gpu_transfer()

    # 测试 2: 模拟推理场景
    baseline_avg, with_fetch_avg, overhead = test_inference_with_fetch()

    # Go/No-Go 决策
    decision, reason, recommendation = make_decision(transfer_results, overhead)

    # 保存结果到文件
    with open('.solar/uma-benchmark-results.txt', 'w') as f:
        f.write("UMA 微基准测试结果 - Task #64\n")
        f.write("=" * 60 + "\n\n")

        f.write("=== 测试 1: 纯传输延迟 ===\n")
        for size_kb, latency in transfer_results:
            f.write(f"{size_kb}KB: {latency:.2f} μs\n")

        f.write("\n=== 测试 2: 模拟推理场景 ===\n")
        f.write(f"Baseline: {baseline_avg:.2f} ms\n")
        f.write(f"With CPU fetch: {with_fetch_avg:.2f} ms\n")
        f.write(f"Overhead: {overhead:.2f}%\n")

        f.write("\n=== Go/No-Go 决策 ===\n")
        f.write(f"决策: {decision}\n")
        f.write(f"理由: {reason}\n")
        f.write(f"建议: {recommendation}\n")

    print("\n✅ 结果已保存到 .solar/uma-benchmark-results.txt")
