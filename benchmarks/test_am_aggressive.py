#!/usr/bin/env python3
"""
AM 压缩强制触发测试 - 使用极长 prompt 和小 max_size

目标：真正触发压缩，测量实际性能影响
"""

import sys
import time
import gc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "mlx-lm-source"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import KVCache, ArraysCache
from mlx_lm.models.compacted_cache import CompactedKVCache


def get_cache_size(cache):
    """获取 cache 中的 token 数量"""
    if isinstance(cache, CompactedKVCache):
        if hasattr(cache, '_keys') and cache._keys is not None:
            return cache._keys.shape[1]  # [num_heads, seq_len, head_dim]
    elif isinstance(cache, KVCache):
        if hasattr(cache, '_keys') and cache._keys is not None:
            return cache._keys.shape[2]  # [batch, num_heads, seq_len, head_dim]
    return 0


def generate_long_prompt(target_tokens=1000):
    """生成指定长度的 prompt"""
    base_text = """Machine learning is a subset of artificial intelligence that focuses on developing algorithms and statistical models. Neural networks are inspired by biological neural networks and consist of interconnected nodes. Deep learning uses multiple layers to progressively extract higher-level features. Supervised learning trains on labeled data with known outputs. Unsupervised learning discovers patterns in unlabeled data. Reinforcement learning uses rewards and penalties for decision making. Transfer learning applies knowledge from one domain to another. Computer vision enables machines to interpret visual information. Natural language processing helps computers understand human language. Recommendation systems personalize content based on user preferences. """

    prompt = ""
    while len(prompt.split()) < target_tokens:
        prompt += base_text

    return prompt


def test_with_compression(model, tokenizer, prompt, max_size=512, compression_ratio=2.0):
    """测试并报告压缩效果"""
    num_layers = len(model.layers)

    print(f"\n{'='*70}")
    print(f"配置: max_size={max_size}, ratio={compression_ratio}")
    print(f"Prompt tokens: {len(tokenizer.encode(prompt))}")
    print(f"{'='*70}")

    # 创建 AM cache
    cache = ArraysCache(size=num_layers)
    for i in range(num_layers):
        cache[i] = CompactedKVCache(
            max_size=max_size,
            compression_ratio=compression_ratio
        )

    # Tokenize
    tokens = mx.array([tokenizer.encode(prompt)])
    prompt_tokens = tokens.shape[1]

    # Phase 1: Prompt Processing
    print(f"\n[Phase 1] Prompt Processing ({prompt_tokens} tokens)...")
    pp_start = time.time()
    logits = model(tokens, cache=cache)
    mx.eval(logits)
    pp_time = time.time() - pp_start

    print(f"  PP time: {pp_time:.3f}s")
    print(f"  PP speed: {prompt_tokens/pp_time:.2f} tok/s")

    # 检查压缩统计
    print(f"\n[Compression Stats]")
    total_compressions = 0
    total_tokens_before = 0
    total_tokens_after = 0

    for i in range(num_layers):
        layer_cache = cache[i]
        if hasattr(layer_cache, 'get_stats'):
            stats = layer_cache.get_stats()
            if stats['num_compressions'] > 0:
                total_compressions += stats['num_compressions']
                total_tokens_before += stats['total_tokens_before']
                total_tokens_after += stats['total_tokens_after']

                if i < 3:  # 只打印前3层的详细信息
                    print(f"  Layer {i}:")
                    print(f"    Compressions: {stats['num_compressions']}")
                    print(f"    Total tokens before: {stats['total_tokens_before']}")
                    print(f"    Total tokens after: {stats['total_tokens_after']}")
                    print(f"    Avg compression ratio: {stats['avg_compression_ratio']:.2f}x")
                    print(f"    Current size: {stats['current_size']}")

    if total_compressions > 0:
        print(f"\n  Total compressions: {total_compressions}")
        print(f"  Total tokens before: {total_tokens_before}")
        print(f"  Total tokens after: {total_tokens_after}")
        avg_ratio = total_tokens_before / total_tokens_after if total_tokens_after > 0 else 0
        print(f"  Avg compression ratio: {avg_ratio:.2f}x")
        memory_saved_pct = (1 - total_tokens_after / total_tokens_before) * 100 if total_tokens_before > 0 else 0
        print(f"  Memory saved: {memory_saved_pct:.1f}%")
    else:
        print(f"  ⚠️  No compressions triggered!")
        print(f"  Hint: Increase prompt length or decrease max_size")

    # Phase 2: Token Generation
    print(f"\n[Phase 2] Token Generation...")
    max_gen_tokens = 50
    generated_tokens = []

    tg_start = time.time()

    # 第一个 token
    next_token = mx.argmax(logits[:, -1, :], axis=-1).item()
    if next_token != tokenizer.eos_token_id:
        generated_tokens.append(next_token)

    # 后续 tokens
    for _ in range(max_gen_tokens - 1):
        tokens = mx.array([[next_token]])
        logits = model(tokens, cache=cache)
        mx.eval(logits)

        next_token = mx.argmax(logits[:, -1, :], axis=-1).item()
        if next_token == tokenizer.eos_token_id:
            break

        generated_tokens.append(next_token)

    tg_time = time.time() - tg_start
    tg_speed = len(generated_tokens) / tg_time if tg_time > 0 else 0

    print(f"  TG time: {tg_time:.3f}s")
    print(f"  TG speed: {tg_speed:.2f} tok/s")
    print(f"  Generated tokens: {len(generated_tokens)}")

    # 输出示例
    output = tokenizer.decode(generated_tokens)
    print(f"\n[Output Sample]")
    print(f"{'-'*70}")
    print(output[:200])
    if len(output) > 200:
        print("...")
    print(f"{'-'*70}")

    return {
        'prompt_tokens': prompt_tokens,
        'pp_time': pp_time,
        'pp_speed': prompt_tokens / pp_time,
        'total_compressions': total_compressions,
        'total_tokens_before': total_tokens_before,
        'total_tokens_after': total_tokens_after,
        'memory_saved_pct': (1 - total_tokens_after / total_tokens_before) * 100 if total_tokens_before > 0 else 0,
        'tg_time': tg_time,
        'tg_speed': tg_speed,
        'output_tokens': len(generated_tokens),
        'output': output
    }


def main():
    print("="*70)
    print("AM 压缩强制触发测试")
    print("="*70)

    model_path = "/Volumes/toshiba/models/qwen3-8b-mlx"

    # 加载模型
    print("\nLoading model...")
    model, tokenizer = load(model_path)

    # 生成极长 prompt（确保触发压缩）
    print("\nGenerating long prompt...")
    prompt = generate_long_prompt(target_tokens=2000)
    actual_tokens = len(tokenizer.encode(prompt))
    print(f"Generated prompt: {actual_tokens} tokens")

    # 测试不同配置
    configs = [
        {'max_size': 1024, 'compression_ratio': 2.0, 'name': 'Conservative'},
        {'max_size': 512, 'compression_ratio': 2.0, 'name': 'Moderate'},
        {'max_size': 256, 'compression_ratio': 3.0, 'name': 'Aggressive'},
    ]

    all_results = {}

    for config in configs:
        result = test_with_compression(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_size=config['max_size'],
            compression_ratio=config['compression_ratio']
        )

        result['config_name'] = config['name']
        result['max_size'] = config['max_size']
        result['compression_ratio'] = config['compression_ratio']

        all_results[config['name']] = result

        # 清理
        gc.collect()
        mx.clear_cache()
        time.sleep(2)

    # 生成报告
    print(f"\n{'='*70}")
    print("测试总结")
    print(f"{'='*70}")

    print(f"\n{'Config':<15} {'Compressions':<15} {'Memory Saved':<15} {'TG Speed':<12}")
    print("-" * 60)

    for name, result in all_results.items():
        if result['total_compressions'] > 0:
            avg_ratio = result['total_tokens_before'] / result['total_tokens_after']
            print(f"{name:<15} "
                  f"{result['total_compressions']:<15} "
                  f"{result['memory_saved_pct']:.1f}%{' '*11} "
                  f"{result['tg_speed']:.2f} tok/s")
        else:
            print(f"{name:<15} "
                  f"0{' '*14} "
                  f"0.0%{' '*11} "
                  f"{result['tg_speed']:.2f} tok/s")

    # 保存详细报告
    report_path = Path(__file__).parent.parent / ".solar" / "am-aggressive-test-report.md"

    with open(report_path, "w") as f:
        f.write("# AM 压缩强制触发测试报告\n\n")
        f.write(f"**模型**: Qwen3-8B\n")
        f.write(f"**Prompt tokens**: {actual_tokens}\n")
        f.write(f"**日期**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## 测试配置\n\n")
        f.write("| 配置 | max_size | ratio | 压缩次数 | 内存节省 | TG Speed |\n")
        f.write("|------|----------|-------|---------|---------|----------|\n")

        for name, result in all_results.items():
            f.write(f"| {name} | {result['max_size']} | {result['compression_ratio']} | "
                   f"{result['total_compressions']} | {result['memory_saved_pct']:.1f}% | "
                   f"{result['tg_speed']:.2f} tok/s |\n")

        f.write("\n## 关键发现\n\n")
        f.write("### 1. 压缩效果\n\n")
        f.write(f"- Prompt tokens: {actual_tokens}\n")

        for name, result in all_results.items():
            if result['total_compressions'] > 0:
                avg_ratio = result['total_tokens_before'] / result['total_tokens_after']
                f.write(f"- {name}: {result['total_compressions']}次压缩, "
                       f"{result['memory_saved_pct']:.1f}%内存节省, "
                       f"{avg_ratio:.2f}x平均压缩率\n")
            else:
                f.write(f"- {name}: 未触发压缩\n")

        f.write("\n### 2. TG 性能影响\n\n")

        for name, result in all_results.items():
            f.write(f"- {name}: {result['tg_speed']:.2f} tok/s\n")

        f.write("\n## 结论\n\n")
        if any(r['total_compressions'] > 0 for r in all_results.values()):
            f.write("1. **压缩成功触发**：部分或全部配置成功触发了压缩\n")
            f.write("2. **内存节省**：AM 压缩能够有效减少 KV cache 内存占用\n")
            f.write("3. **TG 性能**：压缩对 TG 速度的影响需要进一步分析\n")
        else:
            f.write("1. **压缩未触发**：所有配置都未能触发压缩\n")
            f.write("2. **可能原因**：Prompt 长度不足或 max_size 设置过大\n")
            f.write("3. **建议**：使用更长的 prompt 或更小的 max_size\n")

    print(f"\n详细报告已保存到: {report_path}")

    print("\n" + "="*70)
    print("测试完成!")
    print("="*70)


if __name__ == "__main__":
    main()
