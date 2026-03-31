#!/usr/bin/env python3
"""
FlashMLX - Context Length Performance Benchmark

测试不同上下文长度（512/1K/2K/4K/8K/16K）的性能：
- PP (Prompt Processing) throughput
- TG (Token Generation) throughput
- TTFT (Time to First Token / 首 token 延迟)
- Memory consumption

对比：Baseline vs Hybrid Cache (with Attention Matching)
"""

import time
import gc
import mlx.core as mx
from mlx_lm import load, stream_generate
from flashmlx.cache import (
    inject_hybrid_cache_manager,
    restore_original_cache,
    get_cache_statistics,
    create_layer_types_from_model,
    HybridCacheConfig
)
import json


def get_memory_mb():
    """Get memory usage in MB."""
    gc.collect()
    mx.clear_cache()
    return mx.metal.get_active_memory() / (1024 ** 2)


def create_prompt(tokenizer, target_length: int):
    """
    Create a prompt with approximately target_length tokens.

    Uses multi-turn conversation to create realistic long context.
    """
    # Base conversation
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant specializing in machine learning and artificial intelligence."},
    ]

    # Add conversation turns until we reach target length
    qa_pairs = [
        ("What is machine learning?", "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It uses algorithms to analyze data, identify patterns, and make decisions with minimal human intervention."),
        ("What are the main types of machine learning?", "There are three main types: 1) Supervised learning - learning from labeled data, 2) Unsupervised learning - finding patterns in unlabeled data, and 3) Reinforcement learning - learning through trial and error with rewards and penalties."),
        ("Can you explain neural networks?", "Neural networks are computing systems inspired by biological neural networks in animal brains. They consist of interconnected nodes (neurons) organized in layers: input layer, hidden layers, and output layer. Each connection has a weight that adjusts as learning proceeds."),
        ("What is deep learning?", "Deep learning is a subset of machine learning that uses neural networks with multiple hidden layers (hence 'deep'). These deep architectures can automatically learn hierarchical representations of data, making them powerful for tasks like image recognition and natural language processing."),
        ("Explain convolutional neural networks.", "CNNs are specialized neural networks designed for processing grid-like data such as images. They use convolutional layers that apply filters to detect features like edges, textures, and patterns. The architecture typically includes convolutional layers, pooling layers, and fully connected layers."),
        ("What are transformers in AI?", "Transformers are a neural network architecture that revolutionized natural language processing. They use self-attention mechanisms to process sequences in parallel, unlike RNNs that process sequentially. This enables them to capture long-range dependencies and scale to massive datasets."),
        ("Explain the attention mechanism.", "The attention mechanism allows models to focus on different parts of the input when producing each part of the output. It computes attention scores that determine how much focus to place on each input element, enabling the model to weigh the importance of different information dynamically."),
        ("What is transfer learning?", "Transfer learning is a technique where a model trained on one task is reused as the starting point for a model on a second task. This is particularly effective in deep learning, where pre-trained models on large datasets can be fine-tuned for specific tasks with limited data."),
        ("Describe gradient descent.", "Gradient descent is an optimization algorithm used to minimize the loss function in machine learning. It iteratively adjusts model parameters in the direction of steepest descent of the loss function, using the gradient (derivative) to determine the direction and magnitude of updates."),
        ("What is overfitting?", "Overfitting occurs when a model learns the training data too well, including noise and outliers, resulting in poor generalization to new data. It's characterized by high training accuracy but low test accuracy. Techniques like regularization, dropout, and cross-validation help prevent overfitting."),
    ]

    current_tokens = 0
    pair_idx = 0

    while current_tokens < target_length:
        q, a = qa_pairs[pair_idx % len(qa_pairs)]
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})

        # Estimate current length
        temp_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        current_tokens = len(tokenizer.encode(temp_prompt))
        pair_idx += 1

        if current_tokens >= target_length:
            break

    # Add final question
    messages.append({"role": "user", "content": "Based on all the concepts we discussed, provide a comprehensive summary of modern machine learning."})

    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    actual_tokens = len(tokenizer.encode(prompt))
    return prompt, actual_tokens


def measure_performance(model, tokenizer, prompt: str, max_tokens: int = 100):
    """
    Measure performance using stream API for accurate TTFT and TBT.

    Returns:
        dict with ttft_ms, avg_tbt_ms, pp_tok_s, tg_tok_s
    """
    prompt_length = len(tokenizer.encode(prompt))

    # Measure memory before
    mem_before = get_memory_mb()

    # Stream generation with timing
    start_time = time.time()
    first_token_time = None
    token_times = []
    last_time = start_time
    token_count = 0

    for token in stream_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens
    ):
        current_time = time.time()

        if first_token_time is None:
            first_token_time = current_time
            ttft = first_token_time - start_time
        else:
            tbt = current_time - last_time
            token_times.append(tbt)

        last_time = current_time
        token_count += 1

    # Measure memory after
    mem_after = get_memory_mb()

    # Calculate metrics
    avg_tbt = sum(token_times) / len(token_times) if token_times else 0

    return {
        "ttft_ms": ttft * 1000,
        "ttft_s": ttft,
        "avg_tbt_ms": avg_tbt * 1000,
        "pp_tok_s": prompt_length / ttft if ttft > 0 else 0,
        "tg_tok_s": 1.0 / avg_tbt if avg_tbt > 0 else 0,
        "prompt_tokens": prompt_length,
        "generated_tokens": token_count,
        "mem_before_mb": mem_before,
        "mem_after_mb": mem_after,
        "mem_used_mb": mem_after - mem_before
    }


def benchmark_context_length(
    model,
    tokenizer,
    context_length: int,
    use_hybrid: bool = False,
    config: HybridCacheConfig = None
):
    """Benchmark single context length."""
    print(f"\n{'='*70}")
    print(f"Context Length: {context_length} tokens")
    print(f"Mode: {'Hybrid Cache' if use_hybrid else 'Baseline'}")
    print(f"{'='*70}")

    # Create prompt
    print(f"Creating prompt with ~{context_length} tokens...")
    prompt, actual_length = create_prompt(tokenizer, context_length)
    print(f"✓ Actual prompt length: {actual_length} tokens")

    # Setup hybrid cache if requested
    cache_list = None
    if use_hybrid and config:
        layer_types = create_layer_types_from_model(model, attention_layer_pattern="every 4th")
        cache_list = inject_hybrid_cache_manager(
            model, config, layer_types, auto_inject=True
        )
        print(f"✓ Hybrid cache enabled (compression={config.compression_ratio}x, budget={config.total_budget_bytes/(1024**2):.0f}MB)")

    # Measure performance
    print("Running performance test...")
    perf = measure_performance(model, tokenizer, prompt, max_tokens=100)

    # Get cache statistics if hybrid
    if cache_list:
        stats = get_cache_statistics(cache_list)

        # Extract SSM statistics
        ssm_hot = stats.get('ssm', {}).get('hot', {})
        ssm_warm = stats.get('ssm', {}).get('warm', {})
        ssm_cold = stats.get('ssm', {}).get('cold', {})

        # Calculate overall SSM hit rate
        total_accesses = ssm_hot.get('total_accesses', 0) + ssm_warm.get('total_accesses', 0) + ssm_cold.get('total_accesses', 0)
        total_hits = ssm_hot.get('hits', 0) + ssm_warm.get('hits', 0) + ssm_cold.get('hits', 0)
        ssm_hit_rate = total_hits / total_accesses if total_accesses > 0 else 0.0

        # Extract Attention compression
        att_stats = stats.get('attention', {})
        attention_compression = att_stats.get('avg_compression_ratio', 0.0)

        perf['cache_stats'] = {
            'ssm_hit_rate': ssm_hit_rate,
            'attention_compression': attention_compression
        }

    # Print results
    print(f"\n{'='*70}")
    print("Results")
    print(f"{'='*70}")
    print(f"TTFT (首 token 延迟):  {perf['ttft_ms']:7.1f} ms")
    print(f"Average TBT:            {perf['avg_tbt_ms']:7.1f} ms")
    print(f"PP Throughput:          {perf['pp_tok_s']:7.1f} tok/s")
    print(f"TG Throughput:          {perf['tg_tok_s']:7.1f} tok/s")
    print(f"Memory Used:            {perf['mem_used_mb']:7.1f} MB")

    if cache_list:
        print(f"\nCache Statistics:")
        print(f"  SSM hit rate:         {perf['cache_stats']['ssm_hit_rate']:7.1%}")
        print(f"  Attention compression:{perf['cache_stats']['attention_compression']:7.2f}x")

        # Restore original cache
        restore_original_cache(model, cache_list)

    # Clear cache
    gc.collect()
    mx.clear_cache()

    return perf


def main():
    print("="*70)
    print("FlashMLX - Context Length Performance Benchmark")
    print("="*70)

    # Load model
    print("\nLoading Qwen3.5-35B-A3B (MLX)...")
    model_path = "/Volumes/toshiba/models/qwen3.5-35b-mlx/"
    model, tokenizer = load(model_path)
    print("✓ Model loaded")

    # Test configurations
    context_lengths = [512, 1024, 2048, 4096, 8192, 16384]

    # Hybrid cache configuration
    hybrid_config = HybridCacheConfig(
        total_budget_bytes=128 * 1024 * 1024,  # 128MB
        compression_ratio=4.0,
        beta_calibration=True
    )

    # Run benchmarks
    results = {}

    for length in context_lengths:
        print(f"\n{'#'*70}")
        print(f"# Testing Context Length: {length} tokens")
        print(f"{'#'*70}")

        # Baseline
        baseline = benchmark_context_length(
            model, tokenizer, length,
            use_hybrid=False
        )

        # Hybrid cache
        hybrid = benchmark_context_length(
            model, tokenizer, length,
            use_hybrid=True,
            config=hybrid_config
        )

        results[length] = {
            'baseline': baseline,
            'hybrid': hybrid
        }

    # Print summary comparison
    print(f"\n{'='*70}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Context':<10} {'Mode':<12} {'TTFT (ms)':<12} {'PP (tok/s)':<12} {'TG (tok/s)':<12} {'Memory (MB)':<12}")
    print("-"*70)

    for length in context_lengths:
        baseline = results[length]['baseline']
        hybrid = results[length]['hybrid']

        print(f"{length:<10} {'Baseline':<12} {baseline['ttft_ms']:<12.1f} {baseline['pp_tok_s']:<12.1f} {baseline['tg_tok_s']:<12.1f} {baseline['mem_used_mb']:<12.1f}")
        print(f"{length:<10} {'Hybrid':<12} {hybrid['ttft_ms']:<12.1f} {hybrid['pp_tok_s']:<12.1f} {hybrid['tg_tok_s']:<12.1f} {hybrid['mem_used_mb']:<12.1f}")

        # Calculate overhead
        ttft_overhead = (hybrid['ttft_ms'] - baseline['ttft_ms']) / baseline['ttft_ms'] * 100
        pp_overhead = (baseline['pp_tok_s'] - hybrid['pp_tok_s']) / baseline['pp_tok_s'] * 100
        tg_overhead = (baseline['tg_tok_s'] - hybrid['tg_tok_s']) / baseline['tg_tok_s'] * 100
        mem_saved = (baseline['mem_used_mb'] - hybrid['mem_used_mb']) / baseline['mem_used_mb'] * 100

        print(f"{'':<10} {'Overhead':<12} {ttft_overhead:+11.1f}% {pp_overhead:+11.1f}% {tg_overhead:+11.1f}% {mem_saved:+11.1f}%")
        print("-"*70)

    # Save results to JSON
    output_file = "benchmark_context_length_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {output_file}")

    print(f"\n{'='*70}")
    print("✓ Benchmark Complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
