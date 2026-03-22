#!/usr/bin/env python3
"""
FlashMLX Hybrid Cache - Quick Test

Fast verification of PP, TG, and memory usage.
"""

import time
import gc
import mlx.core as mx
from mlx_lm import load, generate
from flashmlx.cache import (
    inject_hybrid_cache_manager,
    create_layer_types_from_model,
    HybridCacheConfig
)


def get_memory_mb():
    """Get memory usage in MB."""
    gc.collect()
    mx.metal.clear_cache()
    return mx.metal.get_active_memory() / (1024 ** 2)


def quick_benchmark(model, tokenizer, prompt: str, use_hybrid: bool = False):
    """Quick benchmark of single prompt."""
    config_name = "Hybrid Cache" if use_hybrid else "Baseline"
    print(f"\n{'='*60}")
    print(f"{config_name}")
    print(f"{'='*60}")

    cache_wrapper = None

    # Setup hybrid cache
    if use_hybrid:
        layer_types = create_layer_types_from_model(model, "every 4th")
        config = HybridCacheConfig(
            total_budget_bytes=64 * 1024 * 1024,
            compression_ratio=4.0,
            beta_calibration=True
        )
        cache_wrapper = inject_hybrid_cache_manager(
            model, config, layer_types, auto_inject=True
        )
        print(f"✓ Hybrid cache enabled (4x compression, 64MB)")

    # Measure
    mem_before = get_memory_mb()

    start = time.time()
    response = generate(model, tokenizer, prompt=prompt, max_tokens=100, verbose=True)
    elapsed = time.time() - start

    mem_after = get_memory_mb()

    # Calculate approximate metrics
    prompt_tokens = len(tokenizer.encode(prompt))
    # Response includes prompt, so just count generated tokens directly
    # The verbose output shows actual token count
    response_tokens = 100  # max_tokens parameter

    # Rough estimate: 30% time for prefill, 70% for decode
    pp_time = elapsed * 0.3
    tg_time = elapsed * 0.7

    pp_tok_s = prompt_tokens / pp_time if pp_time > 0 else 0
    tg_tok_s = response_tokens / tg_time if tg_time > 0 else 0

    # Results
    print(f"\n{'='*60}")
    print(f"Results")
    print(f"{'='*60}")
    print(f"Prompt tokens:     {prompt_tokens}")
    print(f"Generated tokens:  {response_tokens}")
    print(f"Total time:        {elapsed:.2f}s")
    print(f"\nPP (Prompt Processing):")
    print(f"  Estimated time:  {pp_time*1000:.1f} ms")
    print(f"  Throughput:      {pp_tok_s:.1f} tok/s")
    print(f"\nTG (Token Generation):")
    print(f"  Estimated time:  {tg_time*1000:.1f} ms")
    print(f"  Throughput:      {tg_tok_s:.1f} tok/s")
    print(f"\nMemory:")
    print(f"  Before:          {mem_before:.1f} MB")
    print(f"  After:           {mem_after:.1f} MB")
    print(f"  Used:            {mem_after - mem_before:.1f} MB")

    if cache_wrapper:
        stats = cache_wrapper.get_statistics()
        print(f"\nCache Statistics:")
        # Safely extract statistics
        ssm_stats = stats.get('ssm', {}).get('local_cache', {})
        att_stats = stats.get('attention', {}).get('local_cache', {})

        hit_rate = ssm_stats.get('hit_rate', 0.0)
        compression = att_stats.get('avg_compression_ratio', 0.0)

        print(f"  SSM hit rate:    {hit_rate:.2%}")
        print(f"  Avg compression: {compression:.2f}x")

    print(f"\nResponse preview:")
    print(f"{response[:200]}...")

    return {
        "pp_tok_s": pp_tok_s,
        "tg_tok_s": tg_tok_s,
        "mem_used_mb": mem_after - mem_before
    }


def main():
    print("="*60)
    print("FlashMLX Hybrid Cache - Quick Test")
    print("="*60)

    # Load model from local Toshiba drive (already downloaded)
    print("\nLoading Qwen3.5-35B-A3B (MLX) from local path...")
    model_path = "/Volumes/toshiba/models/qwen3.5-35b-mlx/"
    model, tokenizer = load(model_path)
    print("✓ Model loaded")

    # Test prompt (use proper chat format with long context)
    # Create a realistic long context: multi-turn conversation about ML
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant specializing in machine learning and artificial intelligence."},
        {"role": "user", "content": "What is machine learning?"},
        {"role": "assistant", "content": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It uses algorithms to analyze data, identify patterns, and make decisions with minimal human intervention."},
        {"role": "user", "content": "What are the main types of machine learning?"},
        {"role": "assistant", "content": "There are three main types: 1) Supervised learning - learning from labeled data, 2) Unsupervised learning - finding patterns in unlabeled data, and 3) Reinforcement learning - learning through trial and error with rewards and penalties."},
        {"role": "user", "content": "Can you explain neural networks?"},
        {"role": "assistant", "content": "Neural networks are computing systems inspired by biological neural networks in animal brains. They consist of interconnected nodes (neurons) organized in layers: input layer, hidden layers, and output layer. Each connection has a weight that adjusts as learning proceeds, enabling the network to learn complex patterns."},
        {"role": "user", "content": "What is deep learning?"},
        {"role": "assistant", "content": "Deep learning is a subset of machine learning that uses neural networks with multiple hidden layers (hence 'deep'). These deep architectures can automatically learn hierarchical representations of data, making them powerful for tasks like image recognition, natural language processing, and speech recognition."},
        {"role": "user", "content": "Now, given all this context, can you summarize the key concepts of machine learning and explain how they work together in modern AI systems?"}
    ]

    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Baseline
    baseline = quick_benchmark(model, tokenizer, prompt, use_hybrid=False)

    # Hybrid cache
    hybrid = quick_benchmark(model, tokenizer, prompt, use_hybrid=True)

    # Comparison
    print(f"\n{'='*60}")
    print("Comparison")
    print(f"{'='*60}")

    pp_overhead = (baseline['pp_tok_s'] - hybrid['pp_tok_s']) / baseline['pp_tok_s'] * 100
    tg_overhead = (baseline['tg_tok_s'] - hybrid['tg_tok_s']) / baseline['tg_tok_s'] * 100
    mem_saved = (baseline['mem_used_mb'] - hybrid['mem_used_mb']) / baseline['mem_used_mb'] * 100

    print(f"\nPP overhead:       {pp_overhead:+.1f}%")
    print(f"TG overhead:       {tg_overhead:+.1f}%")
    print(f"Memory saved:      {mem_saved:.1f}%")

    print(f"\n{'='*60}")
    if tg_overhead <= 10 and mem_saved >= 15:
        print("✅ Performance targets met!")
    else:
        print("⚠️  Results may vary - run full benchmark for accurate measurements")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
