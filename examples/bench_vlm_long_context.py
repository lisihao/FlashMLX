"""
Benchmark VLM with Long Context

Tests FlashMLX cache performance with different context lengths (512, 2K, 4K).
Measures speed, memory usage, and generation quality.
"""

import sys
from pathlib import Path
import time
import mlx.core as mx

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

sys.path.insert(0, str(project_root / "src/flashmlx/models"))
sys.path.insert(0, str(project_root / "src/flashmlx/generation"))
sys.path.insert(0, str(project_root / "examples"))

from test_real_weights import download_and_load_model
from vlm_generator import VLMGenerator
from vlm_cache import create_vlm_cache, get_vlm_cache_info


def create_long_context_prompt(length: int) -> str:
    """Create a prompt with specified token length.

    Args:
        length: Target token count (~4 tokens per word)

    Returns:
        Prompt string with context + question
    """
    # Base context about MLX and vision models (~100 words = ~400 tokens)
    base_context = """
    MLX is an array framework for machine learning research on Apple Silicon.
    MLX is designed by machine learning researchers for machine learning researchers.
    The framework is intended to be user-friendly, but still efficient to train and
    deploy models. The design of the framework itself is also conceptually simple.

    Vision-Language Models (VLMs) combine computer vision and natural language processing
    to understand both images and text. These models can perform tasks like image captioning,
    visual question answering, and multimodal reasoning. Popular VLM architectures include
    CLIP, BLIP, Flamingo, and Qwen-VL.

    The key innovation in VLMs is the cross-modal attention mechanism that allows the model
    to align visual features with textual representations. This enables the model to generate
    contextually relevant text based on visual input.
    """

    # Calculate how many times to repeat to reach target length
    base_tokens = len(base_context.split())  # Approximate token count
    repeats = max(1, (length - 100) // base_tokens)  # Leave room for question

    # Build context
    context_parts = []
    for i in range(repeats):
        context_parts.append(f"[Context Part {i+1}]\n{base_context}")

    full_context = "\n\n".join(context_parts)

    # Add question at the end
    question = "\n\nBased on the above context, explain what makes MLX unique for Apple Silicon."

    return full_context + question


def benchmark_cache_strategy(
    model, tokenizer, config,
    cache_name: str,
    context_length: int,
    max_tokens: int = 100,
) -> dict:
    """Benchmark a cache strategy with specific context length.

    Args:
        model: VLM model
        tokenizer: Tokenizer
        config: Model config
        cache_name: Cache strategy name
        context_length: Target prompt token count
        max_tokens: Max tokens to generate

    Returns:
        Benchmark results
    """
    print(f"\n{'='*60}")
    print(f"Cache: {cache_name} | Context: ~{context_length} tokens")
    print(f"{'='*60}")

    # Create cache
    cache = create_vlm_cache(model, kv_cache=cache_name)
    cache_info = get_vlm_cache_info(cache)

    # Create prompt
    prompt = create_long_context_prompt(context_length)
    actual_tokens = len(tokenizer.encode(prompt))
    print(f"  Actual prompt tokens: {actual_tokens}")

    # Create generator
    generator = VLMGenerator(
        model=model,
        tokenizer=tokenizer,
        image_token_id=config.image_token_id,
        max_tokens=max_tokens,
    )

    # Measure memory before generation
    mx.eval(model.parameters())
    mem_before = mx.metal.get_active_memory() / 1024**2  # MB

    # Generate
    print(f"  Generating...")
    start_time = time.time()

    response = generator.generate(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.0,
        cache=cache,
        use_chat_template=True,  # Use proper chat format
    )

    mx.eval(response)  # Ensure computation is done
    elapsed = time.time() - start_time

    # Measure memory after generation
    mem_after = mx.metal.get_active_memory() / 1024**2  # MB
    mem_peak = mx.metal.get_peak_memory() / 1024**2  # MB
    mem_cache = mem_peak - mem_before

    # Calculate metrics
    response_tokens = len(tokenizer.encode(response))
    prompt_time = elapsed * (actual_tokens / (actual_tokens + response_tokens))  # Estimate
    gen_time = elapsed - prompt_time

    prompt_tokens_per_sec = actual_tokens / prompt_time if prompt_time > 0 else 0
    gen_tokens_per_sec = response_tokens / gen_time if gen_time > 0 else 0

    print(f"  Response: {response[:100]}...")
    print(f"  Response tokens: {response_tokens}")
    print(f"  Time: {elapsed:.2f}s (Prompt: {prompt_time:.2f}s, Gen: {gen_time:.2f}s)")
    print(f"  Prompt speed: {prompt_tokens_per_sec:.1f} tok/s")
    print(f"  Gen speed: {gen_tokens_per_sec:.1f} tok/s")
    print(f"  Cache memory: {mem_cache:.1f} MB")
    print(f"  Peak memory: {mem_peak:.1f} MB")

    return {
        "cache_name": cache_name,
        "cache_type": cache_info.get("cache_type", "N/A"),
        "context_length": actual_tokens,
        "response_tokens": response_tokens,
        "total_time": elapsed,
        "prompt_time": prompt_time,
        "gen_time": gen_time,
        "prompt_tok_per_sec": prompt_tokens_per_sec,
        "gen_tok_per_sec": gen_tokens_per_sec,
        "cache_memory_mb": mem_cache,
        "peak_memory_mb": mem_peak,
        "response_preview": response[:150],
    }


def main():
    """Run long-context benchmarks."""
    print("="*60)
    print("VLM Long Context Benchmark")
    print("="*60)

    # Load model
    print("\n[1/3] Loading model...")
    model, tokenizer, processor, config = download_and_load_model(use_4bit=False)

    # Benchmark configurations
    print("\n[2/3] Running benchmarks...")

    # Test different context lengths with different caches
    benchmarks = [
        # Short context (512 tokens)
        ("standard", 512),
        ("triple_pq", 512),

        # Medium context (2K tokens)
        ("standard", 2048),
        ("triple_pq", 2048),

        # Long context (4K tokens)
        ("standard", 4096),
        ("triple_pq", 4096),
    ]

    results = []

    for cache_name, context_len in benchmarks:
        try:
            result = benchmark_cache_strategy(
                model, tokenizer, config,
                cache_name=cache_name,
                context_length=context_len,
                max_tokens=50,  # Shorter generation for faster benchmarks
            )
            results.append(result)
        except Exception as e:
            print(f"\n  ❌ Error: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "="*60)
    print("[3/3] Summary")
    print("="*60)

    if len(results) > 0:
        print("\n| Context | Cache | Prompt tok/s | Gen tok/s | Cache Mem (MB) |")
        print("|---------|-------|--------------|-----------|----------------|")

        for r in results:
            print(f"| {r['context_length']:7} | {r['cache_name']:8} | "
                  f"{r['prompt_tok_per_sec']:12.1f} | {r['gen_tok_per_sec']:9.1f} | "
                  f"{r['cache_memory_mb']:14.1f} |")

        # Calculate memory savings
        print("\n### Memory Savings by Context Length")
        print("=" * 60)

        context_lengths = sorted(set(r['context_length'] for r in results))

        for ctx_len in context_lengths:
            standard_mem = next((r['cache_memory_mb'] for r in results
                               if r['context_length'] == ctx_len and r['cache_name'] == 'standard'), None)
            compressed_mem = next((r['cache_memory_mb'] for r in results
                                 if r['context_length'] == ctx_len and r['cache_name'] == 'triple_pq'), None)

            if standard_mem and compressed_mem:
                savings = (1 - compressed_mem / standard_mem) * 100
                speedup_prompt = next(r['prompt_tok_per_sec'] for r in results
                                    if r['context_length'] == ctx_len and r['cache_name'] == 'triple_pq') / \
                               next(r['prompt_tok_per_sec'] for r in results
                                    if r['context_length'] == ctx_len and r['cache_name'] == 'standard')

                print(f"\n{ctx_len} tokens:")
                print(f"  Standard: {standard_mem:.1f} MB")
                print(f"  Compressed: {compressed_mem:.1f} MB")
                print(f"  Memory savings: {savings:.1f}%")
                print(f"  Prompt speedup: {(speedup_prompt-1)*100:+.1f}%")

    print("\n✅ Long context benchmark complete!")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
