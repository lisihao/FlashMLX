"""
Generate with Compaction - Wrapper for compaction-enabled generation

This module provides a drop-in replacement for standard generation that
automatically manages KV cache compaction using the CompactionEngine.

Usage:
    from mlx_lm.generate_with_compaction import generate_with_compaction

    output = generate_with_compaction(
        model,
        tokenizer,
        prompt="What is machine learning?",
        max_tokens=256,
        use_compaction=True,
        compaction_config={
            "max_size": 2048,
            "compression_ratio": 5.0,
            "num_queries": 128,
            "check_interval": 256,
        }
    )
"""

from typing import Union, List, Optional, Dict, Any
import mlx.core as mx
import mlx.nn as nn
from transformers import PreTrainedTokenizer

from .generate import stream_generate
from .tokenizer_utils import TokenizerWrapper
from .models.cache import ArraysCache
from .models.compacted_cache import CompactedKVCache
from .models.compaction_engine import CompactionEngine


def generate_with_compaction(
    model: nn.Module,
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    prompt: Union[str, List[int]],
    max_tokens: int = 256,
    use_compaction: bool = True,
    compaction_config: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
    **kwargs,
) -> str:
    """
    Generate text with automatic KV cache compaction.

    This function is a drop-in replacement for mlx_lm.generate() that
    automatically manages KV cache compaction using CompactionEngine.

    Parameters
    ----------
    model : nn.Module
        The language model
    tokenizer : Union[PreTrainedTokenizer, TokenizerWrapper]
        The tokenizer
    prompt : Union[str, List[int]]
        The input prompt
    max_tokens : int, default=256
        Maximum number of tokens to generate
    use_compaction : bool, default=True
        Enable KV cache compaction
    compaction_config : Optional[Dict[str, Any]], optional
        Compaction configuration with keys:
        - max_size: int, default=2048
        - compression_ratio: float, default=5.0
        - num_queries: int, default=128
        - check_interval: int, default=256
        - use_quality_path: bool, default=True
        - quality_fit_beta: bool, default=True
        - quality_fit_c2: bool, default=True
    verbose : bool, default=False
        Print generation progress and compaction stats
    **kwargs
        Additional arguments passed to stream_generate

    Returns
    -------
    str
        Generated text

    Examples
    --------
    >>> from mlx_lm import load
    >>> from mlx_lm.generate_with_compaction import generate_with_compaction
    >>>
    >>> model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")
    >>>
    >>> # With compaction (default)
    >>> output = generate_with_compaction(
    ...     model, tokenizer,
    ...     prompt="What is machine learning?",
    ...     max_tokens=256,
    ...     use_compaction=True
    ... )
    >>>
    >>> # Custom compaction config
    >>> output = generate_with_compaction(
    ...     model, tokenizer,
    ...     prompt="Explain neural networks",
    ...     max_tokens=512,
    ...     compaction_config={
    ...         "max_size": 4096,
    ...         "compression_ratio": 3.0,
    ...         "num_queries": 256,
    ...     }
    ... )
    """
    if not use_compaction:
        # Fall back to standard generation
        if verbose:
            print("Compaction disabled, using standard generation")
        text = ""
        for response in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens, **kwargs):
            if verbose:
                print(response.text, end="", flush=True)
            text += response.text
        return text

    # Parse compaction config
    config = compaction_config or {}
    max_size = config.get("max_size", 2048)
    compression_ratio = config.get("compression_ratio", 5.0)
    num_queries = config.get("num_queries", 128)
    check_interval = config.get("check_interval", 256)
    use_quality_path = config.get("use_quality_path", True)
    quality_fit_beta = config.get("quality_fit_beta", True)
    quality_fit_c2 = config.get("quality_fit_c2", True)

    if verbose:
        print("=" * 70)
        print("Generation with Compaction")
        print("=" * 70)
        print(f"Compaction config:")
        print(f"  max_size: {max_size}")
        print(f"  compression_ratio: {compression_ratio}")
        print(f"  num_queries: {num_queries}")
        print(f"  check_interval: {check_interval}")
        print(f"  use_quality_path: {use_quality_path}")
        print("=" * 70)

    # Create compacted cache
    num_layers = len(model.layers)
    cache = ArraysCache(size=num_layers)
    for i in range(num_layers):
        cache[i] = CompactedKVCache(
            max_size=max_size,
            compression_ratio=compression_ratio,
            enable_compression=True,
            use_quality_path=use_quality_path,
            quality_fit_beta=quality_fit_beta,
            quality_fit_c2=quality_fit_c2,
        )

    # Create compaction engine
    engine = CompactionEngine(
        max_size=max_size,
        compression_ratio=compression_ratio,
        num_queries=num_queries,
        check_interval=check_interval,
    )

    # Pass cache to generation
    kwargs["prompt_cache"] = cache

    # Generate with periodic compaction
    text = ""
    token_count = 0

    for response in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens, **kwargs):
        if verbose and token_count == 0:
            print(f"\nPrompt: {response.prompt_tokens} tokens")
            print(f"Starting generation...")

        if verbose:
            print(response.text, end="", flush=True)
        text += response.text
        token_count += 1

        # Periodic compaction check (during TG phase)
        if token_count % check_interval == 0:
            if engine.should_compact(cache[0]):
                if verbose:
                    print(f"\n[Token {token_count}] Triggering compaction...")
                queries = engine.sample_queries(cache[0])
                num_compressed, compress_time = engine.compact_all_layers(
                    cache, queries, verbose=verbose
                )
                if verbose:
                    print(f"[Token {token_count}] Compaction complete")

    # Final compaction if needed
    if engine.should_compact(cache[0]):
        if verbose:
            print(f"\n[Final] Triggering compaction...")
        queries = engine.sample_queries(cache[0])
        num_compressed, compress_time = engine.compact_all_layers(
            cache, queries, verbose=verbose
        )

    if verbose:
        print()
        print("=" * 70)
        print("Generation Statistics")
        print("=" * 70)

        # Get cache stats
        stats = cache[0].get_stats()
        print(f"Cache statistics (Layer 0):")
        print(f"  Compressions: {stats['num_compressions']}")
        print(f"  Tokens before: {stats['total_tokens_before']}")
        print(f"  Tokens after: {stats['total_tokens_after']}")
        print(f"  Avg compression ratio: {stats['avg_compression_ratio']:.2f}x")

        # Get engine stats
        engine_stats = engine.get_stats()
        print(f"\nCompaction engine statistics:")
        print(f"  Total compactions: {engine_stats['total_compactions']}")
        print(f"  Total time: {engine_stats['total_compaction_time']*1000:.2f}ms")
        print(f"  Avg time: {engine_stats['avg_compaction_time']*1000:.2f}ms")

        print("=" * 70)

    return text
