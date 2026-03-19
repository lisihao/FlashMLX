"""
FlashMLX Utils - Utility functions
"""

import time
from typing import Callable, Any
from contextlib import contextmanager


@contextmanager
def timer(name: str = "Operation"):
    """
    Context manager for timing code blocks

    Args:
        name: Name of the operation being timed

    Example:
        with timer("Model inference"):
            output = model.generate(prompt)
    """
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"{name} took {elapsed*1000:.2f}ms")


def benchmark(
    func: Callable[..., Any],
    *args,
    warmup: int = 3,
    iterations: int = 10,
    **kwargs
) -> float:
    """
    Benchmark a function

    Args:
        func: Function to benchmark
        warmup: Number of warmup iterations
        iterations: Number of benchmark iterations
        *args, **kwargs: Arguments to pass to the function

    Returns:
        Average execution time in milliseconds
    """
    # Warmup
    for _ in range(warmup):
        func(*args, **kwargs)

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # Convert to ms

    return sum(times) / len(times)
