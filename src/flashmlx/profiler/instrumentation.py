"""
Function instrumentation (hooking)
"""

import time
import functools
from typing import Callable, Optional, Any
import mlx.core as mx


# Global profiler instance (set by Profiler context manager)
_active_profiler: Optional[Any] = None


def set_active_profiler(profiler):
    """Set the active profiler instance"""
    global _active_profiler
    _active_profiler = profiler


def get_active_profiler():
    """Get the active profiler instance"""
    return _active_profiler


def profile(name: Optional[str] = None, capture_args: bool = False):
    """
    Decorator to profile a function

    Args:
        name: Custom name for the function (defaults to function.__name__)
        capture_args: Whether to capture function arguments

    Example:
        @profile("my_function")
        def my_function(x):
            return x * 2
    """
    def decorator(func: Callable) -> Callable:
        func_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            profiler = get_active_profiler()

            if profiler is None:
                # No profiler active, run normally
                return func(*args, **kwargs)

            # Capture input shapes if MLX arrays
            input_shapes = None
            if capture_args:
                def collect_shapes(value):
                    if isinstance(value, mx.array):
                        return list(value.shape)
                    if isinstance(value, (list, tuple)):
                        return [collect_shapes(item) for item in value]
                    return None

                input_shapes = []
                for arg in args:
                    shapes = collect_shapes(arg)
                    if shapes is not None:
                        input_shapes.append(shapes)

                for key, value in kwargs.items():
                    shapes = collect_shapes(value)
                    if shapes is not None:
                        input_shapes.append({key: shapes})

            # Time the function
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)

                # Force evaluation if MLX array
                if isinstance(result, mx.array):
                    mx.eval(result)

                return result
            finally:
                duration_ms = (time.perf_counter() - start) * 1000

                # Log to profiler
                profiler.log_function_call(
                    function_name=func_name,
                    duration_ms=duration_ms,
                    input_shapes=input_shapes
                )

        return wrapper
    return decorator


def instrument_function(
    func: Callable,
    name: Optional[str] = None,
    capture_args: bool = False,
) -> Callable:
    """
    Instrument a function without using decorator

    Args:
        func: Function to instrument
        name: Custom name (defaults to func.__name__)

    Returns:
        Instrumented function

    Example:
        original_matmul = mx.matmul
        mx.matmul = instrument_function(original_matmul, "matmul")
    """
    return profile(name=name, capture_args=capture_args)(func)


def instrument_module(module, functions: list, prefix: str = "", capture_args: bool = False):
    """
    Instrument multiple functions in a module (monkey patching)

    Args:
        module: Module to instrument (e.g., mx, mx.fast)
        functions: List of function names to instrument
        prefix: Prefix for function names in logs

    Example:
        instrument_module(mx, ["matmul", "conv2d"], prefix="mx.")
    """
    original_functions = {}

    for func_name in functions:
        if not hasattr(module, func_name):
            continue

        original_func = getattr(module, func_name)
        instrumented_func = instrument_function(
            original_func,
            name=f"{prefix}{func_name}",
            capture_args=capture_args,
        )

        # Replace with instrumented version
        setattr(module, func_name, instrumented_func)

        # Store original for restoration
        original_functions[func_name] = original_func

    return original_functions


def restore_module(module, original_functions: dict):
    """
    Restore original functions in a module

    Args:
        module: Module to restore
        original_functions: Dictionary of original functions
    """
    for func_name, original_func in original_functions.items():
        setattr(module, func_name, original_func)
