"""
Simple profiling example
"""

import mlx.core as mx
from flashmlx.profiler import Profiler, ProfileAnalyzer, InstrumentationLevel


def simple_matmul():
    """Simple matrix multiplication"""
    a = mx.random.normal((512, 512))
    b = mx.random.normal((512, 512))
    c = mx.matmul(a, b)
    mx.eval(c)
    return c


def attention_like():
    """Attention-like operation"""
    q = mx.random.normal((1, 128, 8, 64))
    k = mx.random.normal((1, 128, 8, 64))
    v = mx.random.normal((1, 128, 8, 64))

    # Simplified attention
    scores = mx.matmul(q, mx.transpose(k, axes=[0, 1, 3, 2]))
    scale = 1.0 / (64 ** 0.5)
    scores = scores * scale
    attn = mx.softmax(scores, axis=-1)
    output = mx.matmul(attn, v)
    mx.eval(output)
    return output


def main():
    print("=" * 60)
    print("FlashMLX Profiler - Simple Example")
    print("=" * 60)

    # Example 1: Basic profiling
    print("\n[Example 1] Basic Profiling")
    with Profiler("simple_matmul", level=InstrumentationLevel.BASIC) as p:
        for _ in range(10):
            simple_matmul()

    # Analyze results
    analyzer = ProfileAnalyzer(str(p.output_file))
    analyzer.print_summary()

    # Example 2: Detailed profiling
    print("\n[Example 2] Detailed Profiling")
    with Profiler("attention_like", level=InstrumentationLevel.DETAILED) as p:
        for _ in range(5):
            attention_like()

    # Analyze results
    analyzer = ProfileAnalyzer(str(p.output_file))
    analyzer.print_summary()

    # Example 3: Manual regions
    print("\n[Example 3] Manual Regions")
    with Profiler("manual_regions") as p:
        with p.region("matmul_phase"):
            for _ in range(5):
                simple_matmul()

        with p.region("attention_phase"):
            for _ in range(3):
                attention_like()

    # Analyze results
    analyzer = ProfileAnalyzer(str(p.output_file))
    analyzer.print_summary()

    # Generate markdown report
    analyzer.generate_report("profile_report.md")


if __name__ == "__main__":
    main()
