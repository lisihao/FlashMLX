#!/usr/bin/env python3
"""
Deep profiling with Python cProfile - 30B A3B Model
"""

import cProfile
import pstats
import io
from pathlib import Path
from mlx_lm import load, stream_generate

MODEL_PATH = Path.home() / "models" / "qwen3.5-35b-mlx"
OUTPUT_DIR = Path(__file__).parent.parent / "profiling_data"
OUTPUT_DIR.mkdir(exist_ok=True)

def run_generation():
    """Run generation for profiling"""
    print("Loading model...")
    model, tokenizer = load(str(MODEL_PATH))

    prompt = "The quick brown fox jumps over the lazy dog. " * 50  # ~250 tokens

    print("Generating tokens...")
    count = 0
    for response in stream_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=50,
    ):
        count += 1

    print(f"Generated {count} tokens")

def main():
    print("=" * 80)
    print("Python cProfile - Deep Function Analysis - 30B A3B Model")
    print("=" * 80)
    print()

    # Create profiler
    profiler = cProfile.Profile()

    # Run with profiling
    print("Running with cProfile...")
    profiler.enable()
    run_generation()
    profiler.disable()

    print()
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print()

    # Create stats
    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats('cumulative')

    # Print top functions by cumulative time
    print("Top 30 functions by cumulative time:")
    print()
    stats.print_stats(30)

    # Save detailed stats
    stats_file = OUTPUT_DIR / "cprofile_stats_35b.txt"
    with open(stats_file, 'w') as f:
        stats_stream = pstats.Stats(profiler, stream=f)
        stats_stream.strip_dirs()
        stats_stream.sort_stats('cumulative')
        stats_stream.print_stats()

    print()
    print(f"\n✅ Full stats saved to: {stats_file}")

    # Also save in a more parseable format
    import json
    stats_dict = {}
    for func, (cc, nc, tt, ct, callers) in stats.stats.items():
        key = f"{func[0]}:{func[1]}:{func[2]}"
        stats_dict[key] = {
            "calls": nc,
            "tottime": tt,
            "cumtime": ct,
            "percall_tot": tt/nc if nc > 0 else 0,
            "percall_cum": ct/nc if nc > 0 else 0,
        }

    json_file = OUTPUT_DIR / "cprofile_stats_35b.json"
    with open(json_file, 'w') as f:
        json.dump(stats_dict, f, indent=2)

    print(f"✅ JSON stats saved to: {json_file}")

if __name__ == "__main__":
    main()
