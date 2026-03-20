# Copyright © 2025 Apple Inc.

import argparse
import sys
from pathlib import Path

from .kvtc_benchmark import (
    _benchmark_generation,
    _benchmark_save_load,
    _cache_layer_stats,
    _fit_calibrations,
    _measure_reconstruction_error,
    _prefill_prompt_tokens,
    _split_prompt_tokens,
    _tokenize_prompt,
)
from .models.kvtc_codec import KVTCCodecConfig
from .utils import load


def setup_arg_parser():
    parser = argparse.ArgumentParser(
        description="Benchmark KVTC compression, reconstruction, and generation curves"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlx_model",
        help="The path to the local model directory or Hugging Face repo.",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        help="Optional path for the trained adapter weights and config.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trusting remote code for tokenizer",
    )
    parser.add_argument(
        "--eos-token",
        type=str,
        default=None,
        help="End of sequence token for tokenizer",
    )
    parser.add_argument(
        "--max-kv-size",
        type=int,
        default=None,
        help="Set the maximum key-value cache size",
    )
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument(
        "--prompt",
        help="Message to be processed by the model ('-' reads from stdin)",
    )
    prompt_group.add_argument(
        "--prompt-file",
        type=str,
        help="Path to a UTF-8 text file containing a real long prompt.",
    )
    parser.add_argument(
        "--cache-codecs",
        nargs="+",
        choices=["plain", "kvtc"],
        default=["plain", "kvtc"],
        help="Cache codecs to benchmark.",
    )
    parser.add_argument(
        "--lengths",
        nargs="+",
        type=int,
        default=[4096, 8192, 16384],
        help="Prompt prefix lengths to benchmark.",
    )
    parser.add_argument("--kvtc-rank", type=int, default=None)
    parser.add_argument("--kvtc-energy", type=float, default=0.995)
    parser.add_argument("--kvtc-bits", type=int, default=4)
    parser.add_argument("--kvtc-group-size", type=int, default=64)
    parser.add_argument("--kvtc-sample-limit", type=int, default=4096)
    parser.add_argument(
        "--continuation-tokens",
        type=int,
        default=64,
        help="Number of trailing prompt tokens kept out of the cache for TTFT / throughput measurement.",
    )
    parser.add_argument(
        "--generation-tokens",
        type=int,
        default=128,
        help="Number of tokens to generate after the cached prefix.",
    )
    return parser


def _load_prompt_text(prompt=None, prompt_file=None):
    if prompt_file is not None:
        return Path(prompt_file).read_text(encoding="utf-8")
    if prompt == "-":
        return sys.stdin.read()
    return prompt or ""


def _normalize_lengths(lengths, total_tokens):
    unique_lengths = []
    seen = set()
    for length in lengths:
        if length in seen:
            continue
        if length <= 0:
            raise ValueError("Prompt lengths must be positive integers")
        seen.add(length)
        if length > total_tokens:
            continue
        unique_lengths.append(length)
    if not unique_lengths:
        raise ValueError(
            f"No requested lengths fit into the available prompt tokens ({total_tokens})"
        )
    return unique_lengths


def main():
    parser = setup_arg_parser()
    args = parser.parse_args()

    tokenizer_config = {"trust_remote_code": True if args.trust_remote_code else None}
    if args.eos_token is not None:
        tokenizer_config["eos_token"] = args.eos_token

    model, tokenizer = load(
        args.model,
        adapter_path=args.adapter_path,
        tokenizer_config=tokenizer_config,
    )

    prompt_text = _load_prompt_text(args.prompt, args.prompt_file)
    prompt_tokens = _tokenize_prompt(tokenizer, prompt_text)
    lengths = _normalize_lengths(args.lengths, len(prompt_tokens))

    codec_config = None
    if "kvtc" in args.cache_codecs:
        codec_config = KVTCCodecConfig(
            energy=args.kvtc_energy,
            rank=args.kvtc_rank,
            bits=args.kvtc_bits,
            group_size=args.kvtc_group_size,
            sample_limit=args.kvtc_sample_limit,
        )

    print(
        f"Prompt source: {'file=' + args.prompt_file if args.prompt_file else 'stdin' if args.prompt == '-' else 'inline'}"
    )
    print(f"Prompt tokens available: {len(prompt_tokens)}")
    print(f"Requested lengths: {', '.join(str(length) for length in args.lengths)}")
    print(f"Effective lengths: {', '.join(str(length) for length in lengths)}")

    for length in lengths:
        window_tokens = prompt_tokens[:length]
        cache_tokens, continuation_tokens = _split_prompt_tokens(
            window_tokens, args.continuation_tokens
        )
        cache = _prefill_prompt_tokens(model, cache_tokens, args.max_kv_size)
        layer_stats = _cache_layer_stats(cache)

        calibrations = None
        if codec_config is not None:
            calibrations = _fit_calibrations(cache, codec_config)

        rows = [
            _benchmark_save_load(cache, codec, calibration=calibrations)
            for codec in args.cache_codecs
        ]

        print(
            f"length={length} "
            f"cache_tokens={len(cache_tokens)} "
            f"continuation_tokens={len(continuation_tokens)} "
            f"compressed_layers={layer_stats.compressed} "
            f"skipped_layers={layer_stats.skipped}"
        )

        print("save_load_curve")
        for row in rows:
            cache_mib = row["cache_nbytes"] / (1024 * 1024)
            file_mib = row["file_size"] / (1024 * 1024)
            ratio = row["cache_nbytes"] / max(1, row["file_size"])
            print(
                f"length={length} codec={row['codec']} "
                f"cache_mib={cache_mib:.2f} file_mib={file_mib:.2f} "
                f"compression={ratio:.2f}x "
                f"save_s={row['save_time']:.3f} load_s={row['load_time']:.3f} "
                f"decode_s={row['decode_time']:.3f}"
            )

        print("generation_curve")
        for row in rows:
            recon_error = (
                0.0
                if row["codec"] == "plain"
                else _measure_reconstruction_error(cache, row["loaded"])
            )
            gen_stats = _benchmark_generation(
                model,
                tokenizer,
                continuation_tokens,
                row["loaded"],
                max_tokens=args.generation_tokens,
            )
            print(
                f"length={length} codec={row['codec']} "
                f"recon_rel_l2={recon_error:.6f} "
                f"ttft_s={gen_stats['ttft']:.3f} "
                f"generation_tps={gen_stats['generation_tps']:.3f} "
                f"gen_tokens={args.generation_tokens}"
            )


if __name__ == "__main__":
    main()
