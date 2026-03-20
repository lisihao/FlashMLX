# Copyright © 2024 Apple Inc.

import argparse
import json
import sys
import time

import mlx.core as mx

from .generate import generate_step
from .models.cache import KVTCPromptCache, make_prompt_cache, save_prompt_cache
from .models.kvtc_codec import KVTCCodecConfig, fit_shared_calibration
from .utils import load

DEFAULT_QUANTIZED_KV_START = 5000


def setup_arg_parser():
    """Set up and return the argument parser."""
    parser = argparse.ArgumentParser(
        description="Cache the state of a prompt to be reused with mlx_lm.generate"
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
    parser.add_argument(
        "--prompt-cache-file",
        help="The file to save the prompt cache in",
        required=True,
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Message to be processed by the model ('-' reads from stdin)",
    )
    parser.add_argument(
        "--kv-bits",
        type=int,
        help="Number of bits for KV cache quantization. "
        "Defaults to no quantization.",
        default=None,
    )
    parser.add_argument(
        "--kv-group-size",
        type=int,
        help="Group size for KV cache quantization.",
        default=64,
    )
    parser.add_argument(
        "--quantized-kv-start",
        help="When --kv-bits is set, start quantizing the KV cache "
        "from this step onwards.",
        type=int,
        default=DEFAULT_QUANTIZED_KV_START,
    )
    parser.add_argument(
        "--cache-codec",
        choices=["plain", "kvtc"],
        default="plain",
        help="How to serialize the prompt cache to disk.",
    )
    parser.add_argument(
        "--kvtc-rank",
        type=int,
        default=None,
        help="Override the PCA rank used by KVTC.",
    )
    parser.add_argument(
        "--kvtc-energy",
        type=float,
        default=0.995,
        help="Explained-variance target when KVTC chooses the rank automatically.",
    )
    parser.add_argument(
        "--kvtc-bits",
        type=int,
        default=4,
        help="Bit-width used for KVTC coefficient quantization.",
    )
    parser.add_argument(
        "--kvtc-group-size",
        type=int,
        default=64,
        help="Group size used for KVTC coefficient quantization.",
    )
    parser.add_argument(
        "--kvtc-sample-limit",
        type=int,
        default=4096,
        help="Maximum number of rows used to fit the KVTC transform.",
    )
    return parser


def main():
    parser = setup_arg_parser()
    args = parser.parse_args()

    # Building tokenizer_config
    tokenizer_config = {"trust_remote_code": True if args.trust_remote_code else None}
    if args.eos_token is not None:
        tokenizer_config["eos_token"] = args.eos_token

    model, tokenizer = load(
        args.model,
        adapter_path=args.adapter_path,
        tokenizer_config=tokenizer_config,
    )

    args.prompt = sys.stdin.read() if args.prompt == "-" else args.prompt

    if tokenizer.has_chat_template:
        messages = [{"role": "user", "content": args.prompt}]
        prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            continue_final_message=True,
        )

    else:
        prompt = tokenizer.encode(args.prompt)

    cache = make_prompt_cache(model, args.max_kv_size)
    y = mx.array(prompt)

    # Process the prompt
    start = time.time()
    max_msg_len = 0

    def callback(processed, total_tokens):
        current = time.time()
        speed = processed / (current - start)
        msg = f"\rProcessed {processed:6d} tokens ({speed:6.2f} tok/s)"
        nonlocal max_msg_len
        max_msg_len = max(max_msg_len, len(msg))
        print(msg + " " * (max_msg_len - len(msg)), end="", flush=True)

    for _ in generate_step(
        y,
        model,
        max_tokens=0,
        prompt_cache=cache,
        kv_bits=args.kv_bits,
        kv_group_size=args.kv_group_size,
        quantized_kv_start=args.quantized_kv_start,
        prompt_progress_callback=callback,
    ):
        pass

    print()
    print(f"Peak memory: {mx.get_peak_memory() / 1e9:.3f} GB")

    print("Saving...")
    if args.cache_codec == "kvtc":
        codec = KVTCCodecConfig(
            energy=args.kvtc_energy,
            rank=args.kvtc_rank,
            bits=args.kvtc_bits,
            group_size=args.kvtc_group_size,
            sample_limit=args.kvtc_sample_limit,
        )
        key_matrices = []
        value_matrices = []
        for layer_cache in cache:
            if layer_cache.empty():
                continue
            keys, values = layer_cache.state
            key_matrices.append(keys.reshape(-1, keys.shape[-1]))
            value_matrices.append(values.reshape(-1, values.shape[-1]))
        calibration = fit_shared_calibration(key_matrices, value_matrices, codec)
        cache = [
            KVTCPromptCache.from_cache(
                c,
                calibration=calibration,
            )
            for c in cache
        ]
    metadata = {}
    metadata["model"] = args.model
    metadata["tokenizer_config"] = json.dumps(tokenizer_config)
    save_prompt_cache(args.prompt_cache_file, cache, metadata)


if __name__ == "__main__":
    print(
        "Calling `python -m mlx_lm.cache_prompt...` directly is deprecated."
        " Use `mlx_lm.cache_prompt...` or `python -m mlx_lm cache_prompt ...` instead."
    )
    main()
