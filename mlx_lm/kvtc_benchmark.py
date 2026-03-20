# Copyright © 2025 Apple Inc.

import argparse
import os
import sys
import tempfile
import time
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from .generate import generate_step, stream_generate
from .models.cache import KVTCPromptCache, make_prompt_cache, load_prompt_cache, save_prompt_cache
from .models.kvtc_codec import (
    KVTCCodecConfig,
    KVTCSharedCalibration,
    _to_numpy,
    fit_shared_calibration,
)
from .utils import load


def setup_arg_parser():
    parser = argparse.ArgumentParser(
        description="Benchmark plain vs KVTC prompt-cache serialization"
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
        "--prompt",
        required=True,
        help="Message to be processed by the model ('-' reads from stdin)",
    )
    parser.add_argument(
        "--cache-codecs",
        nargs="+",
        choices=["plain", "kvtc"],
        default=["plain", "kvtc"],
        help="Cache codecs to benchmark.",
    )
    parser.add_argument("--kvtc-rank", type=int, default=None)
    parser.add_argument("--kvtc-energy", type=float, default=0.995)
    parser.add_argument("--kvtc-bits", type=int, default=4)
    parser.add_argument("--kvtc-group-size", type=int, default=64)
    parser.add_argument("--kvtc-sample-limit", type=int, default=4096)
    return parser


@dataclass(frozen=True)
class CacheLayerStats:
    compressed: int
    skipped: int


class _NoEOSTokenizer:
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self.eos_token_ids = set()

    def __getattr__(self, name):
        return getattr(self._tokenizer, name)


def _split_prompt_tokens(prompt_tokens, continuation_tokens: int):
    prompt_tokens = list(prompt_tokens)
    if not prompt_tokens:
        raise ValueError("Prompt must contain at least one token")
    if len(prompt_tokens) == 1:
        return prompt_tokens, prompt_tokens[-1:]
    if continuation_tokens <= 0:
        return prompt_tokens, prompt_tokens[-1:]
    split = max(1, len(prompt_tokens) - continuation_tokens)
    if split >= len(prompt_tokens):
        split = len(prompt_tokens) - 1
    return prompt_tokens[:split], prompt_tokens[split:]


def _tokenize_prompt(tokenizer, prompt_text):
    if tokenizer.has_chat_template:
        messages = [{"role": "user", "content": prompt_text}]
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            continue_final_message=True,
        )
    return tokenizer.encode(prompt_text)


def _prefill_prompt(model, tokenizer, prompt_text, max_kv_size=None):
    prompt = _tokenize_prompt(tokenizer, prompt_text)
    return _prefill_prompt_tokens(model, prompt, max_kv_size), prompt


def _prefill_prompt_tokens(model, prompt_tokens, max_kv_size=None):
    cache = make_prompt_cache(model, max_kv_size)
    y = mx.array(prompt_tokens)
    for _ in generate_step(y, model, max_tokens=0, prompt_cache=cache):
        pass
    return cache


def _is_kvtc_supported(layer_cache):
    state = layer_cache.state
    if not isinstance(state, tuple) or len(state) != 2:
        return False
    keys, values = state
    return getattr(keys, "ndim", 0) == 4 and getattr(values, "ndim", 0) == 4


def _cache_layer_stats(cache):
    compressed = sum(1 for layer_cache in cache if _is_kvtc_supported(layer_cache))
    return CacheLayerStats(compressed=compressed, skipped=len(cache) - compressed)


def _fit_calibrations(cache, codec):
    calibrations = {}
    groups = {}
    for layer_cache in cache:
        if layer_cache.empty() or not _is_kvtc_supported(layer_cache):
            continue
        keys, values = layer_cache.state
        group_key = (keys.shape[-1], values.shape[-1])
        groups.setdefault(group_key, {"keys": [], "values": []})
        groups[group_key]["keys"].append(keys.reshape(-1, keys.shape[-1]))
        groups[group_key]["values"].append(values.reshape(-1, values.shape[-1]))

    for group_key, group in groups.items():
        calibrations[group_key] = fit_shared_calibration(
            group["keys"], group["values"], codec
        )
    return calibrations


def _layer_calibration(layer_cache, calibrations):
    if isinstance(calibrations, KVTCSharedCalibration):
        return calibrations
    keys, values = layer_cache.state
    return calibrations[(keys.shape[-1], values.shape[-1])]


def _benchmark_save_load(cache, codec, calibration=None, calibrations=None):
    calibration = calibration if calibration is not None else calibrations
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, f"cache_{codec}.safetensors")
        save_input = cache
        if codec == "kvtc":
            save_input = [
                (
                    KVTCPromptCache.from_cache(
                        c, calibration=_layer_calibration(c, calibration)
                    )
                    if _is_kvtc_supported(c)
                    else c
                )
                for c in cache
            ]
        t0 = time.perf_counter()
        save_prompt_cache(path, save_input)
        save_time = time.perf_counter() - t0
        file_size = os.path.getsize(path)

        t1 = time.perf_counter()
        loaded = load_prompt_cache(path)
        load_time = time.perf_counter() - t1

        decode_time = 0.0
        decoded_nbytes = 0
        if codec == "kvtc":
            t2 = time.perf_counter()
            decoded = [c.decompress() if isinstance(c, KVTCPromptCache) else c for c in loaded]
            decode_time = time.perf_counter() - t2
            decoded_nbytes = sum(c.nbytes for c in decoded)
        else:
            decoded_nbytes = sum(c.nbytes for c in loaded)

        return {
            "codec": codec,
            "save_time": save_time,
            "load_time": load_time,
            "decode_time": decode_time,
            "file_size": file_size,
            "cache_nbytes": sum(c.nbytes for c in cache),
            "decoded_nbytes": decoded_nbytes,
            "loaded": loaded,
        }


def _measure_reconstruction_error(original_cache, loaded_cache):
    rel_errors = []
    for original, loaded in zip(original_cache, loaded_cache):
        if not _is_kvtc_supported(original):
            continue
        decoded = loaded.decompress() if isinstance(loaded, KVTCPromptCache) else loaded
        orig_keys, orig_values = original.state
        dec_keys, dec_values = decoded.state
        for orig, dec in ((orig_keys, dec_keys), (orig_values, dec_values)):
            orig_np = _to_numpy(orig)
            dec_np = _to_numpy(dec)
            denom = max(1e-12, float(np.linalg.norm(orig_np)))
            rel_errors.append(float(np.linalg.norm(orig_np - dec_np) / denom))
    if not rel_errors:
        return 0.0
    return sum(rel_errors) / len(rel_errors)


def _benchmark_generation(model, tokenizer, prompt_tokens, prompt_cache, max_tokens=32):
    # Benchmark fixed-length generation so EOS does not skew throughput comparisons.
    tokenizer = _NoEOSTokenizer(tokenizer)
    tic = time.perf_counter()
    first_token_tic = None
    generated_tokens = 0
    for response in stream_generate(
        model,
        tokenizer,
        prompt_tokens,
        max_tokens=max_tokens,
        prompt_cache=prompt_cache,
    ):
        generated_tokens += 1
        if first_token_tic is None:
            first_token_tic = time.perf_counter()
    if generated_tokens == 0:
        raise RuntimeError("Generation produced no response")
    mx.synchronize()
    end_tic = time.perf_counter()
    ttft = (first_token_tic - tic) if first_token_tic is not None else 0.0
    return {
        "ttft": ttft,
        "prompt_tps": len(prompt_tokens) / max(ttft, 1e-12) if prompt_tokens else 0.0,
        "generation_tps": generated_tokens / max(end_tic - first_token_tic, 1e-12),
    }


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

    args.prompt = sys.stdin.read() if args.prompt == "-" else args.prompt
    cache, _ = _prefill_prompt(model, tokenizer, args.prompt, args.max_kv_size)
    layer_stats = _cache_layer_stats(cache)

    calibrations = None
    if "kvtc" in args.cache_codecs:
        codec = KVTCCodecConfig(
            energy=args.kvtc_energy,
            rank=args.kvtc_rank,
            bits=args.kvtc_bits,
            group_size=args.kvtc_group_size,
            sample_limit=args.kvtc_sample_limit,
        )
        calibrations = _fit_calibrations(cache, codec)

    rows = [_benchmark_save_load(cache, codec, calibrations) for codec in args.cache_codecs]

    print(f"Prompt tokens: {len(_tokenize_prompt(tokenizer, args.prompt))}")
    print(
        f"KVTC layers: compressed={layer_stats.compressed}, "
        f"skipped={layer_stats.skipped}"
    )
    for row in rows:
        cache_mib = row["cache_nbytes"] / (1024 * 1024)
        file_mib = row["file_size"] / (1024 * 1024)
        ratio = row["cache_nbytes"] / max(1, row["file_size"])
        print(
            f"{row['codec']}: "
            f"cache={cache_mib:.2f} MiB, "
            f"file={file_mib:.2f} MiB, "
            f"compression={ratio:.2f}x, "
            f"save={row['save_time']:.3f}s, "
            f"load={row['load_time']:.3f}s, "
            f"decode={row['decode_time']:.3f}s"
        )


if __name__ == "__main__":
    main()
