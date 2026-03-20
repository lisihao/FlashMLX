# Copyright © 2024 Apple Inc.

import os
import tempfile
import unittest

import numpy as np
import mlx.core as mx

from mlx_lm.models.cache import (
    KVCache,
    KVTCPromptCache,
    materialize_prompt_cache,
    load_prompt_cache,
    save_prompt_cache,
)
from mlx_lm.kvtc_benchmark import (
    _benchmark_save_load,
    _measure_reconstruction_error,
    _split_prompt_tokens,
)
from mlx_lm.kvtc_metrics_benchmark import _load_prompt_text
from mlx_lm.models.kvtc_codec import KVTCCodecConfig, fit_shared_calibration, plan_bit_allocation


class TestKVTCPromptCache(unittest.TestCase):
    def test_round_trip(self):
        rng = np.random.default_rng(7)

        latent = rng.normal(size=(1, 4, 128, 8)).astype(np.float32)
        wk = rng.normal(size=(8, 32)).astype(np.float32)
        wv = rng.normal(size=(8, 32)).astype(np.float32)

        keys = np.einsum("bhtf,fd->bhtd", latent, wk)
        values = np.einsum("bhtf,fd->bhtd", latent, wv)
        keys += 0.01 * rng.normal(size=keys.shape).astype(np.float32)
        values += 0.01 * rng.normal(size=values.shape).astype(np.float32)

        cache = KVCache()
        cache.update_and_fetch(mx.array(keys), mx.array(values))
        orig_keys, orig_values = cache.state

        compressed = KVTCPromptCache.from_cache(
            cache,
            rank=8,
            bits=4,
            group_size=16,
            sample_limit=256,
        )

        self.assertLess(compressed.nbytes, cache.nbytes)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "prompt_cache.safetensors")
            save_prompt_cache(path, [compressed])
            loaded = load_prompt_cache(path)[0]

            self.assertIsInstance(loaded, KVTCPromptCache)
            self.assertGreater(loaded.nbytes, 0)

            materialized = loaded.decompress()
            self.assertEqual(materialized.size(), cache.size())
            mat_keys, mat_values = materialized.state

            key_rel_err = (
                np.linalg.norm(np.asarray(mat_keys) - np.asarray(orig_keys))
                / np.linalg.norm(np.asarray(orig_keys))
            )
            value_rel_err = (
                np.linalg.norm(np.asarray(mat_values) - np.asarray(orig_values))
                / np.linalg.norm(np.asarray(orig_values))
            )
            self.assertLess(key_rel_err, 1.1)
            self.assertLess(value_rel_err, 1.1)

            tok = mx.array(rng.normal(size=(1, 4, 1, 32)).astype(np.float32))
            ref_k, ref_v = cache.update_and_fetch(tok, tok)
            got_k, got_v = loaded.update_and_fetch(tok, tok)

            self.assertEqual(ref_k.shape, got_k.shape)
            self.assertEqual(ref_v.shape, got_v.shape)

    def test_shared_calibration(self):
        rng = np.random.default_rng(13)
        codec = KVTCCodecConfig(rank=8, bits=4, group_size=16, sample_limit=256)

        caches = []
        key_mats = []
        value_mats = []
        for scale in (1.0, 1.15):
            latent = rng.normal(size=(1, 4, 96, 8)).astype(np.float32)
            wk = (scale * rng.normal(size=(8, 32))).astype(np.float32)
            wv = (scale * rng.normal(size=(8, 32))).astype(np.float32)
            keys = np.einsum("bhtf,fd->bhtd", latent, wk)
            values = np.einsum("bhtf,fd->bhtd", latent, wv)
            cache = KVCache()
            cache.update_and_fetch(mx.array(keys), mx.array(values))
            caches.append(cache)
            key_mats.append(cache.state[0].reshape(-1, cache.state[0].shape[-1]))
            value_mats.append(cache.state[1].reshape(-1, cache.state[1].shape[-1]))

        calibration = fit_shared_calibration(key_mats, value_mats, codec)
        wrapped = [
            KVTCPromptCache.from_cache(cache, calibration=calibration)
            for cache in caches
        ]

        self.assertEqual(wrapped[0]._shared_calibration_id, wrapped[1]._shared_calibration_id)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "shared_prompt_cache.safetensors")
            save_prompt_cache(path, wrapped)
            loaded = load_prompt_cache(path)

            self.assertIsInstance(loaded[0], KVTCPromptCache)
            self.assertIsInstance(loaded[1], KVTCPromptCache)
            self.assertEqual(loaded[0]._shared_calibration_id, loaded[1]._shared_calibration_id)
            self.assertIs(loaded[0]._shared_calibration, loaded[1]._shared_calibration)

    def test_lazy_quantized_attrs_do_not_decode_plain_cache(self):
        rng = np.random.default_rng(17)
        latent = rng.normal(size=(1, 2, 64, 8)).astype(np.float32)
        wk = rng.normal(size=(8, 16)).astype(np.float32)
        wv = rng.normal(size=(8, 16)).astype(np.float32)
        keys = np.einsum("bhtf,fd->bhtd", latent, wk)
        values = np.einsum("bhtf,fd->bhtd", latent, wv)

        cache = KVCache()
        cache.update_and_fetch(mx.array(keys), mx.array(values))

        compressed = KVTCPromptCache.from_cache(
            cache,
            rank=8,
            bits=4,
            group_size=16,
            sample_limit=128,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "prompt_cache.safetensors")
            save_prompt_cache(path, [compressed])
            loaded = KVTCPromptCache.from_state(
                compressed.state, compressed.meta_state
            )

            self.assertFalse(hasattr(loaded, "bits"))
            self.assertFalse(hasattr(loaded, "group_size"))
            self.assertIsNone(loaded._decoded)

    def test_benchmark_helper(self):
        rng = np.random.default_rng(21)
        latent = rng.normal(size=(1, 2, 64, 8)).astype(np.float32)
        wk = rng.normal(size=(8, 16)).astype(np.float32)
        wv = rng.normal(size=(8, 16)).astype(np.float32)
        keys = np.einsum("bhtf,fd->bhtd", latent, wk)
        values = np.einsum("bhtf,fd->bhtd", latent, wv)

        cache = KVCache()
        cache.update_and_fetch(mx.array(keys), mx.array(values))

        codec = KVTCCodecConfig(rank=8, bits=4, group_size=8, sample_limit=128)
        calibration = fit_shared_calibration(
            [cache.state[0].reshape(-1, cache.state[0].shape[-1])],
            [cache.state[1].reshape(-1, cache.state[1].shape[-1])],
            codec,
        )

        plain = _benchmark_save_load([cache], "plain")
        kvtc = _benchmark_save_load([cache], "kvtc", calibration=calibration)

        self.assertGreater(plain["file_size"], 0)
        self.assertGreater(kvtc["file_size"], 0)
        self.assertGreaterEqual(kvtc["cache_nbytes"], kvtc["decoded_nbytes"])
        self.assertGreaterEqual(plain["cache_nbytes"], plain["decoded_nbytes"])

    def test_metrics_helpers(self):
        prefix, continuation = _split_prompt_tokens([1, 2, 3, 4], 2)
        self.assertEqual(prefix, [1, 2])
        self.assertEqual(continuation, [3, 4])

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "prompt.txt")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("hello world")
            self.assertEqual(_load_prompt_text(prompt_file=path), "hello world")

        rng = np.random.default_rng(31)
        latent = rng.normal(size=(1, 2, 32, 8)).astype(np.float32)
        wk = rng.normal(size=(8, 16)).astype(np.float32)
        wv = rng.normal(size=(8, 16)).astype(np.float32)
        keys = np.einsum("bhtf,fd->bhtd", latent, wk)
        values = np.einsum("bhtf,fd->bhtd", latent, wv)

        cache = KVCache()
        cache.update_and_fetch(mx.array(keys), mx.array(values))

        codec = KVTCCodecConfig(rank=8, bits=4, group_size=8, sample_limit=128)
        calibration = fit_shared_calibration(
            [cache.state[0].reshape(-1, cache.state[0].shape[-1])],
            [cache.state[1].reshape(-1, cache.state[1].shape[-1])],
            codec,
        )

        loaded = _benchmark_save_load([cache], "kvtc", calibration=calibration)["loaded"]
        error = _measure_reconstruction_error([cache], loaded)

        self.assertGreaterEqual(error, 0.0)
        self.assertLess(error, 1.1)

    def test_materialize_prompt_cache(self):
        rng = np.random.default_rng(47)
        latent = rng.normal(size=(1, 2, 64, 8)).astype(np.float32)
        wk = rng.normal(size=(8, 16)).astype(np.float32)
        wv = rng.normal(size=(8, 16)).astype(np.float32)
        keys = np.einsum("bhtf,fd->bhtd", latent, wk)
        values = np.einsum("bhtf,fd->bhtd", latent, wv)

        cache = KVCache()
        cache.update_and_fetch(mx.array(keys), mx.array(values))

        compressed = KVTCPromptCache.from_cache(
            cache,
            rank=8,
            bits=4,
            group_size=16,
            sample_limit=128,
        )
        materialized = materialize_prompt_cache([compressed])

        self.assertIsInstance(materialized[0], KVCache)
        self.assertNotIsInstance(materialized[0], KVTCPromptCache)

    def test_plan_prefers_bits_over_zero(self):
        coeffs = np.ones((2, 32), dtype=np.float32)
        config = KVTCCodecConfig(
            bits=4,
            allowed_bits=(0, 2, 4),
            zero_bit_energy_fraction=0.001,
        )
        plan = plan_bit_allocation(coeffs, config)
        self.assertGreater(len(plan), 0)
        self.assertNotIn(0, plan[:, 2])
