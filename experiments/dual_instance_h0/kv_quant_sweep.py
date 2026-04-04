#!/usr/bin/env python3
"""
Quick KV quant sweep: test all supported bit widths and group sizes.
Also test with diverse prompt vs repetitive prompt.
"""

from __future__ import annotations
import sys, argparse

sys.path.insert(0, "/Users/lisihao/FlashMLX/mlx-lm-source")
sys.path.insert(0, "/Users/lisihao/FlashMLX/src")

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import KVCache, make_prompt_cache
from mlx_lm.sample_utils import make_sampler

GREEDY = make_sampler(temp=0.0)

DIVERSE_PROMPT = """The history of artificial intelligence began in antiquity, with myths, stories and rumors of artificial beings endowed with intelligence or consciousness by master craftsmen. The seeds of modern AI were planted by philosophers who attempted to describe the process of human thinking as the mechanical manipulation of symbols. This work culminated in the invention of the programmable digital computer in the 1940s, a machine based on the abstract essence of mathematical reasoning. This device and the ideas behind it inspired a handful of scientists to begin seriously discussing the possibility of building an electronic brain.

The field of AI research was founded at a workshop held on the campus of Dartmouth College, USA during the summer of 1956. Those who attended would become the leaders of AI research for decades. Many of them predicted that a machine as intelligent as a human being would exist in no more than a generation, and they were given millions of dollars to make this vision come true. Eventually, it became obvious that commercial developers and researchers had grossly underestimated the difficulty of the project.

In 1973, in response to the criticism from James Lighthill and ongoing pressure from congress, the U.S. and British Governments stopped funding undirected research into artificial intelligence, and the difficult years that followed became known as an "AI winter". Seven years later, a visionary initiative by the Japanese Government inspired governments and industry to provide AI with billions of dollars, but by the late 1980s the investors became disillusioned and withdrew funding again.

Investment and interest in AI boomed in the first decades of the 21st century when machine learning was successfully applied to many problems in academia and industry due to new methods, the application of powerful computer hardware, and the collection of immense data sets."""


def prefill(model, tokenizer, prompt):
    cache = make_prompt_cache(model)
    tokens = mx.array(tokenizer.encode(prompt))
    out = model(tokens.reshape(1, -1), cache=cache)
    mx.eval(out)
    return cache, len(tokenizer.encode(prompt))


def gen_from_cache(model, tokenizer, cache, question, max_tokens=50):
    q = mx.array(tokenizer.encode(question))
    result = []
    for tok, _ in generate_step(q, model, max_tokens=max_tokens,
                                 sampler=GREEDY, prompt_cache=cache):
        result.append(tok)
    return result, tokenizer.decode(result)


def clone_quant(exact_cache, n_layers, bits, gs):
    new_cache = [KVCache() for _ in range(n_layers)]
    for i in range(n_layers):
        k, v = exact_cache[i].state
        mx.eval(k, v)
        qk = mx.quantize(k, group_size=gs, bits=bits)
        qv = mx.quantize(v, group_size=gs, bits=bits)
        kd = mx.dequantize(*qk, group_size=gs, bits=bits)
        vd = mx.dequantize(*qv, group_size=gs, bits=bits)
        mx.eval(kd, vd)
        new_cache[i].state = (kd, vd)
    mx.eval([c.keys for c in new_cache] + [c.values for c in new_cache])
    return new_cache


def exact_clone(exact_cache, n_layers):
    new_cache = [KVCache() for _ in range(n_layers)]
    for i in range(n_layers):
        k, v = exact_cache[i].state
        mx.eval(k, v)
        new_cache[i].state = (k, v)
    mx.eval([c.keys for c in new_cache] + [c.values for c in new_cache])
    return new_cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    model, tokenizer = load(args.model)
    inner = model.model if hasattr(model, 'model') else model
    n_layers = len(inner.layers)
    head_dim = inner.layers[0].self_attn.rope.dims

    print(f"Model: {args.model.split('/')[-1]}, layers={n_layers}, head_dim={head_dim}")

    question = "\nQ: Summarize the main topic.\nA:"

    for prompt_name, prompt_text in [
        ("repetitive (fox)", "The quick brown fox jumps over the lazy dog. " * 50),
        ("diverse (AI history)", DIVERSE_PROMPT),
    ]:
        tokens_len = len(tokenizer.encode(prompt_text))
        print(f"\n{'═' * 70}")
        print(f"Prompt: {prompt_name} ({tokens_len} tokens)")
        print(f"{'═' * 70}")

        exact_cache, _ = prefill(model, tokenizer, prompt_text)

        # Baseline
        base_c = exact_clone(exact_cache, n_layers)
        base_tok, base_text = gen_from_cache(model, tokenizer, base_c, question)
        print(f"Baseline: {base_text[:100]}")

        print(f"\n  {'bits':>4} {'gs':>4} │ {'match':>8} │ {'KV MB':>8} │ {'ratio':>5}")
        print(f"  {'─' * 4} {'─' * 4}─┼{'─' * 8}─┼{'─' * 8}─┼{'─' * 5}")

        for bits in [8, 6, 5, 4, 3, 2]:
            for gs in [32, 64, 128]:
                if gs > head_dim:
                    continue
                try:
                    qc = clone_quant(exact_cache, n_layers, bits, gs)
                    gen_tok, _ = gen_from_cache(model, tokenizer, qc, question)
                    match = sum(1 for a, b in zip(base_tok, gen_tok) if a == b)
                    total = min(len(base_tok), len(gen_tok))
                    exact = base_tok == gen_tok
                    label = "EXACT" if exact else f"{match}/{total}"

                    # Size
                    nbytes = 0
                    for i in range(n_layers):
                        k, v = exact_cache[i].state
                        qk = mx.quantize(k, group_size=gs, bits=bits)
                        qv = mx.quantize(v, group_size=gs, bits=bits)
                        nbytes += sum(x.nbytes for x in qk) + sum(x.nbytes for x in qv)
                    exact_bytes = sum(c.keys.nbytes + c.values.nbytes for c in exact_cache)
                    ratio = exact_bytes / nbytes

                    print(f"  int{bits:1d} {gs:>4} │ {label:>8} │ {nbytes/1024/1024:>7.1f}M │ {ratio:>4.1f}×")
                except Exception as e:
                    print(f"  int{bits:1d} {gs:>4} │ ERROR: {e}")

    print()


if __name__ == "__main__":
    main()
