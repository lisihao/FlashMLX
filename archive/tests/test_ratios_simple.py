"""
Test different compression ratios
"""
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
import sys
sys.path.insert(0, '/Users/lisihao/FlashMLX/src')

from flashmlx.cache import (
    create_compaction_algorithm,
    create_compacted_cache_list,
    patch_attention_for_compacted_cache,
)
from mlx_lm.models.cache import make_prompt_cache

# Load model
model, tokenizer = load("/Users/lisihao/.omlx/models/Qwen3-1.7B-MLX-4bit")
patch_attention_for_compacted_cache(model, verbose=False)

prefix = "Quantum computing uses qubits that can exist in superposition states."
question = "\n\nQuestion: Explain quantum superposition.\nAnswer:"
full_prompt = prefix + question

# Baseline
sampler = make_sampler(temp=0.0)
baseline = generate(model, tokenizer, prompt=full_prompt, max_tokens=30, sampler=sampler, verbose=False)
print(f"Baseline: {baseline[:60]}...")
baseline_tokens = tokenizer.encode(baseline)

# Test ratios
for ratio in [2, 4]:
    cache = make_prompt_cache(model)
    y = mx.array([tokenizer.encode(prefix)])
    _ = model(y, cache=cache)
    
    T = cache[0].keys.shape[-2]
    t = T // ratio
    algo = create_compaction_algorithm()
    
    compacted_data = []
    for lc in cache:
        K, V = lc.keys, lc.values
        B, nh, _, hd = K.shape
        C1 = mx.zeros((B, nh, t, hd))
        beta = mx.zeros((B, nh, t))
        C2 = mx.zeros((B, nh, t, hd))
        
        for h in range(nh):
            c1, b, c2, _ = algo.compute_compacted_cache(K[0,h], V[0,h], K[0,h,-30:], t)
            C1[0,h] = c1
            beta[0,h] = b
            C2[0,h] = c2
        
        compacted_data.append((C1, beta, C2))
    
    cc = create_compacted_cache_list(compacted_data, T)
    out = generate(model, tokenizer, prompt=tokenizer.encode(question), prompt_cache=cc, max_tokens=30, sampler=sampler, verbose=False)
    
    tokens = tokenizer.encode(out)
    ml = min(len(baseline_tokens), len(tokens))
    ov = sum(1 for i in range(ml) if baseline_tokens[i]==tokens[i])/ml*100 if ml>0 else 0
    
    print(f"{ratio}x: {out[:60]}... | Overlap: {ov:.1f}%")
