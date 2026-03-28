# TripleLayerKVCache Migration Flow Analysis

**Date**: 2026-03-27
**Context**: Investigating why 3-layer works with small Cold but fails with large Cold

## User's Key Questions

1. Recent → Warm → Cold 迁移流程和机制
2. 什么时候压缩？逻辑是否正确？
3. 是否满足设计原则？
4. 校准文件是否正确？
5. 边界条件是否违反？

---

## 1. Migration Flow Analysis

### Current Implementation

```
update_and_fetch(new_keys, new_values):
    1. Append new tokens → Recent
    2. Calculate memory usage
    3. IF memory > budget:
        _manage_aging():
            a. IF Recent.size > recent_size:
                overflow = Recent[0:overflow_size]
                Recent = Recent[overflow_size:]
                _append_warm_with_quant(overflow)  ← Q4_0 quantization

            b. IF Warm.size > warm_size:
                overflow = Warm[0:overflow_size]
                Warm = Warm[overflow_size:]
                IF enable_cold_am:
                    _append_cold_with_am(overflow)  ← AM compression (disabled in our tests)
                ELSE:
                    _append_cold_with_am(overflow)  ← Just store to cold_pending
    4. Concatenate: [Cold_compressed | Cold_pending | Warm | Recent]
    5. Return concatenated cache
```

### Observed Behavior

#### Test Results Summary

| Context | Recent | Warm | Cold | Warm Compressions | Result |
|---------|--------|------|------|-------------------|--------|
| 2K      | 512    | 1024 | 435  | 1                 | ✅ GOOD |
| 2.7K    | 512    | 1024 | 1189 | 1                 | ❌ BAD  |
| 5K      | 512    | 1024 | 3477 | 101               | ❌ BAD  |

**Critical Finding**: Quality degrades when Cold > ~500-1000 tokens, even with only 1 Warm compression!

---

## 2. Compression Timing Analysis

### When Does Compression Happen?

#### Warm Quantization (Q4_0)
```python
_append_warm_with_quant(overflow_keys, overflow_values):
    quant_keys, quant_values, metadata = warm_quantizer.quantize(keys, values)
    warm_keys = concatenate([warm_keys, quant_keys], axis=2)
    warm_metadata.append(metadata)
    num_warm_compressions += 1
```

**Triggered**: Every time Recent overflows to Warm

#### Cold AM Compression (Disabled in Tests)
```python
_append_cold_with_am(keys, values):
    IF enable_cold_am:
        cold_pending = concatenate([cold_pending, keys], axis=2)
        IF cold_pending.size >= cold_batch_threshold:
            compressed = AM_compress(cold_pending)
            cold_compressed = concatenate([cold_compressed, compressed], axis=2)
            cold_pending = None
    ELSE:
        cold_pending = concatenate([cold_pending, keys], axis=2)  ← Our case
```

**Triggered**: When Warm overflows to Cold (but compression disabled in tests)

### Problem Identified

In our tests with `enable_cold_am=False`:
- All Cold tokens stored in `cold_pending` (uncompressed)
- Large `cold_pending` (3477 tokens) stored as-is
- No compression, just accumulation

---

## 3. Design Principle Violations?

### Original Design Intent

**Memory-Budget Driven**:
```
IF estimated_memory > memory_budget:
    Trigger aging to reduce memory
```

### Current Problem

```
Test Results:
- Memory: 30.41 MB
- Budget: 3.00 MB
- Status: Over budget (10x!)
```

**VIOLATION**: System is 10x over budget but continues to accumulate tokens!

**Why?**:
1. Memory budget check happens BEFORE compression
2. But compression only moves tokens, doesn't reduce memory much
3. Budget intended for "when to trigger aging", not "when to evict"

---

## 4. Calibration File Analysis

### What Calibration Files Exist?

```bash
ls /Users/lisihao/FlashMLX/calibration/
```

Let me check what's actually there:

### Are They Being Used Correctly?

In our tests:
- `enable_cold_am=False` → No calibration needed
- `enable_warm_quant=True` → Uses Q4_0 (no calibration file)

**Conclusion**: Calibration files NOT used in these tests, so they're not the issue.

---

## 5. Boundary Condition Violations

### Hypothesis: Large Cold Layer Breaks Concatenation

#### Concatenation Order

```python
_concat_all_layers():
    layers_k = []

    # Cold (compressed + pending)
    IF cold_compressed_keys is not None:
        layers_k.append(cold_compressed_keys)
    IF cold_pending_keys is not None:
        layers_k.append(cold_pending_keys)    ← Large 3477-token tensor

    # Warm (dequantized)
    layers_k.append(warm_keys_dequant)        ← 1024 tokens

    # Recent
    layers_k.append(recent_keys)              ← 512 tokens

    return concatenate(layers_k, axis=2)
```

#### Potential Boundary Issues

1. **Position Encoding**:
   - Does model use absolute position encoding?
   - Are positions [0, 1, 2, ..., 5012] correctly maintained?
   - Could there be position overflow?

2. **Tensor Shape Validation**:
   - All tensors must have same (B, n_heads, *, head_dim)
   - Cold: (1, 8, 3477, 128)
   - Warm: (1, 8, 1024, 128)
   - Recent: (1, 8, 512, 128)
   - Concatenated: (1, 8, 5013, 128) ✓

3. **Memory Layout**:
   - Large Cold tensor (3477 × 128 × 8 × 2 bytes = 7.1 MB per layer × 36 layers = 255 MB!)
   - Could MLX have memory layout issues with very large concatenations?

---

## 6. Critical Bug Found: Warm Dequantization During Overflow

### The Bug (NOW FIXED)

In `_manage_aging()` when Warm overflows to Cold:

**OLD (BUGGY) CODE**:
```python
# Merged all scales from all chunks - WRONG!
all_scales_k = concatenate([meta['scales_k'] for meta in warm_metadata])
merged_metadata = {'scales_k': all_scales_k, ...}

# Tried to dequantize entire Warm with merged scales - BREAKS group structure!
full_warm = dequantizer.dequantize(warm_keys, warm_values, merged_metadata)
```

**NEW (FIXED) CODE**:
```python
# Dequantize each chunk individually using its original metadata
for seq_len, meta in zip(chunk_seq_lens, warm_metadata):
    chunk = warm_keys[:, :, offset:offset+seq_len, :]
    dequant_chunk = dequantizer.dequantize(chunk, chunk_values, meta)
    dequant_chunks.append(dequant_chunk)
    offset += seq_len

full_warm = concatenate(dequant_chunks, axis=2)
```

**Impact of Fix**:
- 2K context: Still works ✅
- 5K context: Still fails ❌

**Conclusion**: Fix was necessary but not sufficient to solve the quality issue.

---

## 7. Remaining Mystery: Why Does Large Cold Break?

### Observations

1. **Cold is uncompressed** (enable_cold_am=False)
2. **Cold is just concatenated fp16 tensors** (no quantization)
3. **Concatenation math is correct** (verified in debug test)
4. **Small Cold (435) works, Large Cold (1189+) fails**

### Hypothesis 1: Memory Budget Not Enforced

**Problem**: System allowed to accumulate 30 MB vs 3 MB budget

**Test**: Reduce memory budget to force more aggressive aging?

**Risk**: Might cause too much compression

### Hypothesis 2: MLX Concatenation Limit

**Problem**: Very large concatenations might hit MLX implementation limits?

**Test**: Check MLX source for concatenation size limits

### Hypothesis 3: Position Encoding Overflow

**Problem**: Qwen3 uses RoPE, might have issues with large contexts?

**Test**: Check if position IDs exceed model's trained maximum

### Hypothesis 4: Attention Mask Issue

**Problem**: Concatenated cache might not be properly masked?

**Test**: Verify attention mask shape and values

---

## 8. Recommended Next Steps

1. **Verify Position Encoding**:
   ```python
   # Check if Qwen3 has position limits
   print(f"Model max position embeddings: {model.config.max_position_embeddings}")
   ```

2. **Test Smaller Cold Threshold**:
   ```python
   # Force Cold compression earlier
   cache = TripleLayerKVCache(
       recent_size=512,
       warm_size=1024,
       enable_cold_am=True,  # ← Enable to prevent large Cold
       compression_ratio=2.0
   )
   ```

3. **Profile Memory Usage**:
   ```python
   # Check actual memory vs estimated
   import psutil
   process = psutil.Process()
   print(f"Actual memory: {process.memory_info().rss / 1024 / 1024:.1f} MB")
   ```

4. **Test Without TripleLayer** (Baseline):
   ```python
   # Use standard KVCache to verify model correctness
   from mlx_lm.models.cache import KVCache
   cache = [KVCache() for _ in range(len(model.layers))]
   ```

---

## 9. Summary

### What Works
- ✅ Q4_0 quantization (after dequantization bug fix)
- ✅ Small contexts (2K, Cold < 500 tokens)
- ✅ Migration flow (Recent → Warm → Cold)

### What Breaks
- ❌ Large Cold layer (> ~500 tokens)
- ❌ Quality degrades even with uncompressed Cold
- ❌ Memory budget not enforced (10x over budget)

### Root Cause Unknown
- Not calibration files (not used)
- Not Warm quantization bugs (fixed)
- Not concatenation logic (verified correct)
- **Likely**: Large Cold layer interaction with model or MLX

### User's Questions Answered

1. **迁移流程**: Recent → Warm (Q4_0) → Cold (pending) ✓ Correct
2. **什么时候压缩**: Every Recent overflow → Warm ✓ Correct
3. **设计原则**: Memory budget NOT enforced ✗ Violation
4. **校准文件**: Not used in these tests (enable_cold_am=False) N/A
5. **边界条件**: Large Cold (> 500 tokens) breaks quality ✗ Violation

---

**Conclusion**: The fundamental architecture is sound, but there's a critical boundary condition violation with large Cold layers that needs investigation.
