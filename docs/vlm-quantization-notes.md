# VLM Quantization Notes

## Current Status (2026-04-10)

### ✅ Working: bf16 Weights
- Full precision bf16 model loads successfully
- All 729 weights loaded without issues
- No quantization parameters to handle

### ⚠️ Partially Working: 4-bit Quantized Weights

**Problem**: Dimension mismatch when loading quantized vision tower weights

**Root Cause**:
MLX 4-bit quantization uses three components per weight:
1. `.weight` - Compressed weight matrix (reduced dimensions)
2. `.scales` - Quantization scales for dequantization
3. `.biases` - Quantization biases for dequantization

Example from vision tower QKV:
```python
vision_tower.blocks.9.attn.qkv.weight: (3840, 160)  # Compressed
vision_tower.blocks.9.attn.qkv.scales: (3840, 20)   # Scales
vision_tower.blocks.9.attn.qkv.biases: (3840, 20)   # Biases
```

**Current Issue**:
Our weight loading filters out `.scales` and `.biases`:
```python
if k.endswith('.biases') or k.endswith('.scales'):
    skipped += 1
    continue
```

This leaves only `.weight` with wrong dimensions (160 vs expected 1280).

**Error**:
```
ValueError: [addmm] Last dimension of first input with shape (1024,1280)
must match second to last dimension of second input with shape (160,3840).
```

Input: (seq_len=1024, hidden_dim=1280)
Weight (transposed): (compressed_dim=160, 3*hidden_dim=3840)

Expected: (hidden_dim=1280, 3*hidden_dim=3840)

## Solutions

### Option 1: Use mlx.nn.QuantizedLinear (Proper Fix)

Replace standard `nn.Linear` with `nn.QuantizedLinear` for quantized models:

```python
# In vision.py VisionAttention.__init__
if quantized:
    self.qkv = nn.QuantizedLinear(
        input_dims=hidden_size,
        output_dims=3 * hidden_size,
        bias=True,
        group_size=32,  # Standard MLX quantization
        bits=4,
    )
else:
    self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
```

**Benefits**:
- Proper quantization support
- Memory efficient (~70% reduction)
- Fast inference

**Challenges**:
- Need to detect quantized weights during loading
- Need to pass quantization flag through model hierarchy
- Need to handle mixed quantization (vision + language)

### Option 2: Use bf16 Model (Current Workaround)

Simply use the bf16 model which has no quantization:

```python
download_and_load_model(use_4bit=False)  # Use bf16
```

**Benefits**:
- Works immediately
- No code changes needed
- Full precision

**Drawbacks**:
- Larger memory footprint (~4GB vs ~1.5GB)
- Slower download

## Implementation Plan for Quantization Support

### Phase 1: Detection
1. Add `is_quantized()` helper to detect `.scales` and `.biases`
2. Set model-level flag during weight loading

### Phase 2: Model Updates
1. Update `VisionAttention` to support `nn.QuantizedLinear`
2. Update `VisionMLP` layers
3. Update `Qwen2Attention` in language model

### Phase 3: Weight Loading
1. Keep `.weight`, `.scales`, `.biases` together
2. Load as `nn.QuantizedLinear` instead of `nn.Linear`
3. Verify dimensions match

### Phase 4: Testing
1. Test with 4-bit model
2. Verify output matches bf16 (within tolerance)
3. Benchmark memory and speed

## References

- MLX Quantization Docs: https://ml-explore.github.io/mlx/build/html/usage/quantization.html
- MLX-LM Quantization: `mlx_lm/models/base.py` QuantizedLinear usage
- MLX Quantized Linear: `mlx.nn.QuantizedLinear` API

## Testing

### Current Tests:
- ✅ bf16 model: All weights load correctly
- ⚠️ 4-bit model: Dimension mismatch in vision tower
- ❌ 4-bit model: Generation fails

### Required Tests After Fix:
- [ ] 4-bit model loads without errors
- [ ] 4-bit generation produces reasonable output
- [ ] 4-bit output matches bf16 (cosine similarity > 0.95)
- [ ] Memory usage ~70% of bf16
- [ ] Inference speed comparable or faster

## Workaround for Now

**For testing and development, use bf16 model:**
```bash
python3 examples/test_real_weights.py  # Now defaults to bf16
```

**For production with memory constraints:**
- Implement proper QuantizedLinear support (Phase 1-4 above)
- Or use MLX-LM's quantization utilities directly
