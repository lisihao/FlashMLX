# OMP Implementation Plan for MLX

## Goal
Implement complete Orthogonal Matching Pursuit (OMP) algorithm for AM key selection in MLX.

## Core Algorithm (from author's omp.py)

### Input
- `queries`: (n, d) - query vectors
- `keys`: (T, d) - all keys
- `budget`: int - number of keys to select
- `scale`: float - attention scale (sqrt(d))

### Output
- `C1`: (budget, d) - selected keys
- `beta`: (budget,) - log weights
- `indices`: list - selected key indices

### Algorithm Steps

```python
# 1. Compute exp_scores and target
scores = (queries @ keys.T) / scale  # (n, T)
max_scores = mx.max(scores, axis=1, keepdims=True)
exp_scores = mx.exp(scores - max_scores)  # (n, T)
target = mx.sum(exp_scores, axis=1)  # (n,) - partition function

# 2. Greedy selection loop
selected_indices = []
current = mx.zeros_like(target)
mask = mx.zeros(T, dtype=bool)

for i in range(budget):
    # 2a. Compute residual
    residual = target - current

    # 2b. Correlation of each key with residual
    corr = (exp_scores * residual[:, None]).sum(axis=0)  # (T,)
    corr[mask] = -inf  # Exclude selected keys

    # 2c. Select key with highest correlation
    idx = mx.argmax(corr)
    selected_indices.append(idx)
    mask[idx] = True

    # 2d. Solve NNLS: M @ B ≈ target
    M = exp_scores[:, selected_indices]  # (n, i+1)
    B = nnls_solve(M, target)

    # 2e. Update approximation
    current = M @ B

# 3. Final beta
beta = mx.log(B)
C1 = keys[selected_indices]
```

## Key Differences from Current Implementation

| Current (Attention-Aware) | OMP (Greedy) |
|---------------------------|--------------|
| Select all keys at once | Iterative selection |
| Based on attention scores | Based on correlation with residual |
| One NNLS solve at end | NNLS solve after each selection |
| Keys may not be optimal for NNLS | Keys optimized for NNLS objective |

## Implementation Phases

### Phase 1: Core OMP Function ✅

**File**: `mlx_lm/compaction/quality.py`

**Function**: `select_keys_omp_mlx(queries, keys, budget, scale)`

**Logic**:
- Implement greedy selection loop
- Call `_nnls_mlx` after each selection
- Return C1, beta, indices

### Phase 2: Integration ✅

**Modify**: `compact_single_head_quality`

**Change**:
- Replace `select_keys_attention_aware` with `select_keys_omp_mlx`
- Beta comes from OMP (don't recompute)
- C2 still uses LSQ fitting

### Phase 3: Optimization (Optional)

**Optimizations** (from author's OMPCompaction):
- `k_choice`: Select top-k keys per iteration (faster)
- `nnls_interval`: Skip NNLS solves occasionally (lazy mode)
- `progressive_schedule`: Different k_choice/interval based on budget

For initial implementation, use simple version (k_choice=1, always solve NNLS).

## MLX-Specific Considerations

### 1. Boolean Masking
MLX may not support boolean indexing exactly like PyTorch.
**Solution**: Use `mx.where` or set masked values to `-inf`

### 2. Dynamic Arrays
Can't use Python list appending then convert to array efficiently.
**Solution**: Pre-allocate `selected_indices` tensor and fill it

### 3. NNLS Efficiency
Calling `_nnls_mlx` in loop may be slow.
**Solution**: Accept slower performance for correctness first; optimize later

## Testing Plan

### Test 1: Simple OMP (budget=5)
- Verify greedy selection works
- Check beta values are reasonable (not all -27)
- Compare correlation-based selection vs attention-based

### Test 2: Quality Test
- Same as before: 100-token generation
- Expected: Better quality than attention-aware (>30%?)
- Check if beta values are in reasonable range (-5 to +5?)

### Test 3: Cumulative Error
- Track quality degradation over compressions
- Compare: OMP vs Attention-Aware vs Direct Copy

## Success Criteria

✅ Beta values in reasonable range (-5 to +5, not -27)
✅ Quality > 30% (better than current 23-25%)
✅ No crashes or numerical errors
✅ Degrades gracefully (no sudden collapse)

## Fallback Plan

If OMP still produces poor quality:
- Try `zerobeta=True` with OMP selection + C2 LSQ
- Try OMP selection + C2 direct copy
- If all fail → Switch to H2O/StreamingLLM (reliable fallback)
