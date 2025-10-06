# Bug Fixes for NeuronRank TF-IDF++ Implementation

## Issues Fixed

### 1. Missing `--save` Default Value (TypeError)
**Problem:** When `--save` argument was not provided, `args.save` was `None`, causing:
```
TypeError: stat: path should be string, bytes, os.PathLike or integer, not NoneType
```

**Fix:** Added `default="out/"` to the `--save` argument in `main.py`:
```python
parser.add_argument('--save', type=str, default="out/", help='Path to save results.')
```

**Location:** `main.py`, line ~186

---

### 2. Enhanced Debug Output
**Problem:** When "No weights were pruned" occurred, there was insufficient information to diagnose the cause.

**Fix:** Added comprehensive debug output throughout the pruning pipeline:

#### Added in `lib/prune.py`:

1. **Hook registration confirmation:**
```python
print(f"✅ Registered {len(handles)} forward hooks")
```

2. **Statistics finalization debug:**
```python
if len(tfidf_stats) > 0:
    print(f"   Sample keys: {list(tfidf_stats.keys())[:3]}")
else:
    print(f"   ⚠️  WARNING: No statistics collected! Check hooks and data flow.")
```

3. **Module skip counters:**
```python
skipped_no_stats = 0
skipped_not_mlp = 0
skipped_should_not_prune = 0
```

4. **Final diagnostic summary:**
```python
print(f"   Debug: skipped_should_not_prune={skipped_should_not_prune}, skipped_not_mlp={skipped_not_mlp}, skipped_no_stats={skipped_no_stats}")
```

---

## Diagnostic Steps

When you run the command now, you'll see output like:

```
🔬 Loading calibration data for NeuronRank TF-IDF++ (doc mode)...
📊 Using document-level TF-IDF
📈 Collecting TF-IDF statistics...
✅ Registered 32 forward hooks          <-- Should match number of layers
  Processed 10/128 calibration batches
  Processed 20/128 calibration batches
  ...
🧮 Finalizing TF-IDF statistics...
✅ Collected TF-IDF statistics for 32 modules    <-- Should be > 0
   Sample keys: ['layer_0.mlp.gate_proj', 'layer_1.mlp.gate_proj', 'layer_2.mlp.gate_proj']
✂️  Applying TF-IDF++ pruning (α=1.0, β=1.0, γ=1.5)...
  [NeuronRank-TFIDF] layer  0 mlp.gate_proj    : pruned  ####### / ####### weights
  ...
```

If "No weights were pruned", you'll see:
```
⚠️  Warning: No weights were pruned!
   Debug: skipped_should_not_prune=X, skipped_not_mlp=Y, skipped_no_stats=Z
```

This tells you exactly what went wrong:
- `skipped_should_not_prune > 0`: `--pruning_last` restriction is too aggressive
- `skipped_not_mlp > 0`: Modules found but they're not MLP modules
- `skipped_no_stats > 0`: Statistics weren't collected (hook or data issue)

---

## Testing Commands

### Quick Test (3 minutes)
```bash
python main.py \
  --model baffo32/decapoda-research-llama-7B-hf \
  --prune_method neuronrank_tfidf \
  --sparsity_ratio 0.5 \
  --pruning_last 3 \
  --nr-tfidf-mode doc \
  --nsamples 32
```

### Full Doc Mode Test
```bash
python main.py \
  --model baffo32/decapoda-research-llama-7B-hf \
  --prune_method neuronrank_tfidf \
  --sparsity_ratio 0.5 \
  --pruning_last 30 \
  --nr-tfidf-mode doc \
  --weight-exp 1.0 --tf-exp 1.0 --idf-exp 1.5 \
  --nsamples 128
```

### Full Topic Mode Test
```bash
python main.py \
  --model baffo32/decapoda-research-llama-7B-hf \
  --prune_method neuronrank_tfidf \
  --sparsity_ratio 0.5 \
  --pruning_last 30 \
  --nr-tfidf-mode topic \
  --nr-tfidf-k 64 \
  --weight-exp 1.0 --tf-exp 1.0 --idf-exp 1.5 \
  --nsamples 128
```

---

## Files Modified

1. **`main.py`**: Added `default="out/"` to `--save` argument
2. **`lib/prune.py`**: Added debug output in `prune_neuronrank_tfidf()` function
3. **`test_tfidf.sh`**: Created test script for quick validation

---

## Expected Behavior

After these fixes:

✅ Commands without `--save` will use `out/` directory by default  
✅ Detailed debug output shows hook registration, statistics collection, and pruning progress  
✅ Clear diagnostic messages when pruning fails  
✅ Statistics collection from forward hooks should work correctly  

---

## Next Steps

1. Run the quick test command to verify basic functionality
2. Check the debug output to confirm:
   - Hooks are registered (should be 32 for LLaMA-7B)
   - Statistics are collected (should be 32 modules)
   - Weights are pruned (should show pruning counts per layer)
3. If still seeing issues, the debug output will pinpoint the exact problem

---

## Potential Issues to Watch For

If you still get "No weights were pruned":

### Issue A: Statistics not collected (skipped_no_stats > 0)
**Cause:** Forward hooks not firing or data not flowing through model  
**Solution:** Check model architecture matches expected structure (has `layer.mlp.gate_proj`)

### Issue B: All modules skipped (skipped_should_not_prune > 0)
**Cause:** `--pruning_last` value too restrictive  
**Solution:** Use `--pruning_last 30` or remove the flag entirely

### Issue C: Wrong module types (skipped_not_mlp > 0)
**Cause:** Only non-MLP modules found  
**Solution:** This would be a model architecture mismatch (shouldn't happen with LLaMA)

---

## Verification Checklist

- [ ] Command runs without TypeError
- [ ] See "✅ Registered N forward hooks" message (N > 0)
- [ ] See "✅ Collected TF-IDF statistics for N modules" (N > 0)
- [ ] See pruning progress messages for individual layers
- [ ] Final sparsity check shows > 0.0% sparsity
- [ ] WikiText perplexity is computed successfully
