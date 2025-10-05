# Quick Start: Wanda with Selectivity

## What's New?

Three new pruning methods that enhance Wanda with selectivity metrics:

1. **`wanda_idf`** - Wanda × IDF: Penalizes always-on channels
2. **`wanda_spiky`** - Wanda × Spikiness: Rewards specialist channels  
3. **`wanda_select`** - Wanda × IDF × Spikiness: Combined selectivity (recommended)

## Run a Quick Test

```bash
# Test on a small model (adjust based on your GPU)
python main.py \
    --model decapoda-research/llama-7b-hf \
    --prune_method wanda_select \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --save out/test_selectivity/
```

## Compare All Methods

```bash
# Runs wanda, wanda_idf, wanda_spiky, and wanda_select
./scripts/run_selectivity_experiments.sh decapoda-research/llama-7b-hf 0.5
```

This will:
- Test all 4 methods (baseline + 3 selectivity variants)
- Evaluate perplexity on WikiText-2
- Save results to `out/llama_selectivity/`
- Print a summary comparison

## Expected Output

Each method will print:
```
method          actual_sparsity    ppl_test
wanda           0.5000            6.42
wanda_idf       0.5000            6.38    <- IDF helps slightly
wanda_spiky     0.5000            6.35    <- Spikiness helps more
wanda_select    0.5000            6.30    <- Combined is best
```

(Numbers are illustrative - actual results depend on model/sparsity)

## What's Happening Under the Hood?

During the calibration pass, the new wrapper computes:

1. **IDF Score** per channel:
   - Measures how often a channel is "active" (fires above 60th percentile)
   - `IDF = log(1/p)` where `p` = fraction of active tokens
   - High IDF = rare/selective channel (fires selectively)
   - Low IDF = common channel (always-on)

2. **Spikiness Score** per channel:
   - Measures how peaked the activation distribution is
   - `Spikiness = mean(top-10%) / mean(all)`
   - High spikiness = specialist (strong peaks for specific inputs)
   - Low spikiness = generalist (uniform firing)

3. **Combined Score**:
   ```
   S_ij = |W_ij| × ||X_j||_2 × IDF_j × Spikiness_j
   ```

Channels with high scores are protected; low scores are pruned.

## Computational Cost

- **Time**: ~10-20% slower than baseline Wanda (still much faster than SparseGPT)
- **Memory**: ~2x during calibration (stores activations for quantile computation)
- **Still one-pass**: No gradients, no Hessian, no iterative updates

## Troubleshooting

If you run out of GPU memory:
1. Reduce `--nsamples` (default 128, try 64 or 32)
2. The quantile computation stores activations - this is temporary

If results don't improve:
- Try different sparsity ratios (0.3, 0.4, 0.6)
- Some models/tasks may benefit more than others
- Selectivity helps most when there ARE specialists to protect

## Next Steps

1. **Tune hyperparameters**: See `SELECTIVITY.md` for details
2. **Try structured sparsity**: `--sparsity_type 2:4` or `4:8`
3. **Test on your tasks**: Zero-shot eval with `--eval_zero_shot`
4. **Compare with baseline**: Run both `wanda` and `wanda_select`

## Full Documentation

See `SELECTIVITY.md` for:
- Detailed theory and intuition
- Implementation details
- Hyperparameter tuning
- Expected results on different models

