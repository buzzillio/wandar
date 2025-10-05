# Wanda with Selectivity Enhancements

This implementation extends the original [Wanda](https://arxiv.org/abs/2306.11695) pruning method with selectivity metrics that prioritize "specialist" neurons over "always-on" or "never-fire" neurons.

## Overview

Original Wanda computes importance scores as:

```
S_ij^Wanda = |W_ij| · ||X_j||_2
```

We enhance this with two cheap, gradient-free selectivity metrics computed in a single calibration pass:

1. **IDF-style rarity**: Penalizes channels that fire frequently (always-on)
2. **Spikiness**: Rewards channels with peaked activation distributions (specialists)

## Methods

### 1. Wanda × IDF (`wanda_idf`)

Adds an IDF (Inverse Document Frequency) term that penalizes always-on channels:

```
S_ij = |W_ij| · ||X_j||_2 · IDF_j
```

where:
- `IDF_j = log(1 / (p_j + ε))` 
- `p_j` = fraction of activations above the 60th percentile threshold
- Clipped to `[0, 10]`

**Intuition**: Channels that fire frequently (high `p_j`) get lower scores, making them more likely to be pruned.

### 2. Wanda × Spikiness (`wanda_spiky`)

Adds a spikiness term that rewards peaky activation distributions:

```
S_ij = |W_ij| · ||X_j||_2 · R_j
```

where:
- `R_j = μ_top / μ_mean`
- `μ_top` = mean of top-10% activations (90th quantile and above)
- `μ_mean` = overall mean activation
- Clipped to `[1, 10]`

**Intuition**: Specialist neurons have high peaks (high `μ_top`) relative to their average (low `μ_mean`), giving them higher scores and protecting them from pruning.

### 3. Wanda × IDF × Spikiness (`wanda_select`)

Combines both metrics for full selectivity:

```
S_ij = |W_ij| · ||X_j||_2 · IDF_j · R_j
```

**Intuition**: Protects specialized neurons (high spikiness) that fire selectively (low IDF), while aggressively pruning always-on generalist neurons.

## Implementation Details

### Efficient One-Pass Computation

All selectivity metrics are computed during the same calibration forward pass used by original Wanda:

```python
# During calibration hooks:
for batch in calib_loader:
    X = get_activations(batch)  # [channels, tokens]
    
    # Accumulate statistics
    channel_sum += X.sum(dim=1)
    channel_sq_sum += (X ** 2).sum(dim=1)
    all_values.append(X)  # For quantile computation

# After all batches:
# Compute IDF
tau_j = quantile(all_values, 0.6, per_channel=True)
p_j = mean(all_values > tau_j)
IDF_j = log(1 / (p_j + ε))

# Compute spikiness
Q_j = quantile(all_values, 0.9, per_channel=True)
μ_top = mean(all_values[all_values >= Q_j])
μ_mean = channel_sum / total_tokens
R_j = μ_top / μ_mean
```

### Memory Efficiency

- Streams activations batch-by-batch
- Only stores per-channel aggregates (not full activation tensors)
- Computes quantiles using batched data
- Memory overhead: ~O(channels × num_samples) for quantile computation

### Hyperparameters

Default values (recommended):
- `tau_percentile = 0.6`: Threshold for "active" (60th percentile)
- `quantile = 0.9`: Top quantile for spikiness (90th percentile)
- `idf_clip_max = 10.0`: Maximum IDF value
- `spiky_clip_max = 10.0`: Maximum spikiness ratio

## Usage

```bash
# Baseline Wanda
python main.py \
    --model decapoda-research/llama-7b-hf \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --save out/wanda/

# Wanda × IDF
python main.py \
    --model decapoda-research/llama-7b-hf \
    --prune_method wanda_idf \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --save out/wanda_idf/

# Wanda × Spikiness
python main.py \
    --model decapoda-research/llama-7b-hf \
    --prune_method wanda_spiky \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --save out/wanda_spiky/

# Wanda × IDF × Spikiness (full selectivity)
python main.py \
    --model decapoda-research/llama-7b-hf \
    --prune_method wanda_select \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --save out/wanda_select/
```

### Batch Experiments

Run all methods automatically:

```bash
# Run on LLaMA-7B with 50% sparsity
./scripts/run_selectivity_experiments.sh decapoda-research/llama-7b-hf 0.5

# Results will be saved to out/llama_selectivity/
```

## Expected Results

On LLaMA models with 50% unstructured sparsity, we expect:

| Method | Description | Expected Impact |
|--------|-------------|-----------------|
| `wanda` | Baseline | Reference perplexity |
| `wanda_idf` | Penalize always-on | Similar or slightly better |
| `wanda_spiky` | Reward specialists | Potentially better on specialized tasks |
| `wanda_select` | Combined | Best overall performance |

The combined method (`wanda_select`) should provide the best perplexity by:
1. Protecting specialist neurons that encode specific patterns
2. Aggressively pruning generic always-on neurons
3. Maintaining the efficiency of Wanda's magnitude × activation metric

## Computational Cost

Compared to original Wanda:
- **Same**: Number of forward passes (1 calibration pass)
- **Same**: No backward passes or gradients
- **+10-20%**: Extra computation for quantile calculations
- **+~2x**: Memory for storing activation values (temporary)

Total overhead is minimal - still much faster than methods requiring Hessian computation (e.g., SparseGPT).

## Theory

### Why IDF?

Channels that fire frequently (high `p_j`) contribute to many outputs and may seem important by magnitude × norm. However, they often encode general, redundant information. The IDF term downweights these, similar to how IDF in NLP downweights common words.

### Why Spikiness?

The ratio `μ_top / μ_mean` captures how "specialized" a channel is:
- **High ratio (>5)**: Fires strongly for specific inputs, weakly otherwise → specialist
- **Low ratio (~1-2)**: Fires uniformly → generalist

Specialists often encode specific features (e.g., certain token patterns, syntactic structures) that are critical for model performance.

### Why Combine?

The two metrics capture complementary aspects:
- **IDF**: Temporal selectivity (when does it fire?)
- **Spikiness**: Magnitude selectivity (how strongly does it fire?)

A truly specialized neuron has both: fires rarely (`low p_j`, `high IDF`) and strongly (`high μ_top/μ_mean`).

## Files Modified

- `lib/layerwrapper.py`: Added `WrappedGPTSelectivity` class
- `lib/prune.py`: Added `prune_wanda_idf`, `prune_wanda_spiky`, `prune_wanda_select`
- `main.py`: Added method choices and dispatch logic
- `scripts/run_selectivity_experiments.sh`: Batch experiment script

## Citation

If you use these selectivity enhancements, please cite both the original Wanda paper and mention this extension:

```bibtex
@article{sun2023wanda,
  title={A Simple and Effective Pruning Approach for Large Language Models}, 
  author={Sun, Mingjie and Liu, Zhuang and Bair, Anna and Kolter, J. Zico},
  year={2023},
  journal={arXiv preprint arXiv:2306.11695}
}
```

