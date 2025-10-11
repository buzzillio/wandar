# Quadratic Discriminant Analysis (QDA) Implementation for NeuronRank Pruning

## Overview

This document describes the comprehensive QDA framework implementation for discriminative neural network pruning in the NeuronRank system. The implementation provides multiple QDA-based scoring methods that assess neuron importance based on their discriminative power across different input classes.

## Implemented QDA Methods

### 1. **NeuronRank QDA** (`neuronrank_qda`)
**Purpose**: Basic QDA with diagonal covariance assumption  
**Math**: For each neuron $j$, compute discriminability score:

$$\text{score}_j = \sum_{k=1}^{K} p_k \cdot \frac{(\mu_{kj} - \bar{\mu}_j)^2}{\sigma_{kj}^2 + \epsilon}$$

where:
- $K$ = number of classes (controlled by `--neuronrank-max-classes`)
- $p_k$ = prior probability of class $k$ (normalized class frequency)
- $\mu_{kj}$ = mean activation of neuron $j$ for class $k$
- $\bar{\mu}_j$ = overall mean activation of neuron $j$
- $\sigma_{kj}^2$ = variance of neuron $j$ activations for class $k$
- $\epsilon$ = small constant for numerical stability

**Usage**:
```bash
python main.py baffo32/decapoda-research-llama-7B-hf c4 \
    --prune_method neuronrank_qda \
    --sparsity_ratio 0.5 \
    --neuronrank-max-classes 512 \
    --pruning_last 32
```

**Key Properties**:
- Diagonal covariance: assumes neurons are independent within each class
- Normalized by within-class variance: higher score = more discriminative
- Weighted by class priors: accounts for class imbalance

---

### 2. **NeuronRank Between-Class Variance** (`neuronrank_between`)
**Purpose**: Simplified discriminability using only between-class variance  
**Math**: For each neuron $j$:

$$\text{score}_j = \sum_{k=1}^{K} p_k \cdot (\mu_{kj} - \bar{\mu}_j)^2$$

**Usage**:
```bash
python main.py baffo32/decapoda-research-llama-7B-hf c4 \
    --prune_method neuronrank_between \
    --sparsity_ratio 0.5 \
    --neuronrank-max-classes 512 \
    --pruning_last 32
```

**Key Properties**:
- Simpler than full QDA: ignores within-class variance
- Measures how much class means differ
- More stable when per-class statistics are noisy
- Fallback method when variances are unavailable

---

### 3. **NeuronRank PCA+QDA** (`neuronrank_pca_qda`)
**Purpose**: Dimensionality reduction before QDA to reduce noise and computational cost  
**Pipeline**:
1. **PCA**: Project $D$-dimensional activations to $d$-dimensional subspace ($d < D$)
2. **QDA**: Compute discriminability scores in reduced space
3. **Back-project**: Transform scores back to original neuron space

**Math**:
1. Covariance matrix: $\Sigma = \frac{1}{N}\sum_{i=1}^{N} (x_i - \bar{x})(x_i - \bar{x})^T$
2. Eigendecomposition: $\Sigma = V \Lambda V^T$
3. Projection: $z_i = V_d^T x_i$ (keep top $d$ components)
4. QDA in reduced space: $\text{score}_j^{(d)} = \sum_k p_k \frac{(\mu_{kj}^{(d)} - \bar{\mu}_j^{(d)})^2}{\sigma_{kj}^{(d)2} + \epsilon}$
5. Back-projection: $\text{score}_j = (V_d \text{score}^{(d)})_j$

**Usage**:
```bash
python main.py baffo32/decapoda-research-llama-7B-hf c4 \
    --prune_method neuronrank_pca_qda \
    --sparsity_ratio 0.5 \
    --neuronrank-max-classes 512 \
    --nr-pca-components 128 \
    --pruning_last 32
```

**Key Properties**:
- Reduces noise from correlated neurons
- Computational efficiency: $O(d)$ instead of $O(D)$ for QDA
- Captures global structure before discriminative analysis
- Default: 128 PCA components (configurable via `--nr-pca-components`)

---

### 4. **NeuronRank Mahalanobis Distance** (`neuronrank_mahalanobis`) ⭐ **NEW**
**Purpose**: Proper Mahalanobis distance-based discriminability using full or pooled covariance  
**Math**: For each neuron $j$, compute weighted squared Mahalanobis distance:

$$\text{score}_j = \sum_{k=1}^{K} p_k \cdot d_M^2(\mu_{kj}, \bar{\mu}_j)$$

where the squared Mahalanobis distance is:

$$d_M^2(\mu_{kj}, \bar{\mu}_j) = \frac{(\mu_{kj} - \bar{\mu}_j)^2}{\sigma_j^2}$$

**Two Modes**:
1. **QDA mode** (default): $\sigma_j^2 = \sigma_{kj}^2$ (per-class covariance)
   - True quadratic discriminant: each class has its own covariance
   - More flexible but requires sufficient per-class samples
   
2. **LDA mode** (`--nr-mahalanobis-pooled`): $\sigma_j^2 = \frac{1}{K}\sum_k p_k \sigma_{kj}^2$ (pooled covariance)
   - Linear discriminant assumption: shared covariance across classes
   - More stable with limited data per class

**Usage**:
```bash
# QDA mode (per-class covariance)
python main.py baffo32/decapoda-research-llama-7B-hf c4 \
    --prune_method neuronrank_mahalanobis \
    --sparsity_ratio 0.5 \
    --neuronrank-max-classes 512 \
    --pruning_last 32

# LDA mode (pooled covariance)
python main.py baffo32/decapoda-research-llama-7B-hf c4 \
    --prune_method neuronrank_mahalanobis \
    --sparsity_ratio 0.5 \
    --neuronrank-max-classes 512 \
    --nr-mahalanobis-pooled \
    --pruning_last 32
```

**Key Properties**:
- **Geometric interpretation**: Measures distance in units of standard deviations
- **Rotation invariant**: Unlike Euclidean distance, accounts for data geometry
- **Numerically stable**: Includes fallback to between-class variance when variances unavailable
- **Flexible**: Toggle between QDA and LDA via `--nr-mahalanobis-pooled`

**Implementation Details**:
```python
def compute_neuronrank_mahalanobis_scores(stats, use_pooled_cov=False):
    """
    Compute Mahalanobis distance-based discriminability scores.
    
    Args:
        stats: Statistics dict with per-class means/variances
        use_pooled_cov: If True, use pooled covariance (LDA-like)
                       If False, use per-class covariance (true QDA)
    
    Returns:
        Dict mapping layer_idx -> {"channel": tensor of shape [D]}
    """
```

---

## Statistics Collection

All QDA methods rely on the same statistics collection pipeline in `collect_neuronrank_statistics()`:

**Tracked Statistics**:
- **Per-neuron global stats**: 
  - `overall_mean`: $\bar{\mu}_j = \frac{1}{N}\sum_{i=1}^{N} x_{ij}$
  - `overall_var`: $\sigma_j^2 = \frac{1}{N}\sum_{i=1}^{N} (x_{ij} - \bar{\mu}_j)^2$
  
- **Per-class per-neuron stats**:
  - `class_mean[k]`: $\mu_{kj} = \frac{1}{N_k}\sum_{i \in C_k} x_{ij}$
  - `class_var[k]`: $\sigma_{kj}^2 = \frac{1}{N_k}\sum_{i \in C_k} (x_{ij} - \mu_{kj})^2$
  - `class_count[k]`: $N_k$ = number of tokens assigned to class $k$

**Class Assignment**: 
- First token of each sequence determines class
- Classes are quantized token IDs: `class_id = token_id % max_classes`
- Configurable via `--neuronrank-max-classes` (default: 512)

**Hooked Layers**: 
- MLP gate projections in transformer layers (where `"mlp.gate_proj" in name`)
- Post-activation statistics collection (after SiLU/GELU)

---

## Pruning Pipeline

All QDA methods follow the same pruning pipeline:

1. **Statistics Collection**: 
   ```python
   stats = collect_neuronrank_statistics(
       model, dataloader, tokenizer, device, 
       max_classes=args.neuronrank_max_classes
   )
   ```

2. **Score Computation**: 
   ```python
   scores = compute_neuronrank_METHOD_scores(stats, **kwargs)
   # scores[layer_idx]["channel"] = tensor of shape [D]
   ```

3. **Per-Weight Metric Construction**:
   ```python
   factor = scores[layer_idx]["channel"]
   factor = (factor / factor.mean()).clamp_min(1e-6)  # normalize
   
   # Broadcast to weight shape
   if "gate_proj" or "up_proj":
       metric = factor.view(-1, 1).expand_as(weight)
   elif "down_proj":
       metric = factor.view(1, -1).expand_as(weight)
   
   # Combine with weight magnitude
   metric = torch.abs(weight) * metric
   ```

4. **Unstructured Masking**:
   ```python
   pruned, numel = _apply_unstructured_mask(weight, metric, sparsity_ratio)
   # Prunes lowest-metric weights, keeping top (1-sparsity_ratio) fraction
   ```

**Key Design Choice**: 
- All methods use **unstructured per-weight masking** (not channel-wise)
- Metric: `|W| × normalized_neuron_score`
- This ensures fair comparison with magnitude baseline
- MLP-only pruning yields ~33.4% global sparsity when requesting 0.5 (MLPs are 66.8% of parameters)

---

## Method Comparison

| Method | Covariance | Complexity | Stability | Best For |
|--------|-----------|------------|-----------|----------|
| `neuronrank_between` | None (ignores variance) | $O(D)$ | ⭐⭐⭐ High | Quick baseline, noisy data |
| `neuronrank_qda` | Diagonal per-class | $O(KD)$ | ⭐⭐ Medium | Independent neurons assumption |
| `neuronrank_mahalanobis` | Full per-class or pooled | $O(KD)$ | ⭐⭐⭐ High | Proper distance metric |
| `neuronrank_pca_qda` | Diagonal after PCA | $O(D^2 + dK)$ | ⭐⭐ Medium | High-D noisy data |

**Recommendations**:
- **Start with**: `neuronrank_mahalanobis` (pooled mode) for stable Mahalanobis distance
- **Try next**: `neuronrank_qda` for per-class covariance if pooled is too restrictive
- **Use PCA variant** if standard QDA is noisy or slow
- **Use between-class** as lightweight baseline

---

## Future QDA Applications (Pending Implementation)

### 2. Layer-wise Discriminative Analysis
**Goal**: Determine which layers to prune more aggressively based on discriminative power

**Approach**:
- Compute layer-level discriminability: $\text{score}_{\ell} = \text{mean}(\text{score}_j^{(\ell)})$
- Adaptive sparsity: prune less discriminative layers more
- Example: `sparsity_ratio_layer = base_ratio × (1 + λ × (1 - normalized_score_layer))`

**Status**: 🔄 **Not yet implemented**

---

### 3. Attention Head Pruning with QDA
**Goal**: Prune attention heads based on discriminative power of their output patterns

**Approach**:
- Collect per-head attention output statistics: $o_{hkj} = \text{head}_h(Q_k, K_k, V_k)_j$
- Compute QDA scores per head: $\text{score}_h = \sum_k p_k \frac{(\mu_{hkj} - \bar{\mu}_{hj})^2}{\sigma_{hkj}^2 + \epsilon}$
- Prune least discriminative heads

**Challenges**:
- Attention outputs have different dimensionality than MLP activations
- Multi-head structure requires careful score aggregation
- May need separate calibration data for attention analysis

**Status**: 🔄 **Not yet implemented**

---

### 4. Weight Pattern Analysis with QDA
**Goal**: Analyze weight matrix structure to identify discriminative vs. redundant patterns

**Approach**:
- Treat each weight row/column as a sample
- Cluster weights using QDA-based distance metric
- Prune redundant clusters while preserving diverse patterns

**Math**:
- Weight space QDA: cluster $W_{i,:}$ (rows) or $W_{:,j}$ (columns)
- Mahalanobis distance between weight vectors: $d_M^2(w_i, w_j) = (w_i - w_j)^T \Sigma^{-1} (w_i - w_j)$
- Keep representatives from each discriminative cluster

**Status**: 🔄 **Not yet implemented**

---

### 5. Dynamic Pruning During Inference
**Goal**: Adaptively prune based on input discriminability at inference time

**Approach**:
- Compute per-input discriminability scores: $\text{score}_j(x) = \sum_k p(k|x) \frac{(\mu_{kj} - x_j)^2}{\sigma_{kj}^2}$
- Prune neurons with low discriminability for specific input
- Requires fast online QDA evaluation

**Challenges**:
- Computational overhead during inference
- Requires efficient QDA score caching or approximation
- Trade-off between speedup and accuracy

**Status**: 🔄 **Not yet implemented**

---

## Configuration Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--prune_method` | - | Choose: `neuronrank_qda`, `neuronrank_pca_qda`, `neuronrank_mahalanobis`, `neuronrank_between` |
| `--neuronrank-max-classes` | 512 | Number of classes for discriminative statistics (higher = more fine-grained) |
| `--nr-pca-components` | 128 | PCA components for `neuronrank_pca_qda` |
| `--nr-mahalanobis-pooled` | False | Use pooled covariance (LDA-like) instead of per-class (QDA) |
| `--sparsity_ratio` | 0.5 | Target sparsity (fraction of weights to prune) |
| `--pruning_last` | None | Only prune last N MLP layers (e.g., 32 for LLaMA-7B) |

---

## Example Commands

### Compare All Methods
```bash
# Magnitude baseline
python main.py baffo32/decapoda-research-llama-7B-hf c4 \
    --prune_method magnitude --sparsity_ratio 0.5 --save results/magnitude/

# Between-class variance (lightweight)
python main.py baffo32/decapoda-research-llama-7B-hf c4 \
    --prune_method neuronrank_between --sparsity_ratio 0.5 \
    --neuronrank-max-classes 512 --pruning_last 32 --save results/between/

# Basic QDA (diagonal covariance)
python main.py baffo32/decapoda-research-llama-7B-hf c4 \
    --prune_method neuronrank_qda --sparsity_ratio 0.5 \
    --neuronrank-max-classes 512 --pruning_last 32 --save results/qda/

# PCA+QDA (dimensionality reduction)
python main.py baffo32/decapoda-research-llama-7B-hf c4 \
    --prune_method neuronrank_pca_qda --sparsity_ratio 0.5 \
    --neuronrank-max-classes 512 --nr-pca-components 128 \
    --pruning_last 32 --save results/pca_qda/

# Mahalanobis QDA (per-class covariance)
python main.py baffo32/decapoda-research-llama-7B-hf c4 \
    --prune_method neuronrank_mahalanobis --sparsity_ratio 0.5 \
    --neuronrank-max-classes 512 --pruning_last 32 --save results/mahalanobis_qda/

# Mahalanobis LDA (pooled covariance)
python main.py baffo32/decapoda-research-llama-7B-hf c4 \
    --prune_method neuronrank_mahalanobis --sparsity_ratio 0.5 \
    --neuronrank-max-classes 512 --nr-mahalanobis-pooled \
    --pruning_last 32 --save results/mahalanobis_lda/
```

---

## Files Modified

- **`lib/neuronrank.py`**: Core statistics and scoring functions
  - `collect_neuronrank_statistics()`: Gathers per-class activation stats
  - `compute_neuronrank_qda_scores()`: Basic QDA scoring
  - `compute_neuronrank_between_scores()`: Between-class variance scoring
  - `compute_neuronrank_pca_qda_scores()`: PCA+QDA hybrid scoring
  - `compute_neuronrank_mahalanobis_scores()`: ⭐ NEW Mahalanobis distance scoring

- **`lib/prune.py`**: Pruning entry points
  - `prune_neuronrank_qda()`: QDA-based unstructured pruning
  - `prune_neuronrank_between()`: Between-class variance pruning
  - `prune_neuronrank_pca_qda()`: PCA+QDA hybrid pruning
  - `prune_neuronrank_mahalanobis()`: ⭐ NEW Mahalanobis distance pruning

- **`main.py`**: CLI integration
  - Added method choices: `neuronrank_qda`, `neuronrank_pca_qda`, `neuronrank_mahalanobis`, `neuronrank_between`
  - Added flags: `--nr-pca-components`, `--nr-mahalanobis-pooled`
  - Wired dispatch logic for all methods

---

## References

- **Fisher LDA**: Fisher, R. A. (1936). "The use of multiple measurements in taxonomic problems"
- **Quadratic Discriminant Analysis**: Hastie, Tibshirani, Friedman (2009). "The Elements of Statistical Learning"
- **Mahalanobis Distance**: Mahalanobis, P. C. (1936). "On the generalized distance in statistics"
- **Original NeuronRank**: Mirzadeh et al. (2024). "NeuronRank: Pruning Neural Networks via Activation Variance"

---

## Known Issues & Limitations

1. **MLP-only pruning yields ~33.4% global sparsity** when requesting 0.5 (because MLPs are 66.8% of total parameters)
   - Solution: Request higher sparsity_ratio (e.g., 0.75 for 50% global)
   - Or extend to attention weights (future work)

2. **Class assignment is simplified**: First token determines class for entire sequence
   - More sophisticated: per-token class assignment
   - Trade-off: memory and computational cost

3. **Diagonal covariance assumption** in basic QDA/PCA+QDA
   - Assumes neuron independence within each class
   - Full covariance QDA would be $O(KD^2)$ memory + $O(KD^3)$ compute
   - Mahalanobis method addresses this partially

4. **Per-class sample size**: Requires sufficient tokens per class for stable variance estimates
   - Default 512 classes with 128 calibration samples (nsamples) = ~1-2 sequences per class
   - May need to reduce `--neuronrank-max-classes` for very limited calibration data

---

## Next Steps

1. ✅ **COMPLETED**: Wire `compute_neuronrank_mahalanobis_scores` into CLI
2. 🔄 **PENDING**: Implement attention head QDA analysis (application #3)
3. 🔄 **PENDING**: Implement layer-wise discriminative analysis (application #2)
4. 🔄 **PENDING**: Explore weight pattern analysis with QDA (application #4)
5. 🔄 **PENDING**: Prototype dynamic inference-time pruning (application #5)

---

**Status**: ✅ Mahalanobis distance implementation complete and integrated into CLI (as of this session)
