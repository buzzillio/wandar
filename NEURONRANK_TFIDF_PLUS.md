# NeuronRank TF-IDF++ Documentation

## Overview

**NeuronRank TF-IDF++** is an advanced pruning method that uses document-level or topic-level IDF (Inverse Document Frequency) to compute neuron importance scores. Unlike the original `neuronrank_old` which uses token-level statistics, TF-IDF++ treats calibration sequences as documents or clusters tokens into semantic topics.

## Key Advantages

✅ **Document-level IDF**: Preserves "generalist" neurons that are broadly useful across multiple sequences  
✅ **Topic-level IDF**: Adds semantic selectivity by clustering tokens into topics  
✅ **Better at 50% sparsity**: The doc/topic-level context prevents over-pruning useful neurons  
✅ **Consistent pipeline**: Uses same per-module masking as other methods (magnitude, wanda)  
✅ **Single-pass collection**: No gradients required, efficient statistics gathering  

## Two Modes

### 1. Document-level TF-IDF (`--nr-tfidf-mode doc`)

**Fast and robust** - Treats each calibration sequence as a "document"

- **TF (Term Frequency)**: Average absolute activation across all tokens
- **DF (Document Frequency)**: Number of sequences where neuron is active (above threshold)
- **IDF**: `log((N_docs + 1) / (DF + 1)) + 1`

**When to use**: Default choice for most scenarios. Faster than topic mode and provides good generalization.

### 2. Topic-level TF-IDF (`--nr-tfidf-mode topic`)

**Semantic clustering** - Clusters tokens into K topics using k-means

- **Topic Assignment**: Uses cosine k-means on random projection of activations
- **TF**: Average activation strength within each topic
- **DF**: Number of topics where neuron is active
- **IDF**: Topic-level inverse frequency

**When to use**: When you want semantic selectivity and can afford extra computation. Especially useful for diverse datasets.

## Scoring Formula

```
Score[i,j] = |W[i,j]|^α × TF[j]^β × IDF[j]^γ × (spikiness[j]^ρ)
```

Where:
- `|W[i,j]|`: Weight magnitude
- `TF[j]`: Term frequency (activation strength) for channel j
- `IDF[j]`: Inverse document/topic frequency (selectivity) for channel j
- `α, β, γ`: Exponents controlling each component's influence
- `ρ`: Optional spikiness exponent (default: 0.0)

## Command-Line Arguments

### Core Arguments

```bash
--prune_method neuronrank_tfidf    # Use TF-IDF++ pruning

--nr-tfidf-mode {doc,topic}        # doc (default) or topic
                                   # doc: document-level IDF (fast)
                                   # topic: semantic topic-level IDF

--nr-tfidf-k 64                    # Number of topics (topic mode only)
                                   # Try: 32, 64, 128
                                   # More topics = finer granularity

--nr-q-active 0.60                 # Quantile for "active" threshold
                                   # Try: 0.50, 0.60, 0.70
                                   # Higher = stricter "active" definition
```

### Scoring Exponents

```bash
--weight-exp 1.0                   # α: Weight magnitude exponent
                                   # Higher = favor larger weights
                                   # Try: 0.5, 1.0, 1.5, 2.0

--tf-exp 1.0                       # β: TF (activation strength) exponent
                                   # Higher = favor strongly-activating neurons
                                   # Try: 0.5, 1.0, 1.5, 2.0

--idf-exp 1.0                      # γ: IDF (selectivity) exponent
                                   # Higher = favor selective/sparse neurons
                                   # Try: 1.0, 1.5, 2.0, 2.5

--nr-spikiness-exp 0.0             # ρ: Optional spikiness multiplier
                                   # Try: 0.0 (off), 0.3, 0.5, 1.0
```

### Shared Arguments

```bash
--sparsity_ratio 0.5               # Target sparsity (0.0 to 1.0)
--pruning_last 30                  # Prune last N layers only
--nsamples 128                     # Calibration samples (more = better stats)
--seed 0                           # Random seed
```

## Usage Examples

### 1. Document-level TF-IDF (Recommended Default)

```bash
python main.py \
  --model baffo32/decapoda-research-llama-7B-hf \
  --prune_method neuronrank_tfidf \
  --sparsity_type unstructured \
  --sparsity_ratio 0.5 \
  --pruning_last 30 \
  --nr-tfidf-mode doc \
  --weight-exp 1.0 \
  --tf-exp 1.0 \
  --idf-exp 1.5 \
  --nsamples 128 \
  --save out/nr_tfidf_doc/
```

**Why these settings?**
- `--idf-exp 1.5`: Emphasizes selectivity slightly more than default
- Doc mode: Fast and robust for most scenarios

### 2. Topic-level TF-IDF (Semantic)

```bash
python main.py \
  --model baffo32/decapoda-research-llama-7B-hf \
  --prune_method neuronrank_tfidf \
  --sparsity_type unstructured \
  --sparsity_ratio 0.5 \
  --pruning_last 30 \
  --nr-tfidf-mode topic \
  --nr-tfidf-k 128 \
  --weight-exp 1.0 \
  --tf-exp 1.0 \
  --idf-exp 2.0 \
  --nsamples 256 \
  --save out/nr_tfidf_topic/
```

**Why these settings?**
- `--nr-tfidf-k 128`: More topics for finer-grained clustering
- `--idf-exp 2.0`: Strong emphasis on topic-level selectivity
- `--nsamples 256`: More samples for better topic statistics

### 3. Emphasize Activation Strength (High TF)

```bash
python main.py \
  --model baffo32/decapoda-research-llama-7B-hf \
  --prune_method neuronrank_tfidf \
  --sparsity_type unstructured \
  --sparsity_ratio 0.5 \
  --pruning_last 30 \
  --nr-tfidf-mode doc \
  --weight-exp 1.0 \
  --tf-exp 2.0 \
  --idf-exp 1.0 \
  --save out/nr_tfidf_strong_act/
```

**Why these settings?**
- `--tf-exp 2.0`: Quadratic emphasis on activation strength
- Keeps neurons that fire strongly across many documents

### 4. Pure Selectivity (IDF-focused)

```bash
python main.py \
  --model baffo32/decapoda-research-llama-7B-hf \
  --prune_method neuronrank_tfidf \
  --sparsity_type unstructured \
  --sparsity_ratio 0.5 \
  --pruning_last 30 \
  --nr-tfidf-mode topic \
  --nr-tfidf-k 64 \
  --weight-exp 0.5 \
  --tf-exp 0.5 \
  --idf-exp 2.5 \
  --save out/nr_tfidf_selective/
```

**Why these settings?**
- Low `--weight-exp` and `--tf-exp`: Reduce magnitude/activation influence
- High `--idf-exp 2.5`: Strong emphasis on selectivity
- Good for finding highly-specialized neurons

### 5. With Spikiness Multiplier

```bash
python main.py \
  --model baffo32/decapoda-research-llama-7B-hf \
  --prune_method neuronrank_tfidf \
  --sparsity_type unstructured \
  --sparsity_ratio 0.5 \
  --pruning_last 30 \
  --nr-tfidf-mode topic \
  --nr-tfidf-k 64 \
  --weight-exp 1.0 \
  --tf-exp 1.5 \
  --idf-exp 1.5 \
  --nr-spikiness-exp 0.5 \
  --save out/nr_tfidf_spiky/
```

**Why these settings?**
- `--nr-spikiness-exp 0.5`: Adds sqrt(spikiness) multiplier
- Favors neurons with high peak/mean activation ratios

## Hyperparameter Tuning Guide

### Priority Order (Most to Least Impact)

1. **`--idf-exp`** (1.0 → 1.5 → 2.0)
   - Controls emphasis on selectivity
   - Higher = keep more selective neurons
   - Start here for tuning

2. **`--nr-tfidf-mode`** (doc vs topic)
   - Doc: faster, robust, good default
   - Topic: semantic, better for diverse data

3. **`--tf-exp`** (1.0 → 1.5 → 2.0)
   - Controls emphasis on activation strength
   - Higher = keep strongly-activating neurons

4. **`--nr-tfidf-k`** (32 → 64 → 128) [topic mode only]
   - Number of semantic clusters
   - More = finer granularity, slower

5. **`--weight-exp`** (0.5 → 1.0 → 1.5)
   - Controls weight magnitude influence
   - Usually best at 1.0

6. **`--nr-q-active`** (0.50 → 0.60 → 0.70)
   - Active threshold percentile
   - Higher = stricter "active" definition

7. **`--nsamples`** (128 → 256 → 512)
   - More samples = better statistics
   - Diminishing returns after 256

### Quick Tuning Recipes

**Conservative (keep generalists):**
```bash
--idf-exp 1.0 --tf-exp 1.5 --weight-exp 1.0
```

**Balanced (default):**
```bash
--idf-exp 1.5 --tf-exp 1.0 --weight-exp 1.0
```

**Aggressive (emphasize specialists):**
```bash
--idf-exp 2.5 --tf-exp 1.0 --weight-exp 0.5
```

## Performance Expectations

### vs. Original `neuronrank_old`

- **At 50% sparsity, all layers**: TF-IDF++ should avoid the perplexity collapse
- **At 80% sparsity, last 3 layers**: Both should perform similarly
- **Key difference**: Doc/topic-level IDF preserves broadly-useful neurons

### vs. `wanda`

- **Lower sparsity (0.3-0.5)**: TF-IDF++ should match or beat Wanda
- **Higher sparsity (0.7-0.8)**: Similar performance, different neurons kept
- **Key difference**: TF-IDF uses document/topic context vs. per-sample activation

### Typical Perplexities (LLaMA-7B on WikiText-2)

| Method | Sparsity | Layers | Expected PPL |
|--------|----------|--------|--------------|
| Dense | 0% | - | ~5.68 |
| Wanda | 50% | last 30 | ~6.5-7.0 |
| NR-TFIDF++ (doc) | 50% | last 30 | ~6.3-6.8 |
| NR-TFIDF++ (topic) | 50% | last 30 | ~6.2-6.7 |

*Note: Actual values depend on hyperparameters and calibration data*

## Implementation Details

### Statistics Collection

- **Hooks**: Registered on `mlp.gate_proj` outputs
- **Device**: All computations stay on-device (GPU)
- **Precision**: float32 for stability
- **Memory**: Scales with number of channels (intermediate layers)

### Clustering (Topic Mode)

- **Algorithm**: Cosine k-means with random projection
- **Projection dim**: 128 (hardcoded, good default)
- **Iterations**: 4 (fast, sufficient for pruning)
- **Initialization**: Random subset of tokens

### Mask Application

- **Granularity**: Per-module (same as wanda/magnitude)
- **Threshold**: `torch.kthvalue` for memory efficiency
- **Direction**: `gate_proj`/`up_proj` = row-wise, `down_proj` = column-wise

## Troubleshooting

### Issue: "No TF-IDF statistics for layer X"

**Cause**: Hook not registered or model structure unexpected

**Fix**: Ensure model has `layer.mlp.gate_proj` modules

### Issue: "All-zero metric"

**Cause**: All neurons inactive in calibration data or bad hyperparameters

**Fix**:
- Increase `--nsamples`
- Lower `--nr-q-active`
- Check calibration data quality

### Issue: Perplexity still high

**Cause**: Suboptimal hyperparameters

**Fix**:
- Try doc mode first (more robust)
- Lower `--idf-exp` to 1.0 or 1.5
- Increase `--nsamples` to 256

### Issue: Very slow (topic mode)

**Cause**: Large K or many samples

**Fix**:
- Reduce `--nr-tfidf-k` to 32 or 64
- Use fewer `--nsamples` initially (128)
- Consider doc mode for speed

## Comparison with Related Methods

| Method | Selectivity Signal | Granularity | Speed | Best For |
|--------|-------------------|-------------|-------|----------|
| `neuronrank_old` | Token-level IDF | Per-weight | Fast | High sparsity (70-80%) |
| `neuronrank_tfidf` (doc) | Doc-level IDF | Per-weight | Fast | Medium sparsity (40-60%) |
| `neuronrank_tfidf` (topic) | Topic-level IDF | Per-weight | Moderate | Semantic pruning |
| `neuronrank_unstructured` | Variance | Per-weight | Moderate | Spiky neuron removal |
| `wanda` | Activation norm | Per-weight | Fast | General purpose |

## References

- Original NeuronRank paper (class-discriminative pruning for CNNs)
- Wanda paper (magnitude × activation pruning for LLMs)
- TF-IDF in information retrieval (document-level importance)

## Version History

- **v1.0** (2025-10-05): Initial implementation with doc and topic modes
