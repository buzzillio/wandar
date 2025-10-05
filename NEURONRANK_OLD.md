# NeuronRank-OLD: TF-IDF Formula for LLM Pruning

## Overview

The `neuronrank_old` method implements the original NeuronRank formula based on TF-IDF (Term Frequency - Inverse Document Frequency) principles. This method scores neurons by combining three key components:

1. **Weight Magnitude** - How large the weights are
2. **TF (Term Frequency)** - How strongly the neuron activates
3. **IDF (Inverse Document Frequency)** - How selective/sparse the neuron is

## Formula

The importance score for each neuron is computed as:

$$\text{score} = |W|^{\alpha} \times \text{TF}^{\beta} \times \text{IDF}^{\gamma}$$

### Components

#### 1. Weight Magnitude: $|W|^{\alpha}$
- **Definition**: L2 norm of the weight vector for each neuron
- **Interpretation**: Neurons with larger weights have more capacity to influence the model
- **Exponent α** (`--weight-exp`): Controls the importance of weight magnitude

#### 2. TF (Term Frequency): $\text{TF}^{\beta}$
- **Definition**: Average absolute activation strength across all tokens
  $$\text{TF} = \frac{1}{T}\sum_{t=1}^{T} |\text{activation}_t|$$
- **Interpretation**: Neurons that activate strongly are more important
- **Exponent β** (`--tf-exp`): Controls the importance of activation strength

#### 3. IDF (Inverse Document Frequency): $\text{IDF}^{\gamma}$
- **Definition**: Log-based selectivity measure
  $$\text{IDF} = \log\left(\frac{T + 1}{n_{\text{active}} + 1}\right) + 1$$
  where $n_{\text{active}}$ is the number of tokens where the neuron fired (activation > 0)
- **Interpretation**: 
  - **High IDF**: Neuron fires rarely (selective) - e.g., only on punctuation
  - **Low IDF**: Neuron fires frequently (not selective) - e.g., on every token
- **Exponent γ** (`--idf-exp`): Controls the importance of selectivity

## Usage

### Basic Command

```bash
python3 main.py \
  --model baffo32/decapoda-research-llama-7B-hf \
  --prune_method neuronrank_old \
  --sparsity_ratio 0.5 \
  --sparsity_type unstructured
```

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--weight-exp` (α) | 1.0 | Exponent for weight magnitude term |
| `--tf-exp` (β) | 1.0 | Exponent for TF (activation strength) |
| `--idf-exp` (γ) | 1.0 | Exponent for IDF (selectivity/sparsity) |
| `--pruning_last` | None | Only prune last X layers |
| `--nsamples` | 128 | Number of calibration samples |

### Pruning Modes

#### Unstructured Pruning (per-weight)
```bash
python3 main.py \
  --model MODEL \
  --prune_method neuronrank_old \
  --sparsity_ratio 0.8 \
  --sparsity_type unstructured
```

#### Structured Pruning (per-neuron)
```bash
python3 main.py \
  --model MODEL \
  --prune_method neuronrank_old \
  --sparsity_ratio 0.5 \
  --sparsity_type unstructured  # Note: structured coming soon
```

#### Layer-Specific Pruning
```bash
# Only prune the last 3 MLP layers
python3 main.py \
  --model MODEL \
  --prune_method neuronrank_old \
  --sparsity_ratio 0.8 \
  --sparsity_type unstructured \
  --pruning_last 3
```

## Tuning Guide

### Exponent Effects

#### Emphasize Weight Magnitude (α > 1)
- Use when you want to preserve neurons with large weights
- Example: `--weight-exp 2.0 --tf-exp 1.0 --idf-exp 1.0`
- **Effect**: Similar to magnitude pruning, but modulated by activation patterns

#### Emphasize Activation Strength (β > 1)
- Use when you want to preserve neurons that activate strongly
- Example: `--weight-exp 1.0 --tf-exp 2.0 --idf-exp 1.0`
- **Effect**: Keeps "loud" neurons that have high average activation

#### Emphasize Selectivity (γ > 1)
- Use when you want to preserve selective/sparse neurons
- Example: `--weight-exp 1.0 --tf-exp 1.0 --idf-exp 2.0`
- **Effect**: Keeps neurons that fire rarely but specifically (e.g., punctuation detectors)

#### De-emphasize a Component (exponent = 0)
- Use to completely remove a component from the formula
- Example: `--weight-exp 0.0` → ignores weight magnitude entirely

### Recommended Starting Points

#### For High Sparsity (70-90%)
```bash
--weight-exp 1.5 --tf-exp 1.0 --idf-exp 1.5
```
- Balances weight magnitude with selectivity
- Preserves important specialized neurons

#### For Moderate Sparsity (40-60%)
```bash
--weight-exp 1.0 --tf-exp 1.0 --idf-exp 1.0
```
- Default balanced formula
- Equal weight to all three components

#### For Low Sparsity (10-30%)
```bash
--weight-exp 2.0 --tf-exp 0.5 --idf-exp 0.5
```
- Focuses on weight magnitude
- Less aggressive than pure magnitude pruning

## Examples

### Example 1: Selective Neuron Preservation
```bash
# Prune last 3 layers at 80% sparsity, emphasizing selectivity
python3 main.py \
  --model baffo32/decapoda-research-llama-7B-hf \
  --prune_method neuronrank_old \
  --sparsity_ratio 0.8 \
  --sparsity_type unstructured \
  --pruning_last 3 \
  --weight-exp 1.0 \
  --tf-exp 1.0 \
  --idf-exp 2.0
```

### Example 2: Pure Activation-Based Pruning
```bash
# Ignore weights, only use activation statistics
python3 main.py \
  --model bafco32/decapoda-research-llama-7B-hf \
  --prune_method neuronrank_old \
  --sparsity_ratio 0.5 \
  --sparsity_type unstructured \
  --weight-exp 0.0 \
  --tf-exp 1.0 \
  --idf-exp 1.0
```

### Example 3: Magnitude + Selectivity
```bash
# Combine weight magnitude with selectivity, ignore TF
python3 main.py \
  --model baffo32/decapoda-research-llama-7B-hf \
  --prune_method neuronrank_old \
  --sparsity_ratio 0.5 \
  --sparsity_type unstructured \
  --weight-exp 1.5 \
  --tf-exp 0.0 \
  --idf-exp 1.5
```

## Interpretation Guide

### Understanding Neuron Roles

**High Weight + High TF + High IDF** (all components high)
- **Role**: Critical specialized detector
- **Example**: Detects specific rare tokens (e.g., code syntax)
- **Pruning**: KEEP these neurons

**High Weight + High TF + Low IDF** (always active)
- **Role**: General feature extractor
- **Example**: Detects common patterns in all text
- **Pruning**: Important but can sometimes be removed

**Low Weight + High TF + High IDF** (selective but small)
- **Role**: Specialized but potentially redundant
- **Example**: Detects rare patterns with small contribution
- **Pruning**: Can remove if other neurons compensate

**Low Weight + Low TF + Low IDF** (inactive)
- **Role**: Dead or nearly-dead neuron
- **Example**: Rarely activates and has minimal impact
- **Pruning**: REMOVE these first

## Comparison with Other Methods

| Method | Formula | Best For |
|--------|---------|----------|
| **magnitude** | $|W|$ | Quick pruning, good baseline |
| **wanda** | $|W| \times \|\|X\|\|$ | Balances weights and activations |
| **neuronrank** | variance-based | Captures activation diversity |
| **neuronrank_old** | $|W|^{\alpha} \times \text{TF}^{\beta} \times \text{IDF}^{\gamma}$ | Flexible, captures selectivity |

### When to Use NeuronRank-OLD

✅ **Use when:**
- You want to preserve selective/sparse neurons
- You need fine control over different importance factors
- You're pruning later layers (selectivity matters more)
- You want to experiment with different scoring strategies

❌ **Don't use when:**
- You need the fastest possible pruning (use magnitude)
- You don't have calibration data
- You're only doing very low sparsity (<10%)

## Technical Details

### Statistics Collection

The method processes calibration data through the model and tracks:
1. **Total tokens processed** ($T$)
2. **Sum of absolute activations** per neuron (for TF)
3. **Count of active tokens** per neuron (for IDF)

All statistics are collected after the SiLU activation function.

### Score Computation

For each neuron $i$ in layer $l$:
1. Compute weight magnitude: $|W_i| = \|\mathbf{w}_i\|_2$
2. Compute TF: $\text{TF}_i = \frac{\sum_t |a_{i,t}|}{T}$
3. Compute IDF: $\text{IDF}_i = \log\left(\frac{T + 1}{n_{i,\text{active}} + 1}\right) + 1$
4. Apply exponents and multiply: $s_i = |W_i|^{\alpha} \times \text{TF}_i^{\beta} \times \text{IDF}_i^{\gamma}$

### Pruning Strategy

- **Unstructured**: Each weight inherits the score of its corresponding neuron
  - `gate_proj`, `up_proj`: score broadcast across columns
  - `down_proj`: score broadcast across rows
- **Structured**: Entire neurons are removed based on scores

## Troubleshooting

**Issue**: All scores are very similar
- **Solution**: Increase one exponent (try α=2.0 or γ=2.0) to amplify differences

**Issue**: Perplexity explodes after pruning
- **Solution**: Reduce sparsity ratio or use `--pruning_last` to prune fewer layers

**Issue**: No difference between exponent values
- **Solution**: Check that neurons have varied activation patterns; try more calibration samples

**Issue**: Some layers have zero TF or IDF
- **Solution**: This is normal; those layers will fall back to weight magnitude only

## References

This method implements the original TF-IDF style scoring from early neural network pruning research, adapted for modern transformer architectures.
