# Implementation Summary: NeuronRank-OLD

## What Was Implemented

A new pruning method `neuronrank_old` that uses the TF-IDF formula:

```
score = |W|^α × TF^β × IDF^γ
```

Where:
- **|W|^α**: Weight magnitude (L2 norm) raised to exponent α
- **TF^β**: Term Frequency (average absolute activation) raised to exponent β  
- **IDF^γ**: Inverse Document Frequency (selectivity measure) raised to exponent γ

## Files Modified

### 1. `lib/neuronrank.py`
**Added:**
- `TFIDFStats` class: Tracks activation statistics for TF-IDF computation
- `collect_neuronrank_old_statistics()`: Collects TF and IDF statistics during forward passes
- `compute_neuronrank_old_scores()`: Computes final scores using the TF-IDF formula with configurable exponents

**Key Features:**
- Tracks total tokens processed
- Computes average activation strength (TF)
- Computes selectivity based on how rarely neurons fire (IDF)
- Applies configurable exponents to each component

### 2. `lib/prune.py`
**Added:**
- `prune_neuronrank_old()`: Main pruning function
  - Supports both structured and unstructured pruning
  - Works with `--pruning_last` flag to prune only specific layers
  - Handles `should_prune_module()` checks for layer selection
  - Broadcasts neuron scores to weight dimensions for unstructured pruning

### 3. `main.py`
**Added:**
- Import for `prune_neuronrank_old`
- Added `"neuronrank_old"` to `--prune_method` choices
- Three new command-line arguments:
  - `--weight-exp`: Exponent α for weight magnitude (default: 1.0)
  - `--tf-exp`: Exponent β for TF/activation strength (default: 1.0)
  - `--idf-exp`: Exponent γ for IDF/selectivity (default: 1.0)
- Dispatch logic to call `prune_neuronrank_old()` when method is selected

## Files Created

### 1. `NEURONRANK_OLD.md`
Comprehensive documentation including:
- Detailed formula explanation
- Usage examples
- Tuning guide with recommended starting points
- Interpretation guide for understanding neuron roles
- Comparison with other methods
- Troubleshooting section

### 2. `NEURONRANK_OLD_QUICK.md`
Quick reference guide with:
- Formula summary
- Command templates
- Key examples
- Tuning tips
- Component interpretation matrix

### 3. `example_neuronrank_old.sh`
Executable bash script with 8 example commands demonstrating:
- Basic usage
- Last-layer pruning
- Different exponent configurations
- Pure magnitude mode
- Pure TF-IDF mode

## How It Works

### Statistics Collection Phase
1. Forward pass calibration data through model
2. Hook into MLP gate projections
3. For each neuron, track:
   - Sum of absolute activations → compute TF
   - Count of tokens where activation > 0 → compute IDF
   - Total tokens processed

### Score Computation Phase
1. Extract weight magnitudes (L2 norm per neuron)
2. Compute TF = sum(|activations|) / total_tokens
3. Compute IDF = log((T+1)/(n_active+1)) + 1
4. Apply exponents: `|W|^α`, `TF^β`, `IDF^γ`
5. Multiply components: `score = |W|^α × TF^β × IDF^γ`

### Pruning Phase
**Unstructured:**
- Broadcast neuron scores to weight dimensions
- Prune weights with lowest scores
- Respects `--pruning_last` flag

**Structured:**
- Prune entire neurons with lowest scores
- Zero out corresponding rows in gate_proj/up_proj
- Zero out corresponding columns in down_proj

## Usage Examples

### Basic Command
```bash
python3 main.py \
  --model baffo32/decapoda-research-llama-7B-hf \
  --prune_method neuronrank_old \
  --sparsity_ratio 0.5 \
  --sparsity_type unstructured
```

### With Custom Exponents
```bash
python3 main.py \
  --model baffo32/decapoda-research-llama-7B-hf \
  --prune_method neuronrank_old \
  --sparsity_ratio 0.8 \
  --sparsity_type unstructured \
  --pruning_last 3 \
  --weight-exp 1.5 \
  --tf-exp 1.0 \
  --idf-exp 2.0
```

## Key Features

✅ **Flexible Scoring**: Three tunable exponents for different pruning strategies
✅ **Layer Selection**: Works with `--pruning_last` flag  
✅ **Both Modes**: Supports structured and unstructured pruning
✅ **Selectivity Aware**: IDF component identifies sparse/selective neurons
✅ **Activation-Based**: TF component captures activation strength
✅ **Well-Documented**: Comprehensive docs and examples

## Testing

All Python files compile without syntax errors:
- ✅ `lib/neuronrank.py`
- ✅ `lib/prune.py`
- ✅ `main.py`

The implementation is ready to use once PyTorch dependencies are available.

## Debug Output

The implementation includes informative print statements:
- TF and IDF ranges per layer during statistics collection
- Score ranges per layer during score computation
- Pruning details per layer

Example output:
```
Layer 29: TF range [0.001234, 0.456789], IDF range [1.234567, 3.456789], tokens=16384
Layer 29 scores: min=1.23e-03, max=4.56e+00, mean=1.23e+00
[NeuronRank-OLD] layer 29 mlp.gate_proj pruned 8806/11008
```

## Next Steps

To test the implementation:
1. Ensure PyTorch and transformers are installed
2. Run one of the examples from `example_neuronrank_old.sh`
3. Experiment with different exponent values
4. Compare results with other pruning methods

## Formula Interpretation

| Exponent | Effect | Use Case |
|----------|--------|----------|
| α > 1 | Emphasize large weights | Low sparsity |
| β > 1 | Emphasize strong activations | Preserve "loud" neurons |
| γ > 1 | Emphasize selectivity | High sparsity, preserve specialists |
| = 0 | Disable component | Focus on other factors |

The default (all 1.0) provides a balanced scoring approach.
