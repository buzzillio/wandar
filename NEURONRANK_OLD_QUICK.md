# NeuronRank-OLD Quick Reference

## Formula
```
score = |W|^α × TF^β × IDF^γ
```

## Command Template
```bash
python3 main.py \
  --model <model_name> \
  --prune_method neuronrank_old \
  --sparsity_ratio <0.0-1.0> \
  --sparsity_type unstructured \
  --weight-exp <α> \
  --tf-exp <β> \
  --idf-exp <γ> \
  [--pruning_last <N>] \
  [--nsamples 128]
```

## Key Arguments

| Arg | Symbol | Default | Description |
|-----|--------|---------|-------------|
| `--weight-exp` | α | 1.0 | Weight magnitude importance |
| `--tf-exp` | β | 1.0 | Activation strength importance |
| `--idf-exp` | γ | 1.0 | Selectivity importance |

## Quick Examples

### 1. Default balanced (α=β=γ=1)
```bash
python3 main.py \
  --model baffo32/decapoda-research-llama-7B-hf \
  --prune_method neuronrank_old \
  --sparsity_ratio 0.5 \
  --sparsity_type unstructured
```

### 2. Last 3 layers, 80% sparsity
```bash
python3 main.py \
  --model baffo32/decapoda-research-llama-7B-hf \
  --prune_method neuronrank_old \
  --sparsity_ratio 0.8 \
  --sparsity_type unstructured \
  --pruning_last 3
```

### 3. Emphasize selectivity (γ=2.0)
```bash
python3 main.py \
  --model baffo32/decapoda-research-llama-7B-hf \
  --prune_method neuronrank_old \
  --sparsity_ratio 0.5 \
  --sparsity_type unstructured \
  --idf-exp 2.0
```

### 4. Pure magnitude (β=γ=0)
```bash
python3 main.py \
  --model baffo32/decapoda-research-llama-7B-hf \
  --prune_method neuronrank_old \
  --sparsity_ratio 0.5 \
  --sparsity_type unstructured \
  --tf-exp 0.0 \
  --idf-exp 0.0
```

### 5. Pure TF-IDF (α=0)
```bash
python3 main.py \
  --model baffo32/decapoda-research-llama-7B-hf \
  --prune_method neuronrank_old \
  --sparsity_ratio 0.5 \
  --sparsity_type unstructured \
  --weight-exp 0.0
```

## Tuning Tips

**High Sparsity (70-90%)**
- Increase selectivity: `--idf-exp 2.0`
- Preserve specialized neurons

**Moderate Sparsity (40-60%)**
- Use defaults: all exponents = 1.0
- Balanced approach

**Low Sparsity (10-30%)**
- Increase magnitude: `--weight-exp 2.0`
- Focus on weight importance

## Understanding Components

### TF (Term Frequency)
- **High**: Neuron activates strongly on average
- **Low**: Neuron has weak activations
- **Formula**: Average absolute activation

### IDF (Inverse Document Frequency)
- **High**: Neuron fires rarely (selective)
- **Low**: Neuron fires often (general)
- **Formula**: log((T+1)/(n_active+1)) + 1

### Interpretation Matrix

| TF | IDF | Importance | Example |
|----|-----|------------|---------|
| High | High | **CRITICAL** | Punctuation detector |
| High | Low | Important | General feature |
| Low | High | Moderate | Rare pattern |
| Low | Low | **PRUNE** | Dead neuron |

## See Full Documentation
- Detailed guide: `NEURONRANK_OLD.md`
- Examples: `example_neuronrank_old.sh`
