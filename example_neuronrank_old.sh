#!/bin/bash

# Example usage of the new NeuronRank-OLD method with TF-IDF formula
# Formula: score = |W|^α × TF^β × IDF^γ

# Basic usage with default exponents (all 1.0)
echo "=== Basic NeuronRank-OLD (structured, 50% sparsity) ==="
python3 main.py \
  --model baffo32/decapoda-research-llama-7B-hf \
  --prune_method neuronrank_old \
  --sparsity_ratio 0.5 \
  --sparsity_type unstructured \
  --nsamples 128

# Unstructured pruning on last 3 layers (80% sparsity)
echo -e "\n=== NeuronRank-OLD: Last 3 layers, 80% sparsity ==="
python3 main.py \
  --model baffo32/decapoda-research-llama-7B-hf \
  --prune_method neuronrank_old \
  --sparsity_ratio 0.8 \
  --sparsity_type unstructured \
  --pruning_last 3 \
  --nsamples 128

# Emphasize weight magnitude (α=2.0)
echo -e "\n=== Emphasize weight magnitude (α=2.0) ==="
python3 main.py \
  --model baffo32/decapoda-research-llama-7B-hf \
  --prune_method neuronrank_old \
  --sparsity_ratio 0.5 \
  --sparsity_type unstructured \
  --weight-exp 2.0 \
  --tf-exp 1.0 \
  --idf-exp 1.0 \
  --nsamples 128

# Emphasize selectivity (γ=2.0) - prefer neurons that fire rarely
echo -e "\n=== Emphasize selectivity/IDF (γ=2.0) ==="
python3 main.py \
  --model baffo32/decapoda-research-llama-7B-hf \
  --prune_method neuronrank_old \
  --sparsity_ratio 0.5 \
  --sparsity_type unstructured \
  --weight-exp 1.0 \
  --tf-exp 1.0 \
  --idf-exp 2.0 \
  --nsamples 128

# Emphasize activation strength (β=2.0) - prefer neurons that activate strongly
echo -e "\n=== Emphasize TF/activation strength (β=2.0) ==="
python3 main.py \
  --model baffo32/decapoda-research-llama-7B-hf \
  --prune_method neuronrank_old \
  --sparsity_ratio 0.5 \
  --sparsity_type unstructured \
  --weight-exp 1.0 \
  --tf-exp 2.0 \
  --idf-exp 1.0 \
  --nsamples 128

# Balanced formula with custom exponents
echo -e "\n=== Balanced custom formula (α=1.5, β=1.0, γ=1.5) ==="
python3 main.py \
  --model baffo32/decapoda-research-llama-7B-hf \
  --prune_method neuronrank_old \
  --sparsity_ratio 0.5 \
  --sparsity_type unstructured \
  --weight-exp 1.5 \
  --tf-exp 1.0 \
  --idf-exp 1.5 \
  --nsamples 128

# Pure weight magnitude (β=0, γ=0) - equivalent to magnitude pruning
echo -e "\n=== Pure weight magnitude (β=0, γ=0) ==="
python3 main.py \
  --model baffo32/decapoda-research-llama-7B-hf \
  --prune_method neuronrank_old \
  --sparsity_ratio 0.5 \
  --sparsity_type unstructured \
  --weight-exp 1.0 \
  --tf-exp 0.0 \
  --idf-exp 0.0 \
  --nsamples 128

# Pure TF-IDF (α=0) - ignore weight magnitude
echo -e "\n=== Pure TF-IDF (α=0) ==="
python3 main.py \
  --model baffo32/decapoda-research-llama-7B-hf \
  --prune_method neuronrank_old \
  --sparsity_ratio 0.5 \
  --sparsity_type unstructured \
  --weight-exp 0.0 \
  --tf-exp 1.0 \
  --idf-exp 1.0 \
  --nsamples 128
