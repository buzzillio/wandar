#!/bin/bash

echo "========================================="
echo "NeuronRank TF-IDF++ Full Model Pruning"
echo "Comparison Test Suite"
echo "========================================="
echo ""

MODEL="baffo32/decapoda-research-llama-7B-hf"
SPARSITY=0.5
NSAMPLES=128

# Test 1: MLP-Only (Baseline)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Test 1: MLP-Only Pruning (Conservative)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python main.py \
  --model $MODEL \
  --prune_method neuronrank_tfidf \
  --sparsity_ratio $SPARSITY \
  --nr-skip-attention \
  --nr-tfidf-mode doc \
  --weight-exp 1.0 --tf-exp 1.0 --idf-exp 1.5 \
  --nsamples $NSAMPLES \
  --save out/compare_mlp_only/

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Test 2: Full Model Pruning (with Attention)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python main.py \
  --model $MODEL \
  --prune_method neuronrank_tfidf \
  --sparsity_ratio $SPARSITY \
  --nr-include-attention \
  --nr-tfidf-mode doc \
  --weight-exp 1.0 --tf-exp 1.0 --idf-exp 1.5 \
  --nsamples $NSAMPLES \
  --save out/compare_full_model/

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Test 3: Full Model + LM Head"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python main.py \
  --model $MODEL \
  --prune_method neuronrank_tfidf \
  --sparsity_ratio $SPARSITY \
  --nr-include-attention \
  --nr-prune-lm-head \
  --nr-tfidf-mode doc \
  --weight-exp 1.0 --tf-exp 1.0 --idf-exp 1.5 \
  --nsamples $NSAMPLES \
  --save out/compare_full_with_lm_head/

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Test 4: Last 30 Layers MLP-Only (Typical)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python main.py \
  --model $MODEL \
  --prune_method neuronrank_tfidf \
  --sparsity_ratio $SPARSITY \
  --pruning_last 30 \
  --nr-tfidf-mode doc \
  --weight-exp 1.0 --tf-exp 1.0 --idf-exp 1.5 \
  --nsamples $NSAMPLES \
  --save out/compare_last30/

echo ""
echo "========================================="
echo "Results Summary"
echo "========================================="
echo ""
echo "Check perplexity in the following files:"
echo "  - out/compare_mlp_only/log_neuronrank_tfidf.txt"
echo "  - out/compare_full_model/log_neuronrank_tfidf.txt"
echo "  - out/compare_full_with_lm_head/log_neuronrank_tfidf.txt"
echo "  - out/compare_last30/log_neuronrank_tfidf.txt"
echo ""
echo "Compare:"
echo "  - MLP-Only: ~32 modules, conservative"
echo "  - Full Model: ~160 modules, aggressive"
echo "  - Full + LM Head: ~161 modules, maximum"
echo "  - Last 30: ~90 modules, typical"
