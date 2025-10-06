#!/bin/bash

echo "========================================="
echo "Testing NeuronRank TF-IDF++ Implementation"
echo "========================================="
echo ""

echo "Test 1: Doc mode with minimal settings"
python main.py \
  --model baffo32/decapoda-research-llama-7B-hf \
  --prune_method neuronrank_tfidf \
  --sparsity_ratio 0.5 \
  --pruning_last 3 \
  --nr-tfidf-mode doc \
  --nsamples 32 \
  --save out/test_doc/

echo ""
echo "========================================="
echo "Test 2: Topic mode with minimal settings"
echo "========================================="
python main.py \
  --model baffo32/decapoda-research-llama-7B-hf \
  --prune_method neuronrank_tfidf \
  --sparsity_ratio 0.5 \
  --pruning_last 3 \
  --nr-tfidf-mode topic \
  --nr-tfidf-k 32 \
  --nsamples 32 \
  --save out/test_topic/
