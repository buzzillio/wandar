#!/bin/bash

echo "========================================="
echo "Hybrid Pruning Method - Comparison Suite"
echo "========================================="
echo ""

MODEL="baffo32/decapoda-research-llama-7B-hf"
SPARSITY=0.5
NSAMPLES=128

# Test 1: Pure Wanda (Baseline)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Test 1: Pure Wanda (Baseline)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python main.py \
  --model $MODEL \
  --prune_method wanda \
  --sparsity_ratio $SPARSITY \
  --nsamples $NSAMPLES \
  --save out/compare_wanda/

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Test 2: Pure NeuronRank TF-IDF++ (Baseline)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python main.py \
  --model $MODEL \
  --prune_method neuronrank_tfidf \
  --sparsity_ratio $SPARSITY \
  --nr-include-attention \
  --nr-tfidf-mode doc \
  --weight-exp 1.0 --tf-exp 1.0 --idf-exp 1.5 \
  --nsamples $NSAMPLES \
  --save out/compare_nr_tfidf/

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Test 3: Hybrid (Wanda + TF-IDF++ Doc)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python main.py \
  --model $MODEL \
  --prune_method hybrid \
  --sparsity_ratio $SPARSITY \
  --hybrid-mlp-method neuronrank_tfidf \
  --nr-tfidf-mode doc \
  --weight-exp 1.0 --tf-exp 1.0 --idf-exp 1.5 \
  --nsamples $NSAMPLES \
  --save out/compare_hybrid_doc/

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Test 4: Hybrid (Wanda + TF-IDF++ Topic)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python main.py \
  --model $MODEL \
  --prune_method hybrid \
  --sparsity_ratio $SPARSITY \
  --hybrid-mlp-method neuronrank_tfidf \
  --nr-tfidf-mode topic \
  --nr-tfidf-k 128 \
  --weight-exp 1.0 --tf-exp 1.5 --idf-exp 1.5 \
  --nsamples 256 \
  --save out/compare_hybrid_topic/

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Test 5: Hybrid (Wanda + NeuronRank OLD)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python main.py \
  --model $MODEL \
  --prune_method hybrid \
  --sparsity_ratio $SPARSITY \
  --hybrid-mlp-method neuronrank_old \
  --weight-exp 1.0 --tf-exp 1.0 --idf-exp 1.0 \
  --nsamples $NSAMPLES \
  --save out/compare_hybrid_old/

echo ""
echo "========================================="
echo "Results Summary"
echo "========================================="
echo ""
echo "Perplexity Comparison:"
echo ""
echo "Method                          | Perplexity"
echo "--------------------------------|------------"
grep -h "^wanda" out/compare_wanda/log_wanda.txt 2>/dev/null | awk '{printf "Pure Wanda                      | %s\n", $3}' || echo "Pure Wanda                      | Not found"
grep -h "^neuronrank_tfidf" out/compare_nr_tfidf/log_neuronrank_tfidf.txt 2>/dev/null | awk '{printf "Pure NeuronRank TF-IDF++        | %s\n", $3}' || echo "Pure NeuronRank TF-IDF++        | Not found"
grep -h "^hybrid" out/compare_hybrid_doc/log_hybrid.txt 2>/dev/null | awk '{printf "Hybrid (Wanda + TF-IDF Doc)     | %s\n", $3}' || echo "Hybrid (Wanda + TF-IDF Doc)     | Not found"
grep -h "^hybrid" out/compare_hybrid_topic/log_hybrid.txt 2>/dev/null | awk '{printf "Hybrid (Wanda + TF-IDF Topic)   | %s\n", $3}' || echo "Hybrid (Wanda + TF-IDF Topic)   | Not found"
grep -h "^hybrid" out/compare_hybrid_old/log_hybrid.txt 2>/dev/null | awk '{printf "Hybrid (Wanda + NR OLD)         | %s\n", $3}' || echo "Hybrid (Wanda + NR OLD)         | Not found"
echo ""
echo "Detailed logs available in:"
echo "  - out/compare_wanda/log_wanda.txt"
echo "  - out/compare_nr_tfidf/log_neuronrank_tfidf.txt"
echo "  - out/compare_hybrid_doc/log_hybrid.txt"
echo "  - out/compare_hybrid_topic/log_hybrid.txt"
echo "  - out/compare_hybrid_old/log_hybrid.txt"
