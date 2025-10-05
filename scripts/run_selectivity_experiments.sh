#!/bin/bash
# Experiment script to compare Wanda with selectivity enhancements
# This script tests the three variants:
# 1. Wanda × IDF(p) - penalizes always-on channels
# 2. Wanda × (top-q / mean) - rewards peaky specialists  
# 3. Wanda × IDF × (top-q / mean) - combined selectivity

MODEL=${1:-"decapoda-research/llama-7b-hf"}
SPARSITY=${2:-"0.5"}
SAVE_DIR="out/llama_selectivity"

echo "Running selectivity experiments on ${MODEL} with sparsity ${SPARSITY}"

# Baseline: Original Wanda
echo "=========================================="
echo "Running baseline: Original Wanda"
echo "=========================================="
python main.py \
    --model ${MODEL} \
    --prune_method wanda \
    --sparsity_ratio ${SPARSITY} \
    --sparsity_type unstructured \
    --save ${SAVE_DIR}/wanda/

# Variant 1: Wanda × IDF
echo "=========================================="
echo "Running Wanda × IDF (penalize always-on)"
echo "=========================================="
python main.py \
    --model ${MODEL} \
    --prune_method wanda_idf \
    --sparsity_ratio ${SPARSITY} \
    --sparsity_type unstructured \
    --save ${SAVE_DIR}/wanda_idf/

# Variant 2: Wanda × Spikiness
echo "=========================================="
echo "Running Wanda × Spikiness (reward specialists)"
echo "=========================================="
python main.py \
    --model ${MODEL} \
    --prune_method wanda_spiky \
    --sparsity_ratio ${SPARSITY} \
    --sparsity_type unstructured \
    --save ${SAVE_DIR}/wanda_spiky/

# Variant 3: Wanda × IDF × Spikiness (full selectivity)
echo "=========================================="
echo "Running Wanda × IDF × Spikiness (combined)"
echo "=========================================="
python main.py \
    --model ${MODEL} \
    --prune_method wanda_select \
    --sparsity_ratio ${SPARSITY} \
    --sparsity_type unstructured \
    --save ${SAVE_DIR}/wanda_select/

echo "=========================================="
echo "All experiments completed!"
echo "Results saved to ${SAVE_DIR}"
echo "=========================================="

# Print summary
echo ""
echo "Summary of results:"
echo "-------------------"
for method in wanda wanda_idf wanda_spiky wanda_select; do
    if [ -f "${SAVE_DIR}/${method}/log_${method}.txt" ]; then
        echo "${method}:"
        cat "${SAVE_DIR}/${method}/log_${method}.txt"
        echo ""
    fi
done

