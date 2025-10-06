# Quick Reference: Full Model Pruning Flags

## TL;DR Command Patterns

### 🎯 Pattern 1: Full Model (Maximum Compression)
```bash
python main.py --model <MODEL> --prune_method neuronrank_tfidf \
  --sparsity_ratio 0.5 --nr-include-attention --nr-prune-lm-head \
  --nr-tfidf-mode doc --nsamples 128
```
**Prunes:** MLPs + Attention + LM Head (225 modules)

---

### 🛡️ Pattern 2: MLP-Only (Conservative)
```bash
python main.py --model <MODEL> --prune_method neuronrank_tfidf \
  --sparsity_ratio 0.5 --nr-skip-attention \
  --nr-tfidf-mode doc --nsamples 128
```
**Prunes:** MLPs only (96 modules)

---

### ⚡ Pattern 3: Last N Layers (Typical)
```bash
python main.py --model <MODEL> --prune_method neuronrank_tfidf \
  --sparsity_ratio 0.5 --pruning_last 30 \
  --nr-tfidf-mode doc --nsamples 128
```
**Prunes:** Last 30 MLP layers (90 modules)

---

### 🧠 Pattern 4: Semantic Full Model (Quality)
```bash
python main.py --model <MODEL> --prune_method neuronrank_tfidf \
  --sparsity_ratio 0.5 --nr-include-attention --nr-tfidf-mode topic \
  --nr-tfidf-k 128 --nsamples 256
```
**Prunes:** MLPs + Attention with semantic clustering (224 modules)

---

## Flag Quick Reference

| Flag | Default | Effect | Use When |
|------|---------|--------|----------|
| `--nr-include-attention` | ✅ ON | Prune attention | Want maximum compression |
| `--nr-skip-attention` | ❌ OFF | Skip attention | Want to preserve attention |
| `--nr-prune-lm-head` | ❌ OFF | Prune LM head | Want absolute maximum pruning |
| `--pruning_last N` | None | Last N MLPs only | Standard pruning approach |
| `--nr-tfidf-mode doc` | ✅ Default | Fast document-level | Most scenarios |
| `--nr-tfidf-mode topic` | Alternative | Semantic clustering | Quality-focused |

---

## Decision Tree

```
Do you want to prune attention layers?
│
├─ YES → Use --nr-include-attention (or omit, it's default)
│   │
│   ├─ Do you also want LM head?
│   │   ├─ YES → Add --nr-prune-lm-head
│   │   └─ NO  → Omit --nr-prune-lm-head
│   │
│   └─ Which mode?
│       ├─ Fast → --nr-tfidf-mode doc
│       └─ Quality → --nr-tfidf-mode topic --nr-tfidf-k 128
│
└─ NO → Use --nr-skip-attention
    │
    └─ All layers or last N?
        ├─ All → Omit --pruning_last
        └─ Last N → Add --pruning_last 30
```

---

## Module Count by Configuration

| Config | MLP | Attention | LM Head | Total | Hooks |
|--------|-----|-----------|---------|-------|-------|
| Full + LM | 96 | 128 | 1 | **225** | 160 |
| Full Model | 96 | 128 | 0 | **224** | 160 |
| MLP-Only | 96 | 0 | 0 | **96** | 32 |
| Last 30 MLP | 90 | 0 | 0 | **90** | 30 |

*For LLaMA-7B with 32 transformer layers*

---

## Performance Expectations (50% Sparsity)

| Config | Hooks | Time | PPL Impact | Speedup | Memory |
|--------|-------|------|------------|---------|--------|
| MLP-Only | 32 | ~15 min | +0.5 | ~1.3× | -30% |
| Full Model | 160 | ~20 min | +1.0 | ~1.8× | -45% |
| Topic Full | 160 | ~30 min | +0.8 | ~1.8× | -45% |

*Estimates for LLaMA-7B on H100 with 128 samples*

---

## Recommended Starting Point

```bash
# Test MLP-only first (safest)
python main.py \
  --model baffo32/decapoda-research-llama-7B-hf \
  --prune_method neuronrank_tfidf \
  --sparsity_ratio 0.5 \
  --nr-skip-attention \
  --nr-tfidf-mode doc \
  --nsamples 128 \
  --save out/test_mlp/

# If PPL is good, try full model
python main.py \
  --model baffo32/decapoda-research-llama-7B-hf \
  --prune_method neuronrank_tfidf \
  --sparsity_ratio 0.5 \
  --nr-include-attention \
  --nr-tfidf-mode doc \
  --nsamples 128 \
  --save out/test_full/
```

---

## Hyperparameter Tuning Priority

When full model PPL is too high:

1. **Increase `--idf-exp`** (1.5 → 2.0 → 2.5)
   - Favors more selective neurons
   
2. **Lower `--sparsity_ratio`** (0.5 → 0.4 → 0.3)
   - Prune less aggressively
   
3. **Try topic mode** (--nr-tfidf-mode topic --nr-tfidf-k 128)
   - Better semantic selectivity
   
4. **Increase `--nsamples`** (128 → 256)
   - Better statistics

5. **Adjust `--tf-exp`** (1.0 → 1.5)
   - Favor strongly-activating neurons

---

## Troubleshooting One-Liners

**Problem:** Attention not being pruned  
**Solution:** Remove `--pruning_last` or add `--nr-include-attention`

**Problem:** Too slow  
**Solution:** Use `--nr-tfidf-mode doc` and `--nsamples 128`

**Problem:** PPL too high  
**Solution:** Use `--nr-skip-attention` or increase `--idf-exp 2.0`

**Problem:** Not enough compression  
**Solution:** Add `--nr-include-attention --nr-prune-lm-head`

---

## See Also

- **`FULL_MODEL_PRUNING.md`** - Comprehensive guide with examples
- **`NEURONRANK_TFIDF_PLUS.md`** - Complete TF-IDF++ documentation
- **`EXTENSION_SUMMARY.md`** - Implementation details
- **`test_full_model_pruning.sh`** - Automated test suite
