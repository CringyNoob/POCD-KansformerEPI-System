# POCD-KansformerEPI v7 — Changelog from v6

## Overview

v7 fixes three critical/moderate issues discovered during a deep audit comparing FYDP-Model against the base KansformerEPI. The most significant change is activating the previously dead sequence branch — the core innovation of this model over the base.

---

## Critical Fix: Dead Sequence Branch → Now Functional

### Problem (v6)
The POCD-ND sequence branch — the entire reason this model exists as a dual-branch architecture — was producing **all-zero tensors** for every sample.

**Root cause chain:**
1. `config.yaml` → `ref_genome: ""` (empty string)
2. `epi_data_pipeline.py` → `_extract_seq()` returned `"NNN...N"` (N-padded dummy) when no reference genome was provided
3. `encoding.py` → POCD-ND encoder's `kmer_to_idx` only contains ACGT k-mers; `"NNN"` is not in the vocabulary
4. `transform()` → returned all-zero tensor `(64, 6000)` for every sample

**Impact:** 128 sequence tokens of pure zeros were concatenated with 500 real epigenetic tokens. The transformer attended over 628 tokens, 20% of which were uninformative noise. The model was **strictly worse** than the single-branch base KansformerEPI.

### Fix (v7)

| File | Change |
|---|---|
| `src/epi_data_pipeline.py` | Replaced `pysam` (Linux-only) with `pyfaidx` (pure Python, Windows-compatible) |
| `src/epi_data_pipeline.py` | Rewrote `_extract_seq()` for pyfaidx slice API with proper `KeyError`/`ValueError` handling |
| `configs/config.yaml` | Set `ref_genome: "../Loco EPI-main/hg19.fa"` (3.2 GB hg19 reference genome) |
| `requirements.txt` | Added `pyfaidx>=0.7.0`, commented out `pysam` |

**Verification:** Smoke tested — 3000bp fetches return 100% ACGT sequences with zero Ns. The `.fai` index is auto-created on first load (~50s), subsequent loads are instant.

---

## Moderate Fix: Distance Loss Weight

### Problem (v6)
`lambda_dist: 0.1` — distance supervision was 10× weaker than the base model's implicit 1.0 weight.

The distance regression head teaches the model about enhancer-promoter spatial relationships. The genomic distance between enhance and promoter is a strong predictor of interaction, and weakening this signal hurts the model's ability to learn spatial context.

### Fix (v7)

| File | Change |
|---|---|
| `configs/config.yaml` | `lambda_dist: 0.1` → `lambda_dist: 1.0` |

**Loss formula now matches base:**
```
loss = BCE(logits, labels) + 1.0 * MSE(dist, pred_dist) + 0.1 * ‖AAᵀ − I‖_F
```

---

## Moderate Fix: Self-Attention Pooling Bias

### Problem (v6)
`SelfAttentionPooling.ws1` and `ws2` used `bias=False`, while the base KansformerEPI uses `bias=True` (PyTorch default). This removed 96 learnable parameters from the attention pooling layer.

### Fix (v7)

| File | Change |
|---|---|
| `src/model.py` | `nn.Linear(d_model, da, bias=False)` → `bias=True` |
| `src/model.py` | `nn.Linear(da, r, bias=False)` → `bias=True` |

---

## Summary of All Changed Files

| File | Changes |
|---|---|
| `src/epi_data_pipeline.py` | pysam → pyfaidx; `_extract_seq()` rewritten for pyfaidx API |
| `src/model.py` | `SelfAttentionPooling` bias=False → bias=True |
| `configs/config.yaml` | `ref_genome` set to hg19.fa path; `lambda_dist` 0.1 → 1.0 |
| `requirements.txt` | Added `pyfaidx>=0.7.0` |

---

## v6 vs v7 Config Comparison

| Parameter | v6 | v7 | Rationale |
|---|---|---|---|
| `ref_genome` | `""` (disabled) | `"../Loco EPI-main/hg19.fa"` | Activates sequence branch |
| `lambda_dist` | 0.1 | 1.0 | Matches base model distance supervision |
| Attention pooling bias | False | True | Matches base model (+96 params) |

---

## What Stays the Same from v6

These v6 features are **retained** as they are beneficial over the base:

| Feature | Value | Why kept |
|---|---|---|
| ReduceLROnPlateau scheduler | factor=0.5, patience=5 | Base has no scheduler; this helps convergence |
| Gradient clipping | 1.0 | Base has none; prevents exploding gradients |
| BiLSTM dropout | 0.1 | Base has 0.0; regularization |
| Data augmentation | RC + noise + shift | Base only has rand_shift |
| 8 epigenetic marks | Includes H3K27ac | Base has 7 (no H3K27ac); H3K27ac marks active enhancers |
| Cross-cell evaluation | Train on 4 cells, test on 2 | Harder than base's cross-chromosome, but demonstrates generalization |
| Epochs / Patience | 100 / 15 | Tuned for RTX 5070 Ti runtime |
| Batch size | 64 | Slightly larger than base's 32 |
| Attention penalty coeff | 0.1 | Same as base |

---

## Architecture (Unchanged)

```
Sequence Branch:  DNA (3000bp enh + 3000bp prom)
                  → POCD-ND encoding (64 channels × 6000 positions)
                  → 2-layer CNN → BiLSTM → 128 tokens × 180d

Epigenetic Branch: 9 channels (1 pos_enc + 8 marks) × 5000 bins
                   → CNN(k=11) → MaxPool1d(10) → BiLSTM → 500 tokens × 180d

Fusion:           cat([128 seq, 500 epi]) = 628 tokens × 180d
                  → Positional Encoding
                  → KAN-Transformer (3 layers, 6 heads, KAN hidden=64)
                  → Self-Attention Pooling (da=64, r=32)
                  → [enh_feat, prom_feat, attn_mean, attn_max] = 720d

Heads:            Classification: Dropout(0.2) → Linear(720,128) → KAN(128,64) → KAN(64,1)
                  Distance:       Dropout(0.2) → KAN(720,180) → KAN(180,1)
```

---

## Expected Impact

| Aspect | v6 (broken) | v7 (fixed) |
|---|---|---|
| Sequence tokens | All zeros (dead) | Real ACGT k-mer patterns |
| Effective input | 500 epi tokens + 128 noise | 500 epi + 128 informative seq |
| Distance learning | Weak (0.1×) | Full strength (1.0×) |
| Attention pooling | Missing bias params | Full capacity |
| Cross-cell generalization | Poor (no sequence signal) | Better (DNA is cell-type-invariant) |

The sequence branch provides cell-type-invariant DNA composition features. Since the model is evaluated on entirely different cell lines (cross-cell), having a strong sequence signal is essential — epigenetic features shift across cell types, but DNA sequence does not.
