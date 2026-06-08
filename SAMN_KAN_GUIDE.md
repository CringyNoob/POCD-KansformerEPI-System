# SAMN-KAN-EPI: Hybrid Model Complete Guide

## Architecture Overview

**SAMN-KAN-EPI** is a hybrid model that combines:
- **SAMN** (Selective Ancestral Memory Network) — replaces global self-attention
  with local windowed attention + survival gates + ancestral memory bin routing
- **KAN** (Kolmogorov-Arnold Network) — preserved as the FFN (feed-forward network)
  with B-spline basis function approximation

This keeps the **mathematical expressiveness** of KAN's B-spline function
approximation while adding SAMN's **noise-filtering local attention** and
**selective memory routing** for long-range enhancer-promoter dependencies.

## Files Created

| File | Purpose |
|------|---------|
| `src/samn_kan_model.py` | Hybrid SAMN-KAN model (4,251,155 params) |
| `configs/config_samn_kan.yaml` | Configuration (5 epochs, patience 15) |
| `train_samn_kan.py` | Training script (same phases as base model) |
| `evaluate_samn_kan.py` | Evaluation script (same metrics as base model) |
| `test_samn_kan_smoke.py` | Smoke test (forward + backward pass) |

## Prerequisites

Activate the virtual environment:
```powershell
cd D:\FYDP\POCD-KansformerEPI
.venv\Scripts\activate
```

Required packages (should already be installed from base model):
- PyTorch >= 2.0, scikit-learn, pyyaml, matplotlib, numpy

## Training

### Quick Start
```powershell
python train_samn_kan.py --train-cells GM12878 HeLa K562 IMR90 --test-cells HMEC NHEK
```

### Full Command with All Options
```powershell
python train_samn_kan.py `
    --config configs/config_samn_kan.yaml `
    --train-cells GM12878 HeLa K562 IMR90 `
    --test-cells HMEC NHEK `
    --epochs 5 `
    --patience 15 `
    --batch-size 64 `
    --device cuda
```

### What Happens During Training

1. **Data Loading**: Loads BENGI files for train cells (GM12878, HeLa, K562, IMR90)
   and test cells (HMEC, NHEK)
2. **POCD-ND Encoder Fitting**: Fits k-mer encoder on training sequences
3. **Train/Val Split**: Splits training data by chromosome (chr11, chr17 → validation)
4. **Training Loop** (per epoch):
   - Forward pass through hybrid SAMN-KAN encoder
   - Loss = BCE + MSE + Frobenius penalty + SAMN auxiliary losses
   - Progress logged every 10 steps
5. **Validation**: Evaluated on held-out chromosomes
6. **Final Evaluation**: Best checkpoint tested on HMEC/NHEK with comprehensive metrics

### Output Files

After training, `output_samn_kan/` will contain:

| File | Description |
|------|-------------|
| `model_best.pth` | Best checkpoint (highest val AUC+AUPR) |
| `model_final.pth` | Final epoch checkpoint |
| `encoder.pkl` | Fitted POCD-ND encoder |
| `loss.png` | Training/validation loss curve |
| `test_all_roc_curve.png` | Combined test ROC curve |
| `test_all_pr_curve.png` | Combined test PR curve |
| `test_HMEC_roc_curve.png` | HMEC-specific ROC curve |
| `test_NHEK_roc_curve.png` | NHEK-specific ROC curve |
| `results.npz` | All numerical results |
| `config_snapshot.yaml` | Frozen config for reproducibility |

## Evaluation Only

To re-evaluate a trained model:
```powershell
python evaluate_samn_kan.py --test-cells HMEC NHEK
python evaluate_samn_kan.py --test-cells HMEC NHEK --checkpoint output_samn_kan/model_best.pth
```

## Metrics Reported (Matching Base Model)

All three models (base Kansformer, pure SAMN, hybrid SAMN-KAN) now report:

| Metric | Description |
|--------|-------------|
| AUROC | Area Under ROC Curve |
| AUPR | Area Under Precision-Recall Curve |
| Accuracy | Overall accuracy |
| Balanced Accuracy | Average of per-class accuracy |
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| F1 Score | Harmonic mean of precision and recall |
| MCC | Matthews Correlation Coefficient |
| Confusion Matrix | TN, FP, FN, TP |

Reports are generated for:
- Combined test set (HMEC + NHEK)
- Per-cell-line breakdown (HMEC, NHEK separately)

## 3-Way Comparison

To run all three models for comparison:

```powershell
# 1. Base Kansformer (if not already trained)
python train.py --train-cells GM12878 HeLa K562 IMR90 --test-cells HMEC NHEK --epochs 5 --patience 15

# 2. Pure SAMN-EPI
python train_samn.py --train-cells GM12878 HeLa K562 IMR90 --test-cells HMEC NHEK

# 3. Hybrid SAMN-KAN-EPI
python train_samn_kan.py --train-cells GM12878 HeLa K562 IMR90 --test-cells HMEC NHEK
```

Results will be in: `output/`, `output_samn/`, `output_samn_kan/`

## Model Comparison Summary

| Feature | Kansformer | SAMN-EPI | SAMN-KAN-EPI |
|---------|-----------|----------|-------------|
| Parameters | 3,529,796 | 4,082,690 | 4,251,155 |
| Attention | Global | Local + Memory | Local + Memory |
| FFN | KAN (B-spline) | MLP (GELU) | KAN (B-spline) |
| Pooling | SelfAttn + Frob | Mean/Max/Bin | SelfAttn + Frob |
| Cls Head | KANLinear | Linear | KANLinear |
| Memory | None | Ancestral bin | Ancestral bin |

## Hyperparameter Tuning

Key hyperparameters in `configs/config_samn_kan.yaml`:

```yaml
# SAMN attention (new)
samn.local_window: 64       # Increase for broader context
samn.bin_slots: 16          # Memory capacity
samn.survivors_per_layer: 8 # Tokens routed to memory per layer

# KAN FFN (preserved from base)
model.kan_hidden: 64        # KAN intermediate dimension

# Loss weights
training.att_penalty_coeff: 0.1    # Frobenius penalty
training.entropy_loss_weight: 0.01 # SAMN gate entropy
training.diversity_loss_weight: 0.05 # SAMN slot diversity
```
