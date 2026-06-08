# SAMN-EPI: Complete Guide

## Replacing KANTransformer with Selective Ancestral Memory Network

This guide covers everything you need to train and evaluate the SAMN-EPI model
for enhancer-promoter interaction prediction using cross-cell-line evaluation.

---

## 1. What Changed

| Component | Original (KansformerEPI) | New (SAMN-EPI) |
|-----------|--------------------------|----------------|
| **Core Encoder** | KANTransformer (3 blocks, global attention + KAN FFN) | SAMN (4 blocks, local windowed attention + survival gates + ancestral memory) |
| **Parameters** | ~2.5M | ~4.1M |
| **Attention** | Full global O(n²) | Local window (±32 tokens) + cross-attention to memory bin |
| **Memory** | None | GRU-updated ancestral memory (16 slots × 96d) |
| **FFN** | KAN (B-spline basis) | Standard GELU FFN |
| **Classification Head** | KANLinear layers | Standard Linear layers |
| **Regularization** | Attention Frobenius penalty | Gate entropy + slot diversity penalty |

**Unchanged**: Dual-branch CNN+BiLSTM feature extraction, POCD-ND encoding,
epigenetic feature processing, enhancer/promoter index extraction.

---

## 2. Files Created

```
POCD-KansformerEPI/
├── src/
│   └── samn_model.py          ← Self-contained SAMN + EPI model (NEW)
├── configs/
│   └── config_samn.yaml       ← SAMN configuration (NEW)
├── train_samn.py              ← Training script (NEW)
├── evaluate_samn.py           ← Evaluation script (NEW)
├── test_samn_smoke.py         ← Smoke test (NEW)
└── SAMN_GUIDE.md              ← This guide (NEW)
```

All original files remain **completely untouched**.

---

## 3. Prerequisites

### Environment
- Python 3.8+
- PyTorch with CUDA support
- GPU: NVIDIA RTX 5070 Ti (12GB VRAM) or equivalent

### Required Packages
The existing `.venv` already has everything needed. If starting fresh:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install numpy scipy scikit-learn matplotlib pyyaml pysam pybigwig
```

### Data Files Required
```
data/
├── BENGI/
│   ├── GM12878.*.tsv.gz       ← Training
│   ├── HeLa.*.tsv.gz          ← Training
│   ├── K562.*.tsv.gz           ← Training
│   ├── IMR90.*.tsv.gz          ← Training
│   ├── HMEC.*.tsv.gz           ← Testing
│   └── NHEK.*.tsv.gz           ← Testing
├── genomic_data/
│   ├── CTCF_DNase_6histone_local.500.json
│   └── processed/
│       └── *.500bp.pt          ← Pre-processed epigenetic tracks
└── hg19.fa                     ← Reference genome
```

---

## 4. Training (Step-by-Step)

### Step 1: Navigate to the project directory
```powershell
cd D:\FYDP\POCD-KansformerEPI
```

### Step 2: Activate virtual environment
```powershell
.venv\Scripts\activate
```

### Step 3: Run the smoke test (verify everything works)
```powershell
python test_samn_smoke.py
```
Expected output: `=== ALL TESTS PASSED ===`

### Step 4: Start training (5 epochs)
```powershell
python train_samn.py --train-cells GM12878 HeLa K562 IMR90 --test-cells HMEC NHEK
```

This will:
1. Load BENGI data for all 6 cell lines
2. Fit the POCD-ND encoder on training sequences
3. Train for 5 epochs with early stopping (patience=15)
4. Validate on chr11/chr17 within training cell lines
5. Test on HMEC and NHEK (completely unseen cell lines)
6. Save model weights, loss plots, ROC/PR curves

### Optional: Customize training
```powershell
# More epochs
python train_samn.py --train-cells GM12878 HeLa K562 IMR90 --test-cells HMEC NHEK --epochs 50

# Custom learning rate
python train_samn.py --train-cells GM12878 HeLa K562 IMR90 --test-cells HMEC NHEK --lr 0.00005

# Larger batch size (if VRAM allows)
python train_samn.py --train-cells GM12878 HeLa K562 IMR90 --test-cells HMEC NHEK --batch-size 128

# Custom output directory
python train_samn.py --train-cells GM12878 HeLa K562 IMR90 --test-cells HMEC NHEK --output-dir ./output_samn_v2

# Force CPU (not recommended)
python train_samn.py --train-cells GM12878 HeLa K562 IMR90 --test-cells HMEC NHEK --device cpu --no-amp
```

---

## 5. What to Expect During Training

You'll see progress updates every 10 steps:
```
  [1/5] Step   10/234 | Loss: 0.7523 (cls:0.693 reg:0.052 samn:-0.007) | Gate: 0.013 | 2.1 batch/s | ETA: 107s
  [1/5] Step   20/234 | Loss: 0.7210 (cls:0.671 reg:0.043 samn:-0.006) | Gate: 0.014 | 2.2 batch/s | ETA: 97s
  ...
  Epoch   1/  5 [120s] | Train(AUC/AUPR): 0.5823/0.4912 | Val(AUC/AUPR): 0.5678/0.4756 | TrLoss: 0.6891 | VlLoss: 0.7012 ★ BEST
```

Key metrics to watch:
- **Loss components**: `cls` (classification), `reg` (distance), `samn` (gate regularization)
- **Gate mean**: Should stabilize around 0.01–0.03 (fraction of tokens surviving)
- **Val AUROC + AUPR**: Monitored for early stopping

---

## 6. Evaluation (After Training)

### Evaluate with best checkpoint
```powershell
python evaluate_samn.py --test-cells HMEC NHEK
```

### Evaluate with a specific checkpoint
```powershell
python evaluate_samn.py --test-cells HMEC NHEK --checkpoint output_samn/samn_model_best.pth
```

### Evaluate on different cell lines
```powershell
python evaluate_samn.py --test-cells GM12878 K562
```

---

## 7. Output Files

After training completes, `output_samn/` will contain:

| File | Description |
|------|-------------|
| `samn_model_best.pth` | Best model weights (by val AUROC+AUPR) |
| `samn_model_final.pth` | Final model weights (last epoch) |
| `encoder.pkl` | Fitted POCD-ND encoder (needed for evaluation) |
| `train_loss.png` | Training/validation loss curves |
| `test_roc_curve.png` | ROC curve with AUROC score |
| `test_pr_curve.png` | Precision-Recall curve with AUPR score |
| `results.npz` | All predictions, labels, and metrics |
| `config_snapshot.yaml` | Exact config used for this run |

---

## 8. Increasing Epochs Later

The config is set to 5 epochs for the initial run. To increase:

### Option A: Edit config file
Open `configs/config_samn.yaml` and change:
```yaml
training:
  epochs: 50    # Changed from 5
```

### Option B: CLI override (no file edit needed)
```powershell
python train_samn.py --train-cells GM12878 HeLa K562 IMR90 --test-cells HMEC NHEK --epochs 50
```

### Option C: Resume from checkpoint
```powershell
# Not yet implemented — would need to add --resume flag
# For now, just re-run with more epochs (training starts fresh)
```

---

## 9. Comparing SAMN-EPI vs Original KansformerEPI

To do a fair comparison, train the original model with the same cell-line split:
```powershell
# Original KansformerEPI
python train.py --train-cells GM12878 HeLa K562 IMR90 --test-cells HMEC NHEK --epochs 5

# SAMN-EPI
python train_samn.py --train-cells GM12878 HeLa K562 IMR90 --test-cells HMEC NHEK --epochs 5
```

Then compare the test AUROC and AUPR from both runs.

---

## 10. SAMN Hyperparameter Tuning

Key SAMN hyperparameters in `configs/config_samn.yaml` under `model.samn`:

| Parameter | Default | What it controls |
|-----------|---------|------------------|
| `num_layers` | 4 | Depth of SAMN encoder (more = deeper routing) |
| `local_window` | 64 | Size of local attention window |
| `bin_slots` | 16 | Number of ancestral memory slots |
| `bin_dim` | 96 | Dimension of memory vectors |
| `survivors_per_layer` | 8 | Tokens selected for memory per layer |
| `gate_temperature` | 1.0 | Lower = sharper survival decisions |
| `novelty_weight` | 0.35 | How much novelty drives survival scoring |
| `prediction_error_weight` | 0.20 | How much prediction error drives survival |
| `bin_decay` | 0.92 | How quickly old memories fade |
| `bin_update` | "gru" | Memory update method: "gru", "fifo", or "attention" |

### Tuning tips:
- If the model struggles with long-range dependencies: increase `bin_slots` (24 or 32)
- If VRAM is tight: reduce `local_window` to 32
- If training is unstable: reduce `gate_temperature` to 0.5
- If memory seems stale: increase `bin_decay` to 0.95

---

## 11. Troubleshooting

### CUDA Out of Memory
```powershell
# Reduce batch size
python train_samn.py --train-cells GM12878 HeLa K562 IMR90 --test-cells HMEC NHEK --batch-size 32

# Or disable mixed precision (uses more VRAM but sometimes helps stability)
python train_samn.py --train-cells GM12878 HeLa K562 IMR90 --test-cells HMEC NHEK --no-amp
```

### Import Errors
Make sure you're using the project's virtual environment:
```powershell
.venv\Scripts\activate
python -c "import torch; print(torch.cuda.is_available())"
```

### No BENGI Files Found
Check that cell line names match exactly: `GM12878`, `HeLa`, `K562`, `IMR90`, `HMEC`, `NHEK`

---

## Quick Start (Copy-Paste)

```powershell
cd D:\FYDP\POCD-KansformerEPI
.venv\Scripts\activate
python test_samn_smoke.py
python train_samn.py --train-cells GM12878 HeLa K562 IMR90 --test-cells HMEC NHEK
```

That's it! 🚀
