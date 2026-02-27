# POCD-KansformerEPI v6 — Execution Guide

## Hardware Requirements
- **GPU**: NVIDIA RTX 5070 Ti (16 GB GDDR7) — sufficient
- **RAM**: 32 GB DDR5
- **OS**: Windows 10/11
- **Python**: 3.13 (venv at `E:\AI Models\.venv`)

## Architecture Summary
Dual-branch model: Sequence (POCD-ND encoding → CNN → BiLSTM) + Epigenetic (CNN → BiLSTM) → KAN-Transformer (3 layers, 6 heads, d_model=180, KAN hidden=64) → Self-Attention Pooling → Enhancer/Promoter index extraction → FC classification head + distance regression head.

## Data
Data is shared from KansformerEPI (hardlinked). No additional downloads needed.

| Directory | Contents |
|---|---|
| `data/BENGI/` | 10 BENGI v3 benchmark TSV.gz files (6 cell lines) |
| `data/genomic_data/processed/` | 48 pre-processed .pt epigenetic signal files (8 marks × 6 cells) |
| `data/genomic_data/CTCF_DNase_6histone_local.500.json` | Feature config mapping cells → marks → .pt files |

### Available Cell Lines
| Cell Line | BENGI Benchmarks | Recommended Role |
|---|---|---|
| GM12878 | HiC, CTCF-ChIAPET, RNAPII-ChIAPET | Train |
| HeLa | HiC, CTCF-ChIAPET, RNAPII-ChIAPET | Train |
| K562 | HiC | Train |
| IMR90 | HiC | Train |
| HMEC | HiC | Test |
| NHEK | HiC | Test |

## Training

### Command
```bash
cd "E:\AI Models\FYDP-Model"
python train.py --train-cells GM12878 HeLa K562 IMR90 --test-cells HMEC NHEK
```

### Full CLI Options
```
--config PATH         Config file (default: configs/config.yaml)
--train-cells CELLS   Cell lines for training (required)
--test-cells CELLS    Cell lines for testing (required)
--epochs N            Override max epochs (default: 100)
--patience N          Override early stopping patience (default: 15)
--batch-size N        Override batch size (default: 64)
--lr FLOAT            Override learning rate (default: 0.0001)
--output-dir PATH     Override output directory (default: ./output)
--device DEVICE       cuda, cpu, or auto (default: auto)
```

### Training Configuration
| Parameter | Value |
|---|---|
| Epochs | 100 (early stopping) |
| Patience | 15 |
| Batch size | 64 |
| Learning rate | 0.0001 (Adam) |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |
| Gradient clipping | 1.0 |
| Augmentation | ON (reverse complement, epi noise, bin shift) |
| Val split | Chromosome-based (chr11, chr17) within train cells |

### Loss Functions
| Loss | Weight | Purpose |
|---|---|---|
| BCEWithLogitsLoss | 1.0 | Binary classification |
| MSELoss | 0.1 (`lambda_dist`) | Distance regression |
| Frobenius norm ‖AAᵀ − I‖_F | 0.1 (`att_penalty_coeff`) | Attention diversity penalty |

### Performance Metrics
- **AUROC** (Area Under ROC Curve)
- **AUPR** (Area Under Precision-Recall Curve)
- Early stopping monitors: val AUROC + val AUPR (combined)

## Evaluation (Standalone)

```bash
python evaluate.py --test-cells HMEC NHEK
```

### Options
```
--config PATH         Config file (default: configs/config.yaml)
--checkpoint PATH     Model checkpoint (default: output/model_best.pth)
--test-cells CELLS    Cell lines to evaluate on (required)
--device DEVICE       cuda or cpu (default: auto)
```

Prints per-cell-line breakdown: AUROC, AUPR, F1, Precision, Recall, Confusion Matrix.

## Output Directory (`./output/`)

| File | Description |
|---|---|
| `model_best.pth` | Best checkpoint (highest val AUROC+AUPR) |
| `model_final.pth` | Final epoch checkpoint |
| `encoder.pkl` | Fitted POCD-ND encoder (needed for evaluation) |
| `loss.png` | Train/val loss curves plot |
| `results.npz` | Predictions, labels, metrics for val and test sets |
| `config_snapshot.yaml` | Full config + cell line assignment used |
| `eval_results.npz` | Standalone evaluation results (after running evaluate.py) |

## Estimated Training Time
- ~4–6 hours on RTX 5070 Ti for 100 epochs (with early stopping likely ~60–80 epochs)
- Data loading: ~2–5 min initial (POCD-ND encoder fitting)
- Per-epoch: ~3–4 min (depending on total train samples from 4 cell lines)

## Troubleshooting

| Issue | Fix |
|---|---|
| `No BENGI files found` | Verify `data/BENGI/` contains .tsv.gz files matching cell names |
| `CUDA out of memory` | Reduce `--batch-size 32` |
| `pysam`/`pyBigWig` import error | Not needed — pre-processed .pt files are used instead |
| Slow data loading | `num_workers: 0` is set for Windows compatibility |
| `KeyError` for cell in feats_config | Ensure JSON keys match BENGI cell names exactly |
