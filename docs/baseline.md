# Baseline — POCD-KansformerEPI Execution Guide

The baseline dual-branch model: POCD-ND sequence encoding (CNN + BiLSTM) and
epigenetic signals (CNN + BiLSTM) fused by a KAN-Transformer, with a
classification head and an auxiliary genomic-distance regression head.

## Files

| Path | Purpose |
|---|---|
| `configs/baseline.yaml` | Hyperparameters and data paths |
| `scripts/train_baseline.py` | Training script |
| `scripts/evaluate_baseline.py` | Standalone evaluation |
| `src/baseline_model.py` | `Kansformer` architecture |
| `src/model_layers.py` | `KANLinear` / `KAN` layers |
| `results/baseline/` | Checkpoints, encoder, plots, and metrics |

## Data

Shared across all stages (see [datasets.md](datasets.md)):

| Directory | Contents |
|---|---|
| `data/BENGI/` | BENGI benchmark `.tsv.gz` files (6 cell lines) |
| `data/genomic_data/processed/` | Pre-processed `.pt` epigenetic signal files |
| `data/genomic_data/*.json` | Feature config mapping cells → marks → `.pt` files |
| `data/hg19.fa` | (Optional) hg19 reference genome for the sequence branch |

### Cell lines

| Cell line | Recommended role |
|---|---|
| GM12878, HeLa, K562, IMR90 | Train |
| HMEC, NHEK | Test |

## Training

```bash
python scripts/train_baseline.py --train-cells GM12878 HeLa K562 IMR90 --test-cells HMEC NHEK
```

### CLI options

```
--config PATH         Config file (default: configs/baseline.yaml)
--train-cells CELLS   Cell lines for training (required)
--test-cells CELLS    Cell lines for testing (required)
--epochs N            Override max epochs
--patience N          Override early-stopping patience
--batch-size N        Override batch size
--lr FLOAT            Override learning rate
--output-dir PATH     Override output directory (default: results/baseline)
--device DEVICE       cuda, cpu, or auto
```

### Loss

```
loss = BCE(logits, labels) + lambda_dist * MSE(dist, pred_dist) + att_penalty * ||AAᵀ − I||_F
```

Early stopping monitors the combined validation AUROC + AUPR. The
chromosome-based validation split (`chr11`, `chr17`) is held out within the
training cell lines.

## Evaluation

```bash
python scripts/evaluate_baseline.py --test-cells HMEC NHEK
python scripts/evaluate_baseline.py --test-cells HMEC NHEK --checkpoint results/baseline/model_best.pth
```

Prints a per-cell-line breakdown: AUROC, AUPR, F1, Precision, Recall, and
confusion matrix.

## Output (`results/baseline/`)

| File | Description |
|---|---|
| `model_best.pth` | Best checkpoint (highest val AUROC + AUPR) |
| `model_final.pth` | Final-epoch checkpoint |
| `encoder.pkl` | Fitted POCD-ND encoder (needed for evaluation) |
| `loss.png` | Train/val loss curves |
| `results.npz` | Predictions, labels, and metrics |
| `config_snapshot.yaml` | Full config + cell-line assignment used for the run |
| `eval_results.npz` | Standalone evaluation results |

## Troubleshooting

| Issue | Fix |
|---|---|
| `No BENGI files found` | Check `data/BENGI/` contains `.tsv.gz` files matching the cell names |
| `CUDA out of memory` | Lower `--batch-size` |
| `pyfaidx` import warning | Optional — without a reference genome the sequence branch uses dummy input |
| Slow data loading | `num_workers: 0` is set for Windows compatibility |
