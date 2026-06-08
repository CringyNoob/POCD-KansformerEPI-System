"""
SAMN-KAN-EPI Evaluation Script — Cross-Cell-Line Testing

Loads a trained SAMN-KAN-EPI checkpoint and evaluates on specified test cell lines.
Generates metrics matching the base POCD-KansformerEPI model:
  AUROC, AUPR, Accuracy, Balanced Accuracy, Precision, Recall, F1, MCC,
  Confusion Matrix, per-cell-line breakdown.

Usage:
    python evaluate_samn_kan.py --test-cells HMEC NHEK
    python evaluate_samn_kan.py --test-cells HMEC NHEK --checkpoint output_samn_kan/model_best.pth
    python evaluate_samn_kan.py --config configs/config_samn_kan.yaml --test-cells HMEC NHEK
"""
import torch
import yaml
import numpy as np
import os
import glob
import pickle
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             average_precision_score, f1_score,
                             precision_score, recall_score,
                             balanced_accuracy_score,
                             confusion_matrix, matthews_corrcoef,
                             roc_curve, precision_recall_curve)
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast

from src.epi_data_pipeline import EPIGenomicDataset
from src.dataset import EPIDataset
from src.samn_kan_model import SAMNKANKansformerEPI
from src.encoding import POCD_ND_Encoder


def filter_bengi_files(bengi_dir, cell_names):
    """Return BENGI file paths matching any of the given cell line names."""
    all_files = sorted(
        glob.glob(os.path.join(bengi_dir, "*.tsv.gz")) +
        glob.glob(os.path.join(bengi_dir, "*.tsv"))
    )
    matched = []
    for f in all_files:
        basename = os.path.basename(f)
        for cell in cell_names:
            if basename.startswith(cell + ".") or basename.startswith(cell + "_"):
                matched.append(f)
                break
    return matched


def evaluate_split(model, loader, device, split_name="Test", use_amp=False):
    """Evaluate model on a DataLoader and compute comprehensive metrics."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            seq = batch["seq"].float().to(device)
            epi = batch["epi"].float().to(device)
            lbl = batch["label"].float().to(device)
            enh_idx = batch["enh_idx"].float().to(device)
            prom_idx = batch["prom_idx"].float().to(device)

            with autocast(device_type="cuda", enabled=use_amp and device.type == "cuda"):
                p_cls, _, _, _ = model(seq, epi, enh_idx, prom_idx)

            all_preds.extend(torch.sigmoid(p_cls.float()).cpu().numpy().flatten())
            all_labels.extend(lbl.cpu().numpy().flatten())

    preds_np = np.array(all_preds)
    labels_np = np.array(all_labels)
    binary_preds = (preds_np > 0.5).astype(int)

    acc = accuracy_score(labels_np, binary_preds)
    try:
        auc = roc_auc_score(labels_np, preds_np)
    except ValueError:
        auc = float("nan")
    try:
        aupr = average_precision_score(labels_np, preds_np)
    except ValueError:
        aupr = float("nan")
    bal_acc = balanced_accuracy_score(labels_np, binary_preds)
    f1 = f1_score(labels_np, binary_preds, zero_division=0)
    prec = precision_score(labels_np, binary_preds, zero_division=0)
    rec = recall_score(labels_np, binary_preds, zero_division=0)
    try:
        mcc = matthews_corrcoef(labels_np, binary_preds)
    except ValueError:
        mcc = 0.0
    cm = confusion_matrix(labels_np, binary_preds)

    print(f"\n{'=' * 50}")
    print(f"  {split_name} Results  ({len(labels_np)} samples)")
    print(f"{'=' * 50}")
    print(f"  AUROC:              {auc:.4f}")
    print(f"  AUPR:               {aupr:.4f}")
    print(f"  Accuracy:           {acc:.4f}")
    print(f"  Balanced Accuracy:  {bal_acc:.4f}")
    print(f"  Precision:          {prec:.4f}")
    print(f"  Recall:             {rec:.4f}")
    print(f"  F1 Score:           {f1:.4f}")
    print(f"  MCC:                {mcc:.4f}")
    print(f"\n  Confusion Matrix:")
    if cm.shape == (2, 2):
        print(f"    TN={cm[0, 0]:6d}  FP={cm[0, 1]:6d}")
        print(f"    FN={cm[1, 0]:6d}  TP={cm[1, 1]:6d}")
    else:
        print(f"    {cm}")
    print(f"\n  Pos: {int(labels_np.sum())} ({100 * labels_np.mean():.1f}%)  "
          f"Neg: {int(len(labels_np) - labels_np.sum())} "
          f"({100 * (1 - labels_np.mean()):.1f}%)")
    print(f"  Predicted Pos: {int(binary_preds.sum())}  "
          f"Predicted Neg: {int(len(binary_preds) - binary_preds.sum())}")
    print(f"{'=' * 50}")

    return {
        "accuracy": acc, "auroc": auc, "aupr": aupr,
        "balanced_accuracy": bal_acc,
        "f1": f1, "precision": prec, "recall": rec, "mcc": mcc,
        "confusion_matrix": cm,
        "predictions": preds_np, "labels": labels_np,
    }


def save_curves(labels, preds, save_dir, prefix="", model_name="SAMN-KAN-EPI"):
    """Generate and save ROC and PR curves."""
    # ROC
    fpr, tpr, _ = roc_curve(labels, preds)
    auc_val = roc_auc_score(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, "b-", linewidth=2,
            label=f"{model_name} (AUROC = {auc_val:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"{prefix}ROC Curve — {model_name}", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"{prefix}roc_curve.png"), dpi=150)
    plt.close(fig)
    print(f"  ROC curve → {prefix}roc_curve.png")

    # PR
    prec_arr, rec_arr, _ = precision_recall_curve(labels, preds)
    aupr_val = average_precision_score(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(rec_arr, prec_arr, "r-", linewidth=2,
            label=f"{model_name} (AUPR = {aupr_val:.4f})")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(f"{prefix}Precision-Recall Curve — {model_name}", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"{prefix}pr_curve.png"), dpi=150)
    plt.close(fig)
    print(f"  PR curve  → {prefix}pr_curve.png")


def main():
    parser = argparse.ArgumentParser(description="Evaluate SAMN-KAN-EPI model")
    parser.add_argument("--config", default="configs/config_samn_kan.yaml")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to model checkpoint (default: best)")
    parser.add_argument("--test-cells", nargs="+", required=True,
                        help="Cell lines to evaluate on")
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device(
        args.device if args.device else
        ("cuda" if torch.cuda.is_available() else "cpu"))
    save_dir = config["paths"]["save_dir"]

    use_amp = config["training"].get("use_amp", True) and not args.no_amp
    if device.type != "cuda":
        use_amp = False

    print("=" * 55)
    print("  SAMN-KAN-EPI Evaluation")
    print("=" * 55)
    print(f"  Device:     {device}")
    if device.type == "cuda":
        print(f"  GPU:        {torch.cuda.get_device_name(0)}")
    print(f"  Mixed Prec: {'ON' if use_amp else 'OFF'}")
    print(f"  Config:     {args.config}")
    print(f"  Test cells: {args.test_cells}")

    # Load encoder
    encoder_path = os.path.join(save_dir, "encoder.pkl")
    if not os.path.exists(encoder_path):
        print(f"ERROR: Encoder not found at {encoder_path}")
        print("Run train_samn_kan.py first to fit and save the encoder.")
        return
    with open(encoder_path, "rb") as f:
        encoder = pickle.load(f)
    print(f"\n  Loaded encoder from {encoder_path}")

    # Load model
    model = SAMNKANKansformerEPI(config).to(device)
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_path = os.path.join(save_dir, "model_best.pth")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(save_dir, "model_final.pth")
    model.load_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=True))
    print(f"  Loaded model from {ckpt_path}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {total_params:,}")

    # Load test data
    bengi_dir = config["paths"].get("bengi_dir", "./data/BENGI")
    feats_config = config["paths"].get("feats_config", "")
    ref_genome = config["paths"].get("ref_genome", "")

    test_bengi = filter_bengi_files(bengi_dir, args.test_cells)
    if len(test_bengi) == 0:
        print(f"ERROR: No BENGI files for test cells {args.test_cells}")
        return

    print(f"\n  Test BENGI files ({len(test_bengi)}):")
    for f in test_bengi:
        print(f"    {os.path.basename(f)}")

    test_genomic = EPIGenomicDataset(
        bengi_paths=test_bengi,
        feats_config_path=feats_config,
        feats_order=config["data"].get("feats_order", None),
        seq_len=config["data"].get("seq_len_bp", 2_500_000),
        bin_size=config["data"].get("bin_size", 500),
        enhancer_window=config["data"].get("enhancer_window", 3000),
        promoter_window=config["data"].get("promoter_window", 3000),
        ref_genome_path=ref_genome if ref_genome else None,
    )

    test_dataset = EPIDataset(config, encoder, source_dataset=test_genomic)
    test_dataset.augment = False

    num_workers = config["training"].get("num_workers", 4)
    batch_size = config["data"]["batch_size"]

    # All test data combined
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             num_workers=num_workers, pin_memory=True)
    combined = evaluate_split(
        model, test_loader, device,
        f"Test (ALL: {', '.join(args.test_cells)})", use_amp=use_amp)

    # Save curves for combined results
    try:
        save_curves(combined["labels"], combined["predictions"],
                    save_dir, prefix="test_all_")
    except Exception as e:
        print(f"  Warning: Could not save combined curves: {e}")

    # Per-cell-line breakdown
    test_cells_in_data = [s.get("cell", "unknown")
                          for s in test_genomic.samples]
    unique_cells = sorted(set(test_cells_in_data))

    per_cell_results = {}
    if len(unique_cells) > 1:
        print(f"\n{'=' * 60}")
        print(f"  PER-CELL-LINE BREAKDOWN")
        print(f"{'=' * 60}")
        for cell in unique_cells:
            cell_idx = [i for i, c in enumerate(test_cells_in_data) if c == cell]
            if len(cell_idx) == 0:
                continue
            cell_subset = Subset(test_dataset, cell_idx)
            cell_loader = DataLoader(cell_subset, batch_size=batch_size,
                                     num_workers=num_workers, pin_memory=True)
            per_cell_results[cell] = evaluate_split(
                model, cell_loader, device, f"Test ({cell})", use_amp=use_amp)
            try:
                save_curves(per_cell_results[cell]["labels"],
                            per_cell_results[cell]["predictions"],
                            save_dir, prefix=f"test_{cell}_")
            except Exception as e:
                print(f"  Warning: Could not save {cell} curves: {e}")

    # Save all results
    results_path = os.path.join(save_dir, "eval_results.npz")
    save_dict = {
        "test_predictions": combined["predictions"],
        "test_labels": combined["labels"],
        "test_auroc": combined["auroc"],
        "test_aupr": combined["aupr"],
        "test_f1": combined["f1"],
        "test_mcc": combined["mcc"],
        "test_balanced_accuracy": combined["balanced_accuracy"],
        "test_cells": args.test_cells,
    }
    for cell, r in per_cell_results.items():
        save_dict[f"{cell}_predictions"] = r["predictions"]
        save_dict[f"{cell}_labels"] = r["labels"]
        save_dict[f"{cell}_auroc"] = r["auroc"]
        save_dict[f"{cell}_aupr"] = r["aupr"]
    np.savez(results_path, **save_dict)
    print(f"\n  Results saved to {results_path}")
    print("\n  Evaluation Complete.")


if __name__ == "__main__":
    main()
