"""
SAMN-EPI Training Script — Cross-Cell-Line Evaluation

Trains the SAMN-based EPI model on selected cell lines and evaluates on
held-out cell lines. Replaces the KANTransformer training pipeline.

Usage:
    python scripts/train_samn.py --train-cells GM12878 HeLa K562 IMR90 --test-cells HMEC NHEK
    python scripts/train_samn.py --config configs/samn.yaml --train-cells GM12878 HeLa K562 IMR90 --test-cells HMEC NHEK
"""
import os
# Reduce CUDA fragmentation so large batches fit on 16 GB. Must be set before
# torch initializes the CUDA caching allocator (i.e. before the first import).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.amp import autocast, GradScaler
import yaml
import os
import glob
import argparse
import time
import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             average_precision_score, f1_score,
                             precision_score, recall_score,
                             balanced_accuracy_score,
                             matthews_corrcoef, confusion_matrix,
                             roc_curve, precision_recall_curve)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.epi_data_pipeline import EPIGenomicDataset
from src.dataset import EPIDataset
from src.samn_model import SAMNKansformerEPI
from src.encoding import POCD_ND_Encoder
from src.visualize import plot_history


def parse_args():
    p = argparse.ArgumentParser(
        description="Train SAMN-EPI (cross-cell-line)")
    p.add_argument("--config", type=str, default="configs/samn.yaml",
                   help="Path to config YAML")
    p.add_argument("--train-cells", nargs="+", required=True,
                   help="Cell lines to train on (e.g. GM12878 HeLa K562 IMR90)")
    p.add_argument("--test-cells", nargs="+", required=True,
                   help="Cell lines to test on (e.g. HMEC NHEK)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--no-amp", action="store_true",
                   help="Disable mixed precision training")
    return p.parse_args()


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


def evaluate_loader(model, loader, device, crit_cls, crit_reg, lambda_dist,
                    use_amp=False):
    """Evaluate model on a DataLoader, return metrics dict."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            seq = batch["seq"].float().to(device)
            epi = batch["epi"].float().to(device)
            lbl = batch["label"].float().to(device)
            dst = batch["dist"].float().to(device)
            enh_idx = batch["enh_idx"].float().to(device)
            prom_idx = batch["prom_idx"].float().to(device)

            with autocast(device_type="cuda", enabled=use_amp and device.type == "cuda"):
                p_cls, p_reg, _ = model(seq, epi, enh_idx, prom_idx)
                loss = crit_cls(p_cls, lbl) + lambda_dist * crit_reg(p_reg, dst)

            total_loss += loss.item()
            all_preds.extend(torch.sigmoid(p_cls.float()).cpu().numpy().flatten())
            all_labels.extend(lbl.cpu().numpy().flatten())

    preds_np = np.array(all_preds)
    labels_np = np.array(all_labels)
    acc = accuracy_score(labels_np, (preds_np > 0.5).astype(int))
    try:
        auc = roc_auc_score(labels_np, preds_np)
    except ValueError:
        auc = 0.0
    try:
        aupr = average_precision_score(labels_np, preds_np)
    except ValueError:
        aupr = 0.0
    avg_loss = total_loss / max(len(loader), 1)
    return {"loss": avg_loss, "auroc": auc, "aupr": aupr, "acc": acc,
            "preds": preds_np, "labels": labels_np}


def save_roc_pr_curves(labels, preds, save_dir, prefix=""):
    """Generate and save ROC and PR curve plots."""
    # ROC Curve
    fpr, tpr, _ = roc_curve(labels, preds)
    auc_val = roc_auc_score(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, "b-", linewidth=2, label=f"SAMN-EPI (AUROC = {auc_val:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"{prefix}ROC Curve", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"{prefix}roc_curve.png"), dpi=150)
    plt.close(fig)

    # PR Curve
    prec_arr, rec_arr, _ = precision_recall_curve(labels, preds)
    aupr_val = average_precision_score(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(rec_arr, prec_arr, "r-", linewidth=2,
            label=f"SAMN-EPI (AUPR = {aupr_val:.4f})")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(f"{prefix}Precision-Recall Curve", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"{prefix}pr_curve.png"), dpi=150)
    plt.close(fig)


def save_loss_plot(train_losses, val_losses, save_dir):
    """Save training and validation loss curves."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_losses, label="Train Loss", linewidth=2)
    ax.plot(val_losses, label="Val Loss", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("SAMN-EPI Training Loss", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "train_loss.png"), dpi=150)
    plt.close(fig)


def main():
    args = parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Apply CLI overrides
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.patience is not None:
        config["training"]["patience"] = args.patience
    if args.batch_size is not None:
        config["data"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["training"]["lr"] = args.lr
    if args.output_dir is not None:
        config["paths"]["save_dir"] = args.output_dir

    save_dir = config["paths"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_amp = config["training"].get("use_amp", True) and not args.no_amp
    if device.type != "cuda":
        use_amp = False

    # GPU throughput flags (Blackwell RTX 5070 Ti). Input shapes are fixed
    # (drop_last on train) so cuDNN autotune pays off; TF32 speeds fp32 matmul/LSTM paths.
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print("=" * 70)
    print("  SAMN-EPI: Selective Ancestral Memory Network for EPI Prediction")
    print("=" * 70)
    print(f"  Device:        {device}")
    if device.type == "cuda":
        print(f"  GPU:           {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:          {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Mixed Prec:    {'ON' if use_amp else 'OFF'}")
    print(f"  Config:        {args.config}")
    print()

    train_cells = args.train_cells
    test_cells = args.test_cells
    print(f"  Train cells:   {train_cells}")
    print(f"  Test cells:    {test_cells}")
    print()

    # ─── Resolve data paths ───
    bengi_dir = config["paths"].get("bengi_dir", "./data/BENGI")
    feats_config = config["paths"].get("feats_config", "")
    ref_genome = config["paths"].get("ref_genome", "")

    train_bengi = filter_bengi_files(bengi_dir, train_cells)
    test_bengi = filter_bengi_files(bengi_dir, test_cells)

    if len(train_bengi) == 0:
        print(f"ERROR: No BENGI files found for train cells {train_cells}")
        return
    if len(test_bengi) == 0:
        print(f"ERROR: No BENGI files found for test cells {test_cells}")
        return

    print(f"Train BENGI files ({len(train_bengi)}):")
    for f in train_bengi:
        print(f"  {os.path.basename(f)}")
    print(f"Test BENGI files ({len(test_bengi)}):")
    for f in test_bengi:
        print(f"  {os.path.basename(f)}")

    # ─── Load datasets ───
    print("\n=== Loading TRAINING data ===")
    train_genomic = EPIGenomicDataset(
        bengi_paths=train_bengi,
        feats_config_path=feats_config,
        feats_order=config["data"].get("feats_order", None),
        seq_len=config["data"].get("seq_len_bp", 2_500_000),
        bin_size=config["data"].get("bin_size", 500),
        enhancer_window=config["data"].get("enhancer_window", 3000),
        promoter_window=config["data"].get("promoter_window", 3000),
        ref_genome_path=ref_genome if ref_genome else None,
    )

    print("\n=== Loading TEST data ===")
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

    # ─── Split train → train/val by chromosome ───
    train_chroms = train_genomic.get_chrom_groups()
    train_labels = train_genomic.get_labels()
    valid_chroms = config["training"].get("valid_chroms", ["chr11", "chr17"])

    train_idx = [i for i, c in enumerate(train_chroms) if c not in valid_chroms]
    val_idx = [i for i, c in enumerate(train_chroms) if c in valid_chroms]

    if len(val_idx) == 0:
        n = len(train_genomic)
        perm = np.random.permutation(n)
        split = int(0.8 * n)
        train_idx = perm[:split].tolist()
        val_idx = perm[split:].tolist()
        print("WARNING: No val chroms matched, using random 80/20 split")

    # ─── Fit POCD-ND encoder ───
    print("\nFitting POCD-ND Encoder on training sequences...")
    encoder = POCD_ND_Encoder(k=config["data"]["kmer_size"])

    train_labels_arr = np.array([train_labels[i] for i in train_idx])
    pos_train_idx = [train_idx[j] for j in np.where(train_labels_arr == 1)[0]]
    neg_train_idx = [train_idx[j] for j in np.where(train_labels_arr == 0)[0]]

    MAX_FIT = config["data"].get("encoder_fit_samples", 5000)
    np.random.seed(42)
    pos_sample = np.random.choice(
        pos_train_idx, size=min(MAX_FIT, len(pos_train_idx)), replace=False)
    neg_sample = np.random.choice(
        neg_train_idx, size=min(MAX_FIT, len(neg_train_idx)), replace=False)

    print(f"  Extracting {len(pos_sample)} pos + {len(neg_sample)} neg sequences...")
    pos_seqs = [train_genomic[i]["enhancer_seq"] + train_genomic[i]["promoter_seq"]
                for i in pos_sample]
    neg_seqs = [train_genomic[i]["enhancer_seq"] + train_genomic[i]["promoter_seq"]
                for i in neg_sample]

    encoder.fit(pos_seqs, neg_seqs, config["data"]["sequence_length"])
    print(f"  Encoder fitted: {len(pos_seqs)} pos + {len(neg_seqs)} neg sequences.")

    encoder_path = os.path.join(save_dir, "encoder.pkl")
    with open(encoder_path, "wb") as f:
        pickle.dump(encoder, f)
    print(f"  Encoder saved to {encoder_path}")

    # ─── Wrap with POCD-ND encoding ───
    train_dataset = EPIDataset(config, encoder, source_dataset=train_genomic)
    test_dataset = EPIDataset(config, encoder, source_dataset=test_genomic)

    train_set = Subset(train_dataset, train_idx)
    val_set = Subset(train_dataset, val_idx)

    n_pos = int(train_labels_arr.sum())
    n_neg = len(train_labels_arr) - n_pos
    print(f"\nTrain: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_dataset)}")
    print(f"Train class balance — pos: {n_pos} ({100*n_pos/len(train_labels_arr):.1f}%), "
          f"neg: {n_neg} ({100*n_neg/len(train_labels_arr):.1f}%)")

    # ─── DataLoaders ───
    num_workers = config["training"].get("num_workers", 8)
    batch_size = config["data"]["batch_size"]
    prefetch = config["training"].get("prefetch_factor", 2)
    persistent = config["training"].get("persistent_workers", True) and num_workers > 0

    # persistent_workers avoids re-pickling the ~760MB dataset into workers every
    # epoch on Windows spawn — the single biggest win for these short runs.
    common = dict(num_workers=num_workers, pin_memory=True)
    if num_workers > 0:
        common["prefetch_factor"] = prefetch
        common["persistent_workers"] = persistent

    # val runs every epoch but is small (chr11+chr17) → fewer workers to bound RAM
    val_common = dict(common)
    val_common["num_workers"] = min(4, num_workers)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              drop_last=True, **common)
    val_loader = DataLoader(val_set, batch_size=batch_size, **val_common)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, **val_common)

    # ─── Model ───
    model = SAMNKansformerEPI(config).to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"].get("weight_decay", 1e-4),
    )

    use_cosine = config["training"].get("use_cosine_scheduler", False)
    if use_cosine:
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=5, T_mult=2)
        print("Scheduler: CosineAnnealingWarmRestarts")
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5)
        print("Scheduler: ReduceLROnPlateau (mode=max, factor=0.5)")

    # Loss functions
    crit_cls = nn.BCEWithLogitsLoss()
    crit_reg = nn.MSELoss()
    lambda_dist = config["training"]["lambda_dist"]
    entropy_weight = config["training"].get("entropy_loss_weight", 0.01)
    diversity_weight = config["training"].get("diversity_loss_weight", 0.05)

    scaler = GradScaler("cuda", enabled=use_amp)

    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {total_params:,} params ({train_params:,} trainable)")
    print(f"Loss: BCE + {lambda_dist}*MSE + SAMN_aux "
          f"(entropy={entropy_weight}, diversity={diversity_weight})")

    # ─── Training Loop ───
    epochs = config["training"]["epochs"]
    patience = config["training"].get("patience", 15)
    train_loss_hist, val_loss_hist = [], []
    best_metric = -float("inf")
    patience_counter = 0
    use_augment = config.get("augmentation", {}).get("enabled", False)

    print(f"\n{'=' * 70}")
    print(f"  Starting Training: {epochs} epochs, patience={patience}, "
          f"augmentation={'ON' if use_augment else 'OFF'}")
    print(f"  Monitor: val AUROC + AUPR")
    print(f"{'=' * 70}")

    for epoch in range(epochs):
        model.train()
        train_dataset.augment = use_augment
        epoch_loss = 0.0
        train_preds_all, train_labels_all = [], []
        num_batches = len(train_loader)
        epoch_start = time.time()

        for step, batch in enumerate(train_loader):
            seq = batch["seq"].float().to(device, non_blocking=True)
            epi = batch["epi"].float().to(device, non_blocking=True)
            lbl = batch["label"].float().to(device, non_blocking=True)
            dst = batch["dist"].float().to(device, non_blocking=True)
            enh_idx = batch["enh_idx"].float().to(device, non_blocking=True)
            prom_idx = batch["prom_idx"].float().to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=use_amp and device.type == "cuda"):
                p_cls, p_reg, aux_terms = model(seq, epi, enh_idx, prom_idx)
                loss_cls = crit_cls(p_cls, lbl)
                loss_reg = crit_reg(p_reg, dst)
                loss_samn = model.samn_auxiliary_loss(
                    aux_terms, entropy_weight, diversity_weight)
                loss = loss_cls + lambda_dist * loss_reg + loss_samn

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config["training"].get("grad_clip", 1.0))
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            with torch.no_grad():
                train_preds_all.extend(
                    torch.sigmoid(p_cls.float()).cpu().numpy().flatten())
                train_labels_all.extend(lbl.cpu().numpy().flatten())

            # Progress logging every 10 steps
            if (step + 1) % 10 == 0 or (step + 1) == num_batches:
                elapsed = time.time() - epoch_start
                avg_loss = epoch_loss / (step + 1)
                speed = (step + 1) / elapsed
                eta = (num_batches - step - 1) / speed if speed > 0 else 0
                gate_mean = aux_terms.get("gate_mean", torch.tensor(0.0)).item()
                print(f"  [{epoch+1}/{epochs}] Step {step+1:4d}/{num_batches} | "
                      f"Loss: {avg_loss:.4f} (cls:{loss_cls.item():.3f} "
                      f"reg:{loss_reg.item():.3f} samn:{loss_samn.item():.3f}) | "
                      f"Gate: {gate_mean:.3f} | "
                      f"{speed:.1f} batch/s | ETA: {eta:.0f}s")

        if use_cosine:
            scheduler.step(epoch)

        # ─── Validation ───
        train_dataset.augment = False
        val_metrics = evaluate_loader(
            model, val_loader, device, crit_cls, crit_reg, lambda_dist,
            use_amp=use_amp)

        avg_train = epoch_loss / max(num_batches, 1)
        train_loss_hist.append(avg_train)
        val_loss_hist.append(val_metrics["loss"])

        # Train metrics
        train_preds_np = np.array(train_preds_all)
        train_labels_np = np.array(train_labels_all)
        try:
            train_auc = roc_auc_score(train_labels_np, train_preds_np)
        except ValueError:
            train_auc = 0.0
        try:
            train_aupr = average_precision_score(train_labels_np, train_preds_np)
        except ValueError:
            train_aupr = 0.0

        epoch_time = time.time() - epoch_start
        print(f"\n  Epoch {epoch+1:3d}/{epochs} [{epoch_time:.0f}s] | "
              f"Train(AUC/AUPR): {train_auc:.4f}/{train_aupr:.4f} | "
              f"Val(AUC/AUPR): {val_metrics['auroc']:.4f}/{val_metrics['aupr']:.4f} | "
              f"TrLoss: {avg_train:.4f} | VlLoss: {val_metrics['loss']:.4f}",
              end="")

        if not use_cosine:
            scheduler.step(val_metrics["auroc"] + val_metrics["aupr"])

        # Early stopping on AUC + AUPR
        current_metric = val_metrics["auroc"] + val_metrics["aupr"]
        if current_metric > best_metric:
            best_metric = current_metric
            torch.save(model.state_dict(),
                       os.path.join(save_dir, "samn_model_best.pth"))
            patience_counter = 0
            print(f" ★ BEST (AUC+AUPR={current_metric:.4f})")
        else:
            patience_counter += 1
            print(f" (patience {patience_counter}/{patience})")

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
        print()

    # ─── Save final model & loss plot ───
    torch.save(model.state_dict(), os.path.join(save_dir, "samn_model_final.pth"))
    save_loss_plot(train_loss_hist, val_loss_hist, save_dir)
    plot_history(train_loss_hist, val_loss_hist, os.path.join(save_dir, "loss.png"))
    print(f"\nFinal model saved to {save_dir}/samn_model_final.pth")
    print(f"Loss plot saved to {save_dir}/train_loss.png")

    # ─── Final Evaluation with best model ───
    print("\n" + "=" * 70)
    print("  FINAL EVALUATION (best checkpoint)")
    print("=" * 70)
    best_path = os.path.join(save_dir, "samn_model_best.pth")
    if os.path.exists(best_path):
        model.load_state_dict(
            torch.load(best_path, map_location=device, weights_only=True))

    # Val set
    val_results = evaluate_loader(
        model, val_loader, device, crit_cls, crit_reg, lambda_dist,
        use_amp=use_amp)
    print(f"\n[Validation] ({len(val_set)} samples, chroms: {valid_chroms})")
    print(f"  AUROC:    {val_results['auroc']:.4f}")
    print(f"  AUPR:     {val_results['aupr']:.4f}")
    print(f"  Accuracy: {val_results['acc']:.4f}")

    # Test set — comprehensive metrics (matching base model)
    test_results = evaluate_loader(
        model, test_loader, device, crit_cls, crit_reg, lambda_dist,
        use_amp=use_amp)
    binary_preds = (test_results["preds"] > 0.5).astype(int)
    test_acc = accuracy_score(test_results["labels"], binary_preds)
    test_bal_acc = balanced_accuracy_score(test_results["labels"], binary_preds)
    f1 = f1_score(test_results["labels"], binary_preds, zero_division=0)
    prec = precision_score(test_results["labels"], binary_preds, zero_division=0)
    rec = recall_score(test_results["labels"], binary_preds, zero_division=0)
    try:
        mcc = matthews_corrcoef(test_results["labels"], binary_preds)
    except ValueError:
        mcc = 0.0
    cm = confusion_matrix(test_results["labels"], binary_preds)

    print(f"\n{'=' * 50}")
    print(f"  Test (ALL: {', '.join(test_cells)}) Results  ({len(test_results['labels'])} samples)")
    print(f"{'=' * 50}")
    print(f"  AUROC:              {test_results['auroc']:.4f}")
    print(f"  AUPR:               {test_results['aupr']:.4f}")
    print(f"  Accuracy:           {test_acc:.4f}")
    print(f"  Balanced Accuracy:  {test_bal_acc:.4f}")
    print(f"  Precision:          {prec:.4f}")
    print(f"  Recall:             {rec:.4f}")
    print(f"  F1 Score:           {f1:.4f}")
    print(f"  MCC:                {mcc:.4f}")
    print(f"\n  Confusion Matrix:")
    if cm.shape == (2, 2):
        print(f"    TN={cm[0, 0]:6d}  FP={cm[0, 1]:6d}")
        print(f"    FN={cm[1, 0]:6d}  TP={cm[1, 1]:6d}")
    print(f"\n  Pos: {int(test_results['labels'].sum())} ({100 * test_results['labels'].mean():.1f}%)  "
          f"Neg: {int(len(test_results['labels']) - test_results['labels'].sum())} "
          f"({100 * (1 - test_results['labels'].mean()):.1f}%)")
    print(f"  Predicted Pos: {int(binary_preds.sum())}  "
          f"Predicted Neg: {int(len(binary_preds) - binary_preds.sum())}")
    print(f"{'=' * 50}")

    # Per-cell-line breakdown
    test_cells_in_data = [s.get("cell", "unknown") for s in test_genomic.samples]
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
                                     num_workers=min(4, num_workers), pin_memory=True)
            cell_res = evaluate_loader(
                model, cell_loader, device, crit_cls, crit_reg, lambda_dist,
                use_amp=use_amp)
            cell_bp = (cell_res["preds"] > 0.5).astype(int)
            cell_acc = accuracy_score(cell_res["labels"], cell_bp)
            cell_bal_acc = balanced_accuracy_score(cell_res["labels"], cell_bp)
            cell_f1 = f1_score(cell_res["labels"], cell_bp, zero_division=0)
            cell_prec = precision_score(cell_res["labels"], cell_bp, zero_division=0)
            cell_rec = recall_score(cell_res["labels"], cell_bp, zero_division=0)
            try:
                cell_mcc = matthews_corrcoef(cell_res["labels"], cell_bp)
            except ValueError:
                cell_mcc = 0.0
            cell_cm = confusion_matrix(cell_res["labels"], cell_bp)

            print(f"\n{'=' * 50}")
            print(f"  Test ({cell}) Results  ({len(cell_res['labels'])} samples)")
            print(f"{'=' * 50}")
            print(f"  AUROC:              {cell_res['auroc']:.4f}")
            print(f"  AUPR:               {cell_res['aupr']:.4f}")
            print(f"  Accuracy:           {cell_acc:.4f}")
            print(f"  Balanced Accuracy:  {cell_bal_acc:.4f}")
            print(f"  Precision:          {cell_prec:.4f}")
            print(f"  Recall:             {cell_rec:.4f}")
            print(f"  F1 Score:           {cell_f1:.4f}")
            print(f"  MCC:                {cell_mcc:.4f}")
            print(f"\n  Confusion Matrix:")
            if cell_cm.shape == (2, 2):
                print(f"    TN={cell_cm[0, 0]:6d}  FP={cell_cm[0, 1]:6d}")
                print(f"    FN={cell_cm[1, 0]:6d}  TP={cell_cm[1, 1]:6d}")
            print(f"\n  Pos: {int(cell_res['labels'].sum())} ({100 * cell_res['labels'].mean():.1f}%)  "
                  f"Neg: {int(len(cell_res['labels']) - cell_res['labels'].sum())} "
                  f"({100 * (1 - cell_res['labels'].mean()):.1f}%)")
            print(f"  Predicted Pos: {int(cell_bp.sum())}  "
                  f"Predicted Neg: {int(len(cell_bp) - cell_bp.sum())}")
            print(f"{'=' * 50}")
            per_cell_results[cell] = cell_res

    # Save ROC and PR curves
    try:
        save_roc_pr_curves(test_results["labels"], test_results["preds"],
                           save_dir, prefix="test_all_")
        print(f"\nROC curve saved to {save_dir}/test_all_roc_curve.png")
        print(f"PR curve saved to {save_dir}/test_all_pr_curve.png")
    except Exception as e:
        print(f"Warning: Could not save curves: {e}")

    for cell, r in per_cell_results.items():
        try:
            save_roc_pr_curves(r["labels"], r["preds"],
                               save_dir, prefix=f"test_{cell}_")
        except Exception as e:
            print(f"Warning: Could not save {cell} curves: {e}")

    # Save results
    results_path = os.path.join(save_dir, "results.npz")
    np.savez(
        results_path,
        val_preds=val_results["preds"], val_labels=val_results["labels"],
        val_auroc=val_results["auroc"], val_aupr=val_results["aupr"],
        test_preds=test_results["preds"], test_labels=test_results["labels"],
        test_auroc=test_results["auroc"], test_aupr=test_results["aupr"],
        test_f1=f1, test_mcc=mcc, test_balanced_accuracy=test_bal_acc,
        train_cells=train_cells, test_cells=test_cells,
        epochs_run=epoch + 1, best_metric=best_metric,
        train_loss_hist=train_loss_hist, val_loss_hist=val_loss_hist,
    )
    for cell, r in per_cell_results.items():
        np.savez(
            os.path.join(save_dir, f"results_{cell}.npz"),
            predictions=r["preds"], labels=r["labels"],
            auroc=r["auroc"], aupr=r["aupr"],
        )
    print(f"\nResults saved to {results_path}")

    # Save config snapshot
    config_snapshot = config.copy()
    config_snapshot["_train_cells"] = train_cells
    config_snapshot["_test_cells"] = test_cells
    with open(os.path.join(save_dir, "config_snapshot.yaml"), "w") as f:
        yaml.dump(config_snapshot, f, default_flow_style=False)

    print("\n" + "=" * 70)
    print("  Training Complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
