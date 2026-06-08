"""
POCD-KansformerEPI v6 — Cell-Line Training Script

Trains on samples from selected cell lines, validates on held-out chromosomes
within those cell lines, and tests on entirely separate cell lines.

Usage:
    python train.py --train-cells GM12878 HeLa K562 IMR90 --test-cells HMEC NHEK
    python train.py --config configs/config.yaml --train-cells GM12878 HeLa K562 IMR90 --test-cells HMEC NHEK --epochs 100 --patience 15
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import yaml
import os
import glob
import argparse
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score

from src.epi_data_pipeline import EPIGenomicDataset
from src.dataset import EPIDataset
from src.model import Kansformer
from src.encoding import POCD_ND_Encoder
from src.visualize import plot_history
import sys as _sys; _sys.path.insert(0, r"E:\AI Models")
from comprehensive_metrics import compute_all_metrics, format_metrics_report
def _print_all_metrics(labels, probs, preds=None, bce=None, mse=None, frob=None, prefix="", is_multilabel=False):
    m = compute_all_metrics(labels, probs, preds, bce_loss_val=bce, mse_loss_val=mse, frob_loss_val=frob, is_multilabel=is_multilabel)
    print(format_metrics_report(m, prefix=prefix))
    return m


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train POCD-KansformerEPI v6 on individual cell lines")
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config YAML')
    parser.add_argument('--train-cells', nargs='+', required=True,
                        help='Cell lines to train on (e.g. GM12878 HeLa K562 IMR90)')
    parser.add_argument('--test-cells', nargs='+', required=True,
                        help='Cell lines to test on (e.g. HMEC NHEK)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override max epochs (default: from config)')
    parser.add_argument('--patience', type=int, default=None,
                        help='Override early stopping patience (default: from config)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch size (default: from config)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate (default: from config)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Override output directory (default: from config)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device: cuda, cpu, or auto (default: auto)')
    return parser.parse_args()


def filter_bengi_files(bengi_dir, cell_names):
    """Return BENGI file paths matching any of the given cell line names."""
    all_files = sorted(
        glob.glob(os.path.join(bengi_dir, '*.tsv.gz')) +
        glob.glob(os.path.join(bengi_dir, '*.tsv'))
    )
    matched = []
    for f in all_files:
        basename = os.path.basename(f)
        for cell in cell_names:
            if basename.startswith(cell + '.') or basename.startswith(cell + '_'):
                matched.append(f)
                break
    return matched


def evaluate_loader(model, loader, device, crit_cls, crit_reg, lambda_dist):
    """Evaluate model on a DataLoader, return metrics dict."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            seq = batch['seq'].float().to(device)
            epi = batch['epi'].float().to(device)
            lbl = batch['label'].float().to(device)
            dst = batch['dist'].float().to(device)
            enh_idx = batch['enh_idx'].float().to(device)
            prom_idx = batch['prom_idx'].float().to(device)
            p_cls, p_reg, _ = model(seq, epi, enh_idx, prom_idx)
            loss = crit_cls(p_cls, lbl) + lambda_dist * crit_reg(p_reg, dst)
            total_loss += loss.item()
            all_preds.extend(torch.sigmoid(p_cls).cpu().numpy().flatten())
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
    return {'loss': avg_loss, 'auroc': auc, 'aupr': aupr, 'acc': acc,
            'preds': preds_np, 'labels': labels_np}


def main():
    args = parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Apply CLI overrides
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.patience is not None:
        config['training']['patience'] = args.patience
    if args.batch_size is not None:
        config['data']['batch_size'] = args.batch_size
    if args.lr is not None:
        config['training']['lr'] = args.lr
    if args.output_dir is not None:
        config['paths']['save_dir'] = args.output_dir

    save_dir = config['paths']['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_cells = args.train_cells
    test_cells = args.test_cells
    print(f"Train cell lines: {train_cells}")
    print(f"Test cell lines:  {test_cells}")

    # ---- Resolve data paths ----
    bengi_dir = config['paths'].get('bengi_dir', './data/BENGI')
    feats_config = config['paths'].get('feats_config', '')
    ref_genome = config['paths'].get('ref_genome', '')

    train_bengi = filter_bengi_files(bengi_dir, train_cells)
    test_bengi = filter_bengi_files(bengi_dir, test_cells)

    if len(train_bengi) == 0:
        print(f"ERROR: No BENGI files found for train cells {train_cells} in {bengi_dir}")
        return
    if len(test_bengi) == 0:
        print(f"ERROR: No BENGI files found for test cells {test_cells} in {bengi_dir}")
        return

    print(f"\nTrain BENGI files ({len(train_bengi)}):")
    for f in train_bengi:
        print(f"  {os.path.basename(f)}")
    print(f"Test BENGI files ({len(test_bengi)}):")
    for f in test_bengi:
        print(f"  {os.path.basename(f)}")

    # ---- Load train dataset ----
    print("\n=== Loading TRAINING data ===")
    train_genomic = EPIGenomicDataset(
        bengi_paths=train_bengi,
        feats_config_path=feats_config,
        feats_order=config['data'].get('feats_order', None),
        seq_len=config['data'].get('seq_len_bp', 2_500_000),
        bin_size=config['data'].get('bin_size', 500),
        enhancer_window=config['data'].get('enhancer_window', 3000),
        promoter_window=config['data'].get('promoter_window', 3000),
        ref_genome_path=ref_genome if ref_genome else None,
    )

    # ---- Load test dataset (separate cell lines) ----
    print("\n=== Loading TEST data ===")
    test_genomic = EPIGenomicDataset(
        bengi_paths=test_bengi,
        feats_config_path=feats_config,
        feats_order=config['data'].get('feats_order', None),
        seq_len=config['data'].get('seq_len_bp', 2_500_000),
        bin_size=config['data'].get('bin_size', 500),
        enhancer_window=config['data'].get('enhancer_window', 3000),
        promoter_window=config['data'].get('promoter_window', 3000),
        ref_genome_path=ref_genome if ref_genome else None,
    )

    # ---- Split train data into train/val by chromosome ----
    train_chroms = train_genomic.get_chrom_groups()
    train_labels = train_genomic.get_labels()
    valid_chroms = config['training'].get('valid_chroms', ['chr11', 'chr17'])

    train_idx = [i for i, c in enumerate(train_chroms) if c not in valid_chroms]
    val_idx = [i for i, c in enumerate(train_chroms) if c in valid_chroms]

    if len(val_idx) == 0:
        # Fallback: random 80/20 split
        n = len(train_genomic)
        perm = np.random.permutation(n)
        split = int(0.8 * n)
        train_idx = perm[:split].tolist()
        val_idx = perm[split:].tolist()
        print("WARNING: No val chroms matched, using random 80/20 split")

    # ---- Fit POCD-ND encoder on training sequences ----
    print("\nFitting POCD-ND Encoder on training sequences...")
    encoder = POCD_ND_Encoder(k=config['data']['kmer_size'])

    train_labels_arr = np.array([train_labels[i] for i in train_idx])
    pos_train_idx = [train_idx[j] for j in np.where(train_labels_arr == 1)[0]]
    neg_train_idx = [train_idx[j] for j in np.where(train_labels_arr == 0)[0]]

    MAX_FIT_SAMPLES = config['data'].get('encoder_fit_samples', 5000)
    np.random.seed(42)
    pos_sample = np.random.choice(pos_train_idx,
                                  size=min(MAX_FIT_SAMPLES, len(pos_train_idx)),
                                  replace=False)
    neg_sample = np.random.choice(neg_train_idx,
                                  size=min(MAX_FIT_SAMPLES, len(neg_train_idx)),
                                  replace=False)

    print(f"  Extracting {len(pos_sample)} pos + {len(neg_sample)} neg sequences...")
    pos_seqs = []
    for i in pos_sample:
        raw = train_genomic[i]
        pos_seqs.append(raw["enhancer_seq"] + raw["promoter_seq"])
    neg_seqs = []
    for i in neg_sample:
        raw = train_genomic[i]
        neg_seqs.append(raw["enhancer_seq"] + raw["promoter_seq"])

    encoder.fit(pos_seqs, neg_seqs, config['data']['sequence_length'])
    print(f"  Encoder fitted: {len(pos_seqs)} pos + {len(neg_seqs)} neg sequences.")

    encoder_path = os.path.join(save_dir, 'encoder.pkl')
    with open(encoder_path, 'wb') as f:
        pickle.dump(encoder, f)
    print(f"  Encoder saved to {encoder_path}")

    # ---- Wrap with POCD-ND encoding ----
    train_dataset = EPIDataset(config, encoder, source_dataset=train_genomic)
    test_dataset = EPIDataset(config, encoder, source_dataset=test_genomic)

    train_set = Subset(train_dataset, train_idx)
    val_set = Subset(train_dataset, val_idx)

    # Class balance stats
    n_pos = int(train_labels_arr.sum())
    n_neg = len(train_labels_arr) - n_pos
    print(f"\nTrain: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_dataset)}")
    print(f"Train class balance — pos: {n_pos} ({100*n_pos/len(train_labels_arr):.1f}%), "
          f"neg: {n_neg} ({100*n_neg/len(train_labels_arr):.1f}%)")

    # ---- DataLoaders ----
    num_workers = config['training'].get('num_workers', 0)
    batch_size = config['data']['batch_size']

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    # ---- Model ----
    model = Kansformer(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'],
                           weight_decay=config['training'].get('weight_decay', 0))

    use_cosine = config['training'].get('use_cosine_scheduler', False)
    if use_cosine:
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=5, T_mult=2)
        print("Scheduler: CosineAnnealingWarmRestarts (T_0=5, T_mult=2)")
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5)
        print("Scheduler: ReduceLROnPlateau (mode=max, factor=0.5, patience=5)")

    # Loss functions
    crit_cls = nn.BCEWithLogitsLoss()
    crit_reg = nn.MSELoss()
    lambda_dist = config['training']['lambda_dist']
    att_coeff = config['training'].get('att_penalty_coeff', 0.1)

    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {total_params:,} params ({train_params:,} trainable)")
    print(f"Loss: BCEWithLogitsLoss + {lambda_dist}*MSELoss + {att_coeff}*FrobeniusPenalty")

    # ---- Training Loop ----
    epochs = config['training']['epochs']
    patience = config['training'].get('patience', 15)
    train_loss_hist, val_loss_hist = [], []
    best_metric = -float('inf')
    patience_counter = 0
    use_augment = config.get('augmentation', {}).get('enabled', False)

    print(f"\nStarting Training: {epochs} epochs, patience={patience}, "
          f"augmentation={'ON' if use_augment else 'OFF'}")
    print(f"Monitor: val AUROC + AUPR")
    print("=" * 80)

    for epoch in range(epochs):
        model.train()
        train_dataset.augment = use_augment
        epoch_loss = 0.0
        train_preds_all, train_labels_all = [], []

        for batch in train_loader:
            seq = batch['seq'].float().to(device)
            epi = batch['epi'].float().to(device)
            lbl = batch['label'].float().to(device)
            dst = batch['dist'].float().to(device)
            enh_idx = batch['enh_idx'].float().to(device)
            prom_idx = batch['prom_idx'].float().to(device)

            optimizer.zero_grad()
            p_cls, p_reg, A = model(seq, epi, enh_idx, prom_idx)

            loss_cls = crit_cls(p_cls, lbl)
            loss_reg = crit_reg(p_reg, dst)
            loss_att = model.attention_penalty(A)
            loss = loss_cls + lambda_dist * loss_reg + att_coeff * loss_att

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config['training'].get('grad_clip', 1.0))
            optimizer.step()
            epoch_loss += loss.item()

            with torch.no_grad():
                train_preds_all.extend(torch.sigmoid(p_cls).cpu().numpy().flatten())
                train_labels_all.extend(lbl.cpu().numpy().flatten())

        if use_cosine:
            scheduler.step(epoch)

        # ---- Validation ----
        train_dataset.augment = False
        val_metrics = evaluate_loader(model, val_loader, device, crit_cls, crit_reg, lambda_dist)

        avg_train = epoch_loss / max(len(train_loader), 1)
        train_loss_hist.append(avg_train)
        val_loss_hist.append(val_metrics['loss'])

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

        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Train(AUC/AUPR): {train_auc:.4f}/{train_aupr:.4f} | "
              f"Val(AUC/AUPR): {val_metrics['auroc']:.4f}/{val_metrics['aupr']:.4f} | "
              f"TrLoss: {avg_train:.4f} | VlLoss: {val_metrics['loss']:.4f}", end='')

        if not use_cosine:
            scheduler.step(val_metrics['auroc'] + val_metrics['aupr'])

        # Early stopping on AUC + AUPR
        current_metric = val_metrics['auroc'] + val_metrics['aupr']
        if current_metric > best_metric:
            best_metric = current_metric
            torch.save(model.state_dict(), os.path.join(save_dir, 'model_best.pth'))
            patience_counter = 0
            print(f" * BEST (AUC+AUPR={current_metric:.4f})")
        else:
            patience_counter += 1
            print(f" (patience {patience_counter}/{patience})")

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    # ---- Save final model & loss plot ----
    torch.save(model.state_dict(), os.path.join(save_dir, 'model_final.pth'))
    plot_history(train_loss_hist, val_loss_hist, os.path.join(save_dir, 'loss.png'))
    print(f"\nFinal model saved to {save_dir}/model_final.pth")
    print(f"Loss plot saved to {save_dir}/loss.png")

    # ---- Final Evaluation with best model ----
    print("\n" + "=" * 80)
    print("FINAL EVALUATION (best checkpoint)")
    print("=" * 80)
    model.load_state_dict(torch.load(os.path.join(save_dir, 'model_best.pth'),
                                      map_location=device, weights_only=True))

    # Val set
    val_results = evaluate_loader(model, val_loader, device, crit_cls, crit_reg, lambda_dist)
    print(f"\n[Validation] ({len(val_set)} samples, chroms: {valid_chroms})")
    print(f"  AUROC:    {val_results['auroc']:.4f}")
    print(f"  AUPR:     {val_results['aupr']:.4f}")
    print(f"  Accuracy: {val_results['acc']:.4f}")

    # Test set (separate cell lines)
    test_results = evaluate_loader(model, test_loader, device, crit_cls, crit_reg, lambda_dist)
    print(f"\n[Test] ({len(test_dataset)} samples, cells: {test_cells})")
    print(f"  AUROC:    {test_results['auroc']:.4f}")
    print(f"  AUPR:     {test_results['aupr']:.4f}")
    print(f"  Accuracy: {test_results['acc']:.4f}")

    # Save results
    results_path = os.path.join(save_dir, 'results.npz')
    np.savez(results_path,
             val_preds=val_results['preds'], val_labels=val_results['labels'],
             val_auroc=val_results['auroc'], val_aupr=val_results['aupr'],
             test_preds=test_results['preds'], test_labels=test_results['labels'],
             test_auroc=test_results['auroc'], test_aupr=test_results['aupr'],
             train_cells=train_cells, test_cells=test_cells,
             epochs_run=epoch+1, best_metric=best_metric)
    print(f"\nResults saved to {results_path}")

    # Save config snapshot
    config_snapshot = config.copy()
    config_snapshot['_train_cells'] = train_cells
    config_snapshot['_test_cells'] = test_cells
    with open(os.path.join(save_dir, 'config_snapshot.yaml'), 'w') as f:
        yaml.dump(config_snapshot, f, default_flow_style=False)

    print("\nTraining Complete.")
    print("\n=== Comprehensive Final Metrics ===")
    print("  (Run evaluate.py for full 11-metric test report)")


if __name__ == '__main__':
    main()