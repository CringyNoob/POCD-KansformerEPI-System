"""
Standalone evaluation script for POCD-KansformerEPI v6.
Loads the best checkpoint and evaluates on specified test cell lines.

Usage:
    python evaluate.py --test-cells HMEC NHEK
    python evaluate.py --test-cells HMEC NHEK --checkpoint output/model_best.pth
    python evaluate.py --config configs/config.yaml --test-cells HMEC NHEK
"""
import torch
import yaml
import numpy as np
import os
import glob
import pickle
import argparse
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             average_precision_score, f1_score,
                             precision_score, recall_score,
                             confusion_matrix)
from torch.utils.data import DataLoader

from src.epi_data_pipeline import EPIGenomicDataset
from src.dataset import EPIDataset
from src.model import Kansformer
from src.encoding import POCD_ND_Encoder

import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from comprehensive_metrics import compute_all_metrics, format_metrics_report


def _print_all_metrics(labels, probs, preds=None, bce=None, mse=None, frob=None, prefix="", is_multilabel=False):
    m = compute_all_metrics(labels, probs, preds, bce_loss_val=bce, mse_loss_val=mse, frob_loss_val=frob, is_multilabel=is_multilabel)
    print(format_metrics_report(m, prefix=prefix))
    return m


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


def evaluate_split(model, loader, device, split_name="Test"):
    """Evaluate model on a DataLoader split and print metrics."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            seq = batch['seq'].float().to(device)
            epi = batch['epi'].float().to(device)
            lbl = batch['label'].float().to(device)
            enh_idx = batch['enh_idx'].float().to(device)
            prom_idx = batch['prom_idx'].float().to(device)
            p_cls, _, _ = model(seq, epi, enh_idx, prom_idx)
            all_preds.extend(torch.sigmoid(p_cls).cpu().numpy().flatten())
            all_labels.extend(lbl.cpu().numpy().flatten())

    preds_np = np.array(all_preds)
    labels_np = np.array(all_labels)
    binary_preds = (preds_np > 0.5).astype(int)

    acc = accuracy_score(labels_np, binary_preds)
    try:
        auc = roc_auc_score(labels_np, preds_np)
    except ValueError:
        auc = float('nan')
    try:
        aupr = average_precision_score(labels_np, preds_np)
    except ValueError:
        aupr = float('nan')
    f1 = f1_score(labels_np, binary_preds, zero_division=0)
    prec = precision_score(labels_np, binary_preds, zero_division=0)
    rec = recall_score(labels_np, binary_preds, zero_division=0)
    cm = confusion_matrix(labels_np, binary_preds)

    print(f"\n{'='*50}")
    print(f"  {split_name} Results  ({len(labels_np)} samples)")
    print(f"{'='*50}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  AUROC:     {auc:.4f}")

    print("\n=== Comprehensive Test Metrics ===")
    try:
        _print_all_metrics(all_labels, all_preds, prefix="[Test]")
    except:
        print("  (Metrics variables not in expected format)")
    print(f"  AUPR:      {aupr:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}  TP={cm[1,1]}")
    print(f"\n  Pos: {int(labels_np.sum())} ({100*labels_np.mean():.1f}%)  "
          f"Neg: {int(len(labels_np)-labels_np.sum())} ({100*(1-labels_np.mean()):.1f}%)")
    print(f"  Predicted Pos: {int(binary_preds.sum())}  "
          f"Predicted Neg: {int(len(binary_preds)-binary_preds.sum())}")
    print(f"{'='*50}")

    return {
        'accuracy': acc, 'auroc': auc, 'aupr': aupr,
        'f1': f1, 'precision': prec, 'recall': rec,
        'confusion_matrix': cm,
        'predictions': preds_np, 'labels': labels_np
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate POCD-KansformerEPI v6")
    parser.add_argument('--config', default='configs/config.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', default=None, help='Path to model checkpoint (default: best)')
    parser.add_argument('--test-cells', nargs='+', required=True,
                        help='Cell lines to evaluate on (e.g. HMEC NHEK)')
    parser.add_argument('--device', default=None, help='Device (cuda/cpu)')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device if args.device else
                          ('cuda' if torch.cuda.is_available() else 'cpu'))
    save_dir = config['paths']['save_dir']
    print(f"Device: {device}")
    print(f"Config: {args.config}")
    print(f"Test cell lines: {args.test_cells}")

    # Load encoder
    encoder_path = os.path.join(save_dir, 'encoder.pkl')
    if not os.path.exists(encoder_path):
        print(f"ERROR: Encoder not found at {encoder_path}")
        print("Run train.py first to fit and save the encoder.")
        return
    with open(encoder_path, 'rb') as f:
        encoder = pickle.load(f)
    print(f"Loaded encoder from {encoder_path}")

    # Load model
    model = Kansformer(config).to(device)
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_path = os.path.join(save_dir, 'model_best.pth')
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(save_dir, 'model_final.pth')
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    print(f"Loaded model from {ckpt_path}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}")

    # Load test data (by cell line)
    bengi_dir = config['paths'].get('bengi_dir', './data/BENGI')
    feats_config = config['paths'].get('feats_config', '')
    ref_genome = config['paths'].get('ref_genome', '')

    test_bengi = filter_bengi_files(bengi_dir, args.test_cells)
    if len(test_bengi) == 0:
        print(f"ERROR: No BENGI files found for test cells {args.test_cells} in {bengi_dir}")
        return

    print(f"\nTest BENGI files ({len(test_bengi)}):")
    for f in test_bengi:
        print(f"  {os.path.basename(f)}")

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

    test_dataset = EPIDataset(config, encoder, source_dataset=test_genomic)
    test_dataset.augment = False

    num_workers = config['training'].get('num_workers', 0)
    batch_size = config['data']['batch_size']

    # Evaluate all test data together
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)
    combined_results = evaluate_split(model, test_loader, device,
                                      f"Test (ALL: {', '.join(args.test_cells)})")

    # Evaluate per cell line
    per_cell_results = {}
    test_cells_in_data = [s.get("cell", "unknown") for s in test_genomic.samples]
    unique_cells = sorted(set(test_cells_in_data))

    if len(unique_cells) > 1:
        print(f"\n{'='*60}")
        print(f"  PER-CELL-LINE BREAKDOWN")
        print(f"{'='*60}")
        for cell in unique_cells:
            cell_idx = [i for i, c in enumerate(test_cells_in_data) if c == cell]
            if len(cell_idx) == 0:
                continue
            from torch.utils.data import Subset
            cell_subset = Subset(test_dataset, cell_idx)
            cell_loader = DataLoader(cell_subset, batch_size=batch_size,
                                     num_workers=num_workers)
            per_cell_results[cell] = evaluate_split(model, cell_loader, device,
                                                     f"Test ({cell})")

    # Save results
    results_path = os.path.join(save_dir, 'eval_results.npz')
    save_dict = {
        'test_predictions': combined_results['predictions'],
        'test_labels': combined_results['labels'],
        'test_auroc': combined_results['auroc'],
        'test_aupr': combined_results['aupr'],
        'test_cells': args.test_cells,
    }
    for cell, r in per_cell_results.items():
        save_dict[f'{cell}_predictions'] = r['predictions']
        save_dict[f'{cell}_labels'] = r['labels']
        save_dict[f'{cell}_auroc'] = r['auroc']
        save_dict[f'{cell}_aupr'] = r['aupr']
    np.savez(results_path, **save_dict)
    print(f"\nResults saved to {results_path}")
    print("\nEvaluation Complete.")


if __name__ == "__main__":
    main()