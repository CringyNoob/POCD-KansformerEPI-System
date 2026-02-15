"""
Standalone evaluation script for POCD-KansformerEPI.
Loads the best checkpoint and evaluates on validation and test chromosomes.

Usage:
    python evaluate.py                       # Uses configs/config.yaml
    python evaluate.py --config path/to/config.yaml
    python evaluate.py --checkpoint path/to/model.pth
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
                             confusion_matrix, classification_report)
from torch.utils.data import DataLoader, Subset

from src.epi_data_pipeline import EPIGenomicDataset
from src.dataset import EPIDataset
from src.model import Kansformer
from src.encoding import POCD_ND_Encoder


def evaluate_split(model, loader, device, split_name="Val"):
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
    parser = argparse.ArgumentParser(description="Evaluate POCD-KansformerEPI")
    parser.add_argument('--config', default='configs/config.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', default=None, help='Path to model checkpoint (default: best)')
    parser.add_argument('--device', default=None, help='Device (cuda/cpu)')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device if args.device else
                          ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Device: {device}")
    print(f"Config: {args.config}")

    # Load encoder
    encoder_path = f"{config['paths']['save_dir']}/encoder.pkl"
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
        ckpt_path = f"{config['paths']['save_dir']}/model_best.pth"
        if not os.path.exists(ckpt_path):
            ckpt_path = f"{config['paths']['save_dir']}/model_final.pth"
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    print(f"Loaded model from {ckpt_path}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}")

    # Load data
    bengi_dir = config['paths'].get('bengi_dir', './data/BENGI')
    feats_config = config['paths'].get('feats_config', '')
    ref_genome = config['paths'].get('ref_genome', '')

    bengi_files = sorted(
        glob.glob(os.path.join(bengi_dir, '*.tsv.gz')) +
        glob.glob(os.path.join(bengi_dir, '*.tsv'))
    )

    if len(bengi_files) == 0 or not os.path.exists(feats_config):
        print("ERROR: BENGI data or feats_config not found.")
        return

    genomic_ds = EPIGenomicDataset(
        bengi_paths=bengi_files,
        feats_config_path=feats_config,
        feats_order=config['data'].get('feats_order', None),
        seq_len=config['data'].get('seq_len_bp', 2_500_000),
        bin_size=config['data'].get('bin_size', 500),
        enhancer_window=config['data'].get('enhancer_window', 3000),
        promoter_window=config['data'].get('promoter_window', 3000),
        ref_genome_path=ref_genome if ref_genome else None,
    )

    dataset = EPIDataset(config, encoder, source_dataset=genomic_ds)
    dataset.augment = False  # No augmentation during evaluation

    chroms = genomic_ds.get_chrom_groups()
    labels = genomic_ds.get_labels()
    valid_chroms = config['training'].get('valid_chroms', ['chr11', 'chr17'])
    test_chroms = config['training'].get('test_chroms', ['chr1', 'chr2'])

    val_idx = [i for i, c in enumerate(chroms) if c in valid_chroms]
    test_idx = [i for i, c in enumerate(chroms) if c in test_chroms]
    train_idx = [i for i, c in enumerate(chroms) if c not in valid_chroms and c not in test_chroms]

    print(f"\nTrain: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    print(f"Val chroms: {valid_chroms}, Test chroms: {test_chroms}")

    num_workers = config['training'].get('num_workers', 0)
    batch_size = config['data']['batch_size']

    results = {}

    # Validate
    if len(val_idx) > 0:
        val_set = Subset(dataset, val_idx)
        val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers)
        results['val'] = evaluate_split(model, val_loader, device, "Validation")

    # Test
    if len(test_idx) > 0:
        test_set = Subset(dataset, test_idx)
        test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)
        results['test'] = evaluate_split(model, test_loader, device, "Test")

    # Save results
    results_path = f"{config['paths']['save_dir']}/eval_results.npz"
    save_dict = {}
    for split_name, r in results.items():
        save_dict[f'{split_name}_predictions'] = r['predictions']
        save_dict[f'{split_name}_labels'] = r['labels']
        save_dict[f'{split_name}_auroc'] = r['auroc']
        save_dict[f'{split_name}_aupr'] = r['aupr']
    np.savez(results_path, **save_dict)
    print(f"\nResults saved to {results_path}")
    print("\nEvaluation Complete.")


if __name__ == '__main__':
    main()