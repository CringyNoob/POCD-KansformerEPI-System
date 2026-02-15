import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
import yaml
import os
import glob
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score

from src.epi_data_pipeline import EPIGenomicDataset
from src.dataset import EPIDataset
from src.model import Kansformer
from src.encoding import POCD_ND_Encoder
from src.visualize import plot_history

# 1. Setup
with open('configs/config.yaml') as f: config = yaml.safe_load(f)
os.makedirs(config['paths']['save_dir'], exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# 2. Data Loading (BEFORE encoder fitting — need real sequences)
bengi_dir = config['paths'].get('bengi_dir', './data/BENGI')
feats_config = config['paths'].get('feats_config', '')
ref_genome = config['paths'].get('ref_genome', '')

bengi_files = sorted(
    glob.glob(os.path.join(bengi_dir, '*.tsv.gz')) +
    glob.glob(os.path.join(bengi_dir, '*.tsv'))
)

USE_REAL_DATA = len(bengi_files) > 0 and os.path.exists(feats_config)

if USE_REAL_DATA:
    print(f"\n=== REAL DATA MODE ===")
    print(f"BENGI files: {len(bengi_files)}")
    for f_path in bengi_files:
        print(f"  {os.path.basename(f_path)}")

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

    # Split by chromosome
    valid_chroms = config['training'].get('valid_chroms', ['chr11', 'chr17'])
    test_chroms = config['training'].get('test_chroms', ['chr1', 'chr2'])
    chroms = genomic_ds.get_chrom_groups()
    labels = genomic_ds.get_labels()

    train_idx = [i for i, c in enumerate(chroms) if c not in valid_chroms and c not in test_chroms]
    val_idx = [i for i, c in enumerate(chroms) if c in valid_chroms]
    test_idx = [i for i, c in enumerate(chroms) if c in test_chroms]

    if len(val_idx) == 0:
        n = len(genomic_ds); train_size = int(0.8 * n)
        perm = np.random.permutation(n)
        train_idx = perm[:train_size].tolist()
        val_idx = perm[train_size:].tolist()

    # --- Fit POCD-ND encoder on REAL training sequences ---
    print("\nFitting POCD-ND Encoder on real training sequences...")
    encoder = POCD_ND_Encoder(k=config['data']['kmer_size'])

    train_labels_arr = np.array([labels[i] for i in train_idx])
    pos_train_idx = [train_idx[j] for j in np.where(train_labels_arr == 1)[0]]
    neg_train_idx = [train_idx[j] for j in np.where(train_labels_arr == 0)[0]]

    # Sample up to N sequences from each class for fitting
    MAX_FIT_SAMPLES = config['data'].get('encoder_fit_samples', 5000)
    np.random.seed(42)
    pos_sample = np.random.choice(pos_train_idx, size=min(MAX_FIT_SAMPLES, len(pos_train_idx)), replace=False)
    neg_sample = np.random.choice(neg_train_idx, size=min(MAX_FIT_SAMPLES, len(neg_train_idx)), replace=False)

    print(f"  Extracting {len(pos_sample)} positive + {len(neg_sample)} negative sequences...")
    pos_seqs = []
    for i in pos_sample:
        raw = genomic_ds[i]
        pos_seqs.append(raw["enhancer_seq"] + raw["promoter_seq"])
    neg_seqs = []
    for i in neg_sample:
        raw = genomic_ds[i]
        neg_seqs.append(raw["enhancer_seq"] + raw["promoter_seq"])

    encoder.fit(pos_seqs, neg_seqs, config['data']['sequence_length'])
    print(f"  Encoder fitted on {len(pos_seqs)} pos + {len(neg_seqs)} neg real sequences.")

    with open(f"{config['paths']['save_dir']}/encoder.pkl", 'wb') as f:
        pickle.dump(encoder, f)
    print("  Encoder saved.")

    # Wrap with POCD-ND encoding
    dataset = EPIDataset(config, encoder, source_dataset=genomic_ds)

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx) if len(test_idx) > 0 else None

    # Compute class balance
    train_labels = [labels[i] for i in train_idx]
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    pos_weight_val = n_neg / max(n_pos, 1)
    print(f"\nTrain: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_idx)} "
          f"(val chroms: {valid_chroms}, test chroms: {test_chroms})")
    print(f"Class balance — pos: {n_pos} ({100*n_pos/len(train_labels):.1f}%), "
          f"neg: {n_neg} ({100*n_neg/len(train_labels):.1f}%), "
          f"pos_weight: {pos_weight_val:.2f}")

else:
    print(f"\n=== SYNTHETIC DATA MODE ===")
    print(f"BENGI dir '{bengi_dir}' not found or feats_config missing.")
    print(f"Using synthetic data for pipeline testing.")
    encoder = POCD_ND_Encoder(k=config['data']['kmer_size'])
    np.random.seed(42)
    dummy = ["".join(np.random.choice(['A','C','G','T'], config['data']['sequence_length'])) for _ in range(50)]
    encoder.fit(dummy, dummy, config['data']['sequence_length'])
    dataset = EPIDataset(config, encoder, num_synthetic=200)
    train_size = int(0.8 * len(dataset))
    train_set, val_set = random_split(dataset, [train_size, len(dataset)-train_size])
    test_set = None
    pos_weight_val = 1.0
    labels = None
    train_idx = list(range(train_size))

num_workers = config['training'].get('num_workers', 0)

# NO WeightedRandomSampler — reference KansformerEPI uses plain shuffle + BCELoss
train_loader = DataLoader(train_set, batch_size=config['data']['batch_size'],
                          shuffle=True, num_workers=num_workers)
print("Using shuffle (no weighted sampler — matches reference KansformerEPI)")

val_loader = DataLoader(val_set, batch_size=config['data']['batch_size'],
                        num_workers=num_workers)
if test_set is not None:
    test_loader = DataLoader(test_set, batch_size=config['data']['batch_size'],
                             num_workers=num_workers)

# 3. Model & Optimizer
model = Kansformer(config).to(device)
optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'],
                       weight_decay=config['training'].get('weight_decay', 1e-4))

# Cosine annealing with warm restarts (reference uses T_0=5, T_mult=2)
use_cosine = config['training'].get('use_cosine_scheduler', True)
if use_cosine:
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2)
    print("Using CosineAnnealingWarmRestarts scheduler (T_0=5, T_mult=2)")
else:
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    print("Using ReduceLROnPlateau scheduler")

# BCELoss (matches reference — no logits, model outputs raw logits so we use BCEWithLogitsLoss)
crit_cls = nn.BCEWithLogitsLoss()
print("Using BCEWithLogitsLoss (no pos_weight — matches reference)")

crit_reg = nn.MSELoss()

total_params = sum(p.numel() for p in model.parameters())
train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel: {total_params:,} params ({train_params:,} trainable)")

# 4. Training Loop
train_loss_hist, val_loss_hist = [], []
best_metric = -float('inf')  # Early stop on AUC + AUPR (like reference)
patience_counter = 0

use_augment = config.get('augmentation', {}).get('enabled', False)
print(f"\nStarting Training for {config['training']['epochs']} epochs...")
print(f"Augmentation: {'ON' if use_augment else 'OFF'}")
print(f"Early stopping: patience={config['training'].get('patience', 10)}, monitor=AUC+AUPR")
for epoch in range(config['training']['epochs']):
    model.train()
    dataset.augment = use_augment
    epoch_loss = 0
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
        
        loss = crit_cls(p_cls, lbl) + (config['training']['lambda_dist'] * crit_reg(p_reg, dst))
        att_penalty = model.attention_penalty(A)
        loss = loss + config['training'].get('att_penalty_coeff', 0.1) * att_penalty
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['training'].get('grad_clip', 1.0))
        optimizer.step()
        epoch_loss += loss.item()
        
        # Collect train predictions for AUC/AUPR
        with torch.no_grad():
            train_preds_all.extend(torch.sigmoid(p_cls).cpu().numpy().flatten())
            train_labels_all.extend(lbl.cpu().numpy().flatten())

    # Step cosine scheduler per epoch
    if use_cosine:
        scheduler.step(epoch)
    
    # Validation
    model.eval()
    dataset.augment = False
    val_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            seq = batch['seq'].float().to(device)
            epi = batch['epi'].float().to(device)
            lbl = batch['label'].float().to(device)
            dst = batch['dist'].float().to(device)
            enh_idx = batch['enh_idx'].float().to(device)
            prom_idx = batch['prom_idx'].float().to(device)
            p_cls, p_reg, _ = model(seq, epi, enh_idx, prom_idx)
            val_loss += (crit_cls(p_cls, lbl) + config['training']['lambda_dist'] * crit_reg(p_reg, dst)).item()
            all_preds.extend(torch.sigmoid(p_cls).cpu().numpy().flatten())
            all_labels.extend(lbl.cpu().numpy().flatten())
            
    avg_train = epoch_loss / max(len(train_loader), 1)
    avg_val = val_loss / max(len(val_loader), 1)
    train_loss_hist.append(avg_train)
    val_loss_hist.append(avg_val)
    
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
    
    # Compute train AUC/AUPR
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
    
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1}/{config['training']['epochs']} | "
          f"train({train_auc:.4f}/{train_aupr:.4f})/vald({auc:.4f}/{aupr:.4f}): "
          f"TrainLoss: {avg_train:.4f} | ValLoss: {avg_val:.4f} | ValAcc: {acc:.4f}")

    if not use_cosine:
        scheduler.step(auc + aupr)
    
    # Early stopping on AUC + AUPR (matches reference)
    current_metric = auc + aupr
    if current_metric > best_metric:
        best_metric = current_metric
        torch.save(model.state_dict(), f"{config['paths']['save_dir']}/model_best.pth")
        patience_counter = 0
        print(f"  -> New best (AUC+AUPR={current_metric:.4f}), checkpoint saved.")
    else:
        patience_counter += 1
    
    if patience_counter >= config['training'].get('patience', 10):
        print(f"Early stopping at epoch {epoch+1}")
        break

# 5. Save & Final Evaluation
torch.save(model.state_dict(), f"{config['paths']['save_dir']}/model_final.pth")
plot_history(train_loss_hist, val_loss_hist, f"{config['paths']['save_dir']}/loss.png")

# --- Evaluate on validation set with best model ---
print("\n=== Final Validation Evaluation (best checkpoint) ===")
model.load_state_dict(torch.load(f"{config['paths']['save_dir']}/model_best.pth",
                                  map_location=device, weights_only=True))
model.eval()
dataset.augment = False
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in val_loader:
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
val_acc = accuracy_score(labels_np, (preds_np > 0.5).astype(int))
val_auc = roc_auc_score(labels_np, preds_np)
val_aupr = average_precision_score(labels_np, preds_np)
print(f"Val Accuracy: {val_acc:.4f}")
print(f"Val AUROC:    {val_auc:.4f}")
print(f"Val AUPR:     {val_aupr:.4f}")

# --- Evaluate on test set (if available) ---
if test_set is not None and len(test_idx) > 0:
    print("\n=== Test Evaluation (best checkpoint) ===")
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
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
    test_acc = accuracy_score(labels_np, (preds_np > 0.5).astype(int))
    test_auc = roc_auc_score(labels_np, preds_np)
    test_aupr = average_precision_score(labels_np, preds_np)
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test AUROC:    {test_auc:.4f}")
    print(f"Test AUPR:     {test_aupr:.4f}")

print("\nTraining Complete.")