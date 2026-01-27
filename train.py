import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import yaml
import os
import numpy as np

from src.dataset import EPIDataset
from src.model import Kansformer
from src.encoding import POCD_ND_Encoder
from src.visualize import plot_history

# 1. Setup
with open('configs/config.yaml') as f: config = yaml.safe_load(f)
os.makedirs(config['paths']['save_dir'], exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Prepare Encoder (Pre-training phase)
print("Initializing Encoder...")
encoder = POCD_ND_Encoder(k=config['data']['kmer_size'])
# Create dummy data for fitting the encoder (In real life, load real FASTA)
dummy_pos = ["".join(np.random.choice(['A','C','G','T'], 1000)) for _ in range(50)]
dummy_neg = ["".join(np.random.choice(['A','C','G','T'], 1000)) for _ in range(50)]
encoder.fit(dummy_pos, dummy_neg, config['data']['sequence_length'])

# 3. Data Loading
dataset = EPIDataset(200, config, encoder)
train_size = int(0.8 * len(dataset))
train_set, val_set = random_split(dataset, [train_size, len(dataset)-train_size])
train_loader = DataLoader(train_set, batch_size=config['data']['batch_size'], shuffle=True)
val_loader = DataLoader(val_set, batch_size=config['data']['batch_size'])

# 4. Model & Opt
model = Kansformer(config).to(device)
optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'])
crit_cls = nn.BCELoss()
crit_reg = nn.MSELoss()

# 5. Loop
train_loss_hist, val_loss_hist = [], []

print("Starting Training...")
for epoch in range(config['training']['epochs']):
    model.train()
    epoch_loss = 0
    
    for batch in train_loader:
        seq = batch['seq'].float().to(device)
        epi = batch['epi'].float().to(device)
        lbl = batch['label'].float().to(device)
        dst = batch['dist'].float().to(device)
        
        optimizer.zero_grad()
        p_cls, p_reg = model(seq, epi)
        
        loss = crit_cls(p_cls, lbl) + (config['training']['lambda_dist'] * crit_reg(p_reg, dst))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            seq = batch['seq'].float().to(device)
            epi = batch['epi'].float().to(device)
            lbl = batch['label'].float().to(device)
            dst = batch['dist'].float().to(device)
            p_cls, p_reg = model(seq, epi)
            val_loss += (crit_cls(p_cls, lbl) + config['training']['lambda_dist'] * crit_reg(p_reg, dst)).item()
            
    avg_train = epoch_loss / len(train_loader)
    avg_val = val_loss / len(val_loader)
    train_loss_hist.append(avg_train)
    val_loss_hist.append(avg_val)
    
    print(f"Epoch {epoch+1} | Train: {avg_train:.4f} | Val: {avg_val:.4f}")

# 6. Save
torch.save(model.state_dict(), f"{config['paths']['save_dir']}/model.pth")
plot_history(train_loss_hist, val_loss_hist, f"{config['paths']['save_dir']}/loss.png")
print("Training Complete.")