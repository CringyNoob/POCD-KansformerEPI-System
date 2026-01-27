import torch
import torch.nn as nn
from src.model_layers import KANLinear

class Kansformer(nn.Module):
    def __init__(self, config):
        super(Kansformer, self).__init__()
        
        # --- Sequence Branch (Inputs: 64 x L) ---
        self.seq_cnn = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 64, kernel_size=5, padding=2),
            nn.ReLU()
        ) # Output L becomes L/2
        
        self.bilstm = nn.LSTM(64, 64, batch_first=True, bidirectional=True) # Out: 128
        
        # --- Epigenetic Branch (Inputs: 8 x Bins) ---
        self.epi_cnn = nn.Sequential(
            nn.Conv1d(8, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # --- Fusion ---
        d_model = config['model']['hidden_dim']
        self.seq_proj = nn.Linear(128, d_model)
        self.epi_proj = nn.Linear(64, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=config['model']['num_heads'], batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config['model']['num_layers'])
        
        # --- KAN Prediction Heads ---
        self.fusion_kan = KANLinear(d_model, 128)
        
        # Classification
        self.classifier = nn.Sequential(
            KANLinear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Regression (Distance)
        self.regressor = nn.Sequential(
            KANLinear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, seq, epi):
        # seq: (B, 64, L)
        x_seq = self.seq_cnn(seq)           # (B, 64, L/2)
        x_seq = x_seq.permute(0, 2, 1)      # (B, L/2, 64)
        x_seq, _ = self.bilstm(x_seq)       # (B, L/2, 128)
        x_seq = self.seq_proj(x_seq)        # (B, L/2, d_model)
        
        # epi: (B, Bins, 8) -> Permute to (B, 8, Bins)
        x_epi = epi.permute(0, 2, 1)
        x_epi = self.epi_cnn(x_epi)         # (B, 64, Bins/2)
        x_epi = x_epi.permute(0, 2, 1)      # (B, Bins/2, 64)
        x_epi = self.epi_proj(x_epi)        # (B, Bins/2, d_model)
        
        # Concat
        z = torch.cat([x_seq, x_epi], dim=1) # (B, Seq+Epi, d_model)
        
        # Transformer
        z_trans = self.transformer(z)
        
        # Global Pooling
        z_pool = z_trans.mean(dim=1)
        
        # KAN Heads
        feats = self.fusion_kan(z_pool)
        
        return self.classifier(feats), self.regressor(feats)