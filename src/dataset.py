import torch
from torch.utils.data import Dataset
import numpy as np

class EPIDataset(Dataset):
    def __init__(self, num_samples, config, encoder):
        self.seq_len = config['data']['sequence_length']
        self.bins = config['data']['epigenetic_bins']
        self.encoder = encoder
        
        print(f"Generating {num_samples} synthetic samples...")
        self.data = []
        bases = ['A','C','G','T']
        
        for _ in range(num_samples):
            # Synthetic DNA
            seq = "".join(np.random.choice(bases, size=self.seq_len))
            # Synthetic Epigenetics
            epi = np.random.rand(self.bins, 8).astype(np.float32)
            # Labels
            label = np.random.randint(0, 2)
            dist = np.random.rand() * 10000
            
            self.data.append((seq, epi, float(label), float(dist)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq, epi, label, dist = self.data[idx]
        seq_enc = self.encoder.transform(seq)
        return {
            'seq': seq_enc, 
            'epi': torch.tensor(epi), 
            'label': torch.tensor([label]), 
            'dist': torch.tensor([dist])
        }