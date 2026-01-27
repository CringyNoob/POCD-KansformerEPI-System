import torch
import yaml
import numpy as np
from src.model import Kansformer
from src.dataset import EPIDataset # Using for dummy generation
from src.encoding import POCD_ND_Encoder
from src.interpretation import Interpreter
from src.visualize import plot_cam

# Load Config & Model
with open('configs/config.yaml') as f: config = yaml.safe_load(f)
device = torch.device('cpu') # Eval on CPU is fine for 1 sample

# Re-init encoder (Must match training state)
encoder = POCD_ND_Encoder(k=config['data']['kmer_size'])
# Fit on dummy again (In prod, save/load encoder pickle)
dummy_pos = ["".join(np.random.choice(['A','C','G','T'], 1000)) for _ in range(50)]
encoder.fit(dummy_pos, dummy_pos, config['data']['sequence_length'])

model = Kansformer(config)
model.load_state_dict(torch.load(f"{config['paths']['save_dir']}/model.pth", map_location=device))

# Interpret one sample
print("Generating Interpretation...")
sample_seq = "ACGT" * 250
sample_epi = torch.randn(config['data']['epigenetic_bins'], 8)
seq_enc = encoder.transform(sample_seq)

interp = Interpreter(model)
cam_score = interp.get_cam(seq_enc, sample_epi)

plot_cam(sample_seq, cam_score, "checkpoints/interpretation_cam.png")
print("Done. Check checkpoints/ folder.")