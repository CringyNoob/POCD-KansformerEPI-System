"""Quick smoke test for SAMN-EPI model."""
import torch
import yaml

from src.samn_model import SAMNKansformerEPI

with open("configs/config_samn.yaml") as f:
    config = yaml.safe_load(f)

print("Creating SAMN-EPI model...")
model = SAMNKansformerEPI(config)

B = 2
seq = torch.randn(B, 64, 6000)
epi = torch.randn(B, 9, 5000)
enh = torch.tensor([2500.0, 2500.0])
prom = torch.tensor([2500.0, 2500.0])

print("Running forward pass...")
cls_out, reg_out, aux = model(seq, epi, enh, prom)

print(f"cls_out shape: {cls_out.shape}")
print(f"reg_out shape: {reg_out.shape}")
print(f"aux keys: {list(aux.keys())}")
print(f"gate_mean: {aux['gate_mean'].item():.4f}")
print(f"gate_entropy: {aux['gate_entropy'].item():.4f}")
print(f"slot_diversity: {aux['slot_diversity_penalty'].item():.4f}")

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal params: {total:,}")
print(f"Trainable:    {trainable:,}")

# Test auxiliary loss
aux_loss = model.samn_auxiliary_loss(aux)
print(f"SAMN aux loss: {aux_loss.item():.6f}")

# Test backward pass
loss = cls_out.sum() + reg_out.sum() + aux_loss
loss.backward()
print("\nBackward pass: OK")
print("\n=== ALL TESTS PASSED ===")
