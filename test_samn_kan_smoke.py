"""Quick smoke test for SAMN-KAN-EPI hybrid model."""
import torch
import yaml

from src.samn_kan_model import SAMNKANKansformerEPI

with open("configs/config_samn_kan.yaml") as f:
    config = yaml.safe_load(f)

print("Creating SAMN-KAN-EPI hybrid model...")
model = SAMNKANKansformerEPI(config)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# Forward pass test
B = 2
seq = torch.randn(B, 64, 6000)      # POCD-ND encoded sequence
epi = torch.randn(B, 9, 5000)       # Epigenetic features
enh_idx = torch.tensor([250, 300], dtype=torch.float)
prom_idx = torch.tensor([2500, 2600], dtype=torch.float)

print("\nForward pass...")
cls_out, reg_out, A, samn_aux = model(seq, epi, enh_idx, prom_idx)
print(f"  cls_out shape: {cls_out.shape}")   # (2, 1)
print(f"  reg_out shape: {reg_out.shape}")   # (2, 1)
print(f"  A shape:       {A.shape}")          # (2, 32, 628)
print(f"  samn_aux keys: {list(samn_aux.keys())}")
print(f"  gate_mean:     {samn_aux['gate_mean'].item():.4f}")
print(f"  gate_entropy:  {samn_aux['gate_entropy'].item():.4f}")
print(f"  slot_div:      {samn_aux['slot_diversity_penalty'].item():.4f}")

# Verify shapes
assert cls_out.shape == (B, 1), f"Expected cls_out (2,1), got {cls_out.shape}"
assert reg_out.shape == (B, 1), f"Expected reg_out (2,1), got {reg_out.shape}"
assert A.shape[0] == B and A.shape[1] == 32, f"Expected A (2,32,S), got {A.shape}"
print("\n[PASS] Forward pass shapes correct")

# Attention penalty test
att_pen = model.attention_penalty(A)
print(f"  Frobenius penalty: {att_pen.item():.4f}")
assert att_pen.shape == (), "Frobenius penalty should be scalar"
print("[PASS] Attention penalty correct")

# SAMN auxiliary loss test
samn_loss = model.samn_auxiliary_loss(samn_aux, 0.01, 0.05)
print(f"  SAMN aux loss:     {samn_loss.item():.4f}")
assert samn_loss.shape == (), "SAMN aux loss should be scalar"
print("[PASS] SAMN auxiliary loss correct")

# Backward pass test
loss = cls_out.sum() + reg_out.sum() + att_pen + samn_loss
loss.backward()
grad_norms = {n: p.grad.norm().item() for n, p in model.named_parameters()
              if p.grad is not None}
print(f"\n[PASS] Backward pass: {len(grad_norms)} params have gradients")

# Check KAN layers have gradients
kan_params = [n for n in grad_norms if "kan" in n.lower()]
print(f"  KAN params with gradients: {len(kan_params)}")
assert len(kan_params) > 0, "KAN layers should have gradients!"
print("[PASS] KAN layers are being trained")

# Check SAMN attention layers have gradients
samn_params = [n for n in grad_norms if "local_attention" in n or "bin_cross" in n
               or "survival" in n]
print(f"  SAMN attention params with gradients: {len(samn_params)}")
assert len(samn_params) > 0, "SAMN attention layers should have gradients!"
print("[PASS] SAMN attention layers are being trained")

print(f"\n{'='*50}")
print(f"  ALL TESTS PASSED — {total_params:,} parameters")
print(f"{'='*50}")
