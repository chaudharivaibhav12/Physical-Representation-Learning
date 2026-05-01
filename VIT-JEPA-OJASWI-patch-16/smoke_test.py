"""
Full smoke test: real data + model + loss + backward
Run this before submitting any GPU job.
"""
import torch
from torch.utils.data import DataLoader
from model   import ViTJEPA
from dataset import ActiveMatterDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load 2 real samples
dataset = ActiveMatterDataset(
    "/scratch/ok2287/data/active_matter/data",
    split="train",
    stride=16,   # large stride = fewer samples = faster test
)
loader = DataLoader(dataset, batch_size=2, shuffle=False)
batch  = next(iter(loader))

ctx = batch["context"].to(device)
tgt = batch["target"].to(device)
print(f"Context: {ctx.shape}, dtype={ctx.dtype}")
print(f"Target:  {tgt.shape}, dtype={tgt.dtype}")
print(f"Any NaN in context: {ctx.isnan().any()}")
print(f"Any NaN in target:  {tgt.isnan().any()}")

# Build model
model = ViTJEPA().to(device)

# Forward
loss, metrics = model(ctx, tgt)
print(f"\nLoss: {loss.item():.4f}")
print(f"NaN loss: {loss.isnan().item()}")  # must be False
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}")

# Backward
loss.backward()
print("\nBackward: OK")

# Check no NaN in gradients
nan_grads = [
    n for n, p in model.named_parameters()
    if p.grad is not None and p.grad.isnan().any()
]
if nan_grads:
    print(f"WARNING: NaN gradients in: {nan_grads}")
else:
    print("Gradients: all clean (no NaN)")

print("\n✓ Smoke test passed — safe to submit GPU job!")
