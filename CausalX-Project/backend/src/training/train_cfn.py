import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.modules.causal_fusion import CausalFusionNetwork

# Load dataset
df = pd.read_csv("data/processed/causal_multimodal_dataset.csv")

# Select features
av_feats = df[["lip_variance", "av_correlation", "av_lag_frames"]].values
phys_feats = df[["jitter_mean", "jitter_std"]].values
labels = df["label"].values

# Convert to tensors
X_av = torch.tensor(av_feats, dtype=torch.float32)
X_phys = torch.tensor(phys_feats, dtype=torch.float32)
y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

dataset = TensorDataset(X_av, X_phys, y)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Model
model = CausalFusionNetwork()
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(20):
    total_loss = 0
    for av, phys, label in loader:
        optimizer.zero_grad()
        preds = model(av, phys)
        loss = criterion(preds, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}: Loss={total_loss/len(loader):.4f}")

print("Learned alpha (AV causal weight):", model.alpha.item())
print("Learned beta (Physical causal weight):", model.beta.item())

import os

MODEL_DIR = os.path.join("models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "cfn.pth")
torch.save(model.state_dict(), MODEL_PATH)

print(f"✔ CFN model saved to {MODEL_PATH}")
