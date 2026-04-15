"""
train.py
Run this ONCE before launching the app.
Trains the surrogate model and saves weights + normalization params.

Usage:
    python train.py
"""

import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

# ── 1. DATA ──────────────────────────────────────────────────────────────────

N = 5000

L = torch.FloatTensor(N).uniform_(2.0, 10.0)
P = torch.FloatTensor(N).uniform_(1000, 50000)
I = torch.FloatTensor(N).uniform_(1e-5, 1e-3)
E = torch.FloatTensor(N).uniform_(190e9, 210e9)
a = torch.FloatTensor(N).uniform_(0.3, 0.7)

a_m  = a * L
b_m  = (1 - a) * L
delta = (P * a_m**2 * b_m**2) / (3 * E * I * L)

noise = torch.randn_like(delta) * 0.02 * delta
delta = torch.clamp(delta + noise, min=1e-9)

X = torch.stack([L, P, I, E, a], dim=1)
y = delta.unsqueeze(1)

# ── 2. LOG NORMALIZE ─────────────────────────────────────────────────────────

X_mean = X.mean(dim=0)
X_std  = X.std(dim=0)
X_norm = (X - X_mean) / X_std

y_log  = torch.log(y)
y_mean = y_log.mean()
y_std  = y_log.std()
y_norm = (y_log - y_mean) / y_std

# ── 3. SHUFFLE + SPLIT ───────────────────────────────────────────────────────

idx    = torch.randperm(N)
X_norm = X_norm[idx]
y_norm = y_norm[idx]

split       = int(0.8 * N)
X_train, X_val = X_norm[:split], X_norm[split:]
y_train, y_val = y_norm[:split], y_norm[split:]

# ── 4. MODEL ─────────────────────────────────────────────────────────────────

class SurrogateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 64),  nn.Tanh(),
            nn.Linear(64, 1)
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.net(x)

# ── 5. PHYSICS LOSS ──────────────────────────────────────────────────────────

def physics_loss(model, X_batch, y_std, X_std):
    X_double = X_batch.clone()
    X_double[:, 1] = X_double[:, 1] + (torch.log(torch.tensor(2.0)) / X_std[1])
    pred_orig   = model(X_batch)
    pred_double = model(X_double)
    target      = pred_orig + torch.log(torch.tensor(2.0)) / y_std
    return ((pred_double - target)**2).mean()

# ── 6. TRAIN ─────────────────────────────────────────────────────────────────

model     = SurrogateModel()
loss_fn   = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3000, eta_min=1e-6)

best_val  = float('inf')
patience  = 200
counter   = 0
BATCH     = 256

print("Training surrogate model...")

for epoch in range(3000):
    model.train()
    idx_b  = torch.randperm(split)[:BATCH]
    Xb, yb = X_train[idx_b], y_train[idx_b]

    optimizer.zero_grad()
    mse  = loss_fn(model(Xb), yb)
    phys = physics_loss(model, Xb, y_std, X_std)
    loss = mse + 0.1 * phys
    loss.backward()
    optimizer.step()
    scheduler.step()

    model.eval()
    with torch.no_grad():
        val_loss = loss_fn(model(X_val), y_val).item()

    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), 'model_weights.pt')
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if epoch % 300 == 0:
        print(f"Epoch {epoch:4d} | Val Loss: {val_loss:.6f}")

# ── 7. SAVE NORMALIZATION PARAMS ─────────────────────────────────────────────

torch.save({
    'X_mean': X_mean,
    'X_std':  X_std,
    'y_mean': y_mean,
    'y_std':  y_std,
}, 'norm_params.pt')

print(f"\nDone. Best val loss: {best_val:.6f}")
print("Saved: model_weights.pt + norm_params.pt")
print("Now run: streamlit run app.py")

