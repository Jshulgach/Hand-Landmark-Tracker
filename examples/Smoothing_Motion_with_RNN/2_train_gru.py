# 2_train_gru.py

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ----------------------------
# Configuration
# ----------------------------
WINDOW_SIZE = 10
BATCH_SIZE = 32
EPOCHS = 200
LR = 1e-3
DATA_DIR = "data"
MODEL_SAVE_PATH = "data/gru_smoother.pth"


# ----------------------------
# GRU Model Definition
# ----------------------------
class GRUSmoother(nn.Module):
    def __init__(self, input_size=63, hidden_size=256, num_layers=2):
        super(GRUSmoother, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, bidirectional=True,
                          dropout=0.3,  # applies dropout between GRU layers
        )
        self.out = nn.Linear(hidden_size*2, input_size)
        self.norm = nn.LayerNorm(hidden_size * 2)

    def forward(self, x):
        y, _ = self.rnn(x)
        y = self.norm(y)        # normalize across features
        return self.out(y)  # Output a residual


# ----------------------------
# Load Dataset
# ----------------------------
print("📂 Loading dataset...")
X = np.load(os.path.join(DATA_DIR, "X_raw.npy"))  # shape: [N, 10, 21, 3]
Y = np.load(os.path.join(DATA_DIR, "Y_smooth.npy"))

# Flatten landmarks to [N, T, 63]
X = X.reshape(X.shape[0], X.shape[1], -1)
Y = Y.reshape(Y.shape[0], Y.shape[1], -1)


# Compute mean and std over the training set
def normalize_input(X, save_dir=None):
    X_mean = np.mean(X, axis=(0, 1), keepdims=True)
    X_std = np.std(X, axis=(0, 1), keepdims=True)
    X_norm = (X - X_mean) / X_std

    if save_dir:
        np.save(os.path.join(DATA_DIR, "X_mean.npy"), X_mean)
        np.save(os.path.join(DATA_DIR, "X_std.npy"), X_std)
    return X_norm, X_mean, X_std


X = normalize_input(X, save_dir=DATA_DIR)[0]  # Normalize input

# Split into train and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                         torch.tensor(Y_train, dtype=torch.float32))
val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                       torch.tensor(Y_val, dtype=torch.float32))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# ----------------------------
# Training Setup
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GRUSmoother().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.SmoothL1Loss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=5)

print(f"🚀 Training GRU smoother on {device}...")

# ----------------------------
# Training Loop
# ----------------------------
best_val_loss = float('inf')
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0

    for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = criterion(pred, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    scheduler.step(avg_val_loss)
    print(f"✅ Epoch {epoch} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"💾 Best model updated and saved to {MODEL_SAVE_PATH}")


# ----------------------------
# Final Evaluation
# ----------------------------
def evaluate_model(model, loader, name="Set"):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            total_loss += loss.item()
    avg = total_loss / len(loader)
    print(f"📊 Final {name} Loss: {avg:.6f}")
    return avg


print("📈 Final model performance:")
evaluate_model(model, train_loader, name="Train")
evaluate_model(model, val_loader, name="Validation")
