# 3_predict.py

import os
import numpy as np
import torch
import torch.nn as nn

# ----------------------------
# Configuration
# ----------------------------
MODEL_PATH = "data/gru_smoother.pth"
INPUT_PATH = "data/X_raw.npy"
OUTPUT_PATH = "data/Y_pred.npy"
NORM_MEAN_PATH = "data/X_mean.npy"
NORM_STD_PATH = "data/X_std.npy"

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
        self.out = nn.Linear(hidden_size * 2, input_size)
        self.norm = nn.LayerNorm(hidden_size * 2)

    def forward(self, x):
        y, _ = self.rnn(x)
        y = self.norm(y)
        return self.out(y)

# ----------------------------
# Load Model
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GRUSmoother().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"🔁 Loaded model from {MODEL_PATH}")

# ----------------------------
# Load Input + Normalize
# ----------------------------
X_raw = np.load(INPUT_PATH)  # [N, 10, 21, 3]
X_input = X_raw.reshape(X_raw.shape[0], X_raw.shape[1], -1)  # [N, 10, 63]

mean = np.load(NORM_MEAN_PATH)
std = np.load(NORM_STD_PATH)
X_input = (X_input - mean) / (std + 1e-6)

X_tensor = torch.tensor(X_input, dtype=torch.float32).to(device)

# ----------------------------
# Run Inference
# ----------------------------
with torch.no_grad():
    Y_pred = model(X_tensor).cpu().numpy()

# Reshape back to [N, T, 21, 3]
Y_pred = Y_pred.reshape(Y_pred.shape[0], Y_pred.shape[1], 21, 3)

# ----------------------------
# Save Output
# ----------------------------
np.save(OUTPUT_PATH, Y_pred)
print(f"✅ Saved predicted smoothed landmarks to: {OUTPUT_PATH}")
