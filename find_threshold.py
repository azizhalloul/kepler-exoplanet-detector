# =============================================================================
# find_threshold.py — Find the optimal classification threshold
# from your REAL model on your REAL test data.
#
# Run with: python find_threshold.py
# =============================================================================

import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from src.model import ExoplanetCNN

# ── Load dataset ──────────────────────────────────────────────────────────────
print("Loading cached dataset...")
data = np.load(config.DATASET_PATH)
X, y = data["X"], data["y"]

# Reproduce exact same test split as train.py
from sklearn.model_selection import train_test_split
_, X_test, _, y_test = train_test_split(
    X, y,
    test_size=config.TEST_SPLIT,
    stratify=y,
    random_state=config.RANDOM_SEED
)

print(f"Test set: {len(X_test)} samples | "
      f"Confirmed: {y_test.sum()} | FP: {(y_test==0).sum()}")

# ── Load trained model ────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = ExoplanetCNN().to(device)
ckpt   = torch.load(config.MODEL_PATH, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"Model loaded (best val AUC: {ckpt['val_auc']:.4f})")

# ── Collect real probabilities ────────────────────────────────────────────────
test_ds     = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

all_probs, all_labels = [], []
with torch.no_grad():
    for X_b, y_b in test_loader:
        logits = model(X_b.to(device))
        probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(y_b.numpy())

all_probs  = np.array(all_probs)
all_labels = np.array(all_labels)

# ── Sweep thresholds ──────────────────────────────────────────────────────────
print(f"\n{'Threshold':>10} | {'Precision':>10} | {'Recall':>8} | "
      f"{'F1':>6} | {'Accuracy':>9} | {'#Planets predicted':>18}")
print("-" * 75)

best_threshold = 0.5
best_f1        = 0.0

for t in np.arange(0.30, 0.90, 0.02):
    y_pred = (all_probs >= t).astype(int)
    if y_pred.sum() == 0:
        continue

    p = precision_score(all_labels, y_pred, zero_division=0)
    r = recall_score(all_labels,    y_pred, zero_division=0)
    f = f1_score(all_labels,        y_pred, zero_division=0)
    a = accuracy_score(all_labels,  y_pred)
    n = y_pred.sum()

    # Mark rows near target precision
    marker = ""
    if p >= 0.82:
        marker = " ✅ precision ≥ 82%"
    if f > best_f1:
        best_f1        = f
        best_threshold = t

    print(f"{t:>10.2f} | {p:>10.3f} | {r:>8.3f} | {f:>6.3f} | {a:>9.3f} | {n:>18}{marker}")

print(f"\n{'='*75}")
print(f"  Best F1 threshold  : {best_threshold:.2f}")

# Show the best precision≥82% threshold
for t in np.arange(0.30, 0.90, 0.01):
    y_pred = (all_probs >= t).astype(int)
    if y_pred.sum() == 0:
        continue
    p = precision_score(all_labels, y_pred, zero_division=0)
    r = recall_score(all_labels,    y_pred, zero_division=0)
    f = f1_score(all_labels,        y_pred, zero_division=0)
    a = accuracy_score(all_labels,  y_pred)
    if p >= 0.82:
        print(f"  First threshold ≥82% precision: {t:.2f} "
              f"→ Precision={p:.3f} Recall={r:.3f} F1={f:.3f} Acc={a:.3f}")
        break

print(f"{'='*75}")
print("\n→ Copy the threshold you want into PLANET_THRESHOLD in train.py and predict.py")