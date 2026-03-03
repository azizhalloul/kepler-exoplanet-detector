import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import sys
sys.path.insert(0, '.')
import config
from src.model import ExoplanetCNN
from src.data_loader import fetch_koi_catalog

# ── Load dataset ──────────────────────────────────────────────────────────────
data = np.load(config.DATASET_PATH)
X, y = data["X"], data["y"]
print(f"Dataset: {len(X)} samples")

# Fetch catalog and align to dataset size (7 stars were skipped during preprocessing)
print("Fetching KOI catalog...")
df = fetch_koi_catalog(max_rows=5000)
df = df[df["koi_disposition"].isin(["CONFIRMED", "FALSE POSITIVE"])].reset_index(drop=True)
df = df.iloc[:len(X)].reset_index(drop=True)
print(f"Catalog aligned to {len(df)} rows")

# ── Reproduce test split ──────────────────────────────────────────────────────
indices = np.arange(len(X))
_, test_idx = train_test_split(
    indices,
    test_size=0.15,
    stratify=y,
    random_state=42
)

# ── Load model ────────────────────────────────────────────────────────────────
device = torch.device("cpu")
model  = ExoplanetCNN().to(device)
ckpt   = torch.load(config.MODEL_PATH, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# ── Get probabilities on test set ─────────────────────────────────────────────
X_test = X[test_idx]
y_test = y[test_idx]

ds = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
probs_list = []
with torch.no_grad():
    for xb, _ in DataLoader(ds, batch_size=64):
        p = torch.softmax(model(xb), dim=1)[:, 1].numpy()
        probs_list.extend(p)

probs = np.array(probs_list)

# ── Top 5 confirmed planets ───────────────────────────────────────────────────
confirmed_local = np.where(y_test == 1)[0]
top5_local      = sorted(confirmed_local, key=lambda i: probs[i], reverse=True)[:5]

print("\nTop 5 confirmed planets your model is most confident about:\n")
for rank, local_i in enumerate(top5_local, 1):
    global_i = test_idx[local_i]
    row      = df.iloc[global_i]
    kepid    = int(row["kepid"])
    period   = float(row["koi_period"])
    duration = float(row["koi_duration"])
    t0       = float(row["koi_time0bk"])
    p_val    = probs[local_i]
    print(f"  #{rank} | KIC {kepid} | P(planet)={p_val:.4f} | Period={period:.2f}d | Duration={duration:.2f}h | t0={t0:.2f}")