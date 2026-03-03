# =============================================================================
# config.py — Central configuration for the Exoplanet Detector project
# All hyperparameters, paths, and API endpoints live here.
# =============================================================================

import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
RAW_DIR         = os.path.join(DATA_DIR, "raw")          # downloaded light curves (.fits)
PROCESSED_DIR   = os.path.join(DATA_DIR, "processed")   # numpy arrays ready for training
MODEL_DIR       = os.path.join(BASE_DIR, "models")
LOG_DIR         = os.path.join(BASE_DIR, "logs")

for d in [RAW_DIR, PROCESSED_DIR, MODEL_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# ── NASA Exoplanet Archive ─────────────────────────────────────────────────────
# TAP (Table Access Protocol) endpoint — no API key required, fully open
NASA_EXOPLANET_API = (
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    "?query=select+kepid,kepoi_name,koi_disposition,koi_period,"
    "koi_time0bk,koi_duration,koi_depth"
    "+from+cumulative"
    "+where+koi_disposition+in+('CONFIRMED','FALSE+POSITIVE')"
    "+order+by+kepid"
    "&format=csv"
)

# ── Dataset Size ──────────────────────────────────────────────────────────────────
MAX_KOIS        = 5000      # use full NASA KOI catalog (was 1500)

# ── Light Curve Preprocessing ──────────────────────────────────────────────────
N_BINS          = 201       # number of phase bins for the folded light curve
BIN_WIDTH_FRAC  = 0.16      # fraction of phase space each bin covers
SIGMA_CLIP      = 5.0       # sigma threshold for outlier removal
POLYFIT_DEGREE  = 3         # polynomial degree for baseline detrending

# ── Model Architecture ─────────────────────────────────────────────────────────
INPUT_LENGTH    = N_BINS    # 1D sequence length fed into the CNN
NUM_CLASSES     = 2         # CONFIRMED=1, FALSE POSITIVE=0

# ── Training ───────────────────────────────────────────────────────────────────
BATCH_SIZE      = 64
EPOCHS          = 40
LEARNING_RATE   = 1e-3
WEIGHT_DECAY    = 1e-4
DROPOUT_RATE    = 0.5
VAL_SPLIT       = 0.15
TEST_SPLIT      = 0.15
RANDOM_SEED     = 42

# ── Saved Artifacts ────────────────────────────────────────────────────────────
MODEL_PATH      = os.path.join(MODEL_DIR, "exoplanet_cnn.pth")
SCALER_PATH     = os.path.join(MODEL_DIR, "scaler.joblib")
DATASET_PATH    = os.path.join(PROCESSED_DIR, "dataset.npz")