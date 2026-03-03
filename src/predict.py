# =============================================================================
# src/predict.py — Inference pipeline
#
# Two modes:
#   1. Cached mode  — if the star is in our training dataset, use the cached
#                     preprocessed curve (avoids train/inference mismatch)
#   2. Live mode    — download and preprocess fresh from NASA MAST
# =============================================================================

import argparse
import os
import sys
import torch
import numpy as np
import requests
from io import StringIO
import pandas as pd
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.model import ExoplanetCNN
from src.data_loader import fetch_and_preprocess_single


def load_model() -> tuple:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = ExoplanetCNN().to(device)

    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at {config.MODEL_PATH}. "
            "Please run src/train.py first."
        )

    ckpt = torch.load(config.MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, device, ckpt.get("val_auc", None)


def lookup_koi_params(kepid: int) -> Optional[dict]:
    """
    Auto-fetch orbital parameters for a given Kepler star from NASA Archive.
    Returns dict with period, t0, duration_hours — or None if not found.
    """
    query_url = (
        "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
        f"?query=select+kepid,koi_period,koi_time0bk,koi_duration"
        f"+from+cumulative+where+kepid={kepid}&format=csv"
    )
    try:
        r = requests.get(query_url, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text)).dropna()
        if df.empty:
            return None
        row = df.iloc[0]
        return {
            "period":         float(row["koi_period"]),
            "t0":             float(row["koi_time0bk"]),
            "duration_hours": float(row["koi_duration"])
        }
    except Exception:
        return None


def load_cached_curve(kepid: int) -> Optional[np.ndarray]:
    """
    Look up a star's preprocessed curve from the cached dataset.
    Returns the curve array (201,) if found, else None.

    This avoids train/inference mismatch — the model sees exactly
    the same representation it was trained on.
    """
    if not os.path.exists(config.DATASET_PATH):
        return None

    try:
        # Load dataset + catalog to match kepid → array index
        data   = np.load(config.DATASET_PATH)
        X, y   = data["X"], data["y"]

        # Fetch catalog to get kepids in order
        import sys
        from src.data_loader import fetch_koi_catalog
        df = fetch_koi_catalog(max_rows=5000)
        df = df[df["koi_disposition"].isin(["CONFIRMED", "FALSE POSITIVE"])].reset_index(drop=True)
        df = df.iloc[:len(X)].reset_index(drop=True)

        matches = df[df["kepid"] == kepid]
        if matches.empty:
            return None

        idx = matches.index[0]
        return X[idx, 0, :]   # shape (201,)

    except Exception:
        return None


def predict_star(
    kepid: int,
    period: float = None,
    t0: float = None,
    duration_hours: float = None
) -> dict:
    """
    Run inference on a single star.

    Strategy:
      1. Try to find the star in the cached dataset → use cached curve
      2. Fall back to live download from NASA MAST
    """
    model, device, val_auc = load_model()

    # ── Fetch orbital parameters if not provided ───────────────────────────────
    params = lookup_koi_params(kepid)
    if params:
        period         = period         or params["period"]
        t0             = t0             or params["t0"]
        duration_hours = duration_hours or params["duration_hours"]

    if not period:
        raise ValueError(f"Could not find orbital parameters for KIC {kepid}.")

    # ── Try cached curve first ─────────────────────────────────────────────────
    cached = load_cached_curve(kepid)
    lc_raw = None

    if cached is not None:
        preprocessed = cached
        source = "cache"
    else:
        # Fall back to live download
        tensor_input, lc_raw = fetch_and_preprocess_single(
            kepid, period, t0, duration_hours
        )
        if tensor_input is None:
            raise ValueError(f"Could not preprocess light curve for KIC {kepid}.")
        preprocessed = tensor_input[0, 0, :]
        source = "live"

    # ── Run inference ──────────────────────────────────────────────────────────
    PLANET_THRESHOLD = 0.80

    x      = torch.FloatTensor(preprocessed).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_class = 1 if probs[1] >= PLANET_THRESHOLD else 0
    confidence = float(probs[pred_class])
    label      = "Confirmed Planet" if pred_class == 1 else "False Positive"

    return {
        "label":         label,
        "confidence":    confidence,
        "probabilities": {
            "confirmed":     float(probs[1]),
            "false_positive": float(probs[0])
        },
        "preprocessed":  preprocessed,
        "curve":         lc_raw,
        "period":        period,
        "duration_hours": duration_hours,
        "t0":            t0,
        "source":        source   # "cache" or "live"
    }


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict exoplanet for a Kepler star")
    parser.add_argument("--kepid",    type=int,   required=True)
    parser.add_argument("--period",   type=float, default=None)
    parser.add_argument("--t0",       type=float, default=None)
    parser.add_argument("--duration", type=float, default=None)
    args = parser.parse_args()

    result = predict_star(args.kepid, args.period, args.t0, args.duration)
    print(f"\nKIC {args.kepid} — {result['label']}")
    print(f"  P(Planet)        : {result['probabilities']['confirmed']:.4f}")
    print(f"  P(False Positive): {result['probabilities']['false_positive']:.4f}")
    print(f"  Confidence       : {result['confidence']:.2%}")
    print(f"  Source           : {result['source']}")