# =============================================================================
# src/predict.py — Inference on a New Star
#
# Usage (CLI):
#   python src/predict.py --kepid 757076 --period 3.522 --t0 2454833.0 --duration 2.1
#
# Or import predict_star() directly from app.py
# =============================================================================

import argparse
import os
import sys
import torch
import numpy as np
import requests
from io import StringIO
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.model import ExoplanetCNN
from src.data_loader import fetch_and_preprocess_single

LABELS = {0: "❌ False Positive", 1: "✅ Confirmed Planet"}


def load_model(device: torch.device) -> ExoplanetCNN:
    """Load the trained model from checkpoint."""
    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(
            f"No model found at {config.MODEL_PATH}. "
            "Run `python src/train.py` first."
        )

    model = ExoplanetCNN().to(device)
    ckpt  = torch.load(config.MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def lookup_koi_params(kepid: int) -> dict | None:
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
            "t0":             float(row["koi_time0bk"]),  # BKJD — no offset needed
            "duration_hours": float(row["koi_duration"])
        }
    except Exception:
        return None


def predict_star(kepid: int, period: float = None, t0: float = None,
                 duration_hours: float = None) -> dict:
    """
    Full inference pipeline for a single star.

    If orbital parameters are not provided, we auto-look them up from the
    NASA Exoplanet Archive (works only for known KOIs).

    Returns
    -------
    dict with keys:
        label         : 'Confirmed Planet' or 'False Positive'
        confidence    : float ∈ [0, 1]
        probabilities : {'confirmed': float, 'false_positive': float}
        curve         : raw lightkurve LightCurve object (for plotting)
        preprocessed  : np.ndarray shape (201,) — the model's actual input
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(device)

    # Auto-lookup if params not provided
    if period is None or t0 is None or duration_hours is None:
        params = lookup_koi_params(kepid)
        if params is None:
            raise ValueError(
                f"Star KIC {kepid} not found in NASA Archive. "
                "Please provide orbital parameters manually."
            )
        period         = params["period"]
        t0             = params["t0"]
        duration_hours = params["duration_hours"]

    # Fetch and preprocess
    tensor_input, lc_raw = fetch_and_preprocess_single(kepid, period, t0, duration_hours)
    if tensor_input is None:
        raise RuntimeError(f"Could not download or preprocess light curve for KIC {kepid}.")

    # Inference
    x = torch.FloatTensor(tensor_input).to(device)   # shape (1, 1, 201)

    with torch.no_grad():
        probs  = model.predict_proba(x).cpu().numpy()[0]  # [p_FP, p_confirmed]

    # Threshold tuned for 82% precision on confirmed planets.
    # At 0.62, the model only predicts "planet" when highly confident,
    # trading recall (~55%) for precision (~82%) — optimal for credible discovery claims.
    PLANET_THRESHOLD = 0.68
    pred_class  = 1 if probs[1] >= PLANET_THRESHOLD else 0
    confidence  = float(probs[pred_class])

    return {
        "label":          "Confirmed Planet" if pred_class == 1 else "False Positive",
        "confidence":     confidence,
        "probabilities":  {"confirmed": float(probs[1]), "false_positive": float(probs[0])},
        "curve":          lc_raw,
        "preprocessed":   tensor_input[0, 0, :],   # shape (201,)
        "period":         period,
        "t0":             t0,
        "duration_hours": duration_hours,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict exoplanet probability for a Kepler star.")
    parser.add_argument("--kepid",    type=int,   required=True,  help="Kepler Input Catalog ID")
    parser.add_argument("--period",   type=float, default=None,   help="Orbital period (days). Auto-fetched if omitted.")
    parser.add_argument("--t0",       type=float, default=None,   help="Time of first transit (BJD)")
    parser.add_argument("--duration", type=float, default=None,   help="Transit duration (hours)")
    args = parser.parse_args()

    result = predict_star(args.kepid, args.period, args.t0, args.duration)

    print(f"\n{'─' * 50}")
    print(f"  Star KIC {args.kepid}")
    print(f"  Prediction  : {result['label']}")
    print(f"  Confidence  : {result['confidence']*100:.1f}%")
    print(f"  P(confirmed): {result['probabilities']['confirmed']*100:.1f}%")
    print(f"  P(false_pos): {result['probabilities']['false_positive']*100:.1f}%")
    print(f"{'─' * 50}\n")