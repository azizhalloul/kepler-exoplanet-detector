# NASA Data Ingestion & Light Curve Preprocessing
#
# Pipeline overview:
#   1. Fetch KOI (Kepler Object of Interest) table from NASA Exoplanet Archive
#      via their TAP API (Table Access Protocol) → labels + orbital parameters
#   2. For each KOI, download the raw light curve from MAST using lightkurve
#   3. Detrend, sigma-clip, phase-fold, and bin the curve → fixed-size vector
#   4. Save the processed (X, y) arrays to disk for fast reloading


import os
import sys
import logging
import requests
import numpy as np
import pandas as pd
import lightkurve as lk
from io import StringIO
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# STEP 1 — Fetch the KOI catalog from NASA


def fetch_koi_catalog(max_rows: int = 2000) -> pd.DataFrame:
    """
    Query the NASA Exoplanet Archive KOI cumulative table.

    The archive exposes a TAP (Table Access Protocol) endpoint that accepts
    ADQL (Astronomical Data Query Language) — basically SQL for astronomy.
    No API key is required; it is completely open.

    Returns a DataFrame with columns:
        kepid          : Kepler Input Catalog star ID
        kepoi_name     : KOI identifier (e.g. K00001.01)
        koi_disposition: 'CONFIRMED' or 'FALSE POSITIVE' (our label)
        koi_period     : orbital period in days — needed for phase folding
        koi_time0bk    : time of first transit in BKJD (Barycentric Kepler JD)
        koi_duration   : transit duration in hours
        koi_depth      : transit depth in parts per million
    """
    log.info("Fetching KOI catalog from NASA Exoplanet Archive ...")
    url = config.NASA_EXOPLANET_API + f"&maxrec={max_rows}"

    response = requests.get(url, timeout=60)
    response.raise_for_status()

    # The API returns plain CSV — parse it directly from the response text
    df = pd.read_csv(StringIO(response.text))

    # Drop rows with missing orbital parameters (can't phase-fold without them)
    df.dropna(subset=["koi_period", "koi_time0bk", "koi_duration"], inplace=True)

    # Encode labels: CONFIRMED → 1, FALSE POSITIVE → 0
    df["label"] = (df["koi_disposition"] == "CONFIRMED").astype(int)

    log.info(f"  → {len(df)} KOIs loaded | "
             f"CONFIRMED: {df['label'].sum()} | "
             f"FALSE POSITIVE: {(df['label'] == 0).sum()}")
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Download a single raw light curve from MAST via lightkurve
# ─────────────────────────────────────────────────────────────────────────────

def download_light_curve(kepid: int, quarter: str = "all") -> lk.LightCurve | None:
    """
    Download Kepler PDCSAP (Pre-search Data Conditioning SAP) flux for a star.

    PDCSAP is preferred over raw SAP flux because NASA's pipeline has already:
      - Removed instrumental systematics (spacecraft motion, thermal drift)
      - Applied crowding corrections
      - Removed some astrophysical variability (but NOT planetary transits)

    Parameters
    ----------
    kepid : Kepler Input Catalog ID (integer)
    quarter : which Kepler observing quarter(s) to use. 'all' stitches them.

    Returns None if no data is found (some stars have gaps in coverage).
    """
    try:
        # Search MAST archive for this star's short-cadence or long-cadence data
        search_result = lk.search_lightcurve(
            f"KIC {kepid}",
            mission="Kepler",
            author="Kepler",   # official pipeline, not community
            cadence="long"     # 30-min cadence covers full mission baseline
        )

        if len(search_result) == 0:
            return None

        # Download all quarters and stitch into one continuous light curve.
        # normalize=True divides each quarter's flux by its median so quarters
        # can be compared on the same scale after stitching.
        lc_collection = search_result.download_all(quality_bitmask="default")
        lc = lc_collection.stitch(corrector_func=lambda x: x.normalize())

        return lc

    except Exception as e:
        log.debug(f"  ✗ Failed KIC {kepid}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Preprocess: detrend → sigma-clip → phase-fold → bin → normalize
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_light_curve(
    lc: lk.LightCurve,
    period: float,
    t0: float,
    duration_hours: float,
    n_bins: int = config.N_BINS
) -> np.ndarray | None:
    """
    Transform a raw light curve into a fixed-length phase-folded vector.

    This is the critical preprocessing step. Here's why each sub-step matters:

    ① Detrending (polynomial baseline removal)
       Stars aren't perfectly constant — they pulsate, have spots, etc.
       These variations are MUCH larger than transit signals and would swamp
       the CNN. We fit a low-degree polynomial to the out-of-transit flux and
       divide it out (flattening the baseline to ~1.0).

    ② Sigma clipping
       Cosmic rays and detector artifacts create single-point flux spikes.
       We replace any point > 5σ from the local median with NaN, then drop it.

    ③ Phase folding
       A planet transits at exactly regular intervals (its orbital period P).
       By folding all time stamps modulo P, ALL transits stack on top of each
       other at phase=0, making the signal N_transits times stronger.

       phase_i = ((time_i - t0) / P) mod 1.0  ∈ [0, 1)
       We shift so transit is centered at phase=0.5.

    ④ Binning
       The folded curve has thousands of points. We bin them into N_BINS=201
       equally-spaced bins by taking the median in each bin. This:
         - Reduces noise (more points per bin → lower uncertainty)
         - Creates a FIXED-SIZE input regardless of mission duration

    ⑤ Normalisation to [0, 1]
       Min-max scale so the CNN sees consistent input magnitudes.
    """
    try:
        # ① Detrend: remove slow stellar variability
        lc_flat = lc.flatten(
            window_length=501,
            polyorder=2,
            break_tolerance=10
        )

        # ② Sigma-clip: use lightkurve's built-in method (avoids astropy version conflicts)
        lc_clean = lc_flat.remove_outliers(sigma=config.SIGMA_CLIP)

        if len(lc_clean) < 200:
            log.warning("  ✗ Too few points after sigma clipping")
            return None

        # ③ Phase-fold
        # CRITICAL FIX: t0 must be in BKJD — use koi_time0bk directly, NO +2454833 offset.
        # The offset was wrongly added in build_dataset; corrected there too.
        lc_folded = lc_clean.fold(
            period=period,
            epoch_time=t0,
            normalize_phase=True    # phase in [-0.5, 0.5]; transit centred at 0
        )

        # ④ Bin to fixed length
        lc_binned = lc_folded.bin(bins=n_bins, aggregate_func=np.nanmedian)
        flux = lc_binned.flux.value

        # ⑤ Normalise to [0, 1]
        f_min, f_max = np.nanmin(flux), np.nanmax(flux)
        if f_max - f_min < 1e-10:
            log.warning("  ✗ Degenerate flat signal — skipping")
            return None

        flux_norm = (flux - f_min) / (f_max - f_min)
        flux_norm = np.nan_to_num(flux_norm, nan=1.0)

        if len(flux_norm) != n_bins:
            flux_norm = np.interp(
                np.linspace(0, 1, n_bins),
                np.linspace(0, 1, len(flux_norm)),
                flux_norm
            )

        return flux_norm.astype(np.float32)

    except Exception as e:
        # Use warning so the error is always visible during testing
        log.warning(f"  ✗ Preprocessing failed — {type(e).__name__}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Build and cache the full dataset
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(max_kois: int = 1500, force_rebuild: bool = False):
    """
    Orchestrate the full pipeline from API → numpy arrays.

    Results are cached to DATASET_PATH as a compressed .npz file.
    Re-running without force_rebuild=True loads from cache (saves ~1h).

    Returns
    -------
    X : np.ndarray, shape (N, 1, N_BINS)  — batch of folded light curves
    y : np.ndarray, shape (N,)             — labels (0=FP, 1=CONFIRMED)
    """
    if not force_rebuild and os.path.exists(config.DATASET_PATH):
        log.info(f"Loading cached dataset from {config.DATASET_PATH}")
        data = np.load(config.DATASET_PATH)
        return data["X"], data["y"]

    # ── Fetch catalog ──────────────────────────────────────────────────────────
    koi_df = fetch_koi_catalog(max_rows=max_kois)

    X_list, y_list = [], []
    skipped = 0

    # ── Iterate over KOIs ─────────────────────────────────────────────────────
    for _, row in tqdm(koi_df.iterrows(), total=len(koi_df), desc="Processing KOIs"):
        kepid          = int(row["kepid"])
        period         = float(row["koi_period"])
        t0             = float(row["koi_time0bk"])  # BKJD — no offset, fold() expects BKJD
        duration_hours = float(row["koi_duration"])
        label          = int(row["label"])

        # Download raw light curve from MAST
        lc = download_light_curve(kepid)
        if lc is None:
            skipped += 1
            continue

        # Preprocess into a fixed-length vector
        curve = preprocess_light_curve(lc, period, t0, duration_hours)
        if curve is None:
            skipped += 1
            continue

        X_list.append(curve)
        y_list.append(label)

    if len(X_list) == 0:
        raise RuntimeError("No samples collected. Check your network connection.")

    # Stack into arrays and add channel dim: (N, N_BINS) → (N, 1, N_BINS)
    # PyTorch Conv1d expects (batch, channels, length)
    X = np.stack(X_list)[:, np.newaxis, :]   # shape: (N, 1, 201)
    y = np.array(y_list, dtype=np.int64)

    log.info(f"\nDataset built: {len(X)} samples | Skipped: {skipped}")
    log.info(f"  CONFIRMED: {y.sum()} | FALSE POSITIVE: {(y == 0).sum()}")

    # Cache to disk
    np.savez_compressed(config.DATASET_PATH, X=X, y=y)
    log.info(f"Saved to {config.DATASET_PATH}")

    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Live prediction helper (used by app.py and predict.py)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_and_preprocess_single(
    kepid: int,
    period: float,
    t0: float,
    duration_hours: float
) -> tuple[np.ndarray | None, lk.LightCurve | None]:
    """
    Download and preprocess ONE star's light curve for live inference.

    Returns
    -------
    curve  : preprocessed np.ndarray of shape (1, 1, N_BINS) — model-ready
    lc_raw : raw stitched LightCurve object for plotting in the UI
    """
    lc = download_light_curve(kepid)
    if lc is None:
        return None, None

    curve = preprocess_light_curve(lc, period, t0, duration_hours)
    if curve is None:
        return None, lc

    # Add batch and channel dims for torch: (N_BINS,) → (1, 1, N_BINS)
    tensor_input = curve[np.newaxis, np.newaxis, :].astype(np.float32)

    return tensor_input, lc