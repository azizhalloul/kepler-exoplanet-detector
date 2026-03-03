# =============================================================================
# app.py — Streamlit Web UI for the Exoplanet Detector
# Run with: streamlit run app.py
# =============================================================================

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.predict import predict_star

# ─────────────────────────────────────────────────────────────────────────────
# Page config — must be first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Kepler Exoplanet Detector",
    page_icon="🪐",
    layout="wide"
)

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.title("🪐 Kepler Exoplanet Detector")
st.markdown("**1D CNN trained on real NASA Kepler data** to detect planets orbiting distant stars.")
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔭 Search a Star")

    EXAMPLE_STARS = {
        "Kepler-22b (KIC 10593626)": 10593626,
        "Kepler-7b  (KIC 5780885)":  5780885,
        "Kepler-11  (KIC 6541920)":  6541920,
        "KIC 7747457": 7747457,
        "KIC 4932348": 4932348,
        "KIC 8240746": 8240746,
        "KIC 5444549": 5444549,
        "KIC 4757331": 4757331,
    }

    example = st.selectbox("🌟 Try a famous system:", list(EXAMPLE_STARS.keys()))
    default_kepid = EXAMPLE_STARS[example]

    kepid = st.number_input("Or enter a custom KIC ID:", min_value=1, value=default_kepid, step=1)

    st.divider()
    st.markdown("### ⚙️ Orbital Parameters")
    st.caption("Leave at 0 to auto-fetch from NASA.")

    period_input   = st.number_input("Period (days)",    min_value=0.0, value=0.0, format="%.4f")
    t0_input       = st.number_input("Epoch T0 (BKJD)", min_value=0.0, value=0.0, format="%.2f")
    duration_input = st.number_input("Duration (hours)", min_value=0.0, value=0.0, format="%.2f")

    analyse_btn = st.button("🚀 Analyse", use_container_width=True, type="primary")

    st.divider()
    st.markdown("### 📋 How it works")
    st.markdown("""
    1. Download PDCSAP light curve from NASA MAST
    2. Detrend & clean the signal
    3. Phase-fold around transit epoch
    4. Bin to 201 fixed-width bins
    5. Classify with the trained 1D CNN
    """)

# ─────────────────────────────────────────────────────────────────────────────
# Main panel — default state
# ─────────────────────────────────────────────────────────────────────────────
if not analyse_btn:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Architecture", "1D CNN")
    col2.metric("Training Stars", "1,494 KOIs")
    col3.metric("Input Size", "201 bins")
    col4.metric("Test Accuracy", "84%")

    st.info("👈 Select a star in the sidebar and click **Analyse** to start.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Run analysis
# ─────────────────────────────────────────────────────────────────────────────
period_arg   = period_input   if period_input   > 0 else None
t0_arg       = t0_input       if t0_input       > 0 else None
duration_arg = duration_input if duration_input > 0 else None

with st.spinner(f"🛸 Fetching light curve for KIC {kepid} from NASA MAST..."):
    try:
        result = predict_star(
            kepid=int(kepid),
            period=period_arg,
            t0=t0_arg,
            duration_hours=duration_arg
        )
    except FileNotFoundError as e:
        st.error(str(e))
        st.info("💡 Train the model first: `python src/train.py`")
        st.stop()
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
is_confirmed = result["label"] == "Confirmed Planet"
emoji        = "✅" if is_confirmed else "❌"

# Verdict
col1, col2, col3 = st.columns(3)
col1.metric("Prediction", f"{emoji} {result['label']}")
col2.metric("Confidence", f"{result['confidence']*100:.1f}%")
col3.metric("P(Planet) / P(False Positive)",
            f"{result['probabilities']['confirmed']*100:.1f}% / {result['probabilities']['false_positive']*100:.1f}%")

if is_confirmed:
    st.success(f"✅ The CNN detected a periodic transit signal consistent with a planet orbiting KIC {kepid}.")
else:
    st.warning(f"❌ The transit signal for KIC {kepid} was classified as a false positive (eclipsing binary or artefact).")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Charts
# ─────────────────────────────────────────────────────────────────────────────
preprocessed = result["preprocessed"]
lc_raw       = result["curve"]
transit_color = "#00d4aa" if is_confirmed else "#ff4b4b"

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=[
        f"Raw Light Curve — KIC {kepid}",
        "Phase-Folded Curve (CNN Input)"
    ]
)

# Left — raw light curve
if lc_raw is not None:
    n = min(3000, len(lc_raw.time))
    fig.add_trace(
        go.Scatter(
            x=lc_raw.time.value[:n],
            y=lc_raw.flux.value[:n],
            mode="lines",
            line=dict(color="#667eea", width=0.8),
            name="PDCSAP Flux"
        ),
        row=1, col=1
    )
    fig.update_xaxes(title_text="Time (BKJD days)", row=1, col=1)
    fig.update_yaxes(title_text="Normalised Flux",  row=1, col=1)

# Right — phase-folded binned curve
phase = np.linspace(-0.5, 0.5, len(preprocessed))
fig.add_trace(
    go.Scatter(
        x=phase,
        y=preprocessed,
        mode="lines+markers",
        marker=dict(size=4, color=transit_color),
        line=dict(color=transit_color, width=2),
        name="Phase-Folded"
    ),
    row=1, col=2
)
fig.update_xaxes(title_text="Orbital Phase", row=1, col=2)
fig.update_yaxes(title_text="Normalised Flux", row=1, col=2)

fig.update_layout(
    template="plotly_dark",
    height=400,
    showlegend=True
)
st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Confidence gauge
# ─────────────────────────────────────────────────────────────────────────────
p_conf = result["probabilities"]["confirmed"] * 100

gauge_fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=p_conf,
    title={"text": "Planet Probability (%)"},
    gauge={
        "axis": {"range": [0, 100]},
        "bar":  {"color": transit_color},
        "steps": [
            {"range": [0,  50], "color": "#2d1b1b"},
            {"range": [50, 75], "color": "#2d2a1b"},
            {"range": [75, 100],"color": "#1b2d25"},
        ],
        "threshold": {
            "line": {"color": "white", "width": 3},
            "thickness": 0.75,
            "value": 50
        }
    }
))
gauge_fig.update_layout(template="plotly_dark", height=280)

g1, g2 = st.columns([1, 2])
with g1:
    st.plotly_chart(gauge_fig, use_container_width=True)
with g2:
    st.markdown("### 📋 Summary")
    st.markdown(f"""
| Parameter | Value |
|---|---|
| Kepler ID | KIC {kepid} |
| Orbital Period | {result['period']:.4f} days |
| Transit Duration | {result['duration_hours']:.2f} hours |
| P(Confirmed Planet) | {result['probabilities']['confirmed']*100:.1f}% |
| P(False Positive) | {result['probabilities']['false_positive']*100:.1f}% |
| Model Verdict | {emoji} {result['label']} |
""")

st.divider()
st.caption("Data: NASA Kepler Mission via MAST Archive & NASA Exoplanet Archive. Model: 1D CNN — 84% accuracy on 225 test stars.")