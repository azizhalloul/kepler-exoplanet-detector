<div align="center">

#  Kepler Exoplanet Detector

### 1D CNN trained on real NASA Kepler data to detect planets orbiting distant stars

[![Python](https://img.shields.io/badge/Python-3+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![NASA](https://img.shields.io/badge/Data-NASA%20Kepler-0B3D91?style=flat-square&logo=nasa&logoColor=white)](https://exoplanetarchive.ipac.caltech.edu/)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)

<br/>

*When a planet orbits a star, it blocks a tiny fraction of its light.*  
*This project teaches a neural network to find that shadow.*

<br/>

![Status](https://drive.google.com/file/d/1JSPrPdEjsvBow0pSFXe29Q5LFDKAfAAZ/view?usp=drive_link)

</div>

---

##  What Is This?

The **Kepler Space Telescope** stared at 150,000 stars for 4 years, recording their brightness every 30 minutes. When a planet passes in front of a star, it causes a tiny, periodic dip in brightness — called a **transit**.

This project trains a **1D Convolutional Neural Network** to look at these brightness recordings and answer one question:

> *"Is that dip caused by a real planet — or is it just noise?"*

Inspired by the [Google Brain paper](https://arxiv.org/abs/1712.05898) that used the same approach to discover 2 new exoplanets.

---

##  Key Results

| Metric | Value |
|--------|-------|
|  AUC-ROC | **90%** |
|  Accuracy | **84%** |
|  Precision (Planet) | **77%** |
|  Recall (Planet) | **78%** |
|  Test Set | 749 stars |
|  Training Data | 4993 Kepler KOIs |
|  Classification Threshold | 0.68 |

---

##  Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/azizhalloul/kepler-exoplanet-detector.git
cd kepler-exoplanet-detector

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

pip install lightkurve
pip install torch torchvision
pip install streamlit plotly scikit-learn tqdm joblib
```

### 2. Train the Model

```bash
python src/train.py
```

>  First run downloads ~1,500 light curves from NASA MAST (~2 hours). Cached forever after.

### 3. Launch the App

```bash
streamlit run app.py
```

Open and enter any Kepler star ID and get a live prediction.

---

##  Data Pipeline

```
NASA Exoplanet Archive (TAP API)
           ↓
  KOI Table — period, epoch, duration, label
           ↓
  NASA MAST Archive (lightkurve)
           ↓
  PDCSAP Flux — pre-corrected by NASA pipeline
           ↓
  Savitzky-Golay Detrending (window=501)
           ↓
  Outlier Removal (5σ clipping)
           ↓
  Phase Folding around transit epoch
           ↓
  Median Binning → 201 fixed-width bins
           ↓
  Min-Max Normalisation → [0, 1]
           ↓
  np.ndarray shape (N, 1, 201) — model ready
```

---

##  Project Structure

```
kepler-exoplanet-detector/
│
├──  config.py              — hyperparameters & paths
├──  app.py                 — Streamlit web UI
├──  find_threshold.py      — optimal threshold finder
├──  requirements.txt
│
├──  src/
│   ├── data_loader.py        — NASA API + preprocessing pipeline
│   ├── model.py              — 1D CNN (ExoplanetCNN)
│   ├── train.py              — training loop & evaluation
│   └── predict.py            — inference on any Kepler star
│
├──  data/
│   └── processed/            — cached dataset.npz (auto-generated)
│
├──  models/
│   └── exoplanet_cnn.pth     — best model checkpoint (auto-generated)
│
└──  logs/
    ├── training_history.png
    └── confusion_matrix.png
```


---

##  Try It — Famous Systems

| Star | KIC ID | Expected Result |
|------|--------|-----------------|
| Kepler-7b | 5780885 |  Confirmed Planet |
| Kepler-11 | 6541920 |  Confirmed Planet |

---

##  Future Improvements

- **Two-branch architecture** (local + global view) as in Shallue & Vanderburg (2018) — captures both fine-grained transit shape and broader orbital context
- **Full KOI catalog** (5,000+ stars) for better generalisation on shallow-transit planets
- **Data augmentation** via phase-shifting and noise injection to expand the confirmed planet training set
- **Uncertainty quantification** via Monte Carlo Dropout for calibrated confidence intervals
- **Grad-CAM visualisation** to highlight which part of the light curve the CNN focused on

---

## References

- Shallue & Vanderburg (2018) — *Identifying Exoplanets with Deep Learning* — [arXiv:1712.05898](https://arxiv.org/abs/1712.05898)
- NASA Exoplanet Archive — https://exoplanetarchive.ipac.caltech.edu
- NASA MAST Archive — https://mast.stsci.edu
- lightkurve — https://lightkurve.github.io

---

##  Author

**Aziz Halloul**  
Engineering student in AI & Data Science — Polytech Nantes

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Aziz%20Halloul-0077B5?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/halloul-aziz/)
[![GitHub](https://img.shields.io/badge/GitHub-azizhalloul-181717?style=flat-square&logo=github)](https://github.com/azizhalloul)

---

<div align="center">

**Python · PyTorch · lightkurve · Streamlit · NASA Open Data**

</div>
