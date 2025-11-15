# analysis_utils.py
"""
Frequency (FFT) and Time-Frequency (STFT) helpers and basic stats.
Contains functions:
 - get_frequency_spectrum(path)
 - compute_stft_for_image(path)
 - extract_stats(path)
 - plotting helpers for distributions
"""
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import signal
from pathlib import Path
from typing import Tuple, Optional

def get_frequency_spectrum(img_path: str, size: Tuple[int,int]=(256,256)) -> Optional[np.ndarray]:
    """Return grayscale image and its shifted magnitude spectrum (log scaled)."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    img = cv2.resize(img, size)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = 20*np.log(np.abs(fshift) + 1e-6)
    return magnitude

def compute_stft_for_image(img_path: str, size: Tuple[int,int]=(256,256)):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None, None, None, None
    img = cv2.resize(img, size)
    sig = img[size[0]//2, :]  # middle row
    f, t, Zxx = signal.stft(sig, fs=1.0, nperseg=64, noverlap=32)
    return img, sig, f, Zxx

def extract_stats(image_path: str) -> dict:
    """Return mean, variance, range, min, max for grayscale image"""
    im = Image.open(image_path).convert("L")
    arr = np.array(im).astype(np.float32)
    return {"mean": float(arr.mean()), "variance": float(arr.var()),
            "range": float(arr.max()-arr.min()), "min": float(arr.min()), "max": float(arr.max())}

def build_stats_dataframe(root_folder: str, classes=('fighting','not_fighting')) -> pd.DataFrame:
    rows = []
    root = Path(root_folder)

    for label in classes:
        folder = root / label
        if not folder.exists():
            continue

        for p in folder.glob("*"):
            if p.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                continue

            # Spatial features
            spatial = extract_stats(str(p))

            # Frequency features
            freq = extract_frequency_features(str(p))

            if freq is None:
                continue

            # Merge both dicts + label
            row = {
                "label": label,
                **spatial,
                **freq,
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    # Keep only numeric cols + label
    numeric_cols = df.select_dtypes(include=["number"]).columns
    df = df[["label"] + list(numeric_cols)]

    return df


def extract_frequency_features(img_path: str, size=(256,256)):
    """Extract frequency-domain features from FFT magnitude spectrum."""
    spec = get_frequency_spectrum(img_path, size=size)
    if spec is None:
        return None

    # Flatten for statistics
    flat = spec.flatten().astype(np.float32)

    # Avoid NaNs
    flat = flat[np.isfinite(flat)]
    if len(flat) == 0:
        return None

    # Frequency energy
    energy = np.sum(flat ** 2)

    # Entropy
    p = flat / np.sum(flat)
    entropy = -np.sum(p * np.log2(p + 1e-12))

    # Split low vs high frequency energy
    h, w = spec.shape
    cy, cx = h // 2, w // 2
    low_freq = spec[cy-20:cy+20, cx-20:cx+20]   # center block
    high_freq = spec.copy()
    high_freq[cy-20:cy+20, cx-20:cx+20] = 0

    low_energy = np.sum(low_freq ** 2)
    high_energy = np.sum(high_freq ** 2)
    total = low_energy + high_energy + 1e-6

    return {
        "fft_mean": float(flat.mean()),
        "fft_var": float(flat.var()),
        "fft_energy": float(energy),
        "fft_entropy": float(entropy),
        "fft_low_freq_ratio": float(low_energy / total),
        "fft_high_freq_ratio": float(high_energy / total),
    }

def plot_distribution(df, metric='mean'):
    plt.figure(figsize=(10,5))
    sns.histplot(data=df, x=metric, hue="label", kde=True, stat="density", common_norm=False)
    plt.title(f"Distribution of {metric}")
    plt.show()

def plot_box(df, metric='mean'):
    plt.figure(figsize=(8,5))
    sns.boxplot(data=df, x='label', y=metric)
    plt.title(f"{metric} by label")
    plt.show()
