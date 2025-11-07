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
        folder = root/label
        if not folder.exists(): continue
        for p in folder.glob("*"):
            if p.suffix.lower() not in (".png",".jpg",".jpeg"): continue
            s = extract_stats(str(p))
            s["label"] = label
            s["path"] = str(p)
            rows.append(s)
    return pd.DataFrame(rows)

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
