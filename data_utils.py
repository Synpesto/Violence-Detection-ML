# data_utils.py
"""
Dataset utilities:
 - prepare_cnn_data: split images into train/val/test folders
 - extract_frames_from_videos: (simple) extract 1 frame/second and save with index names
 - build_feature_table: compute spatial + frequency features and save CSV
 - split_frequency_dataset: split frequency spectrum dataset into train/val/test
"""
import os, shutil, random
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from typing import Tuple, List
from analysis_utils import get_frequency_spectrum, extract_stats
from torchvision import transforms
from PIL import Image

def prepare_cnn_data(source_dir: str, target_dir: str,
                     train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    random.seed(seed)
    splits = ['train','val','test']
    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir,d))]
    for sp in splits:
        for c in classes:
            os.makedirs(os.path.join(target_dir, sp, c), exist_ok=True)

    report = []
    for c in classes:
        files = [f for f in os.listdir(os.path.join(source_dir,c))
                 if f.lower().endswith(('.jpg','.jpeg','.png'))]
        random.shuffle(files)
        total = len(files)
        t = int(total*train_ratio)
        v = int(total*val_ratio)
        train = files[:t]
        val = files[t:t+v]
        test = files[t+v:]
        for f in train:
            shutil.copy2(os.path.join(source_dir,c,f), os.path.join(target_dir,'train',c,f))
        for f in val:
            shutil.copy2(os.path.join(source_dir,c,f), os.path.join(target_dir,'val',c,f))
        for f in test:
            shutil.copy2(os.path.join(source_dir,c,f), os.path.join(target_dir,'test',c,f))
        report.append({'class':c,'total':total,'train':len(train),'val':len(val),'test':len(test)})
    return report

def split_frequency_dataset(source_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split frequency spectrum dataset into train/val/test sets

    Args:
        source_dir: Path to Frequency_Spectrums folder with class subfolders
        output_dir: Path to output folder (will create train/val/test structure)
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
    """
    import random
    import shutil
    from pathlib import Path

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1.0"

    source_path = Path(source_dir)
    output_path = Path(output_dir)

    class_folders = [d for d in source_path.iterdir() if d.is_dir()]

    for class_folder in class_folders:
        class_name = class_folder.name
        print(f"[INFO] Processing class: {class_name}")

        images = list(class_folder.glob("*.*"))
        random.shuffle(images)

        total = len(images)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        train_imgs = images[:train_end]
        val_imgs = images[train_end:val_end]
        test_imgs = images[val_end:]

        for split_name, split_imgs in [('train', train_imgs), ('val', val_imgs), ('test', test_imgs)]:
            split_dir = output_path / split_name / class_name
            split_dir.mkdir(parents=True, exist_ok=True)

            for img_path in split_imgs:
                shutil.copy2(img_path, split_dir / img_path.name)

        print(f"   >> Train: {len(train_imgs)} | Val: {len(val_imgs)} | Test: {len(test_imgs)}")

    print(f"[SUCCESS] Dataset split completed at {output_path}")


def extract_frames_from_videos(video_path: str, out_dir: str, per_second: int = 1):
    """
    Extract frames at `per_second` frames per second.
    Filenames: 0.jpg, 1.jpg, ... where number corresponds to second index.
    """
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = int(round(fps / per_second))
    idx = 0
    saved = 0
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            fname = os.path.join(out_dir, f"{idx}.jpg")
            cv2.imwrite(fname, frame)
            idx += 1
            saved += 1
        frame_idx += 1
    cap.release()
    return saved

def build_feature_table(images_root: str, out_csv: str, size=(224,224)):
    """
    For each image compute:
      - spatial stats: mean, var, min, max, range (grayscale)
      - frequency stats: mean, var, range on magnitude spectrum
    Save combined table to out_csv.
    """
    rows = []
    root = Path(images_root)
    classes = [d.name for d in root.iterdir() if d.is_dir()]
    for cls in classes:
        for p in (root/cls).glob("*"):
            if p.suffix.lower() not in ('.jpg','.jpeg','.png'): continue
            try:
                spatial = extract_stats(str(p))
                mag = get_frequency_spectrum(str(p), size=size)
                if mag is None:
                    continue
                mag_stats = {
                    "freq_mean": float(np.mean(mag)),
                    "freq_var": float(np.var(mag)),
                    "freq_min": float(np.min(mag)),
                    "freq_max": float(np.max(mag)),
                    "freq_range": float(np.max(mag)-np.min(mag))
                }
                row = {
                    "path": str(p),
                    "label": cls,
                    "spatial_mean": spatial["mean"],
                    "spatial_var": spatial["variance"],
                    "spatial_min": spatial["min"],
                    "spatial_max": spatial["max"],
                    "spatial_range": spatial["range"],
                }
                row.update(mag_stats)
                rows.append(row)
            except Exception as e:
                print("Error", p, e)
                continue
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df
