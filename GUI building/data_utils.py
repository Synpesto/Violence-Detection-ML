# data_utils.py
"""
Data preparation helpers: directory split and a small OpenCV augmentation script.
This file intentionally keeps things simple and robust.
"""
import os
import random
import shutil
from typing import List, Dict

def prepare_cnn_data(source_dir: str, target_dir: str,
                     train_ratio: float=0.7, val_ratio: float=0.15):
    """
    Copy files from source_dir/class -> target_dir/{train,val,test}/class.
    Returns summary list.
    """
    os.makedirs(target_dir, exist_ok=True)
    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir,d))]
    splits = ['train','val','test']
    for s in splits:
        for c in classes:
            os.makedirs(os.path.join(target_dir,s,c), exist_ok=True)

    summary = []
    for c in classes:
        cls_path = os.path.join(source_dir, c)
        files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg','.jpeg','.png','.bmp','.tiff'))]
        random.shuffle(files)
        n = len(files)
        n_train = int(n*train_ratio); n_val = int(n*val_ratio)
        train_files = files[:n_train]
        val_files = files[n_train:n_train+n_val]
        test_files = files[n_train+n_val:]
        for f in train_files:
            shutil.copy2(os.path.join(cls_path,f), os.path.join(target_dir,'train',c,f))
        for f in val_files:
            shutil.copy2(os.path.join(cls_path,f), os.path.join(target_dir,'val',c,f))
        for f in test_files:
            shutil.copy2(os.path.join(cls_path,f), os.path.join(target_dir,'test',c,f))
        summary.append({'class':c,'total':len(files),'train':len(train_files),'val':len(val_files),'test':len(test_files)})
    return summary

# Minimal OpenCV augmentation (optional)
import cv2, numpy as np
def augment_and_save(src_dir: str, dst_dir: str, n_aug:int=3):
    """
    Very small augment loop: flip, small rotate, resize.
    Saves augmented images into dst_dir/<class>/
    """
    os.makedirs(dst_dir, exist_ok=True)
    classes = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir,d))]
    for c in classes:
        os.makedirs(os.path.join(dst_dir,c), exist_ok=True)
        files = [f for f in os.listdir(os.path.join(src_dir,c)) if f.lower().endswith(('.jpg','.png','.jpeg'))]
        for fn in files:
            img = cv2.imread(os.path.join(src_dir,c,fn))
            if img is None:
                continue
            base = os.path.splitext(fn)[0]
            # save original resized
            out0 = os.path.join(dst_dir,c, base + "_orig.jpg")
            cv2.imwrite(out0, cv2.resize(img,(224,224)))
            for i in range(n_aug):
                out = img.copy()
                if random.random() < 0.5:
                    out = cv2.flip(out,1)
                angle = random.uniform(-15,15)
                h,w = out.shape[:2]
                M = cv2.getRotationMatrix2D((w/2,h/2), angle, 1.0)
                out = cv2.warpAffine(out, M, (w,h))
                out = cv2.resize(out,(224,224))
                cv2.imwrite(os.path.join(dst_dir,c, f"{base}_aug{i}.jpg"), out)
    return True
