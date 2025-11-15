# predict_utils.py
"""
Prediction wrappers:
 - predict_image_cnn (wrap cnn_utils.predict_image_cnn if needed)
 - predict_image_yolo (wrap yolo_utils.detect_fight_with_yolo)
"""
from typing import Dict, Any
import torch
from PIL import Image
import numpy as np
import os

from cnn_utils import FightDetectionCNN, load_cnn_model, get_transforms
from yolo_utils import load_yolo_model, detect_fight_with_yolo

def predict_image_cnn_wrapper(weights_path: str, image_path: str, device='cpu', class_names=None):
    """
    Loads model state-dict and predicts (convenience wrapper).
    """
    model = load_cnn_model(weights_path, device=device)
    _, val_tf = get_transforms(224)
    res = model_predict_image(model, image_path, val_tf, device, class_names)
    return res

def model_predict_image(model, image_path, transform, device='cpu', class_names=None):
    # keep exact signature as your predict in cnn_utils
    return __import__('cnn_utils').predict_image_cnn(model, image_path, transform, device, class_names)

def predict_image_yolo_wrapper(weights_path: str, image_path: str):
    model = load_yolo_model(weights_path)
    fight, info = detect_fight_with_yolo(image_path, model)
    return {"fight": fight, "info": info}
