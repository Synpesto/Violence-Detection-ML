# yolo_utils.py
"""
YOLO helpers: load model, person detection, overlap-based fight rules,
and simple annotation/saving helpers.
Requires: ultralytics YOLO (pip install ultralytics), OpenCV, numpy
"""
from typing import List, Tuple, Dict, Any
import os
import numpy as np
import cv2

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None  # allow import if user doesn't have ultralytics installed

def load_yolo_model(weights_path: str):
    """Load YOLO model (ultralytics). Returns model or raises."""
    if YOLO is None:
        raise ImportError("ultralytics YOLO not installed. pip install ultralytics")
    return YOLO(weights_path)

# --- overlap helpers ---
def compute_overlap_percentage(box1: List[int], box2: List[int]) -> float:
    """
    Overlap area relative to the smaller box.
    box = [x1, y1, x2, y2]
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    if x1_inter >= x2_inter or y1_inter >= y2_inter:
        return 0.0
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    area2 = (box2[2]-box2[0])*(box2[3]-box2[1])
    smaller = min(area1, area2)
    return inter_area / smaller if smaller > 0 else 0.0

def compute_horizontal_overlap_percentage(box1, box2):
    x1 = max(box1[0], box2[0]); x2 = min(box1[2], box2[2])
    if x1 >= x2: return 0.0
    overlap = x2 - x1
    w1 = box1[2]-box1[0]; w2 = box2[2]-box2[0]
    denom = min(w1, w2) if min(w1,w2)>0 else 1
    return overlap/denom

def compute_vertical_overlap_percentage(box1, box2):
    y1 = max(box1[1], box2[1]); y2 = min(box1[3], box2[3])
    if y1 >= y2: return 0.0
    overlap = y2 - y1
    h1 = box1[3]-box1[1]; h2 = box2[3]-box2[1]
    denom = min(h1, h2) if min(h1,h2)>0 else 1
    return overlap/denom

# --- detection / rule logic ---
def get_person_boxes_from_results(results) -> List[List[int]]:
    """
    Given ultralytics Results or similar (results[0].boxes),
    extract person bounding boxes as [x1,y1,x2,y2] ints.
    """
    pboxes = []
    for r in results:
        for box in r.boxes:
            if int(box.cls) == 0:  # 0 is person in COCO
                coords = list(map(int, box.xyxy[0].tolist()))
                pboxes.append(coords)
    return pboxes

def detect_fight_with_yolo(image_path: str, model, *,
                           overlap_threshold: float = 0.3,
                           horizontal_threshold: float = 0.6,
                           vertical_threshold: float = 0.7,
                           distance_threshold: float = 75,
                           expand_ratio: float = 0.05,
                           min_box_area: int = 1000) -> Tuple[bool, Dict[str,Any]]:
    """
    Run YOLO on an image path and determine pairwise interactions
    using overlap/distance heuristics. Returns (is_fight, info_dict).
    """
    results = model(image_path)
    pboxes = []
    for r in results:
        for box in r.boxes:
            if int(box.cls) == 0:
                x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                area = (x2-x1)*(y2-y1)
                if area >= min_box_area:
                    pboxes.append([x1,y1,x2,y2])

    info = {
        "person_boxes": pboxes,
        "pair_details": [],
        "max_overall_overlap": 0.0,
        "max_horizontal_overlap": 0.0,
        "max_vertical_overlap": 0.0,
        "min_distance": float("inf"),
        "thresholds": {
            "overlap": overlap_threshold,
            "horizontal": horizontal_threshold,
            "vertical": vertical_threshold,
            "distance": distance_threshold
        }
    }

    if len(pboxes) < 2:
        info["reason"] = "insufficient_people"
        return False, info

    # Expand boxes slightly
    expanded = []
    for (x1,y1,x2,y2) in pboxes:
        w, h = x2-x1, y2-y1
        x1e = max(0, x1 - int(w*expand_ratio))
        y1e = max(0, y1 - int(h*expand_ratio))
        x2e = x2 + int(w*expand_ratio)
        y2e = y2 + int(h*expand_ratio)
        expanded.append([x1e,y1e,x2e,y2e])

    fight = False
    for i in range(len(expanded)):
        for j in range(i+1, len(expanded)):
            b1, b2 = expanded[i], expanded[j]
            overall = compute_overlap_percentage(b1,b2)
            horiz = compute_horizontal_overlap_percentage(b1,b2)
            vert = compute_vertical_overlap_percentage(b1,b2)
            cx1, cy1 = (b1[0]+b1[2])/2, (b1[1]+b1[3])/2
            cx2, cy2 = (b2[0]+b2[2])/2, (b2[1]+b2[3])/2
            dist = ((cx1-cx2)**2 + (cy1-cy2)**2)**0.5

            info["pair_details"].append({
                "pair": (i,j),
                "overall_overlap": overall,
                "horizontal_overlap": horiz,
                "vertical_overlap": vert,
                "distance": dist
            })
            info["max_overall_overlap"] = max(info["max_overall_overlap"], overall)
            info["max_horizontal_overlap"] = max(info["max_horizontal_overlap"], horiz)
            info["max_vertical_overlap"] = max(info["max_vertical_overlap"], vert)
            info["min_distance"] = min(info["min_distance"], dist)

            conds = [
                overall >= overlap_threshold,
                (horiz >= horizontal_threshold and vert >= vertical_threshold),
                (dist <= distance_threshold and overall > 0.05)
            ]
            if any(conds):
                fight = True
                # add optional reason
                reason = []
                if overall >= overlap_threshold: reason.append(f"high_overlap_{overall:.3f}")
                if horiz >= horizontal_threshold and vert >= vertical_threshold:
                    reason.append(f"significant_h{horiz:.3f}_v{vert:.3f}")
                if dist <= distance_threshold and overall > 0.05: reason.append(f"close_distance_{dist:.1f}")
                info["pair_details"][-1]["fight_reason"] = reason

    info["fight_detected"] = fight
    info["person_count"] = len(pboxes)
    return fight, info

# --- annotation helper ---
def annotate_and_save(image_path: str, results, out_path: str):
    """
    Given ultralytics results object, annotate and save as out_path.
    """
    res = results[0]
    arr = res.plot()  # BGR numpy
    cv2.imwrite(out_path, arr)
    return out_path
