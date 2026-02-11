#!/usr/bin/env python3
"""
material_price_estimator.py

Uso:
  python material_price_estimator.py --image path/to/photo.jpg --real_width_cm 20
  python material_price_estimator.py --webcam 0 --real_width_cm 20

Se non passi --real_width_cm lo script stima area in "pixel" e comunque fornisce una stima di prezzo usando dimensioni di default.
"""

import argparse
import cv2
import numpy as np
import math
import torch
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray
from tqdm import tqdm

# --------------------
# Config / parametri
# --------------------
CONFIG = {
    "material_base_cost_eur_per_cm2": {
        "leather": 0.06,
        "fabric": 0.015,
        "canvas": 0.012,
        "denim": 0.02,
        "suede": 0.05,
        "synthetic": 0.009,
        "plastic": 0.008,
        "metal": 0.02,   # per cm2 metal visible (hardware)
        "wood": 0.03
    },
    "hardware_fixed_cost": {  # zipper, button etc default add-on range
        "min": 0.5,
        "typ": 2.5,
        "max": 6.0
    },
    "labor_eur_per_hour": 18.0,
    "time_minutes_per_cm2_by_material": {
        # estimated minutes of work per cm2 (very coarse)
        "leather": 0.09,
        "fabric": 0.04,
        "canvas": 0.03,
        "denim": 0.05,
        "suede": 0.10,
        "synthetic": 0.03,
        "plastic": 0.02,
        "metal": 0.01,
        "wood": 0.05
    },
    "markup_percent": { "low": 0.15, "typ": 0.35, "high": 0.6 }  # margins
}

# --------------------
# Utility: ImageNet labels
# --------------------
# We'll use torchvision's pretrained ResNet50 to get top-k generic labels and map them (heuristic).
IMAGENET_TO_MATERIAL_HINT = {
    # some example mappings (heuristic). Many imagenet classes not directly material names.
    "wallet": ["leather", "synthetic"],
    "handbag": ["leather", "fabric", "canvas"],
    "purse": ["leather", "fabric"],
    "backpack": ["canvas", "fabric", "synthetic"],
    "jean": ["denim"],
    "jeans": ["denim"],
    "sandal": ["leather", "synthetic"],
    "shoe": ["leather", "synthetic"],
    "chain": ["metal"],
    "zipper": ["metal", "plastic"],
    # fallback mapping for words containing...
}

# --------------------
# Helpers: load model
# --------------------
def load_imagenet_model(device):
    model = models.resnet50(pretrained=True)
    model.eval()
    model.to(device)
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    # load labels
    from torchvision import datasets
    # get labels file via torchvision mapping
    try:
        idx2label = {i: s for i, s in enumerate(
            open(torch.hub.get_dir() + "/checkpoints/imagenet_classes.txt").read().splitlines())}
    except Exception:
        # fallback built-in mapping (very small). For robust usage download imagenet_classes.txt
        idx2label = {i: f"class_{i}" for i in range(1000)}
    return model, transform, idx2label

# --------------------
# Segmentation: simple largest-contour
# --------------------
def segment_largest_object(img_bgr):
    img = img_bgr.copy()
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    # adaptive threshold to separate object from background
    th = cv2.adaptiveThreshold(blur, 255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 51, 6)
    # morphological ops to clean
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    close = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
    # find contours
    contours, _ = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        mask = np.ones((h,w), dtype=np.uint8) * 255
        return mask
    # take largest contour by area
    largest = max(contours, key=cv2.contourArea)
    mask = np.zeros((h,w), dtype=np.uint8)
    cv2.drawContours(mask, [largest], -1, 255, thickness=-1)
    # smooth mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask

# --------------------
# Feature extraction: texture, specular, laplacian var (sharpness)
# --------------------
def texture_and_shine_features(img_rgb, mask):
    # img_rgb: HxWx3 in RGB
    gray = rgb2gray(img_rgb)
    # Local Binary Pattern (skimage)
    lbp = local_binary_pattern((gray*255).astype(np.uint8), P=8, R=1, method='uniform')
    # histogram
    lbp_hist, _ = np.histogram(lbp[mask>0].ravel(), bins=59, range=(0,58))
    lbp_hist = lbp_hist.astype(float) / (lbp_hist.sum()+1e-9)
    # laplacian variance (texture / smoothness)
    lap = cv2.Laplacian((gray*255).astype(np.uint8), cv2.CV_64F)
    lap_var = float(np.var(lap[mask>0]))
    # specularity: fraction of very bright pixels in masked area
    img_gray = (gray*255).astype(np.uint8)
    _, spec_mask = cv2.threshold(img_gray, 220, 255, cv2.THRESH_BINARY)
    spec_frac = np.sum((spec_mask>0)&(mask>0)) / (np.sum(mask>0)+1e-9)
    # color dominant
    pixels = img_rgb[mask>0].reshape(-1,3)
    mean_color = pixels.mean(axis=0) if pixels.size else np.array([0,0,0])
    return {"lbp_hist": lbp_hist, "lap_var": lap_var, "spec_frac": spec_frac, "mean_color": mean_color}

# --------------------
# Map imagenet predictions -> material hints
# --------------------
def imagenet_material_hints(model, transform, idx2label, pil_image, device, topk=5):
    input_t = transform(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(input_t)
        probs = torch.nn.functional.softmax(out[0], dim=0)
        topk_idx = torch.topk(probs, topk).indices.cpu().numpy().tolist()
    hints = {}
    for idx in topk_idx:
        label = idx2label.get(idx, f"class_{idx}")
        # try simple normalization: split words and check mapping
        lw = label.lower()
        mapped = []
        for key in IMAGENET_TO_MATERIAL_HINT.keys():
            if key in lw:
                mapped.extend(IMAGENET_TO_MATERIAL_HINT[key])
        # if mapping found, add with weight from prob
        hints[label] = { "prob": float(probs[idx].cpu().numpy()), "mapped_materials": mapped }
    return hints

# --------------------
# Combine heuristics to score materials
# --------------------
def score_materials(feats, imagenet_hints):
    # Candidate materials
    mats = list(CONFIG["material_base_cost_eur_per_cm2"].keys())
    scores = {m: 0.0 for m in mats}
    # Heuristic 1: specularity favors leather, synthetic, plastic, metal
    spec = feats["spec_frac"]
    if spec > 0.12:
        for m in ["leather", "synthetic", "plastic", "metal"]:
            scores[m] += spec * 2.0
    # Heuristic 2: laplacian variance: low -> smooth (leather), high -> textured (fabric, denim)
    lap = feats["lap_var"]
    if lap < 50:
        scores["leather"] += 1.5
        scores["suede"] += 0.6
    else:
        for m in ["fabric", "canvas", "denim"]:
            scores[m] += 1.0 * min(1.0, lap/400.0)
    # Heuristic 3: imagenet label hints
    for label, info in imagenet_hints.items():
        prob = info["prob"]
        for mm in info["mapped_materials"]:
            if mm in scores:
                scores[mm] += prob * 3.0
    # Heuristic 4: color/hue (very rough): tan/brown favors leather/suede
    mean_color = feats["mean_color"]
    # convert to approximate H,S,V by simple rgb->hsv
    mean_rgb = np.uint8([[mean_color]])
    hsv = cv2.cvtColor(mean_rgb, cv2.COLOR_RGB2HSV)[0,0]
    h,s,v = hsv.astype(float)
    if (h >= 5 and h <= 30) and v > 80:
        scores["leather"] += 0.6
        scores["suede"] += 0.5
    # normalize scores
    total = sum(scores.values()) + 1e-9
    scores_norm = {m: float(scores[m]/total) for m in mats}
    # sort
    sorted_scores = dict(sorted(scores_norm.items(), key=lambda kv: kv[1], reverse=True))
    return sorted_scores

# --------------------
# Area estimation (pixels -> cm2 if real_width_cm provided)
# --------------------
def estimate_area_cm2(mask, real_width_cm=None):
    h,w = mask.shape
    # bounding rect width in px
    ys, xs = np.where(mask>0)
    if xs.size==0 or ys.size==0:
        return None, None, None
    minx, maxx = xs.min(), xs
