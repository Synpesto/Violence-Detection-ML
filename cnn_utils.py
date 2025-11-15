# cnn_utils.py
"""
CNN helpers: transforms, model builders (ResNet wrapper),
training/validation loops (two-phase), simple CNN/ANN baselines, predictors.
"""
from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader
import copy
import os
from PIL import Image
import numpy as np

# Transforms
def get_transforms(img_size: int = 224):
    train_transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.15,0.15,0.15,0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    return train_transform, val_transform

# Model
class FightDetectionCNN(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def freeze_backbone(self):
        for p in self.backbone.parameters(): p.requires_grad = False
        for p in self.backbone.fc.parameters(): p.requires_grad = True

    def unfreeze_backbone(self):
        for p in self.backbone.parameters(): p.requires_grad = True

    def forward(self, x):
        return self.backbone(x)

def load_cnn_model(weights_path, device="cpu"):
    model = FightDetectionCNN(num_classes=2, pretrained=False)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model

# Training utilities
def train_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0; correct = 0; total = 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += images.size(0)
    return running_loss/total if total else 0.0, correct/total if total else 0.0

def validate_epoch(model: nn.Module, loader: DataLoader, criterion, device):
    model.eval()
    running_loss = 0.0; correct = 0; total = 0
    all_targets = []; all_preds = []
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += images.size(0)
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    return running_loss/total if total else 0.0, correct/total if total else 0.0, all_targets, all_preds

def two_phase_train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                    device: str = "cpu",
                    phase1_epochs: int = 8, phase2_epochs: int = 12,
                    lr_p1: float = 1e-3, lr_p2: float = 1e-4,
                    weight_decay: float = 1e-4):
    """
    Simplified two-phase training:
      - Phase 1: freeze backbone, train classifier
      - Phase 2: unfreeze and fine-tune
    Returns: trained_model, history dict
    """
    model.to(device)
    model.freeze_backbone()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_p1, weight_decay=weight_decay)

    history = {"train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[]}
    best_val = -1; best_state = None

    # Phase 1
    for ep in range(phase1_epochs):
        tr_l, tr_a = train_epoch(model, train_loader, criterion, optimizer, device)
        val_l, val_a, _, _ = validate_epoch(model, val_loader, criterion, device)
        history["train_loss"].append(tr_l); history["train_acc"].append(tr_a)
        history["val_loss"].append(val_l); history["val_acc"].append(val_a)
        if val_a > best_val:
            best_val = val_a; best_state = copy.deepcopy(model.state_dict())

    # Phase 2
    model.load_state_dict(best_state)
    model.unfreeze_backbone()
    optimizer = optim.Adam(model.parameters(), lr=lr_p2, weight_decay=weight_decay)
    for ep in range(phase2_epochs):
        tr_l, tr_a = train_epoch(model, train_loader, criterion, optimizer, device)
        val_l, val_a, _, _ = validate_epoch(model, val_loader, criterion, device)
        history["train_loss"].append(tr_l); history["train_acc"].append(tr_a)
        history["val_loss"].append(val_l); history["val_acc"].append(val_a)
        if val_a > best_val:
            best_val = val_a; best_state = copy.deepcopy(model.state_dict())

    # Load best
    model.load_state_dict(best_state)
    return model, history

# Simple CNN/ANN baselines
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.MaxPool2d(2,2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*28*28, 256),  # adjust if input size different
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self,x):
        x = self.conv_layers(x); return self.fc(x)

class SimpleANN(nn.Module):
    def __init__(self, input_shape=(3,128,128)):
        super().__init__()
        in_dim = input_shape[0]*input_shape[1]*input_shape[2]
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim,512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512,128), nn.ReLU(),
            nn.Linear(128,1), nn.Sigmoid()
        )
    def forward(self,x): return self.net(x)

# Predict helper
def predict_image_cnn(model: nn.Module, image_path: str, transform, device='cpu', class_names=None) -> Dict[str,Any]:
    model.eval()
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1) if logits.shape[1]>1 else torch.cat([1-logits, logits], dim=1)
        pred_idx = int(torch.argmax(probs, dim=1).item())
        conf = float(probs[0,pred_idx].cpu().numpy())
    if class_names is None:
        class_names = ["not_fighting","fighting"] if logits.shape[1]>1 else ["0","1"]
    return {"predicted_class": class_names[pred_idx], "confidence": conf, "probs": probs.cpu().numpy().tolist()}
