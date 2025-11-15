# train_sequence_models.py
"""
Train sequence models using frame sequences:
 - CNN + LSTM
 - CNN + GRU (RNN)
 - CNN + Transformer
"""

import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from cnn_utils import FightDetectionCNN, get_transforms


# Dataset: loads frames 0.jpg,1.jpg,... and returns (T,C,H,W)
class VideoFramesDataset(Dataset):
    def __init__(self, base_dir, list_file, transform, max_frames=16):
        """
        list_file format:
            videoA,1
            videoB,0
        """
        self.base_dir = base_dir
        self.transform = transform
        self.max_frames = max_frames
        self.samples = []

        with open(list_file, "r") as f:
            for line in f:
                name, label = line.strip().split(",")
                self.samples.append((name, int(label)))

    def __len__(self):
        return len(self.samples)

    def load_frames(self, folder):
        files = [f for f in os.listdir(folder) if f.endswith(('.jpg','.png'))]
        files = sorted(files, key=lambda x: int(os.path.splitext(x)[0]))

        frames = []
        for f in files[:self.max_frames]:
            img = Image.open(os.path.join(folder, f)).convert("RGB")
            frames.append(self.transform(img))

        # Pad if fewer frames
        if len(frames) < self.max_frames:
            pad = [torch.zeros_like(frames[0])] * (self.max_frames - len(frames))
            frames += pad

        return torch.stack(frames)  # (T, C, H, W)

    def __getitem__(self, idx):
        name, label = self.samples[idx]
        folder = os.path.join(self.base_dir, name)
        frames = self.load_frames(folder)
        return frames, label


# Backbone: CNN feature extractor
def load_backbone(device="cpu"):
    backbone = FightDetectionCNN(num_classes=2, pretrained=True).to(device)
    backbone.backbone.fc = nn.Identity()   # remove final classifier
    return backbone


# CNN + LSTM
class CNN_LSTM(nn.Module):
    def __init__(self, backbone, emb_size=2048, hidden=256, num_classes=2):
        super().__init__()
        self.cnn = backbone
        self.lstm = nn.LSTM(emb_size, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        # x: (B,T,C,H,W)
        B,T,C,H,W = x.shape
        x = x.view(B*T, C, H, W)
        emb = self.cnn(x)                 # (B*T, emb)
        emb = emb.view(B, T, -1)
        out, _ = self.lstm(emb)
        return self.fc(out[:, -1])


# CNN + GRU (RNN)
class CNN_GRU(nn.Module):
    def __init__(self, backbone, emb_size=2048, hidden=256, num_classes=2):
        super().__init__()
        self.cnn = backbone
        self.gru = nn.GRU(emb_size, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        B,T,C,H,W = x.shape
        x = x.view(B*T, C, H, W)
        emb = self.cnn(x)
        emb = emb.view(B, T, -1)
        out, _ = self.gru(emb)
        return self.fc(out[:, -1])


# CNN + Transformer
class CNN_Transformer(nn.Module):
    def __init__(self, backbone, emb_size=2048, num_classes=2, nhead=8, layers=4):
        super().__init__()
        self.cnn = backbone
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_size, nhead=nhead, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.fc = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        B,T,C,H,W = x.shape
        x = x.view(B*T, C, H, W)
        emb = self.cnn(x)
        emb = emb.view(B, T, -1)
        out = self.transformer(emb)
        return self.fc(out[:, -1])


# Factory
def build_model(model_type, backbone):
    model_type = model_type.lower()
    if model_type == "lstm":
        return CNN_LSTM(backbone)
    elif model_type == "gru" or model_type == "rnn":
        return CNN_GRU(backbone)
    elif model_type == "transformer" or model_type == "trans":
        return CNN_Transformer(backbone)
    else:
        raise ValueError("Unknown model_type. Use: lstm, gru, transformer")


# Train Loop
def train(model, loader, device="cpu", epochs=5, lr=1e-4):
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total, correct = 0, 0
        for frames, labels in loader:
            frames = frames.to(device)
            labels = labels.to(device)

            optim.zero_grad()
            out = model(frames)
            loss = loss_fn(out, labels)
            loss.backward()
            optim.step()

            total += labels.size(0)
            correct += (out.argmax(1) == labels).sum().item()

        acc = correct / total * 100
        print(f"[Epoch {epoch+1}] Loss={loss.item():.4f} Acc={acc:.2f}%")

    return model
