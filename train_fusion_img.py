# train_fusion_img.py
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# --- 1. DATASET ---
class DualImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, spec_size=(128, 128)):
        self.dataset = ImageFolder(root_dir)
        self.transform = transform
        self.spec_h = spec_size[0]
        self.spec_w = spec_size[1]

    def __len__(self):
        return len(self.dataset)

    def compute_spectrogram(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None: return np.zeros((self.spec_h, self.spec_w), dtype=np.float32)

        # Resize to (Time, Freq) dimensions
        img_resized = cv2.resize(img, (self.spec_w, self.spec_h))
        # Normalize
        spec = img_resized.astype(np.float32) / 255.0
        return spec

    def __getitem__(self, idx):
        path, label = self.dataset.samples[idx]

        # Spatial
        img_pil = Image.open(path).convert('RGB')
        if self.transform:
            spatial_tensor = self.transform(img_pil)
        else:
            spatial_tensor = transforms.ToTensor()(img_pil)

        # Frequency
        spec_matrix = self.compute_spectrogram(path)
        spec_tensor = torch.tensor(spec_matrix.T, dtype=torch.float32)

        return spatial_tensor, spec_tensor, label


# --- 2. MODEL ---
class FusionModel(nn.Module):
    def __init__(self, num_classes, model_type='lstm', spec_input_dim=128, hidden_dim=128):
        super(FusionModel, self).__init__()
        self.model_type = model_type.lower()

        # Spatial Branch (ResNet)
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-1])
        for param in self.cnn_backbone.parameters(): param.requires_grad = False
        self.cnn_out_dim = 2048

        # Frequency Branch
        if self.model_type == 'lstm':
            self.seq_model = nn.LSTM(input_size=spec_input_dim, hidden_size=hidden_dim, batch_first=True)
        elif self.model_type == 'gru':
            self.seq_model = nn.GRU(input_size=spec_input_dim, hidden_size=hidden_dim, batch_first=True)
        elif self.model_type == 'transformer':
            layer = nn.TransformerEncoderLayer(d_model=spec_input_dim, nhead=4, batch_first=True)
            self.seq_model = nn.TransformerEncoder(layer, num_layers=2)
            hidden_dim = spec_input_dim

            # Fusion
        self.classifier = nn.Sequential(
            nn.Linear(self.cnn_out_dim + hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x_spatial, x_seq):
        with torch.no_grad():
            c_out = self.cnn_backbone(x_spatial).view(x_spatial.size(0), -1)

        if self.model_type == 'transformer':
            s_out = self.seq_model(x_seq).mean(dim=1)
        else:
            out, _ = self.seq_model(x_seq)
            s_out = out[:, -1, :]

        return self.classifier(torch.cat((c_out, s_out), dim=1))


# --- 3. MAIN (TRAIN + EVALUATE) ---
def main(args):
    device = torch.device(args.device)
    print(f"[INFO] Mode: {args.model.upper()} | Device: {device}")

    # Setup
    train_dir = os.path.join(args.data, 'train')
    val_dir = os.path.join(args.data, 'val')

    spatial_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_ds = DualImageDataset(train_dir, transform=spatial_tf)
    val_ds = DualImageDataset(val_dir, transform=spatial_tf)
    classes = train_ds.dataset.classes

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    model = FusionModel(len(classes), model_type=args.model).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_weights_path = f"best_fusion_{args.model}.pth"

    # --- TRAINING LOOP ---
    print("\n=== START TRAINING ===")
    for epoch in range(args.epochs):
        model.train()
        total, correct = 0, 0
        loss_sum = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for img, spec, label in pbar:
            img, spec, label = img.to(device), spec.to(device), label.to(device)

            optimizer.zero_grad()
            output = model(img, spec)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            _, pred = torch.max(output, 1)
            correct += (pred == label).sum().item()
            total += label.size(0)
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for img, spec, label in val_loader:
                img, spec, label = img.to(device), spec.to(device), label.to(device)
                output = model(img, spec)
                _, pred = torch.max(output, 1)
                val_correct += (pred == label).sum().item()
                val_total += label.size(0)

        val_acc = val_correct / val_total
        print(f"   >> Train Acc: {correct / total:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_weights_path)

    # --- EVALUATION (AUTO RUN AFTER TRAIN) ---
    print(f"\n=== FINAL EVALUATION (Best Model: {best_acc:.4f}) ===")

    # Load Best Weights
    model.load_state_dict(torch.load(best_weights_path))
    model.eval()

    y_true = []
    y_pred = []

    print("[INFO] Generating Report...")
    with torch.no_grad():
        for img, spec, label in tqdm(val_loader, desc="Testing"):
            img, spec, label = img.to(device), spec.to(device), label.to(device)
            output = model(img, spec)
            _, pred = torch.max(output, 1)

            y_true.extend(label.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

    # 1. Classification Report
    print("\n" + classification_report(y_true, y_pred, target_names=classes))

    # 2. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Fusion Confusion Matrix ({args.model.upper()})')

    save_img = f"confusion_matrix_fusion_{args.model}.png"
    plt.savefig(save_img)
    print(f"[SUCCESS] Matrix saved to {save_img}")
    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="./CNN_Data")
    parser.add_argument("--model", type=str, default="lstm", choices=['lstm', 'gru', 'transformer'])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    main(args)