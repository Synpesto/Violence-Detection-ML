# train_simple.py
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- IMPORT FROM YOUR EXISTING FILES ---
from cnn_utils import SimpleCNN, SimpleANN, get_transforms
from data_utils import prepare_cnn_data


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        labels = labels.float().unsqueeze(1)  # Resize label to match sigmoid output

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return running_loss / len(loader), correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            labels = labels.float().unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return running_loss / len(loader), correct / total


def main(args):
    # 1. Setup Data
    if not os.path.exists(os.path.join(args.data_dir, 'train')):
        print(f"[INFO] Folder '{args.data_dir}' not found. Running prepare_cnn_data...")
        source_img_dir = "./Processed_Images"
        if os.path.exists(source_img_dir):
            prepare_cnn_data(source_img_dir, args.data_dir)
        else:
            print(f"[ERROR] Source folder '{source_img_dir}' not found.")
            return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # 2. Prepare Loader
    img_size = 224
    print(f"[INFO] Using image size: {img_size}x{img_size}")

    train_tf, val_tf = get_transforms(img_size=img_size)

    train_ds = ImageFolder(os.path.join(args.data_dir, 'train'), transform=train_tf)
    val_ds = ImageFolder(os.path.join(args.data_dir, 'val'), transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(f"[INFO] Classes: {train_ds.classes}")

    # 3. Initialize Model
    if args.model == 'cnn':
        print("[INFO] Initializing SimpleCNN...")
        # SimpleCNN in cnn_utils.py expects 224x224 input (output feature 100352)
        model = SimpleCNN().to(device)

    elif args.model == 'ann':
        print("[INFO] Initializing SimpleANN...")
        # Updated input_shape to match the new img_size (224)
        model = SimpleANN(input_shape=(3, img_size, img_size)).to(device)
    else:
        print(f"[ERROR] Unknown model type: {args.model}")
        return

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 4. Training Loop
    best_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    print(f"\n[START] Training {args.model.upper()} for {args.epochs} epochs...")

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(
            f"Epoch {epoch + 1:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            save_name = f"best_simple_{args.model}.pth"
            torch.save(model.state_dict(), save_name)

    print(f"\n[DONE] Best Validation Accuracy: {best_acc:.4f}")
    print(f"Model saved to: best_simple_{args.model}.pth")

    # 5. Plotting
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title(f'{args.model.upper()} Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title(f'{args.model.upper()} Accuracy')
    plt.legend()

    plot_name = f"history_{args.model}.png"
    plt.savefig(plot_name)
    print(f"Plot saved to: {plot_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "ann"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--data_dir", type=str, default="./CNN_Data")

    args = parser.parse_args()
    main(args)