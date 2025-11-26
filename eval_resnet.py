# eval_resnet.py
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from cnn_utils import FightDetectionCNN, get_transforms


def evaluate_resnet(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # 1. Setup Data (Test Set)
    test_dir = os.path.join(args.data_dir, 'test')
    if not os.path.exists(test_dir):
        print(f"[ERROR] Test folder not found: {test_dir}")
        return

    # ResNet uses 224x224
    _, val_tf = get_transforms(img_size=224)

    test_ds = ImageFolder(test_dir, transform=val_tf)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    class_names = test_ds.classes
    print(f"[INFO] Classes: {class_names}")

    # 2. Load Model
    print("[INFO] Loading ResNet50...")
    # Initialize model structure
    model = FightDetectionCNN(num_classes=len(class_names), pretrained=False)
    model.to(device)

    # Load trained weights
    weights_path = args.weights
    if not os.path.exists(weights_path):
        print(f"[ERROR] Weights file '{weights_path}' not found.")
        return

    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"[INFO] Weights loaded from {weights_path}")
    except RuntimeError as e:
        print(f"[CRITICAL] Error loading weights: {e}")
        return

    model.eval()

    # 3. Prediction Loop
    y_true = []
    y_pred = []

    print(f"[INFO] Evaluating on Test set...")

    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            # Output is raw logits (not sigmoid/softmax yet depending on loss used)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # 4. Report & Confusion Matrix
    print("\n" + "=" * 40)
    print(f"RESNET50 EVALUATION REPORT")
    print("=" * 40)

    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - ResNet50')

    out_img = "confusion_matrix_resnet.png"
    plt.tight_layout()
    plt.savefig(out_img)
    print(f"[INFO] Confusion matrix saved to {out_img}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./CNN_Data")
    parser.add_argument("--weights", type=str, default="./checkpoints/resnet50_best.pth")

    args = parser.parse_args()
    evaluate_resnet(args)