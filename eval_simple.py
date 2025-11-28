# eval_simple.py
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

from cnn_utils import SimpleCNN, SimpleANN, get_transforms


def evaluate_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # 1. Determine settings based on model type
    if args.model == 'cnn':
        img_size = 224
        weights_path = "best_simple_cnn.pth"
        print(f"[INFO] Config: SimpleCNN @ {img_size}x{img_size}")
    elif args.model == 'ann':
        img_size = 128
        weights_path = "best_simple_ann.pth"
        print(f"[INFO] Config: SimpleANN @ {img_size}x{img_size}")
    else:
        return

    # 2. Setup Data (Use Test set)
    if os.path.exists(os.path.join(args.data_dir, 'test')):
        test_dir = os.path.join(args.data_dir, 'test')
    else:
        test_dir = args.data_dir

    print(f"[INFO] Data source: {test_dir}")

    if not os.path.exists(test_dir):
        print(f"[ERROR] Folder not found: {test_dir}")
        return

    _, val_tf = get_transforms(img_size=img_size)

    test_ds = ImageFolder(test_dir, transform=val_tf)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

    class_names = test_ds.classes
    print(f"[INFO] Classes: {class_names}")

    # 3. Load Model & Weights
    if args.model == 'cnn':
        model = SimpleCNN().to(device)
    elif args.model == 'ann':
        model = SimpleANN(input_shape=(3, img_size, img_size)).to(device)

    if not os.path.exists(weights_path):
        print(f"[ERROR] Weights file '{weights_path}' not found. Did you train it?")
        return

    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print("[INFO] Weights loaded successfully.")
    except RuntimeError as e:
        print(f"\n[CRITICAL ERROR] Shape mismatch. You might need to retrain.\nDetails: {e}")
        return

    model.eval()

    # 4. Prediction Loop
    y_true = []
    y_pred = []

    print(f"[INFO] Evaluating {args.model.upper()} on Test set...")

    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(device)

            outputs = model(images)
            # Sigmoid output -> Binary label (threshold 0.5)
            preds = (outputs > 0.5).float().cpu().numpy().flatten()

            y_true.extend(labels.numpy())
            y_pred.extend(preds)

    # 5. Report & Confusion Matrix
    print("\n" + "=" * 40)
    print(f"EVALUATION REPORT: {args.model.upper()}")
    print("=" * 40)

    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {args.model.upper()}')

    out_img = f"confusion_matrix_{args.model}.png"
    plt.tight_layout()
    plt.savefig(out_img)
    print(f"[INFO] Confusion matrix saved to {out_img}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["cnn", "ann"], help="Model to evaluate")
    parser.add_argument("--data_dir", type=str, default="./CNN_Data")

    args = parser.parse_args()
    evaluate_model(args)