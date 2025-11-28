import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from torchvision import transforms

# Import architecture & dataset definition
from train_fusion_img import FusionModel, DualImageDataset


def evaluate_fusion(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device} | Model Type: {args.model.upper()}")

    # 1. Data Setup
    if os.path.exists(os.path.join(args.data_dir, 'test')):
        test_dir = os.path.join(args.data_dir, 'test')
    else:
        test_dir = args.data_dir

    print(f"[INFO] Evaluating data from: {test_dir}")
    if not os.path.exists(test_dir):
        print(f"[ERROR] Directory not found: {test_dir}")
        return

    # Transformations matching training configuration
    spatial_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Initialize Dataset and DataLoader
    # Spec size set to (128, 128) to match training default
    test_ds = DualImageDataset(test_dir, transform=spatial_tf, spec_size=(128, 128))
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    class_names = test_ds.dataset.classes
    print(f"[INFO] Classes detected: {class_names}")

    # 2. Model Initialization
    # Initialize the specific fusion architecture (LSTM/GRU/Transformer)
    model = FusionModel(num_classes=len(class_names), model_type=args.model, spec_input_dim=128).to(device)

    if not os.path.exists(args.weights):
        print(f"[ERROR] Weights file not found: {args.weights}")
        return

    try:
        # Load state dictionary
        model.load_state_dict(torch.load(args.weights, map_location=device))
        print(f"[INFO] Successfully loaded weights from {args.weights}")
    except Exception as e:
        print(f"[CRITICAL] Failed to load weights: {e}")
        print("Ensure the model architecture matches the checkpoint file.")
        return

    model.eval()

    # 3. Inference Loop
    y_true = []
    y_pred = []

    print("[INFO] Starting inference...")
    with torch.no_grad():
        for img, spec, label in tqdm(test_loader):
            img = img.to(device)
            spec = spec.to(device)

            output = model(img, spec)
            _, pred = torch.max(output, 1)

            y_true.extend(label.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

    # 4. Metrics and Reporting
    print("\n" + "=" * 50)
    print(f"FUSION MODEL EVALUATION REPORT: {args.model.upper()}")
    print("=" * 50)

    # Print Classification Report
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Generate Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    plt.title(f'Confusion Matrix - Fusion ({args.model.upper()})')

    out_img = f"confusion_matrix_fusion_{args.model}.png"
    plt.tight_layout()
    plt.savefig(out_img)
    print(f"[SUCCESS] Confusion matrix saved to: {out_img}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Fusion Models (CNN + RNN/Transformer)")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to image dataset directory")
    parser.add_argument("--model", type=str, required=True, choices=['lstm', 'gru', 'transformer'],
                        help="Sequence model type")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained .pth weights file")

    args = parser.parse_args()
    evaluate_fusion(args)