# train_cnn.py
"""
Train ResNet50 two-phase using cnn_utils.two_phase_train
Usage:
    python train_cnn.py --data ./CNN_Data --out checkpoints/resnet50_best.pth
"""
import argparse, os
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from cnn_utils import FightDetectionCNN, get_transforms, two_phase_train


def main(args):
    # 1. Setup Data
    # Use the same image size as the model expects (usually 224 for ResNet)
    train_tf, val_tf = get_transforms(img_size=224)

    train_path = os.path.join(args.data, 'train')
    val_path = os.path.join(args.data, 'val')

    # Check paths
    if not os.path.exists(train_path):
        print(f"[ERROR] Train folder not found: {train_path}")
        return

    train_ds = ImageFolder(train_path, transform=train_tf)
    val_ds = ImageFolder(val_path, transform=val_tf)

    print(f"[INFO] Training on {len(train_ds)} images, Validating on {len(val_ds)} images.")
    print(f"[INFO] Device: {args.device}")

    # --- CRITICAL FIX FOR WINDOWS: num_workers=0 ---
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    # 2. Setup Model (ResNet50)
    print("[INFO] Initializing ResNet50 (Transfer Learning)...")
    model = FightDetectionCNN(num_classes=len(train_ds.classes), pretrained=True)

    # 3. Start Training
    # Phase 1: Freeze backbone, train head
    # Phase 2: Unfreeze, fine-tune all
    model, history = two_phase_train(
        model,
        train_loader,
        val_loader,
        device=args.device,
        phase1_epochs=args.p1,
        phase2_epochs=args.p2,
        lr_p1=args.lr1,
        lr_p2=args.lr2,
        weight_decay=args.wd
    )

    # 4. Save Results
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"\n[SUCCESS] Saved best model weights to: {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="./CNN_Data")
    p.add_argument("--out", default="./checkpoints/resnet50_best.pth")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch", type=int, default=32)
    # Phase 1 settings
    p.add_argument("--p1", type=int, default=5, help="Epochs for Phase 1 (Frozen backbone)")
    p.add_argument("--lr1", type=float, default=1e-3, help="Learning rate for Phase 1")
    # Phase 2 settings
    p.add_argument("--p2", type=int, default=10, help="Epochs for Phase 2 (Fine-tuning)")
    p.add_argument("--lr2", type=float, default=1e-4, help="Learning rate for Phase 2")
    p.add_argument("--wd", type=float, default=1e-4, help="Weight decay")

    args = p.parse_args()
    main(args)