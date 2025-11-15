# train_cnn.py
"""
Train ResNet50 two-phase using cnn_utils.two_phase_train
Usage:
    python train_cnn.py --data ./CNN_Data --out checkpoints/resnet_best.pth
"""
import argparse, os
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from cnn_utils import FightDetectionCNN, get_transforms, two_phase_train

def main(args):
    train_tf, val_tf = get_transforms(args.img_size)
    train_ds = ImageFolder(os.path.join(args.data,'train'), transform=train_tf)
    val_ds = ImageFolder(os.path.join(args.data,'val'), transform=val_tf)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=4)

    model = FightDetectionCNN(num_classes=len(train_ds.classes), pretrained=True)
    model, history = two_phase_train(model, train_loader, val_loader, device=args.device,
                                     phase1_epochs=args.p1, phase2_epochs=args.p2,
                                     lr_p1=args.lr1, lr_p2=args.lr2, weight_decay=args.wd)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print("Saved best state_dict to", args.out)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="./CNN_Data")
    p.add_argument("--out", default="./checkpoints/resnet50_best.pth")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--p1", type=int, default=8)
    p.add_argument("--p2", type=int, default=12)
    p.add_argument("--lr1", type=float, default=1e-3)
    p.add_argument("--lr2", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    args = p.parse_args()
    main(args)
