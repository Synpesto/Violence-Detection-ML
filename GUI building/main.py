# main.py
"""
Example usage of the minimal module split.
Edit the paths below and run the parts you need.
"""
from pathlib import Path
import torch
import os
from torch.serialization import add_safe_globals

from yolo_utils import load_yolo_model, detect_fight_with_yolo
from cnn_utils import FightDetectionCNN, get_transforms, predict_image_cnn
from data_utils import prepare_cnn_data
from analysis_utils import build_stats_dataframe, plot_distribution

# Add to safe globals immediately after import
add_safe_globals([FightDetectionCNN])

# Change these paths to your dataset/model paths
IMAGES_DIR = "./Images"
PROCESSED_DIR = "./Processed_Images"
FREQ_DIR = "./Frequency_Spectrums"
YOLO_WEIGHTS = "yolo11n.pt"

#  Saved ONLY weights for ResNet, not full model
RESNET_WEIGHTS = "best_fight_detection_cnn.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def example_cnn_predict():
    print("\nRunning CNN example prediction...\n")
    model = FightDetectionCNN(num_classes=2, pretrained=False)
    if os.path.exists(RESNET_WEIGHTS):
        print(f"Loading weights from {RESNET_WEIGHTS}")
        state = torch.load(RESNET_WEIGHTS, map_location=DEVICE, weights_only=False)
        model.load_state_dict(state, strict=False)
    else:
        print(" ERROR: Weight file not found.")
        return
    model.to(DEVICE).eval()
    train_tf, val_tf = get_transforms(224)
    fighting_dir = os.path.join(IMAGES_DIR, "fighting")
    files = [f for f in os.listdir(fighting_dir) if f.lower().endswith(("jpg", "jpeg", "png"))]

    if not files:
        print("No images found inside Images/fighting/")
        return

    sample = os.path.join(fighting_dir, files[0])
    result = predict_image_cnn(
        model,
        sample,
        transform=val_tf,
        device=DEVICE,
        class_names=["not_fighting", "fighting"]
    )

    print("CNN Prediction Complete:")
    print(result)

def example_yolo_run():
    model = load_yolo_model(YOLO_WEIGHTS)
    sample = os.path.join(PROCESSED_DIR, "fighting", os.listdir(os.path.join(PROCESSED_DIR,"fighting"))[0])
    fight, info = detect_fight_with_yolo(sample, model)
    print("YOLO fight:", fight)
    print(info)

def example_frequency_analysis():
    df = build_stats_dataframe(FREQ_DIR)
    print(df.groupby("label").mean())
    plot_distribution(df, metric="mean")
    plot_distribution(df, metric="variance")

if __name__ == "__main__":
    print("Example runs (edit paths in main.py before running).")
    # example_yolo_run()
    # example_cnn_predict()
    # example_frequency_analysis()