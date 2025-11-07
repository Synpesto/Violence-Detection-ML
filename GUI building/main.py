# main.py
"""
Example usage of the minimal module split.
Edit the paths below and run the parts you need.
"""
from pathlib import Path
import torch
import os

# Change these paths to your dataset/model paths
IMAGES_DIR = "./Images"
PROCESSED_DIR = "./Processed_Images"
FREQ_DIR = "./Frequency_Spectrums"
YOLO_WEIGHTS = "yolo11n.pt"   # adjust
RESNET_WEIGHTS = "best_fight_detection_cnn.pth"  # if you saved full model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Example imports from our modules
from yolo_utils import load_yolo_model, detect_fight_with_yolo
from cnn_utils import FightDetectionCNN, get_transforms, predict_image_cnn
from data_utils import prepare_cnn_data
from analysis_utils import build_stats_dataframe, plot_distribution

def example_yolo_run():
    model = load_yolo_model(YOLO_WEIGHTS)
    sample = os.path.join(PROCESSED_DIR, "fighting", os.listdir(os.path.join(PROCESSED_DIR,"fighting"))[0])
    fight, info = detect_fight_with_yolo(sample, model)
    print("YOLO fight:", fight)
    print(info)

def example_cnn_predict():
    # Build model skeleton and load weights (if you saved full model as state_dict, change accordingly)
    model = FightDetectionCNN(num_classes=2, pretrained=False, device=DEVICE)
    if os.path.exists(RESNET_WEIGHTS):
        # If you saved full model via torch.save(model), you can load entire model
        try:
            model = torch.load(RESNET_WEIGHTS, map_location=DEVICE)
        except Exception:
            model.load_state_dict(torch.load(RESNET_WEIGHTS, map_location=DEVICE))
    train_tf, val_tf = get_transforms(224)
    sample = os.path.join(IMAGES_DIR, "fighting", os.listdir(os.path.join(IMAGES_DIR,"fighting"))[0])
    res = predict_image_cnn(model, sample, val_tf, device=DEVICE, class_names=['not_fighting','fighting'])
    print("CNN predict:", res)

def example_frequency_analysis():
    df = build_stats_dataframe(FREQ_DIR)
    print(df.groupby("label").mean())
    plot_distribution(df, metric="mean")
    plot_distribution(df, metric="variance")

if __name__ == "__main__":
    print("Example runs (edit paths in main.py before running).")
    # example_yolo_run()      # uncomment to test YOLO (needs ultralytics installed)
    # example_cnn_predict()   # uncomment to test CNN (needs trained model)
    # example_frequency_analysis()
