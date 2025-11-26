import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import argparse

# Import from your existing utils
from yolo_utils import load_yolo_model, detect_fight_with_yolo


def evaluate_yolo_dataset(data_dir, model_path, output_img_path="yolo_confusion_matrix.png"):
    """
    Iterates through the dataset, runs detect_fight_with_yolo, and evaluates performance.
    Expected data_dir structure:
       data_dir/
          fight/
             img1.jpg...
          not_fight/
             img2.jpg...
    """

    # 1. Load Model
    print(f"[INFO] Loading YOLO model from {model_path}...")
    try:
        model = load_yolo_model(model_path)
    except Exception as e:
        print(f"[ERROR] Could not load model: {e}")
        return

    y_true = []
    y_pred = []

    # Define labels: not_fighting = 0, fighting = 1
    categories = {
        "not_fighting": 0,
        "fighting": 1
    }

    print("\n[INFO] Starting evaluation...")

    for category, label in categories.items():
        folder_path = os.path.join(data_dir, category)
        if not os.path.exists(folder_path):
            print(f"[WARNING] Folder not found: {folder_path}. Skipping...")
            continue

        # Get list of images
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"[INFO] Processing {category} ({len(image_files)} images)...")

        # Loop through images with progress bar
        for img_name in tqdm(image_files, desc=f"Evaluating {category}"):
            img_path = os.path.join(folder_path, img_name)

            try:
                is_fight, _ = detect_fight_with_yolo(img_path, model)

                pred_label = 1 if is_fight else 0

                y_true.append(label)
                y_pred.append(pred_label)
            except Exception as e:
                print(f"[ERROR] Failed to process {img_name}: {e}")

    if not y_true:
        print("[ERROR] No data processed. Check your dataset path and structure.")
        return

    # 2. Report Results
    print("\n" + "=" * 40)
    print("YOLO HEURISTIC EVALUATION RESULTS")
    print("=" * 40)

    target_names = ["Non-Fight", "Fight"]
    print(classification_report(y_true, y_pred, target_names=target_names))

    # 3. Plot Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    plt.title('Confusion Matrix - YOLO Heuristic Approach')

    plt.tight_layout()
    plt.savefig(output_img_path)
    print(f"[INFO] Confusion Matrix saved to: {output_img_path}")
    plt.show()


if __name__ == "__main__":
    # Configuration
    DATASET_DIR = "./Processed_Images"
    YOLO_WEIGHTS = "yolo11n.pt"

    if not os.path.exists(DATASET_DIR):
        print(f"[ERROR] Dataset directory not found: {DATASET_DIR}")
    elif not os.path.exists(YOLO_WEIGHTS):
        print(f"[ERROR] YOLO weights not found: {YOLO_WEIGHTS}")
    else:
        evaluate_yolo_dataset(DATASET_DIR, YOLO_WEIGHTS)