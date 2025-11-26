import os
import numpy as np
import cv2
from tqdm import tqdm


def create_spectrogram(img_path, save_path):
    """
    Reads an image, computes FFT, applies fixed scaling, and saves as a heatmap.
    """
    try:
        # 1. Read as grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return

        # 2. Resize to 224x224 (standard input for ResNet)
        img = cv2.resize(img, (224, 224))

        # 3. Compute 2D FFT
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude = 20 * np.log(np.abs(fshift) + 1e-6)

        # 4. Global Scaling (Fixed Clipping)
        norm_spec = np.clip(magnitude, 0, 255)

        # Convert to uint8 for image saving
        norm_spec = norm_spec.astype(np.uint8)

        # 5. Apply JET Colormap (Blue=Low Energy, Red=High Energy)
        heatmap = cv2.applyColorMap(norm_spec, cv2.COLORMAP_JET)

        # 6. Save result
        cv2.imwrite(save_path, heatmap)

    except Exception as e:
        print(f"[ERROR] Could not process {img_path}: {e}")


def main():
    # Config paths
    SOURCE_DIR = "./CNN_Data"
    TARGET_DIR = "./Spectrogram_Data"

    print(f"[INFO] Generating spectrograms from {SOURCE_DIR} to {TARGET_DIR}...")

    # Walk through all files in source directory
    for root, dirs, files in os.walk(SOURCE_DIR):
        for file in tqdm(files, desc="Processing"):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                src_path = os.path.join(root, file)

                # Replicate folder structure (train/val/test) in target directory
                rel_path = os.path.relpath(root, SOURCE_DIR)
                target_folder = os.path.join(TARGET_DIR, rel_path)
                os.makedirs(target_folder, exist_ok=True)

                dst_path = os.path.join(target_folder, file)

                # Generate spectrogram
                create_spectrogram(src_path, dst_path)

    print(f"\n[SUCCESS] Spectrogram dataset ready at: {TARGET_DIR}")


if __name__ == "__main__":
    main()