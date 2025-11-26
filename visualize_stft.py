# visualize_stft.py
import os
import random
import matplotlib.pyplot as plt
import numpy as np
from analysis_utils import get_frequency_spectrum, compute_stft_for_image


def plot_comparison(fight_path, non_fight_path):
    paths = [("Fight", fight_path), ("Non-Fight", non_fight_path)]

    # Create a figure with 2 rows (Images, FFT) and 2 columns (Fight, Non-Fight)
    # Adjusted figure size for better visibility
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))

    for i, (label, path) in enumerate(paths):
        print(f"[INFO] Processing {label}: {path}")

        # 1. Original Image
        try:
            img = plt.imread(path)
            axes[0, i].imshow(img)
            axes[0, i].set_title(f"Original: {label}")
            axes[0, i].axis('off')
        except Exception as e:
            print(f"[WARNING] Could not read image {path}: {e}")
            continue

        # 2. FFT Magnitude Spectrum
        try:
            fft_mag = get_frequency_spectrum(path)
            if fft_mag is not None:
                axes[1, i].imshow(fft_mag, cmap='inferno')
                axes[1, i].set_title(f"FFT Spectrum: {label}")
                axes[1, i].axis('off')
        except Exception as e:
            print(f"[WARNING] FFT Error: {e}")

        # 3. STFT Spectrogram (Robust Plotting)
        try:
            # Attempt to unpack 4 values (f, t, Zxx, extra)
            # If that fails, unpack 3 values
            try:
                f, t, Zxx, _ = compute_stft_for_image(path)
            except ValueError:
                f, t, Zxx = compute_stft_for_image(path)

            if Zxx is not None:
                stft_mag = np.abs(Zxx)
                log_stft = 20 * np.log10(stft_mag + 1e-6)

                # CHECK DIMENSIONS to choose plot type
                if log_stft.ndim == 2:
                    # 2D Data -> Heatmap (Spectrogram)
                    axes[2, i].imshow(log_stft, aspect='auto', origin='lower', cmap='jet')
                    axes[2, i].set_title(f"STFT Spectrogram: {label}")
                    axes[2, i].set_ylabel('Frequency Bin')
                    axes[2, i].set_xlabel('Time Segment')
                else:
                    # 1D Data -> Line Plot (Spectrum)
                    # This handles the "Invalid shape (33,)" error
                    axes[2, i].plot(log_stft, color='blue')
                    axes[2, i].fill_between(range(len(log_stft)), log_stft, color='blue', alpha=0.3)
                    axes[2, i].set_title(f"STFT Mean Spectrum (1D): {label}")
                    axes[2, i].set_xlabel('Frequency Bin')
                    axes[2, i].set_ylabel('Log Magnitude')
                    axes[2, i].grid(True, alpha=0.5)

        except Exception as e:
            print(f"[WARNING] STFT Plot Error for {path}: {e}")
            axes[2, i].text(0.5, 0.5, "STFT Error", ha='center')

    plt.tight_layout()
    out_file = "stft_analysis.png"
    plt.savefig(out_file)
    print(f"[SUCCESS] Saved analysis chart to {out_file}")
    # plt.show() # Uncomment to view the window popup


def main():
    # Point to your data directory
    # Adjust this if your folder structure is different
    data_dir = "./CNN_Data/test"

    fight_dir = os.path.join(data_dir, "fighting")
    non_fight_dir = os.path.join(data_dir, "not_fighting")

    # Handle potential folder name variations
    if not os.path.exists(non_fight_dir):
        non_fight_dir = os.path.join(data_dir, "non_fighting")
    if not os.path.exists(fight_dir):
        fight_dir = os.path.join(data_dir, "fight")

    if not os.path.exists(fight_dir) or not os.path.exists(non_fight_dir):
        print(f"[ERROR] Could not find data folders in {data_dir}")
        return

    # Pick random images with validation
    try:
        f_files = [f for f in os.listdir(fight_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        nf_files = [f for f in os.listdir(non_fight_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

        if not f_files or not nf_files:
            print("[ERROR] One of the image folders is empty.")
            return

        f_img = random.choice(f_files)
        nf_img = random.choice(nf_files)

        plot_comparison(
            os.path.join(fight_dir, f_img),
            os.path.join(non_fight_dir, nf_img)
        )
    except Exception as e:
        print(f"[ERROR] {e}")


if __name__ == "__main__":
    main()