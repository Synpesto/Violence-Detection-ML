# run_ml_pipeline.py
import os
import pandas as pd
from data_utils import build_feature_table
from ml_train import train_classic_ml_models


def main():
    # 1. Configuration
    DATA_DIRS = ["./CNN_Data/train", "./CNN_Data/test"]
    OUTPUT_CSV = "final_features.csv"

    all_dfs = []

    print("[INFO] Step 1: Extracting Features (Spatial + Frequency)...")

    for d in DATA_DIRS:
        if os.path.exists(d):
            print(f"[INFO] Processing folder: {d}")
            folder_name = os.path.basename(d.rstrip('/\\'))
            temp_csv_path = f"{folder_name}_temp_features.csv"

            try:
                result = build_feature_table(d, temp_csv_path)

                # Logic to get the DataFrame (handle if function returns None or DF)
                if isinstance(result, pd.DataFrame):
                    df_part = result
                elif os.path.exists(temp_csv_path):
                    # If function saves file but returns None, read the file
                    df_part = pd.read_csv(temp_csv_path)
                else:
                    print("   -> [Error] Function returned None and file was not created.")
                    continue

                if df_part is not None and not df_part.empty:
                    all_dfs.append(df_part)
                    print(f"   -> Successfully extracted {len(df_part)} samples.")
                else:
                    print("   -> [Warning] Extracted DataFrame is empty.")

                # Clean up temp file (optional, remove if you want to keep them)
                if os.path.exists(temp_csv_path):
                    os.remove(temp_csv_path)

            except Exception as e:
                print(f"   -> [Error] processing {d}: {e}")
        else:
            print(f"[WARNING] Folder not found: {d}")

    if not all_dfs:
        print("[ERROR] No data found. Cannot proceed with training.")
        return

    # Combine train and test data into one single DataFrame
    full_df = pd.concat(all_dfs, ignore_index=True)

    # Save combined features to final CSV
    full_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[SUCCESS] All features extracted and saved to {OUTPUT_CSV}")
    print(f"Total samples: {len(full_df)}")
    print("First 5 rows:")
    print(full_df.head())

    # 2. Train Machine Learning Models
    print("\n[INFO] Step 2: Training Classic ML Models...")
    train_classic_ml_models(full_df)


if __name__ == "__main__":
    main()