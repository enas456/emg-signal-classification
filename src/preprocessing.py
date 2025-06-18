import os
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from tqdm import tqdm

def filter_all_emg_data():
    # 📁 Project-relative paths
    base_folder = os.path.join("data")
    output_path = os.path.join("data", "filtered_EMG_dataset.csv")

    # 🎚 Filter parameters
    fs = 962.46
    fpass = [50, 400]
    fpass_norm = [f / (fs / 2) for f in fpass]
    b, a = butter(4, fpass_norm, btype='bandpass')

    all_filtered = []

    print(f"\n🚀 Starting EMG filtering from: {base_folder}")

    for folder in tqdm(sorted(os.listdir(base_folder))):
        folder_path = os.path.join(base_folder, folder)
        if not os.path.isdir(folder_path):
            continue

        txt_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".txt")])
        if not txt_files:
            print(f"⚠️ No .txt files in {folder_path}")
            continue

        file_path = os.path.join(folder_path, txt_files[0])
        print(f"\n📂 Processing {file_path}")

        try:
            df = pd.read_csv(file_path, sep="\t", engine="python")

            # Handle both formats (with or without index column)
            if df.shape[1] == 10:
                df.columns = ["index", "channel1", "channel2", "channel3", "channel4",
                              "channel5", "channel6", "channel7", "channel8", "class"]
                df.drop("index", axis=1, inplace=True)
            elif df.shape[1] == 9:
                df.columns = ["channel1", "channel2", "channel3", "channel4",
                              "channel5", "channel6", "channel7", "channel8", "class"]
            else:
                print(f"⚠️ Skipped {file_path}: unexpected column count ({df.shape[1]})")
                continue

            # Apply filtering per channel
            filtered_channels = {}
            for ch in [f"channel{i}" for i in range(1, 9)]:
                filtered = filtfilt(b, a, df[ch].values)
                filtered_channels[ch] = filtered

            # Rebuild sample rows
            for i in range(len(df)):
                sample = [filtered_channels[f"channel{j}"][i] for j in range(1, 9)]
                label = int(df["class"].iloc[i]) if not pd.isna(df["class"].iloc[i]) else -1
                all_filtered.append(sample + [label])

        except Exception as e:
            print(f"❌ Error reading {file_path}: {e}")
            continue

    # 🧪 Final results
    print(f"\n🧪 Total filtered samples: {len(all_filtered)}")
    if not all_filtered:
        print("❌ No data processed. Please check your files.")
        return

    columns = [f"emg{i}" for i in range(1, 9)] + ["label"]
    filtered_df = pd.DataFrame(all_filtered, columns=columns)

    filtered_df.to_csv(output_path, index=False)
    print(f"\n✅ All filtered EMG data saved to: {output_path}")

if __name__ == "__main__":
    filter_all_emg_data()
