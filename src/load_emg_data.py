import os
import pandas as pd
import numpy as np

def load_emg_dataset(base_folder):
    data = []

    for folder_name in sorted(os.listdir(base_folder)):
        folder_path = os.path.join(base_folder, folder_name)
        if not os.path.isdir(folder_path):
            continue

        txt_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.txt')])
        if not txt_files:
            continue

        # Use only the first .txt file
        file_path = os.path.join(folder_path, txt_files[0])
        try:
            df = pd.read_csv(file_path, sep="\t", engine="python")

            if "channel1" not in df.columns and df.shape[1] == 10:
                df.columns = ["index", "channel1", "channel2", "channel3", "channel4",
                              "channel5", "channel6", "channel7", "channel8", "class"]
                df.drop("index", axis=1, inplace=True)

            for _, row in df.iterrows():
                signal = row[["channel1", "channel2", "channel3", "channel4",
                              "channel5", "channel6", "channel7", "channel8"]].values.astype(float)
                label = int(row["class"])
                data.append({"emg": signal.tolist(), "label": label})

        except Exception as e:
            print(f"⚠️ Error reading {file_path}: {e}")

    return pd.DataFrame(data)

# TEST
if __name__ == "__main__":
    df = load_emg_dataset("data")
    print(df.head())
    print("✅ Total samples loaded:", len(df))
