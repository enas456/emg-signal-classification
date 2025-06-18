import os
import pandas as pd
import numpy as np

def load_emg_dataset(base_folder):
    data = []

    for root, dirs, files in os.walk(base_folder):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                try:
                    df = pd.read_csv(file_path, sep="\t", engine="python")
                    
                    # Handle cases where the first column is an index
                    if "channel1" not in df.columns and df.shape[1] == 10:
                        df.columns = ["index", "channel1", "channel2", "channel3", "channel4",
                                      "channel5", "channel6", "channel7", "channel8", "class"]
                        df.drop("index", axis=1, inplace=True)

                    for _, row in df.iterrows():
                        signal = row[["channel1", "channel2", "channel3", "channel4",
                                      "channel5", "channel6", "channel7", "channel8"]].values.astype(float)
                        label = int(row["class"])
                        data.append({
                            "emg": signal.tolist(),
                            "label": label
                        })
                except Exception as e:
                    print(f"⚠️ Failed to load {file_path}: {e}")
    
    return pd.DataFrame(data)
