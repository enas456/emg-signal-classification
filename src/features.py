import pandas as pd
import numpy as np
from scipy.signal import welch
import os

# File paths
input_file_path = "data/filtered_EMG_dataset.csv"
output_file_path = "data/featured_EMG_dataset.csv"

# Load dataset
df = pd.read_csv(input_file_path)
print("✅ EMG data loaded successfully.")

fs = 250  # Sampling frequency
window_size = 100  # Number of samples per segment
channels = [f"emg{i}" for i in range(1, 9)]

# --- Feature functions ---

def compute_time_features(signal):
    mean = np.mean(signal)
    rms = np.sqrt(np.mean(signal**2))
    std_dev = np.std(signal)
    shape_factor = rms / mean if mean != 0 else 0
    snr = mean / std_dev if std_dev != 0 else 0
    thd = np.sum(np.abs(signal - mean)) / np.abs(mean) if mean != 0 else 0
    sinad = snr
    peak_value = np.max(signal)
    crest_factor = peak_value / rms if rms != 0 else 0
    clearance_factor = peak_value / (np.mean(np.sqrt(np.abs(signal)))) if np.mean(np.sqrt(np.abs(signal))) != 0 else 0
    impulse_factor = peak_value / np.mean(np.abs(signal)) if np.mean(np.abs(signal)) != 0 else 0
    return [mean, rms, std_dev, shape_factor, snr, thd, sinad, peak_value, crest_factor, clearance_factor, impulse_factor]

def compute_frequency_features(signal, fs):
    freqs, psd = welch(signal, fs=fs, nperseg=min(len(signal), 64))
    mean_freq = np.sum(freqs * psd) / np.sum(psd) if np.sum(psd) != 0 else 0
    median_freq = freqs[np.where(np.cumsum(psd) >= np.sum(psd) / 2)[0][0]]
    band_power = np.sum(psd)
    occupied_bw = freqs[np.argmax(psd)]
    power_bw = np.max(psd)
    peak_amp = np.max(psd)
    peak_loc = freqs[np.argmax(psd)]
    return [mean_freq, median_freq, band_power, occupied_bw, power_bw, peak_amp, peak_loc]

# --- Feature extraction loop ---

feature_rows = []
for gesture in sorted(df["label"].unique()):
    gesture_df = df[df["label"] == gesture].reset_index(drop=True)
    num_windows = len(gesture_df) // window_size

    for w in range(num_windows):
        window_data = gesture_df.iloc[w * window_size : (w + 1) * window_size]
        window_features = []
        for ch in channels:
            signal = window_data[ch].values
            feats = compute_time_features(signal) + compute_frequency_features(signal, fs)
            window_features.extend(feats)
        window_features.append(gesture)
        feature_rows.append(window_features)

# --- Column names ---

feature_columns = []
for ch in channels:
    feature_columns += [
        f"{ch}_Mean", f"{ch}_RMS", f"{ch}_STD", f"{ch}_ShapeFactor", f"{ch}_SNR", f"{ch}_THD", f"{ch}_SINAD",
        f"{ch}_Peak", f"{ch}_CrestFactor", f"{ch}_ClearanceFactor", f"{ch}_ImpulseFactor",
        f"{ch}_MeanFreq", f"{ch}_MedianFreq", f"{ch}_BandPower", f"{ch}_OccupiedBW", f"{ch}_PowerBW",
        f"{ch}_PeakAmp", f"{ch}_PeakLoc"
    ]
feature_columns.append("label")

# --- Save output ---

features_df = pd.DataFrame(feature_rows, columns=feature_columns)
features_df.to_csv(output_file_path, index=False)
print(f"✅ Feature extraction completed. Saved to {output_file_path}")
