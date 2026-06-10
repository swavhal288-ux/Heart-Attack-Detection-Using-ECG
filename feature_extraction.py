#feature extraction of one
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

def extract_features():
    # Step 1: Read preprocessed ECG data
    df = pd.read_csv("ecg_preprocessed.csv")
    signal = df["ECG_Preproc"].values

    time = pd.to_datetime(df["Time"], errors="coerce")
    time_seconds = (time - time.iloc[0]).dt.total_seconds()

    print("✅ Loaded preprocessed ECG data.")

    # Step 2: Detect R-peaks again for feature extraction
    peaks, _ = find_peaks(signal, distance=150, prominence=200)
    rr_intervals = np.diff(time_seconds[peaks])  # time between beats in seconds

    # Step 3: Compute Heart Rate (HR) in BPM
    heart_rate = 60 / rr_intervals if len(rr_intervals) > 0 else []
    avg_hr = np.mean(heart_rate) if len(heart_rate) > 0 else 0

    # Step 4: Compute basic signal features
    features = {
        "Total_Samples": len(signal),
        "Detected_Peaks": len(peaks),
        "Average_HeartRate_BPM": round(avg_hr, 2),
        "RR_Interval_Mean_s": round(np.mean(rr_intervals), 4) if len(rr_intervals) > 0 else 0,
        "RR_Interval_SD_s": round(np.std(rr_intervals), 4) if len(rr_intervals) > 0 else 0,
        "Signal_Mean": round(np.mean(signal), 2),
        "Signal_StdDev": round(np.std(signal), 2),
        "Signal_Min": np.min(signal),
        "Signal_Max": np.max(signal)
    }

    # Step 5: Save features to CSV
    features_df = pd.DataFrame([features])
    features_df.to_csv("ecg_features.csv", index=False)
    print("\n✅ ECG features extracted and saved to 'ecg_features.csv'")

    # Step 6: Optional — Plot R-peaks again
    plt.figure(figsize=(12, 4))
    plt.plot(time_seconds, signal, label="Filtered ECG", alpha=0.7)
    plt.scatter(time_seconds[peaks], signal[peaks], color="red", label="R-peaks")
    plt.xlabel("Time (s)")
    plt.ylabel("ECG (filtered units)")
    plt.title("Detected R-peaks in Preprocessed ECG")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    extract_features()