//preprocessing
# ecg.py
# Beginner-friendly preprocessing + simple R-peak detector for your saved ecg_data.csv
# Requirements:
#   pip install pandas numpy scipy matplotlib

import os
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import signal
import matplotlib.pyplot as plt

CSV_IN = "ecg_data.csv"            # your recorded CSV (do not change filename unless you want to)
CSV_PRE = "ecg_preprocessed.csv"
CSV_PEAKS = "ecg_peaks.csv"

def estimate_fs(times_series):
    """Estimate sampling frequency from timestamp strings."""
    times = pd.to_datetime(times_series)
    diffs = times.diff().dt.total_seconds().dropna()
    if len(diffs) == 0:
        return None, None
    median_dt = diffs.median()
    fs = 1.0 / median_dt if median_dt > 0 else None
    return fs, median_dt

def remove_baseline(data, fs, cutoff=0.5):
    """High-pass to remove baseline wander (cutoff in Hz)."""
    if fs is None:
        return data
    nyq = 0.5 * fs
    wp = cutoff / nyq
    b, a = signal.butter(2, wp, btype='high')
    return signal.filtfilt(b, a, data)

def notch_50hz(data, fs, freq=50.0, Q=30.0):
    """Notch filter to remove mains interference (50Hz in India)."""
    if fs is None:
        return data
    w0 = freq / (fs / 2)  # normalized
    b, a = signal.iirnotch(w0, Q)
    return signal.filtfilt(b, a, data)

def bandpass(data, fs, low=0.5, high=40.0, order=4):
    """Bandpass between low and high Hz."""
    if fs is None:
        return data
    nyq = 0.5 * fs
    lown = low / nyq
    highn = high / nyq
    b, a = signal.butter(order, [lown:=lown, highn], btype='band')
    return signal.filtfilt(b, a, data)

def detect_rpeaks(ecg, fs):
    """Simple peak detection using prominence and minimum distance."""
    if fs is None:
        return np.array([], dtype=int)
    min_distance_samples = int(0.25 * fs)  # at least 0.25s between peaks
    prom = max(0.5 * np.std(ecg), 0.1)     # heuristic prominence
    peaks, props = signal.find_peaks(ecg, distance=min_distance_samples, prominence=prom)
    return peaks

def main_preprocess():
    if not os.path.exists(CSV_IN):
        print(f"File '{CSV_IN}' not found in the current folder. First run your serial-to-csv part to create it.")
        return

    print(f"Loading {CSV_IN} ...")
    df = pd.read_csv(CSV_IN)
    # Expecting columns: 'Time' and 'ECG_Value'
    if 'Time' not in df.columns or 'ECG_Value' not in df.columns:
        print("CSV format unexpected. It should have columns 'Time' and 'ECG_Value'.")
        print("First 5 columns found:", df.columns.tolist())
        return

    # Estimate sampling rate
    fs, median_dt = estimate_fs(df['Time'])
    if fs is None:
        print("Could not estimate sampling rate from timestamps.")
    else:
        print(f"Estimated sampling frequency: {fs:.2f} Hz (median dt = {median_dt:.6f} s)")

    raw = df['ECG_Value'].astype(float).values

    # Preprocessing pipeline (beginner-friendly)
    print("1) Removing baseline wander ...")
    cleaned = remove_baseline(raw, fs, cutoff=0.5)

    print("2) Applying 50 Hz notch (to remove mains) ...")
    cleaned = notch_50hz(cleaned, fs, freq=50.0, Q=30.0)

    print("3) Applying bandpass 0.5 - 40 Hz ...")
    try:
        # note: some scipy versions require explicit variable names; bandpass uses lown, highn inside
        cleaned = bandpass(cleaned, fs, low=0.5, high=40.0, order=4)
    except Exception:
        # fallback simpler bandpass
        nyq = 0.5 * fs if fs is not None else 1
        b, a = signal.butter(4, [0.5/nyq, 40.0/nyq], btype='band')
        cleaned = signal.filtfilt(b, a, cleaned)

    # Save preprocessed CSV
    out_df = pd.DataFrame({"Time": df['Time'], "ECG_Preproc": cleaned})
    out_df.to_csv(CSV_PRE, index=False)
    print(f"Saved preprocessed data to '{CSV_PRE}'")

    # Detect R-peaks
    print("4) Detecting R-peaks ...")
    peaks = detect_rpeaks(cleaned, fs)
    print(f"Detected {len(peaks)} peaks (R-peaks).")

    peak_times = pd.to_datetime(df['Time']).iloc[peaks].astype(str).tolist()
    peaks_df = pd.DataFrame({"peak_index": peaks, "peak_time": peak_times})
    peaks_df.to_csv(CSV_PEAKS, index=False)
    print(f"Saved peak indices to '{CSV_PEAKS}'")

    # Basic features (RR intervals and heart rate)
    if len(peaks) >= 2 and fs is not None:
        rr_samples = np.diff(peaks)
        rr_sec = rr_samples / fs
        hr_bpm = 60.0 / rr_sec
        print("Basic HR stats from detected R-peaks:")
        print(f"  Mean HR (bpm): {np.mean(hr_bpm):.1f}, Median HR (bpm): {np.median(hr_bpm):.1f}")
    else:
        print("Not enough peaks to compute HR statistics.")

    # Plot a short segment to visually verify (first 10 seconds or whole if shorter)
    print("5) Plotting raw vs filtered signal with detected peaks (close the plot to finish).")
    t = pd.to_datetime(df['Time'])
    # Convert to seconds relative to start for plotting
    t_sec = (t - t.iloc[0]).dt.total_seconds().values

    # choose segment length to plot (first 10 seconds or full if shorter)
    max_plot_time = 10.0
    if t_sec[-1] > max_plot_time:
        mask = t_sec <= max_plot_time
    else:
        mask = slice(None)

    plt.figure(figsize=(12, 4))
    plt.plot(t_sec[mask], raw[mask], label='Raw', alpha=0.6)
    plt.plot(t_sec[mask], cleaned[mask], label='Filtered', linewidth=1.2)
    if len(peaks) > 0:
        # only show peaks that are within plotted time window
        peak_mask = peaks[(t_sec[peaks] <= (t_sec[mask][-1] if isinstance(mask, slice) else t_sec[mask][-1]))]
        plt.scatter(t_sec[peak_mask], cleaned[peak_mask], color='red', marker='x', label='R-peaks')
    plt.xlabel("Time (s)")
    plt.ylabel("ECG (ADC counts or units)")
    plt.title("Raw vs Filtered ECG (first segment)")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # If you have both serial-reading code and this in same file:
    # - Make sure your serial code writes ecg_data.csv before calling preprocess
    # - This script will just preprocess the CSV if it exists
    main_preprocess()