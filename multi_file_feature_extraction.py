//feature of all
import pandas as pd
import numpy as np

# List of your ECG files
files = [
    "ecg_data1.csv",
    "ecg_data2.csv",
    "ecg_data3.csv",
    "ecg_data4.csv",
    "ecg_data5.csv",
    "ecg_data6.csv",
    "ecg_data7.csv"
]

# Empty list to store features
features = []

for file in files:
    # Read ECG data
    data = pd.read_csv(file)

    # Assuming your ECG signal column name is 'ECG' or similar
    # If your file has two columns (time, ECG_value), we take the 2nd column
    ecg_signal = data.iloc[:, 1]  # take second column (ECG values)
    
    # Calculate features
    mean_val = np.mean(ecg_signal)
    std_val = np.std(ecg_signal)
    max_val = np.max(ecg_signal)
    min_val = np.min(ecg_signal)
    skewness = ecg_signal.skew()
    kurtosis = ecg_signal.kurtosis()
    rms = np.sqrt(np.mean(ecg_signal ** 2))

    # Store features for this file
    feature_dict = {
        "File_Name": file,
        "Mean": mean_val,
        "Standard_Deviation": std_val,
        "Maximum": max_val,
        "Minimum": min_val,
        "Skewness": skewness,
        "Kurtosis": kurtosis,
        "RMS": rms,
        "Label": "Normal"  # You can change this to 'Heart_Attack' if needed
    }
    
    features.append(feature_dict)

# Convert all features to one DataFrame
features_df = pd.DataFrame(features)

# Save the extracted features into a single CSV
features_df.to_csv("ecg_features.csv", index=False)

print("✅ Feature extraction completed successfully!")
print("📁 Saved as 'ecg_features.csv'")
print(features_df)