import serial
import pandas as pd
from datetime import datetime
import time

# Step 1: Arduino connection setup
arduino_port = 'COM3'   # Change this if your Arduino uses a different COM port
baud_rate = 9600
file_name = "ecg_data.csv"

print("Step 1: Trying to connect to Arduino...")

try:
    ser = serial.Serial(arduino_port, baud_rate)
    time.sleep(2)  # Wait for Arduino to reset
    print("Connected to Arduino on", arduino_port)
except Exception as e:
    print("Error connecting to Arduino:", e)
    exit()  # Stop program if error occurs

print("Step 2: Collecting ECG data... Press Ctrl+C to stop.\n")

data = []

# Step 3: Read data continuously
try:
    while True:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        print("Raw data:", line)  # Shows what’s coming from Arduino
        if line.isdigit():        # Only record numeric ECG values
            timestamp = datetime.now().strftime("%H:%M:%S.%f")
            data.append([timestamp, int(line)])
except KeyboardInterrupt:
    print("\nStopped recording. Saving to CSV file...")
    ser.close()

    # Step 4: Save data to CSV
    df = pd.DataFrame(data, columns=['Time', 'ECG_Value'])
    df.to_csv(file_name, index=False)
    print("Data saved successfully to", file_name)