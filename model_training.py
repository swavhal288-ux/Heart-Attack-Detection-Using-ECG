#ml
# ✅ AI-Based Heart Attack Detection using ECG Signal Analysis
# --- Machine Learning Model ---

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Step 1: Load the extracted features
data = pd.read_csv("ecg_features.csv")

print("✅ ECG Feature Data Loaded Successfully!\n")
print(data.head())  # Show first 5 rows

# Step 2: Separate input features (X) and output label (y)
X = data.drop(["File_Name", "Label"], axis=1)
y = data["Label"]

# Step 3: Encode the label (Normal → 0, Heart_Attack → 1)
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Step 4: Split data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train ML Model (Random Forest Classifier)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("\n✅ Model Training Completed!")

# Step 6: Test the model
y_pred = model.predict(X_test)

# Step 7: Evaluate performance
print("\n📊 Model Evaluation Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 8: Save the trained model for future use
joblib.dump(model, "heart_attack_model.pkl")
print("\n💾 Trained model saved as 'heart_attack_model.pkl'")
# Step 9 (Optional): Save encoder for decoding labels later
joblib.dump(encoder, "label_encoder.pkl")
print("💾 Label encoder saved as 'label_encoder.pkl'")