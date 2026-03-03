import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("patient_data.csv")

# Rename column if needed
data.rename(columns={'C': 'Gender'}, inplace=True)

# Remove duplicates
data.drop_duplicates(inplace=True)

# -------------------------------
# ENCODING SECTION
# -------------------------------

# Binary columns
binary_cols = ['History', 'Patient', 'TakeMedication',
               'BreathShortness', 'VisualChanges',
               'NoseBleeding', 'ControlledDiet']

for col in binary_cols:
    if col in data.columns:
        data[col] = data[col].map({'No': 0, 'Yes': 1})

# Gender
if 'Gender' in data.columns:
    data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})

# Age
if 'Age' in data.columns:
    data['Age'] = data['Age'].map({
        '18-34': 1,
        '35-50': 2,
        '51-64': 3,
        '65+': 4
    })

# Severity
if 'Severity' in data.columns:
    data['Severity'] = data['Severity'].map({
        'Mild': 0,
        'Moderate': 1,
        'Severe': 2
    })

# ✅ FIXED COLUMN NAME
if 'Whendiagnoused' in data.columns:
    data['Whendiagnoused'] = data['Whendiagnoused'].map({
        '<1 Year': 0,
        '1-5 Years': 1,
        '5+ Years': 2
    })

# Encode Systolic BP
if 'Systolic' in data.columns:
    data['Systolic'] = data['Systolic'].str.replace('[^0-9]', '', regex=True)
    data['Systolic'] = pd.to_numeric(data['Systolic'], errors='coerce')

# Encode Diastolic BP
if 'Diastolic' in data.columns:
    data['Diastolic'] = data['Diastolic'].str.replace('[^0-9]', '', regex=True)
    data['Diastolic'] = pd.to_numeric(data['Diastolic'], errors='coerce')

# Target Encoding
data['Stages'] = data['Stages'].map({
    'Normal': 0,
    'Stage-1': 1,
    'Stage-2': 2,
    'Crisis': 3
})

# Fill NaN values
data.fillna(0, inplace=True)
# -------------------------------
# CHECK FOR ANY REMAINING STRINGS
# -------------------------------
print("Remaining Data Types:")
print(data.dtypes)

# Fill any NaN created during mapping
data.fillna(0, inplace=True)

# -------------------------------
# SPLIT FEATURES & TARGET
# -------------------------------
X = data.drop("Stages", axis=1)
y = data["Stages"]

# Scaling
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# TRAIN MODEL
# -------------------------------
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model
pickle.dump(model, open("model.pkl", "wb"))

# Save scaler
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("Model and Scaler saved successfully!")