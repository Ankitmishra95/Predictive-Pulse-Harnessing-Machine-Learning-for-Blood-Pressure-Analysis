import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# ==============================
# LOAD DATASET
# ==============================

data = pd.read_csv("patient_data.csv")

# Rename column
data.rename(columns={'C': 'Gender'}, inplace=True)

# Remove duplicates
data.drop_duplicates(inplace=True)

print("\nOriginal Stage Values:")
print(data["Stages"].unique())


# ==============================
# ENCODING
# ==============================

binary_cols = [
    'History','Patient','TakeMedication',
    'BreathShortness','VisualChanges',
    'NoseBleeding','ControlledDiet'
]

for col in binary_cols:
    if col in data.columns:
        data[col] = data[col].map({'No':0,'Yes':1})


# Gender
if 'Gender' in data.columns:
    data['Gender'] = data['Gender'].map({'Male':0,'Female':1})


# Age
if 'Age' in data.columns:
    data['Age'] = data['Age'].map({
        '18-34':1,
        '35-50':2,
        '51-64':3,
        '65+':4
    })


# Severity
if 'Severity' in data.columns:
    data['Severity'] = data['Severity'].map({
        'Mild':0,
        'Moderate':1,
        'Severe':2
    })


# Time diagnosed
if 'Whendiagnoused' in data.columns:
    data['Whendiagnoused'] = data['Whendiagnoused'].map({
        '<1 Year':0,
        '1-5 Years':1,
        '5+ Years':2
    })


# Systolic BP
if 'Systolic' in data.columns:
    data['Systolic'] = data['Systolic'].astype(str).str.replace('[^0-9]','',regex=True)
    data['Systolic'] = pd.to_numeric(data['Systolic'], errors='coerce')


# Diastolic BP
if 'Diastolic' in data.columns:
    data['Diastolic'] = data['Diastolic'].astype(str).str.replace('[^0-9]','',regex=True)
    data['Diastolic'] = pd.to_numeric(data['Diastolic'], errors='coerce')


# ==============================
# TARGET ENCODING
# ==============================

data['Stages'] = data['Stages'].astype(str).str.upper().str.strip()

def encode_stage(value):
    if "NORMAL" in value:
        return 0
    elif "STAGE-1" in value:
        return 1
    elif "STAGE-2" in value:
        return 2
    elif "CRISIS" in value:
        return 3
    else:
        return np.nan

data['Stages'] = data['Stages'].apply(encode_stage)

# Remove invalid rows
data = data.dropna(subset=["Stages"])

print("\nStage Distribution After Encoding:")
print(data["Stages"].value_counts())


# ==============================
# HANDLE MISSING VALUES
# ==============================

data.fillna(0, inplace=True)


# ==============================
# SPLIT FEATURES & TARGET
# ==============================

X = data.drop("Stages", axis=1)
y = data["Stages"]


# ==============================
# FEATURE SCALING
# ==============================

scaler = MinMaxScaler()
X = scaler.fit_transform(X)


# ==============================
# TRAIN TEST SPLIT
# ==============================

if len(y.unique()) < 2:
    raise ValueError("Dataset has only one class after encoding. Check dataset labels.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ==============================
# MODEL COMPARISON
# ==============================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC()
}

results = {}

print("\n==============================")
print("MODEL COMPARISON RESULTS")
print("==============================")

for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    results[name] = acc

    print("\nModel:", name)
    print("Accuracy:", acc)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("Classification Report:")
    print(classification_report(y_test, y_pred))


# ==============================
# BEST MODEL SELECTION
# ==============================

best_model_name = max(results, key=results.get)

print("\nBest Model:", best_model_name)

best_model = models[best_model_name]

# retrain best model on full training set
best_model.fit(X_train, y_train)


# ==============================
# OVERFITTING CHECK
# ==============================

train_acc = best_model.score(X_train, y_train)
test_acc = best_model.score(X_test, y_test)

print("\nTraining Accuracy:", train_acc)
print("Testing Accuracy:", test_acc)

if train_acc - test_acc > 0.05:
    print("Warning: Possible Overfitting")
else:
    print("Model is well generalized")


# ==============================
# SAVE MODEL & SCALER
# ==============================

pickle.dump(best_model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("\nModel and scaler saved successfully!")