import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import pickle

# Step 1: Load Dataset
df = pd.read_csv("patient_data.csv")

print("First 5 rows:")
print(df.head())

# Step 2: Convert Categorical Columns to Numbers
label_encoders = {}

for column in df.columns:
    if df[column].dtype == "object":
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

# Step 3: Separate Features and Target
X = df.drop("Stages", axis=1)
y = df["Stages"]

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Train Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 6: Save Model
pickle.dump((model, label_encoders), open("hypertension_model.pkl", "wb"))

print("Model trained and saved successfully!")