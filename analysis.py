import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("patient_data.csv")

print(df.head())
print(df.columns)

sns.countplot(x="C", data=df)
plt.title("Gender Distribution")
plt.savefig("gender_distribution.png")
plt.show()

sns.countplot(x="Stages", data=df)
plt.title("Hypertension Stage Distribution")
plt.xticks(rotation=45)
plt.savefig("Hypertension Stage Distribution.png")
plt.show()

sns.countplot(x="Age", hue="Stages", data=df)
plt.title("Age Group vs Hypertension Stage")
plt.xticks(rotation=45)
plt.savefig("Age Group vs Hypertension Stage ..png")
plt.show()

sns.countplot(x="TakeMedication", hue="Stages", data=df)
plt.title("Medication vs Hypertension Stage")
plt.savefig("medication_vs_stage.png")
plt.close()

sns.countplot(x="Severity", hue="Stages", data=df)
plt.title("Severity vs Hypertension Stage")
plt.savefig("severity_vs_stage.png")
plt.close()

sns.countplot(x="ControlledDiet", hue="Stages", data=df)
plt.title("Controlled Diet vs Hypertension Stage")
plt.savefig("diet_vs_stage.png")
plt.close()

# Remove spaces
df["Systolic"] = df["Systolic"].str.replace(" ", "")
df["Diastolic"] = df["Diastolic"].str.replace(" ", "")

# Function to convert range or single+ value to midpoint
def convert_bp(value):
    if "-" in value:
        parts = value.split("-")
        return (int(parts[0]) + int(parts[1])) / 2
    elif "+" in value:
        return int(value.replace("+", ""))
    else:
        return int(value)

# Apply function
df["Systolic_mid"] = df["Systolic"].apply(convert_bp)
df["Diastolic_mid"] = df["Diastolic"].apply(convert_bp)

# Scatter plot
sns.scatterplot(x="Systolic_mid", y="Diastolic_mid", hue="Stages", data=df)
plt.title("Systolic vs Diastolic Across Stages")
plt.savefig("systolic_vs_diastolic.png")
plt.close()