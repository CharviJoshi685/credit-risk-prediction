import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# === 1. Load Dataset ===
df = pd.read_csv("german_credit_data.csv")

print("📌 First 5 rows of data:")
print(df.head(), "\n")

print("📌 Dataset Info:")
print(df.info(), "\n")

print("📌 Summary Statistics:")
print(df.describe(), "\n")

# === 2. Check Missing Values ===
print("📌 Missing Values Count:")
print(df.isnull().sum(), "\n")

# Visualize missing values
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values Heatmap")
plt.show()

# === 3. Drop ID column ===
if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)

# === 4. Target Column (Risk) ===
if "Risk" in df.columns:
    # Convert 'good'/'bad' to numeric 1/0
    df["Risk"] = df["Risk"].map({"good": 1, "bad": 0})
else:
    # Create dummy target for testing only (remove if using real dataset)
    np.random.seed(42)
    df["Risk"] = np.random.randint(0, 2, size=len(df))

# === 5. Encode Categorical Columns ===
categorical_cols = ["Sex", "Housing", "Saving accounts", "Checking account", "Purpose"]
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype(str)  # Ensure string type

df = pd.get_dummies(df, columns=[col for col in categorical_cols if col in df.columns], drop_first=True)

# === 6. Handle Missing Values ===
df.dropna(inplace=True)

# === 7. Save Cleaned Data ===
df.to_csv("cleaned_credit_data.csv", index=False)

print("✅ Data cleaned and saved to data/cleaned_credit_data.csv")
print("📌 Cleaned Data Preview:")
print(df.head())

# === 8. Quick Visualization of Target ===
sns.countplot(x="Risk", data=df, palette="coolwarm")
plt.title("Distribution of Credit Risk")
plt.show()

# === 9. Correlation Heatmap ===
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
plt.title("Feature Correlation Heatmap")
plt.show()
