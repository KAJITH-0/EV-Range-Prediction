# ===========================================
# Week 1 - Data Understanding & Cleaning
# ===========================================

# 1️⃣ Import Required Libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ===========================================
# 2️⃣ Load the Dataset (from your actual path)
# ===========================================
data_path = r"E:\sills4future\electric_vehicles_spec_2025.csv(1).csv"

if not os.path.exists(data_path):
    raise FileNotFoundError(f"❌ File not found at: {data_path}")

df = pd.read_csv(data_path)
print("✅ Dataset loaded successfully!")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# ===========================================
# 3️⃣ Handle Empty Spaces as Missing Values
# ===========================================
df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
print("\n🚨 Missing values after converting blanks:")
print(df.isnull().sum())

# ===========================================
# 4️⃣ Basic Info and Summary
# ===========================================
print("\n🔍 Dataset Info:")
df.info()

print("\n📊 Summary Statistics:")
print(df.describe(include='all'))

# ===========================================
# 5️⃣ Handle Missing Values
# ===========================================
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
for col in numeric_cols:
    df[col].fillna(df[col].median(), inplace=True)

categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

print("\n✅ Missing values handled successfully!")

# ===========================================
# 6️⃣ Clean Numeric Columns (Remove Units)
# ===========================================
def clean_numeric(col):
    return (
        col.astype(str)
        .str.replace(" km/h", "", regex=False)
        .str.replace(" km", "", regex=False)
        .str.replace(" sec", "", regex=False)
        .str.replace(" Wh/km", "", regex=False)
        .str.replace(" kW", "", regex=False)
        .str.replace("*", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.extract("([-+]?\d*\.?\d+)")[0]
        .astype(float)
    )

cols_to_clean = ['Range', 'Range (km)', 'Battery Capacity (kWh)', 'Top Speed', 'Efficiency', '0 - 100']
for col in cols_to_clean:
    if col in df.columns:
        df[col] = clean_numeric(df[col])

print("\n🧹 Numeric columns cleaned (units removed).")

# ===========================================
# 7️⃣ One-Hot Encode Brand Column
# ===========================================
if 'Brand' in df.columns:
    df = pd.get_dummies(df, columns=['Brand'], prefix='Brand', drop_first=True)
    print("✅ 'Brand' column one-hot encoded successfully!")
else:
    print("⚠️ 'Brand' column not found in dataset, skipping encoding.")

# ===========================================
# 8️⃣ Detect and Visualize Outliers
# ===========================================
plt.figure(figsize=(8, 5))
range_col = 'Range (km)' if 'Range (km)' in df.columns else 'Range'
if range_col in df.columns:
    sns.boxplot(x=df[range_col])
    plt.title('📈 Range Distribution with Outliers')
    os.makedirs(r"E:\sills4future", exist_ok=True)
    plt.savefig(r"E:\sills4future\range_distribution.png")
    plt.show()
else:
    print("⚠️ Range column not found, skipping outlier plot.")

# ===========================================
# 🔟 Save Cleaned Dataset
# ===========================================
cleaned_path = r"E:\sills4future\cleaned_ev_data.csv"
os.makedirs(os.path.dirname(cleaned_path), exist_ok=True)
df.to_csv(cleaned_path, index=False)
print(f"\n💾 Cleaned dataset saved as: {cleaned_path}")

print("\n✅ Week 1 - Data Understanding & Cleaning completed successfully!")
