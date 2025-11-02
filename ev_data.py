import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Load dataset ===
file_path = r"E:\sills4future\cleaned_ev_data.csv"
df = pd.read_csv(file_path)

# === Clean column names ===
df.columns = df.columns.str.strip().str.lower()  # remove spaces + lowercase

print("\n✅ Available columns in dataset:\n", df.columns.tolist())

# === Basic cleaning ===
df = df.drop_duplicates()
df = df.dropna(how='all')
df = df.fillna(df.mean(numeric_only=True))

# Fill missing categorical with mode
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# === Detect closest column names ===
brand_col = next((col for col in df.columns if 'brand' in col), None)
range_col = next((col for col in df.columns if 'range' in col), None)
price_col = next((col for col in df.columns if 'price' in col or 'cost' in col), None)

print(f"\nDetected columns → Brand: {brand_col}, Range: {range_col}, Price: {price_col}")

# === One-hot encode Brand column ===
if brand_col:
    df = pd.get_dummies(df, columns=[brand_col], prefix='Brand')
else:
    print("⚠️ 'Brand' column not found — skipping one-hot encoding.")

# === PLOTS ===

# 1️⃣ Top 10 Brands
brand_cols = [col for col in df.columns if col.startswith('Brand_')]
if brand_cols:
    brand_counts = df[brand_cols].sum().sort_values(ascending=False).head(10)
    plt.figure(figsize=(10,5))
    sns.barplot(x=brand_counts.index.str.replace('Brand_', ''), y=brand_counts.values)
    plt.title("Top 10 Electric Vehicle Brands")
    plt.ylabel("Number of Models")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ No brand columns found to plot.")

# 2️⃣ Range Distribution
if range_col:
    plt.figure(figsize=(8,5))
    sns.histplot(df[range_col], bins=20, kde=True)
    plt.title("Distribution of EV Range (km)")
    plt.xlabel("Range (km)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ No range column found to plot.")

# 3️⃣ Price vs Range
if price_col and range_col:
    plt.figure(figsize=(8,5))
    sns.scatterplot(x=df[range_col], y=df[price_col])
    plt.title("Price vs Range of Electric Vehicles")
    plt.xlabel("Range (km)")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ Price or range column missing, skipping scatter plot.")

print("\n✅ Done — plots displayed successfully!")
