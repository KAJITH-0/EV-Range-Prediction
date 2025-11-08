 ===========================================
# Week 2 - Model Training & Evaluation
# ===========================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings("ignore")


data_path = r"E:\sills4future\cleaned_ev_data.csv"

if not os.path.exists(data_path):
    raise FileNotFoundError(f"❌ File not found at: {data_path}")

df = pd.read_csv(data_path)
print("✅ Dataset loaded successfully!")
print("Shape:", df.shape)

df = df.drop_duplicates()
for col in df.select_dtypes(include=np.number).columns:
    df[col] = df[col].fillna(df[col].median())

for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].fillna(df[col].mode()[0])


label_enc = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = label_enc.fit_transform(df[col].astype(str))

target = "range_km"
if target not in df.columns:
    raise ValueError("❌ 'range_km' column not found in dataset!")

y = df[target]
X = df.drop([target], axis=1)

print(f"\n🎯 Target Variable: {target}")
print(f"📊 Total Features: {X.shape[1]}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n✅ Data Split Complete — Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# --- Linear Regression ---
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

# --- Random Forest Regressor ---
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

def evaluate(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"\n📘 {name} Performance:")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.3f}")
    return mae, rmse, r2

mae_lr, rmse_lr, r2_lr = evaluate(y_test, y_pred_lr, "Linear Regression")
mae_rf, rmse_rf, r2_rf = evaluate(y_test, y_pred_rf, "Random Forest Regressor")

metrics = ['MAE', 'RMSE', 'R²']
lr_scores = [mae_lr, rmse_lr, r2_lr]
rf_scores = [mae_rf, rmse_rf, r2_rf]
x = np.arange(len(metrics))

plt.figure(figsize=(8,5))
plt.bar(x - 0.2, lr_scores, width=0.4, label='Linear Regression', color='skyblue')
plt.bar(x + 0.2, rf_scores, width=0.4, label='Random Forest', color='orange')
plt.xticks(x, metrics)
plt.title('Model Performance Comparison (Range Prediction)')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_rf, alpha=0.7, color='teal')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Range (km)")
plt.ylabel("Predicted Range (km)")
plt.title("Random Forest: Actual vs Predicted Range")
plt.grid(True)
plt.show()

feature_importance = pd.Series(rf.feature_importances_, index=X.columns)
top_features = feature_importance.nlargest(10)
plt.figure(figsize=(8,5))
top_features.plot(kind='barh', color='green')
plt.title('Top 10 Important Features Affecting Range')
plt.xlabel('Importance Score')
plt.show()

os.makedirs(r"E:\sills4future\models", exist_ok=True)
joblib.dump(lr, r"E:\sills4future\models\linear_regression_model.pkl")
joblib.dump(rf, r"E:\sills4future\models\random_forest_model.pkl")
joblib.dump(scaler, r"E:\sills4future\models\scaler.pkl")

print("\n💾 Models and scaler saved successfully!")
