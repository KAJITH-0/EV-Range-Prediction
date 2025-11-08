
This project aims to develop a machine learning model that predicts the driving range (in kilometers) of an electric vehicle (EV) based on its technical specifications.
The dataset contains information such as battery capacity, motor power, top speed, weight, and efficiency for various EV models.
Predicting range is crucial for manufacturers, buyers, and policymakers to understand performance, optimize design, and plan charging infrastructure.
To build a regression model capable of predicting the driving range of EVs using the given specifications and identify which factors most strongly affect the range.

---

## Tools and Technologies:

- **Programming Language:** Python
- **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn[pip install pandas numpy matplotlib seaborn scikit-learn]
- **Environment:** Jupyter Notebook

---
## Data Analysis:
- Battery capacity and motor power show strong positive correlation with driving range.
- Energy consumption (Wh/km) has an inverse relationship with range.
- Vehicles with AWD drivetrain tend to have slightly lower range due to higher energy demand.
- Distribution of range is slightly right-skewed — indicating a few high-end long-range EVs.

---
## Models Implemented:
- **Linear Regression** – establish a baseline performance model.
- **Random Forest Regressor** – capture non-linear relationships and feature interactions.

## Project Flow

```

EV-Range-Prediction/
│
├── data/
│   ├── ev_specifications_2025.csv
│   └── cleaned_ev_data.csv
├── notebooks/
│   └── Week1_Data_Preparation.ipynb
├── images/
│   ├── range_distribution.png
│   ├── model_comparison.png
│   ├── rf_predictions.png
│   └── feature_importance.png
├── src/
│   └── __init__.py
├── requirements.txt
└── README.md

```

---
## Workflow and Progress
- Data Understanding & Cleaning.
- Load and inspect dataset using pandas.
- Handle missing and inconsistent values.
- Encode categorical variables such as brand and drive type.
- Detect and visualize outliers.
- Perform Exploratory Data Analysis (EDA).
- Save cleaned dataset for modeling.
- Loaded the cleaned dataset from Week 1 (cleaned_ev_data.csv).
- Split data into training (80%) and testing (20%) subsets.
- Applied feature scaling using StandardScaler.
- Trained both Linear Regression and Random Forest models.
- Evaluated using performance metrics:
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
  - R² (Coefficient of Determination)
- Visualized model comparison and feature importance.

---
## Results:
- Cleaned dataset: cleaned_ev_data.csv.
- Visualizations: Distribution of range.

- **Week 2 summary:**
 - Linear Regression → MAE: 17.57 | RMSE: 21.59 | R²: 0.956
 - Random Forest     → MAE: 13.22 | RMSE: 18.59 | R²: 0.967
---
## Jupyter notebook:
- Week1_Data_Preparation.ipynb




