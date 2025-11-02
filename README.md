EV-Range-Prediction
This project aims to develop a machine learning model that predicts the driving range (in kilometers) of an electric vehicle (EV) based on its technical specifications.
The dataset contains information such as battery capacity, motor power, top speed, weight, and efficiency for various EV models.
Predicting range is crucial for manufacturers, buyers, and policymakers to understand performance, optimize design, and plan charging infrastructure.
To build a regression model capable of predicting the driving range of EVs using the given specifications and identify which factors most strongly affect the range.

Tools and Technologies:
Programming Language: Python
Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn
Environment: Jupyter Notebook
Version Control: Git & GitHub

Data Analysis:
Battery capacity and motor power show strong positive correlation with driving range.
Energy consumption (Wh/km) has an inverse relationship with range.
Vehicles with AWD drivetrain tend to have slightly lower range due to higher energy demand.
Distribution of range is slightly right-skewed — indicating a few high-end long-range EVs.

Workflow and Progress
Data Understanding & Cleaning.
Load and inspect dataset using pandas.
Handle missing and inconsistent values.
Encode categorical variables such as brand and drive type.
Detect and visualize outliers.
Perform Exploratory Data Analysis (EDA).
Save cleaned dataset for modeling.
Results:
Cleaned dataset: cleaned_ev_data.csv.
Visualizations: Distribution of range.
Jupyter notebook: Week1_Data_Preparation.ipynb
Correlation heatmap of features

Jupyter notebook: Week1_Data_Pre
