import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

df = pd.read_csv("Cleaned_TimeSeries_Data_Enhanced.csv")
df.shape

df = df.sort_values(["Country Name", "Year"]).reset_index(drop=True)

df["Country_encoded"] = df["Country Name"].astype("category").cat.codes
df["Internet_Growth_Rate"] = df["Internet_Growth_Rate"].clip(-200, 200)

df.head()

df.isnull().sum()

y = df["GDP_per_Capita_USD"]

features = [
    "Internet_Usage_%",
    "Internet_Smoothed",
    "Internet_lag_1",
    "GDP_lag_1",
    "Internet_MA_3",
    "Internet_Growth_Rate",
    "Year",
    "Country_encoded"
]
X = df[features]

X = X.fillna(method="ffill").fillna(method="bfill")
y = y.fillna(method="ffill").fillna(method="bfill")

split_index = int(len(X) * 0.8)

X_train = X.iloc[:split_index]
X_test  = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test_poly  = y.iloc[split_index:]

model = Pipeline([
    ("scaler", StandardScaler()),
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("ridge", Ridge(alpha=1.0))
])

model.fit(X_train, y_train)

pred_poly = model.predict(X_test)
print(pred_poly[:5])

import matplotlib.pyplot as plt
residuals = y_test_poly - pred_poly
plt.figure(figsize=(6,6))
plt.scatter(y_test_poly, pred_poly, alpha=0.5)
plt.xlabel("Actual GDP")
plt.ylabel("Predicted GDP")
plt.title("Actual vs Predicted GDP")
plt.show()
plt.hist(residuals, bins=30)
plt.title("Residual Distribution")
plt.show()

feature_names = model.named_steps["poly"].get_feature_names_out(X.columns)
coefs = model.named_steps["ridge"].coef_

importance = sorted(zip(feature_names, coefs), key=lambda x: abs(x[1]), reverse=True)

for f, c in importance[:10]:
    print(f, c)
