import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("diabetes_dataset.csv")
print("Shape:", df.shape)
print(df.head())
print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicates:", df.duplicated().sum())
print(df.describe().T)

df = df.drop_duplicates().reset_index(drop=True)
df = df.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)

target = "target"
X = df.drop(columns=[target])
y = df[target]

# EDA
plt.figure(figsize=(8, 5))
plt.hist(y, bins=25)
plt.title("Target Distribution")
plt.xlabel("Target")
plt.ylabel("Frequency")
plt.show()

corr = df.corr(numeric_only=True)[target].sort_values(ascending=False)
print("\nCorrelation with target:\n", corr)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

models = {
    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),
    "Random Forest": RandomForestRegressor(
        n_estimators=300, random_state=42, n_jobs=-1
    )
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    results.append([name, mae, rmse, r2])

results_df = pd.DataFrame(results, columns=["Model", "MAE", "RMSE", "R2"])
print("\nModel Performance:")
print(results_df.sort_values("R2", ascending=False).to_string(index=False))

best_name = results_df.sort_values("R2", ascending=False).iloc[0]["Model"]
print("\nBest model:", best_name)
