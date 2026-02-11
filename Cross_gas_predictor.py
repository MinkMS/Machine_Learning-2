import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Predicting CO(GT) from other gas sensor readings

DATA_PATH = r"C:\Users\Mink\OneDrive\Documents\GitHub\Machine-Learning-2\Project\AirQualityUCI.csv"
OUT_DIR = r"C:\Users\Mink\OneDrive\Documents\GitHub\Machine-Learning-2\Project\results_cross_gas"

TARGET = "CO(GT)"
FEATURES = ["C6H6(GT)", "NO2(GT)", "NOx(GT)"]

os.makedirs(OUT_DIR, exist_ok=True)
plt.rcParams["figure.figsize"] = (6, 6)
plt.rcParams["axes.grid"] = True

df = pd.read_csv(DATA_PATH)

cols = [TARGET] + FEATURES
df = df[cols].replace(-200, np.nan).dropna()

models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(
        n_estimators=200, random_state=42
    ),
    "SVR": SVR(kernel="rbf", C=10, epsilon=0.1)
}

for feat in FEATURES:
    print(f"\n=== Predicting {TARGET} from {feat} ===")

    subdir = os.path.join(OUT_DIR, feat.replace("(", "").replace(")", ""))
    os.makedirs(subdir, exist_ok=True)

    X = df[[feat]]
    y = df[TARGET]

    split = int(0.8 * len(df))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    records = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        records.append([name, rmse, mae, r2])

        plt.figure()
        plt.scatter(y_test, y_pred, alpha=0.6)
        lims = [
            min(y_test.min(), y_pred.min()),
            max(y_test.max(), y_pred.max())
        ]
        plt.plot(lims, lims, "k--", linewidth=1)

        plt.xlabel("True CO(GT)")
        plt.ylabel("Predicted CO(GT)")
        plt.title(
            f"{name}\nRMSE={rmse:.2f} | MAE={mae:.2f} | R²={r2:.2f}"
        )

        plt.tight_layout()
        plt.savefig(os.path.join(subdir, f"{name}_scatter.png"))
        plt.close()

    metrics_df = pd.DataFrame(
        records, columns=["Model", "RMSE", "MAE", "R2"]
    )
    metrics_df.to_csv(os.path.join(subdir, "metrics.csv"), index=False)

print("\nTRAINING FINISHED")
