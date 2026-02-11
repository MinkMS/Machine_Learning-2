import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ===============================
# 0. SETUP
# ===============================
DATA_PATH = r"C:\Users\Mink\OneDrive\Documents\GitHub\Machine-Learning-2\Project\AirQualityUCI.csv"
OUT_DIR = r"C:\Users\Mink\OneDrive\Documents\GitHub\Machine-Learning-2\Project\results_CO_timeline"
TARGET = "CO(GT)"
LAGS = [1, 2, 3]
RANDOM_STATE = 42

os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams["figure.figsize"] = (10, 4)
plt.rcParams["axes.grid"] = True

df = pd.read_csv(DATA_PATH)

df = df[[TARGET]].copy()

for lag in LAGS:
    df[f"{TARGET}_lag{lag}"] = df[TARGET].shift(lag)

df = df.dropna().reset_index(drop=True)

X = df[[f"{TARGET}_lag{lag}" for lag in LAGS]]
y = df[TARGET]

split_idx = int(0.8 * len(df))
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),
    "SVR": Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="rbf", C=10, epsilon=0.1))
    ])
}

results = []

for model_name, model in models.items():
    print(f"Training {model_name}...")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics (version-safe)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results.append([model_name, rmse, mae, r2])

    plt.figure()
    plt.plot(y_test.values, label="True", linewidth=2)
    plt.plot(y_pred, label="Predicted", linestyle="--")

    plt.title(
        f"{model_name} | "
        f"RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.2f}"
    )
    plt.xlabel("Time index")
    plt.ylabel("CO(GT)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{model_name}_timeseries.png")
    plt.close()

results_df = pd.DataFrame(
    results,
    columns=["Model", "RMSE", "MAE", "R2"]
)

results_df.to_csv(f"{OUT_DIR}/metrics_summary.csv", index=False)

print("\nTraining finished.")
print(results_df)
print(f"\nAll results saved in '{OUT_DIR}/'")
