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

# Predicting further into the future using lagged CO(GT) values

DATA_PATH = r"C:\Users\Mink\OneDrive\Documents\GitHub\Machine-Learning-2\Project\AirQualityUCI.csv"
OUT_DIR = r"C:\Users\Mink\OneDrive\Documents\GitHub\Machine-Learning-2\Project\results_CO_further"
TARGET = "CO(GT)"
LAGS = [1, 2, 3]
HORIZONS = [1, 3, 6]
RANDOM_STATE = 42

os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams["figure.figsize"] = (7, 5)
plt.rcParams["axes.grid"] = True

df_raw = pd.read_csv(DATA_PATH)
df_raw = df_raw[[TARGET]].copy()

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

all_results = []

for horizon in HORIZONS:
    print(f"\n=== Horizon +{horizon} ===")

    df = df_raw.copy()

    df["target_future"] = df[TARGET].shift(-horizon)

    for lag in LAGS:
        df[f"{TARGET}_lag{lag}"] = df[TARGET].shift(lag)

    df = df.dropna().reset_index(drop=True)

    X = df[[f"{TARGET}_lag{lag}" for lag in LAGS]]
    y = df["target_future"]

    split_idx = int(0.8 * len(df))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    horizon_results = []

    for model_name, model in models.items():
        print(f"Training {model_name} (horizon +{horizon})")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrics (version-safe)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        all_results.append([horizon, model_name, rmse, mae, r2])
        horizon_results.append([model_name, rmse])

    horizon_df = pd.DataFrame(
        horizon_results,
        columns=["Model", "RMSE"]
    )

    plt.figure()
    plt.bar(horizon_df["Model"], horizon_df["RMSE"])
    plt.title(f"RMSE Comparison – Horizon +{horizon}")
    plt.xlabel("Model")
    plt.ylabel("RMSE")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/RMSE_horizon_{horizon}.png")
    plt.close()

results_df = pd.DataFrame(
    all_results,
    columns=["Horizon", "Model", "RMSE", "MAE", "R2"]
)

results_df.to_csv(f"{OUT_DIR}/metrics_multi_horizon.csv", index=False)

plt.figure()

for model_name in results_df["Model"].unique():
    sub = results_df[results_df["Model"] == model_name]
    plt.plot(sub["Horizon"], sub["RMSE"], marker="o", label=model_name)

plt.xlabel("Forecast Horizon (hours)")
plt.ylabel("RMSE")
plt.title("RMSE vs Forecast Horizon")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/RMSE_vs_Horizon.png")
plt.close()

print("\nTraining finished.")
print(results_df)
print(f"\nAll results saved in '{OUT_DIR}/'")
