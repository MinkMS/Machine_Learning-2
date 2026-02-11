import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

DATA_PATH = r"C:\Users\Mink\OneDrive\Documents\GitHub\Machine-Learning-2\Project\AirQualityUCI.csv"
OUT_DIR = r"C:\Users\Mink\OneDrive\Documents\GitHub\Machine-Learning-2\Project\results_CO_predictor"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams["figure.figsize"] = (7, 7)
plt.rcParams["axes.grid"] = True

df = pd.read_csv(DATA_PATH)

df = df.select_dtypes(include=[np.number])

TARGET = "CO(GT)"

FEATURES = [
    "NO2(GT)",
    "C6H6(GT)",
    "NOx(GT)",
    "PT08.S1(CO)",
    "PT08.S2(NMHC)",
    "PT08.S3(NOx)",
    "PT08.S4(NO2)",
    "PT08.S5(O3)"
]

FEATURES = [c for c in FEATURES if c in df.columns]

df = df[FEATURES + [TARGET]].dropna()

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
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

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results.append([model_name, rmse, mae, r2])

    plt.figure()
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        linestyle="--"
    )

    plt.xlabel("True CO(GT)")
    plt.ylabel("Predicted CO(GT)")
    plt.title(
        f"{model_name}\n"
        f"RMSE={rmse:.2f} | MAE={mae:.2f} | R2={r2:.2f}"
    )
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{model_name}_scatter.png")
    plt.close()

results_df = pd.DataFrame(
    results,
    columns=["Model", "RMSE", "MAE", "R2"]
)

results_df.to_csv(f"{OUT_DIR}/metrics_summary.csv", index=False)

print("\nTraining finished.")
print(results_df)
print(f"\nAll results saved in '{OUT_DIR}/'")
