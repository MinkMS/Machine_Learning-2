import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# SETUP
DATA_PATH = r"C:\Users\Mink\OneDrive\Documents\GitHub\Machine-Learning-2\Project\AirQualityUCI.csv"
RESULT_DIR = r"C:\Users\Mink\OneDrive\Documents\GitHub\Machine-Learning-2\Project\results_regression_step1"
os.makedirs(RESULT_DIR, exist_ok=True)

plt.rcParams["figure.figsize"] = (8, 6)
plt.rcParams["axes.grid"] = True

# LOAD & BASIC CLEAN
df = pd.read_csv(DATA_PATH)
df = df.replace(-200, np.nan)

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# REGRESSION FUNCTION
def run_regression(gt_col, sensor_col, folder_name):
    print(f"Running regression: {gt_col} <- {sensor_col}")

    out_dir = os.path.join(RESULT_DIR, folder_name)
    os.makedirs(out_dir, exist_ok=True)

    sub = df[[gt_col, sensor_col]].dropna()

    X = sub[[sensor_col]].values
    y = sub[gt_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # LINEAR REGRESSION
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    r2_lr = r2_score(y_test, y_pred_lr)
    bias_lr = np.mean(y_pred_lr - y_test)

    # RANDOM FOREST
    rf = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    r2_rf = r2_score(y_test, y_pred_rf)
    bias_rf = np.mean(y_pred_rf - y_test)

    # SVR (RBF)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    svr = SVR(kernel="rbf", C=100, epsilon=0.1)
    svr.fit(X_train_s, y_train)
    y_pred_svr = svr.predict(X_test_s)

    rmse_svr = np.sqrt(mean_squared_error(y_test, y_pred_svr))
    r2_svr = r2_score(y_test, y_pred_svr)
    bias_svr = np.mean(y_pred_svr - y_test)

    # SCATTER: SENSOR vs GT
    corr = np.corrcoef(sub[sensor_col], sub[gt_col])[0, 1]

    plt.scatter(sub[sensor_col], sub[gt_col], alpha=0.4)
    plt.xlabel(sensor_col)
    plt.ylabel(gt_col)
    plt.title(
        f"{gt_col} vs {sensor_col}\n"
        f"Correlation = {corr:.3f}"
    )
    plt.savefig(os.path.join(out_dir, "scatter_gt_vs_sensor.png"))
    plt.close()

    # PREDICTION VS GT
    def plot_pred(y_true, y_pred, model_name, rmse, r2):
        plt.scatter(y_true, y_pred, alpha=0.4)
        min_v = min(y_true.min(), y_pred.min())
        max_v = max(y_true.max(), y_pred.max())
        plt.plot([min_v, max_v], [min_v, max_v], linestyle="--")

        plt.xlabel("Ground Truth")
        plt.ylabel("Prediction")
        plt.title(
            f"{model_name}\n"
            f"RMSE={rmse:.3f}, R²={r2:.3f}"
        )
        plt.savefig(os.path.join(out_dir, f"prediction_vs_gt_{model_name.lower()}.png"))
        plt.close()

    plot_pred(y_test, y_pred_lr, "Linear", rmse_lr, r2_lr)
    plot_pred(y_test, y_pred_rf, "RF", rmse_rf, r2_rf)
    plot_pred(y_test, y_pred_svr, "SVR", rmse_svr, r2_svr)

    # ERROR DISTRIBUTION (SVR)
    plt.hist(y_pred_svr - y_test, bins=40)
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.title("SVR Error Distribution")
    plt.savefig(os.path.join(out_dir, "error_distribution_svr.png"))
    plt.close()

    # SAVE METRICS
    with open(os.path.join(out_dir, "metrics.txt"), "w") as f:
        f.write(f"Samples used: {len(sub)}\n\n")

        f.write("Linear Regression:\n")
        f.write(f"  RMSE = {rmse_lr:.4f}\n")
        f.write(f"  R2   = {r2_lr:.4f}\n")
        f.write(f"  Bias = {bias_lr:.4f}\n\n")

        f.write("Random Forest:\n")
        f.write(f"  RMSE = {rmse_rf:.4f}\n")
        f.write(f"  R2   = {r2_rf:.4f}\n")
        f.write(f"  Bias = {bias_rf:.4f}\n\n")

        f.write("SVR (RBF):\n")
        f.write(f"  RMSE = {rmse_svr:.4f}\n")
        f.write(f"  R2   = {r2_svr:.4f}\n")
        f.write(f"  Bias = {bias_svr:.4f}\n")

# RUN STEP 1
run_regression("CO(GT)", "PT08.S1(CO)", "CO_GT_vs_PT08_S1")
run_regression("NO2(GT)", "PT08.S4(NO2)", "NO2_GT_vs_PT08_S4")

print("Step 1 regression (Linear + RF + SVR) finished.")
