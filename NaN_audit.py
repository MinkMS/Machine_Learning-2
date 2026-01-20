import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===============================
# 0. SETUP
# ===============================
DATA_PATH = r"C:\Users\Mink\Documents\GitHub\Dataset-Save-Place\Air Quality\AirQualityUCI_cleaned.csv"
RESULT_DIR = "results_nan_audit"
os.makedirs(RESULT_DIR, exist_ok=True)

plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.grid"] = True

# ===============================
# 1. LOAD DATA
# ===============================
df = pd.read_csv(DATA_PATH, sep=",", decimal=".")

# Replace dataset error marker
df = df.replace(-200, np.nan)

# ===============================
# 2. DROP DATE & TIME
# ===============================
df_sensor = df.iloc[:, 2:]  # skip Date, Time

# Force numeric conversion on sensor data only
for col in df_sensor.columns:
    df_sensor[col] = pd.to_numeric(df_sensor[col], errors="coerce")

# ===============================
# 3. NaN AUDIT
# ===============================
total_rows = len(df_sensor)
total_values = df_sensor.size

nan_per_col = df_sensor.isna().sum()
nan_total = nan_per_col.sum()

rows_with_nan = (df_sensor.isna().sum(axis=1) > 0).sum()

nan_percentage_total = (nan_total / total_values) * 100
nan_percentage_col = (nan_per_col / total_rows) * 100

# ===============================
# 4. SAVE TABLES
# ===============================
summary = pd.DataFrame({
    "Metric": [
        "Total samples",
        "Samples with at least one NaN",
        "Total NaN values",
        "NaN percentage (overall)"
    ],
    "Value": [
        total_rows,
        rows_with_nan,
        nan_total,
        round(nan_percentage_total, 2)
    ]
})

summary.to_csv(f"{RESULT_DIR}/nan_summary.csv", index=False)

nan_col_df = pd.DataFrame({
    "Column": nan_per_col.index,
    "NaN_Count": nan_per_col.values,
    "NaN_Percentage (%)": nan_percentage_col.round(2).values
})

nan_col_df.to_csv(f"{RESULT_DIR}/nan_per_column.csv", index=False)

# ===============================
# 5. VISUALIZATION
# ===============================
plt.bar(nan_per_col.index, nan_per_col.values)
plt.xticks(rotation=90)

for i, (cnt, pct) in enumerate(zip(nan_per_col.values, nan_percentage_col.values)):
    plt.text(i, cnt, f"{cnt}\n({pct:.1f}%)",
             ha="center", va="bottom", fontsize=8)

plt.title("NaN count and percentage per sensor column")
plt.ylabel("NaN count")
plt.tight_layout()
plt.savefig(f"{RESULT_DIR}/nan_per_column.png")
plt.close()

# ===============================
# 6. CONSOLE OUTPUT
# ===============================
print("==== NaN AUDIT REPORT (DATE & TIME SKIPPED) ====")
print(f"Total samples                     : {total_rows}")
print(f"Samples with at least one NaN     : {rows_with_nan}")
print(f"Total NaN values                  : {nan_total}")
print(f"Overall NaN percentage            : {nan_percentage_total:.2f}%")
print(f"Results saved in '{RESULT_DIR}/'")
