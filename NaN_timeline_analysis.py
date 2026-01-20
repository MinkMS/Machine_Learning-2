import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===============================
# 0. SETUP
# ===============================
DATA_PATH = r"C:\Users\Mink\Documents\GitHub\Dataset-Save-Place\Air Quality\AirQualityUCI_cleaned.csv"
RESULT_DIR = "results_nan_timeline"
os.makedirs(RESULT_DIR, exist_ok=True)

plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["axes.grid"] = True

# ===============================
# 1. LOAD DATA
# ===============================
df = pd.read_csv(DATA_PATH, sep=",", decimal=".")

# Replace dataset error marker
df = df.replace(-200, np.nan)

# ===============================
# 2. PARSE DATETIME
# ===============================
# Combine Date + Time into one datetime column
df["Datetime"] = pd.to_datetime(
    df["Date"] + " " + df["Time"],
    dayfirst=True,
    errors="coerce"
)

# Drop rows with invalid datetime
df = df.dropna(subset=["Datetime"])

# ===============================
# 3. SENSOR DATA ONLY
# ===============================
sensor_df = df.iloc[:, 2:-1]  # skip Date, Time, Datetime

for col in sensor_df.columns:
    sensor_df[col] = pd.to_numeric(sensor_df[col], errors="coerce")

# ===============================
# 4. NaN COUNT PER TIMESTAMP
# ===============================
df["NaN_count"] = sensor_df.isna().sum(axis=1)

# Aggregate by day (mean NaN per day)
daily_nan = df.groupby(df["Datetime"].dt.date)["NaN_count"].mean()

# ===============================
# 5. SAVE DATA
# ===============================
daily_nan_df = daily_nan.reset_index()
daily_nan_df.columns = ["Date", "Average_NaN_per_sample"]
daily_nan_df.to_csv(f"{RESULT_DIR}/nan_daily_timeline.csv", index=False)

# ===============================
# 6. VISUALIZATION
# ===============================
plt.plot(daily_nan.index, daily_nan.values)

# Annotate peaks
threshold = daily_nan.mean() + 2 * daily_nan.std()
for date, value in daily_nan.items():
    if value > threshold:
        plt.text(date, value, f"{value:.1f}", fontsize=8)

plt.xlabel("Date")
plt.ylabel("Average NaN per sample")
plt.title("NaN Timeline (Sensor Data)")
plt.tight_layout()
plt.savefig(f"{RESULT_DIR}/nan_timeline.png")
plt.close()

# ===============================
# 7. CONSOLE OUTPUT
# ===============================
print("==== NaN TIMELINE ANALYSIS ====")
print(f"Total days analyzed : {len(daily_nan)}")
print(f"Mean NaN per sample : {daily_nan.mean():.2f}")
print(f"Max NaN per sample  : {daily_nan.max():.2f}")
print(f"Results saved in '{RESULT_DIR}/'")
