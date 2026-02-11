import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = r"C:\Users\Mink\OneDrive\Documents\GitHub\Machine-Learning-2\Project\AirQualityUCI.csv"
OUT_DIR = r"C:\Users\Mink\OneDrive\Documents\GitHub\Machine-Learning-2\Project\results_correlation"
os.makedirs(OUT_DIR, exist_ok=True)

GT_COLS = [
    "CO(GT)",
    "NMHC(GT)",
    "C6H6(GT)",
    "NOx(GT)",
    "NO2(GT)"
]

df = pd.read_csv(DATA_PATH)

df_gt = df[GT_COLS].copy()
df_gt = df_gt.replace(-200, np.nan)
df_gt = df_gt.dropna()

corr = df_gt.corr(method="pearson")
corr.to_csv(f"{OUT_DIR}/correlation_matrix_GT.csv")

plt.figure(figsize=(8, 6))
im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)

plt.colorbar(im, fraction=0.046, pad=0.04)
plt.xticks(range(len(GT_COLS)), GT_COLS, rotation=45, ha="right")
plt.yticks(range(len(GT_COLS)), GT_COLS)

for i in range(len(GT_COLS)):
    for j in range(len(GT_COLS)):
        plt.text(
            j, i,
            f"{corr.iloc[i, j]:.2f}",
            ha="center", va="center",
            color="black", fontsize=9
        )

plt.title("Pearson Correlation Heatmap (GT gases)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/GT_correlation_heatmap.png", dpi=300)
plt.close()

print("Correlation heatmap DONE.")
print(f"Results saved in: {OUT_DIR}")
