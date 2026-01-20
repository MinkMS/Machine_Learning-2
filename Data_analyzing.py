import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ===============================
# 0. SETUP
# ===============================
DATA_PATH = r"C:\Users\Mink\Documents\GitHub\Dataset-Save-Place\Air Quality\AirQualityUCI_cleaned.csv"
RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.grid"] = True

steps = [
    "Load & clean data",
    "Basic statistics",
    "Correlation analysis",
    "Feature scaling",
    "PCA",
    "Clustering"
]

pbar = tqdm(total=len(steps), desc="Pre-analysis pipeline")

# ===============================
# 1. LOAD & CLEAN DATA
# ===============================
df = pd.read_csv(DATA_PATH, sep=",", decimal=".")

# Drop empty columns
df = df.dropna(axis=1, how="all")

# Replace dataset missing value marker
df = df.replace(-200, np.nan)

# Force numeric conversion
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Keep numeric only
num_df = df.select_dtypes(include=[np.number])

# Drop columns that are entirely NaN
num_df = num_df.dropna(axis=1, how="all")

# Fill remaining NaN with median
num_df = num_df.fillna(num_df.median())

# Drop zero-variance columns
zero_var_cols = num_df.columns[num_df.nunique() <= 1]
num_df = num_df.drop(columns=zero_var_cols)

print(f"[INFO] Removed {len(zero_var_cols)} zero-variance columns")

# Save cleaned data
num_df.to_csv(f"{RESULT_DIR}/cleaned_data.csv", index=False)

pbar.update(1)

# ===============================
# 2. BASIC STATISTICS
# ===============================
stats = num_df.describe().T
stats.to_csv(f"{RESULT_DIR}/statistics.csv")

pbar.update(1)

# ===============================
# 3. CORRELATION (NUMERIC TABLE)
# ===============================
corr = num_df.corr()
corr.to_csv(f"{RESULT_DIR}/correlation_matrix.csv")

fig, ax = plt.subplots()
ax.axis("off")

table = ax.table(
    cellText=np.round(corr.values, 2),
    colLabels=corr.columns,
    rowLabels=corr.columns,
    loc="center"
)

table.scale(1, 1.4)
table.auto_set_font_size(False)
table.set_fontsize(8)

plt.title("Correlation Matrix (Numeric Values)")
plt.savefig(f"{RESULT_DIR}/correlation_matrix_numeric.png", bbox_inches="tight")
plt.close()

pbar.update(1)

# ===============================
# 4. FEATURE SCALING
# ===============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(num_df)

# Safety check
assert not np.isnan(X_scaled).any(), "NaN still exists after scaling!"

pbar.update(1)

# ===============================
# 5. PCA
# ===============================
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

explained = np.cumsum(pca.explained_variance_ratio_)

plt.plot(explained, marker="o")
for i, v in enumerate(explained):
    plt.text(i, v, f"{v:.2f}")

plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance")
plt.savefig(f"{RESULT_DIR}/pca_explained_variance.png")
plt.close()

# Save PCA info
pca_info = pd.DataFrame({
    "Component": np.arange(1, len(explained) + 1),
    "Cumulative Explained Variance": explained
})
pca_info.to_csv(f"{RESULT_DIR}/pca_explained_variance.csv", index=False)

pbar.update(1)

# ===============================
# 6. CLUSTERING (KMEANS)
# ===============================
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_pca)

num_df["Cluster"] = clusters

# PCA scatter
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("KMeans Clustering (PCA Space)")
plt.savefig(f"{RESULT_DIR}/clustering_pca.png")
plt.close()

# Cluster statistics
cluster_stats = num_df.groupby("Cluster").mean()
cluster_stats.to_csv(f"{RESULT_DIR}/cluster_statistics.csv")

pbar.update(1)
pbar.close()

print("\n[SUCCESS] Pre-analysis pipeline finished.")
print(f"All results saved in '{RESULT_DIR}/'")
