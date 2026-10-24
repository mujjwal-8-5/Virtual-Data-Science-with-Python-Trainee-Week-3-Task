"""
Week 3: Unsupervised Learning and Clustering Analysis
Dataset: Iris measurements (public dataset distributed with scikit-learn).
Run from project root: python src/clustering_analysis.py

The target/species labels are deliberately not used in clustering. The script
exports clustering metrics, assigned cluster IDs, and figures.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
df = pd.read_csv(DATA / "iris_measurements.csv")
features = [c for c in df.columns if c != "sample_id"]
X = df[features]
Z = StandardScaler().fit_transform(X)

rows, fitted = [], {}
for k in range(2, 7):
    model = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = model.fit_predict(Z)
    fitted[k] = (model, labels)
    rows.append({
        "k": k,
        "silhouette": silhouette_score(Z, labels),
        "davies_bouldin": davies_bouldin_score(Z, labels),
        "calinski_harabasz": calinski_harabasz_score(Z, labels)
    })
metrics = pd.DataFrame(rows)
metrics.to_csv(DATA / "k_selection_metrics.csv", index=False)
best_k = int(metrics.loc[metrics["silhouette"].idxmax(), "k"])
model, labels = fitted[best_k]
result = df.copy()
result["cluster"] = labels
result.to_csv(DATA / "clustered_iris_results.csv", index=False)

plt.figure(figsize=(7, 4.5))
plt.plot(metrics["k"], metrics["silhouette"], marker="o")
plt.xticks(range(2, 7))
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette score")
plt.title("Choosing k with the Silhouette Score")
plt.grid(True, alpha=.3)
plt.tight_layout()
plt.savefig(FIG / "silhouette_scores.png", dpi=180)
plt.close()

pca = PCA(n_components=2, random_state=42)
pcs = pca.fit_transform(Z)
plt.figure(figsize=(7, 5))
for cluster_id in sorted(set(labels)):
    mask = labels == cluster_id
    plt.scatter(pcs[mask, 0], pcs[mask, 1],
                label=f"Cluster {cluster_id} (n={mask.sum()})", alpha=.8)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
plt.title(f"K-Means Clusters in PCA Projection (k={best_k})")
plt.legend()
plt.tight_layout()
plt.savefig(FIG / "pca_cluster_plot.png", dpi=180)
plt.close()

profiles = result.groupby("cluster")[features].mean()
ax = profiles.T.plot(kind="bar", figsize=(9, 5))
ax.set_title("Average Flower Measurements by Cluster")
ax.set_ylabel("Mean measurement (cm)")
ax.set_xlabel("Feature")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(FIG / "cluster_feature_means.png", dpi=180)
plt.close()

Z_link = linkage(Z, method="ward")
plt.figure(figsize=(10, 5))
dendrogram(Z_link, truncate_mode="lastp", p=25,
           show_leaf_counts=True, leaf_rotation=45)
plt.title("Ward Hierarchical Clustering Dendrogram (truncated)")
plt.xlabel("Merged sample groups")
plt.ylabel("Ward distance")
plt.tight_layout()
plt.savefig(FIG / "hierarchical_dendrogram.png", dpi=180)
plt.close()

print(f"Selected k={best_k}; silhouette={silhouette_score(Z, labels):.3f}")
print("Cluster counts:")
print(result["cluster"].value_counts().sort_index())
print("Cluster means in original measurement units:")
print(profiles.round(2))
