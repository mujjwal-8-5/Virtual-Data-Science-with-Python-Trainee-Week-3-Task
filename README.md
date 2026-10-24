# Week 3: Unsupervised Learning and Clustering Analysis

## Project overview
This project applies K-Means clustering to the public Iris measurements dataset. It also includes a Ward hierarchical-clustering dendrogram for comparison. The species labels are **not used** to fit either clustering method; this is an unsupervised analysis.

## Dataset
- Dataset: Iris flower measurements (150 rows, four numeric features)
- Source: Iris dataset distributed with scikit-learn: https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html
- Local copy: `data/iris_measurements.csv`
- Features: sepal length, sepal width, petal length, petal width (centimetres)
- `sample_id` is an identifier and is not used as a model feature.

## Setup and run
Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
python src/clustering_analysis.py
```

The script writes:
- `data/k_selection_metrics.csv`
- `data/clustered_iris_results.csv`
- charts under `figures/`

## Method summary
1. Read the measurements.
2. Standardize features using `StandardScaler` so variables with different spread do not dominate Euclidean distance.
3. Fit K-Means for k=2 through k=6 using `random_state=42` and `n_init=20`.
4. Select k with the highest silhouette score among the tested values.
5. Evaluate with silhouette score (higher is better), Davies–Bouldin index (lower is better), and Calinski–Harabasz score (higher is generally better).
6. Visualize the assignments with PCA, plus a Ward-linkage hierarchical dendrogram.

## Reproducibility
A fixed random seed is used. Cluster IDs are arbitrary and may be permuted by another implementation/version; interpret clusters by their feature profiles, not by their numeric ID.
