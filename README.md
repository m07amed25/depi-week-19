# Clustering Report on Country Dataset

## Overview
This project applies clustering techniques to a dataset containing country-level statistics. The goal is to group countries based on socio-economic indicators, which may help in identifying similar nations for policy-making, investment, or research purposes.

---

## Dataset Description
**Filename**: `Country-data.csv`

**Features**:
- `country`: Name of the country (excluded from clustering)
- `child_mort`: Death of children under 5 years per 1000 live births
- `exports`: Exports of goods and services as % of GDP
- `health`: Total health spending as % of GDP
- `imports`: Imports of goods and services as % of GDP
- `income`: Net income per person
- `inflation`: Annual inflation rate
- `life_expec`: Life expectancy at birth
- `total_fer`: Total fertility rate
- `gdpp`: GDP per capita

---

## Data Preprocessing
1. **Missing Values**: Checked and found none.
2. **Feature Selection**: Dropped the `country` column for clustering purposes.
3. **Feature Scaling**: Applied `StandardScaler` from `sklearn` to standardize features.

---

## Clustering Techniques Applied

### 1. K-Means Clustering
- **Elbow Method** used to determine the optimal number of clusters.
- **Chosen Clusters**: 3
- **Implementation**:
```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(scaled_features)
```

### 2. PCA for Visualization
- **Reduced Features** to 2 principal components for plotting.
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(scaled_features)
```
- **Scatter Plot**:
```python
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=df['cluster'], cmap='viridis')
plt.title('K-Means Clusters (PCA Reduced)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()
```

---

## Results & Interpretation
- Countries were successfully grouped into 3 distinct clusters.
- These clusters can represent economic tiers or health/infrastructure similarities.
- The visualization shows good separation among the clusters.

---

## Next Steps
- Try other algorithms: DBSCAN, Hierarchical Clustering.
- Perform cluster profiling (mean stats for each cluster).
- Visualize country clusters on a world map.

---

## Requirements
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

You can install them with:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## Author
Generated using Python, assisted by ChatGPT.

---

## License
This project is open-source and free to use.

