import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Load dataset
df = pd.read_csv("data/processed/cleaned_supermarket_sales.csv")

# Features for clustering
features = df[["Unit price", "Quantity", "Sales", "Rating"]]

# Scale features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)

# -----------------------------
# Elbow Method
# -----------------------------
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1,11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.grid(True)
plt.show()

# -----------------------------
# KMeans Clustering
# -----------------------------
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(scaled_data)

# Silhouette Score
score = silhouette_score(scaled_data, df["Cluster"])
print("Silhouette Score:", round(score,3))

# Cluster Summary
print("\nCluster Summary")
print(df.groupby("Cluster")[["Sales","Quantity","Rating"]].mean())

# -----------------------------
# PCA Visualization
# -----------------------------
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

plt.figure(figsize=(8,6))
plt.scatter(pca_data[:,0], pca_data[:,1], c=df["Cluster"])
plt.title("Customer Segmentation using K-Means")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True)
plt.show()

# Save clustered dataset
df.to_csv("data/processed/customer_segments.csv", index=False)

print("\nCustomer Segmentation Completed Successfully.")