import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
data = pd.read_csv("kmeans_data.csv")
X = data[["Study_Hours", "Marks"]]
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
data["KMeans_Cluster"] = kmeans.fit_predict(X)
print("K-Means Cluster Assignments:")
print(data)
plt.scatter(X["Study_Hours"], X["Marks"], c=data["KMeans_Cluster"])
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("K-Means Clustering")
plt.show()
hdata = pd.read_csv("hierarchical_data.csv")
HX = hdata[["Annual_Income", "Spending_Score"]]
hc = AgglomerativeClustering(n_clusters=3, linkage="ward")
hdata["Cluster"] = hc.fit_predict(HX)
print("\nHierarchical Cluster Assignments:")
print(hdata)
plt.scatter(HX["Annual_Income"], HX["Spending_Score"], c=hdata["Cluster"])
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Hierarchical Clustering")
plt.show()
pdata = pd.read_csv("pca_data.csv")
PX = pdata[["Maths", "Physics", "Chemistry", "Biology"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(PX)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA - 2D Representation")
plt.show() 