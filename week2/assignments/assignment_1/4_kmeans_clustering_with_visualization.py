# Import necessary libraries
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Generate random 2D dataset with 3 centers
X, y_true = make_blobs(n_samples=300, 
                        centers=3,
                        n_features=2,
                        cluster_std=0.60, 
                        random_state=0)

# Apply KMeans clustering with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)


# Plot the clusters with different colors
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
plt.show()