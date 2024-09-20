from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class DBSCANModel:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples)

    def train(self, X):
        self.labels = self.model.fit_predict(X)

    def evaluate(self, X):
        # Exclude noise points (-1) for Silhouette Score calculation
        if len(set(self.labels)) > 1 and len(self.labels[self.labels != -1]) > 0:
            self.silhouette_score = silhouette_score(X[self.labels != -1], self.labels[self.labels != -1])
            print(f"DBSCAN Silhouette Score: {self.silhouette_score}")
        else:
            print("Silhouette score cannot be computed with only one cluster or all points labeled as noise.")

    def visualize(self, X):
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=self.labels, cmap='viridis')
        plt.title('DBSCAN Clustering')
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.show()