from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class KMeansModel:
    def __init__(self, n_clusters=5, random_state=42):
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=self.n_clusters, random_state=random_state)

    def train(self, X):
        self.labels = self.model.fit_predict(X)

    def evaluate(self, X):
        self.silhouette_score = silhouette_score(X, self.labels)
        print("K-Means Silhouette Score:", self.silhouette_score)

    def visualize(self, X):
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=self.labels, cmap='viridis')
        plt.title(f'K-Means Clustering (Silhouette: {self.silhouette_score:.2f})')
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.show()
