import numpy as np
import matplotlib.pyplot as plt
from minisom import MiniSom

class SOMModel:
    def __init__(self, x=10, y=10, input_len=10, sigma=1.0, learning_rate=0.5):
        self.som = MiniSom(x=x, y=y, input_len=input_len, sigma=sigma, learning_rate=learning_rate)
        self.x = x
        self.y = y

    def train(self, X, num_iterations=100):
        self.som.random_weights_init(X)
        self.som.train_random(X, num_iterations)
        self.labels = np.array([self.som.winner(x) for x in X])
        print(f"SOM trained with {num_iterations} iterations")

    def evaluate(self, X):
        quantization_error = np.mean([np.linalg.norm(x - self.som.get_weights()[self.som.winner(x)]) for x in X])
        print(f"Quantization Error: {quantization_error}")

        # Number of unique nodes (clusters)
        unique_nodes = len(set(map(tuple, self.labels)))
        print(f"SOM Number of unique nodes (clusters): {unique_nodes}")


    def visualize(self, X, data_labels=None):
        plt.figure(figsize=(10, 10))
        # Get the SOM grid
        distance_map = self.som.distance_map().T

        # Heatmap creation
        plt.imshow(distance_map, cmap='bone_r', interpolation='nearest')
        plt.colorbar(label='Distance from Neurons')

        # Plot points on the heatmap, using colors based on labels or assigning random colors if no labels are available
        for i, (x, label) in enumerate(zip(X, data_labels if data_labels is not None else [str(i) for i in range(len(X))])):
            w = self.som.winner(x)  # Get the winning node
            plt.text(w[0], w[1], str(label), color='red', ha='center', va='center', fontsize=12,
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='red', boxstyle='round,pad=0.2'))

        plt.title('Self-Organizing Map (SOM) Visualization')
        plt.show()
