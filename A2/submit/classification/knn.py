from classification.model_base import ModelBase
from sklearn.neighbors import KNeighborsClassifier

class KNNClassifier(ModelBase):
    def __init__(self, df_preprocessed):
        super().__init__(df_preprocessed)

    def train(self):
        """Train the KNN model with grid search."""
        knn = KNeighborsClassifier()
        pipeline = self.build_pipeline(knn)

        # Define parameter grid for KNN
        param_grid = {
            'model__n_neighbors': [3, 5, 7],
            'model__weights': ['uniform', 'distance']
        }

        best_model, best_params = self.grid_search_cv(pipeline, param_grid)
        print(f"Best KNN Params: {best_params}")
        self.evaluate(best_model)
        return best_model, best_params