from classification.model_base import ModelBase
from sklearn.svm import SVC


class SVMClassifier(ModelBase):
    def __init__(self, df_preprocessed):
        super().__init__(df_preprocessed)

    def train(self):
        """Train the SVM model with grid search."""
        svm = SVC()
        pipeline = self.build_pipeline(svm)

        # Define parameter grid for SVM
        param_grid = {
            'model__C': [0.1, 1, 10],
            'model__kernel': ['linear', 'rbf']
        }

        best_model, best_params = self.grid_search_cv(pipeline, param_grid)
        print(f"Best SVM Params: {best_params}")
        self.evaluate(best_model)
        return best_model, best_params