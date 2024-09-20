from classification.model_base import ModelBase
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

class DecisionTreeClassifierModel(ModelBase):
    def __init__(self, df_preprocessed):
        super().__init__(df_preprocessed)

    def train(self):
        """Train the Decision Tree model with grid search."""
        dt = DecisionTreeClassifier()
        pipeline = self.build_pipeline(dt)

        # Define parameter grid for Decision Tree
        param_grid = {
            'model__max_depth': [3, 5, 7],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4]
        }

        best_model, best_params = self.grid_search_cv(pipeline, param_grid)
        print(f"Best Decision Tree Params: {best_params}")
        self.evaluate(best_model)
        return best_model, best_params
    
class RandomForestClassifierModel(ModelBase):
    def __init__(self, df_preprocessed):
        super().__init__(df_preprocessed)

    def train(self):
        """Train the Random Forest model with grid search."""
        rf = RandomForestClassifier()
        pipeline = self.build_pipeline(rf)

        # Define parameter grid for Random Forest
        param_grid = {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [3, 5, 7],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4]
        }

        best_model, best_params = self.grid_search_cv(pipeline, param_grid)
        print(f"Best Random Forest Params: {best_params}")
        self.evaluate(best_model)
        return best_model, best_params