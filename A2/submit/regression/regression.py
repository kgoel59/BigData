from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd

class Regression:
    def __init__(self, df, features, target, embedding_col, test_size=0.2, random_state=42):
        """
        Initialize the ModelPipeline with data and parameters.
        
        Args:
        df: DataFrame containing the dataset.
        features: List of feature column names.
        target: Target column name.
        embedding_col: Column name for word embeddings.
        test_size: Proportion of data to be used for testing.
        random_state: Random state for reproducibility.
        """
        self.df = df
        self.features = features
        self.target = target
        self.embedding_col = embedding_col
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        
        # Prepare data
        self.X, self.y = self._prepare_features_labels()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Apply SMOTE to balance the training set
        smote = SMOTE(random_state=self.random_state)
        self.X_train_smote, self.y_train_smote = smote.fit_resample(self.X_train, self.y_train)
        
        # Scale the features
        self.X_train_smote = self.scaler.fit_transform(self.X_train_smote)
        self.X_test = self.scaler.transform(self.X_test)
        
    def _prepare_features_labels(self):
        """Prepares features (X) and labels (y) for the model."""
        X = np.hstack((
            self.df[self.features].values, 
            np.array(self.df[self.embedding_col].tolist())  # Convert word embeddings to numpy array
        ))
        y = self.df[self.target]
        return X, y

    def logistic_regression(self):
        """Train and evaluate a Logistic Regression model."""
        param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
        grid_search = GridSearchCV(
            LogisticRegression(max_iter=1000, class_weight='balanced'), 
            param_grid, cv=5, scoring='f1_weighted'
        )
        grid_search.fit(self.X_train_smote, self.y_train_smote)
        
        # Best estimator
        best_log_reg = grid_search.best_estimator_
        print("Best parameters for Logistic Regression: ", grid_search.best_params_)
        
        # Make predictions
        y_pred_logistic = best_log_reg.predict(self.X_test)
        print("\nClassification Report (Logistic Regression with SMOTE):")
        print(classification_report(self.y_test, y_pred_logistic))
        
        # Cross-Validation
        cv_scores = cross_val_score(best_log_reg, self.X, self.y, cv=5, scoring='f1_weighted')
        print(f'Logistic Regression Cross-Validation F1 scores: {cv_scores}')
        print(f'Mean F1 score (Logistic Regression): {np.mean(cv_scores)}')

    def linear_regression(self):
        """Train and evaluate a Linear Regression model used for classification."""
        # Initialize and train Linear Regression model
        linear_reg = LinearRegression()
        linear_reg.fit(self.X_train_smote, self.y_train_smote)
        
        # Make predictions (continuous values)
        y_pred_linear_continuous = linear_reg.predict(self.X_test)
        
        # Convert continuous predictions to binary (classification threshold = 0.5)
        y_pred_linear_class = (y_pred_linear_continuous >= 0.5).astype(int)
        
        print("\nClassification Report (Linear Regression used for Classification):")
        print(classification_report(self.y_test, y_pred_linear_class))
        
        # Cross-Validation
        cv_scores_linear = cross_val_score(linear_reg, self.X, self.y, cv=5, scoring='f1_weighted')
        print(f'Linear Regression Cross-Validation F1 scores: {cv_scores_linear}')
        print(f'Mean F1 score (Linear Regression): {np.mean(cv_scores_linear)}')