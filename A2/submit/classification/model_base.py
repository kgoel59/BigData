from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

class ModelBase:
    def __init__(self, df_preprocessed):
        self.df = df_preprocessed
        self.X_train, self.X_test, self.y_train, self.y_test = self.prepare_data()

    def prepare_data(self):
        """Prepare data by splitting into features and labels, and then into train/test sets."""
        # Separate features and target
        X = self.df.drop(columns='gender')  # Assuming 'gender' is the label
        y = self.df['gender']

        # Flatten the `bow_feature` and `word2vec_embeddings` columns and combine with other numerical features
        X_bow = pd.DataFrame(X['bow_feature'].tolist(), index=X.index)
        X_w2v = pd.DataFrame(X['word2vec_embeddings'].tolist(), index=X.index)
        X.drop(columns=['bow_feature', 'word2vec_embeddings', 'text'], inplace=True)  # Drop unnecessary columns

        # Concatenate the numeric columns and the flattened BOW and Word2Vec features
        X_combined = pd.concat([X, X_bow, X_w2v], axis=1)
        
        # Ensure all feature names are strings
        X_combined.columns = X_combined.columns.astype(str)
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)
        
        return X_train, X_test, y_train, y_test

    def build_pipeline(self, model):
        """Build a pipeline for scaling numeric features and applying the model."""
        # Standard scaling for numerical features
        pipeline_steps = [
            ('scaler', StandardScaler()),  # Scale all the features
            ('model', model)               # The classifier
        ]
        return Pipeline(pipeline_steps)

    def grid_search_cv(self, pipeline, param_grid):
        """Perform K-fold cross-validation with grid search."""
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        grid_search = GridSearchCV(pipeline, param_grid, cv=kfold, scoring='accuracy')
        grid_search.fit(self.X_train, self.y_train)
        return grid_search.best_estimator_, grid_search.best_params_

    def evaluate(self, model):
        """Evaluate the model on the test data using accuracy, F1, recall, and confusion matrix."""
        y_pred = model.predict(self.X_test)

        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        recall = recall_score(self.y_test, y_pred, average='weighted')
        cm = confusion_matrix(self.y_test, y_pred)

        print(f"Accuracy: {accuracy}")
        print(f"F1 Score: {f1}")
        print(f"Recall: {recall}")
        self.plot_confusion_matrix(cm)

    @staticmethod
    def plot_confusion_matrix(cm):
        """Plot confusion matrix using seaborn."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()
