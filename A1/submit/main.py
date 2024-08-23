"""NewChic Dataset Analysis"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.tree import DecisionTreeClassifier


class Setup:
    """Class to Setup"""

    def __init__(self):
        pass

    @staticmethod
    def section(text):
        """section text"""
        n = len(text)
        print()
        print("-" * n)
        print(text)
        print("-" * n)
        print()


class Task1:
    """Problem Analysis and Data Preprocess"""

    def __init__(self, base_path, files):
        self.base_path = base_path
        self.files = files
        self.df = None
        self.scaler = StandardScaler()
        self.le_category = LabelEncoder()
        self.le_name = LabelEncoder()

        self.load_and_concatenate_data()

    def load_and_concatenate_data(self):
        """Load data from files and concatenate into a single DataFrame"""
        print(f"Loading data from files: {self.files}")
        data_frames = {
            file.split('.')[0]: pd.read_csv(os.path.join(self.base_path, file))
            for file in self.files
        }
        self.df = pd.concat(data_frames.values(), ignore_index=True)

    def preprocess_data_frame(self, scale_data=True):
        """Select features, fill missing values, scale features, encode labels, and remove outliers"""
        # Selecting features and filling missing values
        chosen_columns = {
            'category', 'name', 'current_price', 'raw_price', 'likes_count', 'discount'
        }
        print(f"Selecting features: {chosen_columns}")
        df = self.df.loc[:, self.df.columns.intersection(chosen_columns)]

        fill_values = {
            'likes_count': 0,
            'discount': 0,
            'current_price': df['current_price'].median()
        }

        for column, value in fill_values.items():
            df[column] = df[column].fillna(value)

        # Apply scaling, encoding, and outlier removal
        if scale_data:
            df = self.scale_features(df)

        df = self.encode_labels(df)
        df = self.remove_outliers(df)

        return df

    def scale_features(self, df):
        """Scale numeric features using StandardScaler"""
        print("Applying Standard Scaler")
        columns_to_scale = ['current_price',
                            'raw_price', 'discount', 'likes_count']
        numerical = df[columns_to_scale]
        scaled_features = self.scaler.fit_transform(numerical)
        scaled_features_df = pd.DataFrame(
            scaled_features, columns=columns_to_scale, index=self.df.index)
        df[columns_to_scale] = scaled_features_df

        return df

    def encode_labels(self, df):
        """Encode categorical features using LabelEncoder"""
        print("Encoding lables")
        df['category'] = self.le_category.fit_transform(df['category'])
        df['name'] = self.le_name.fit_transform(df['name'])

        return df

    def remove_outliers(self, df):
        """Remove outliers based on Z-score"""
        print("Removed outliers")
        df_numerical = df.select_dtypes(include=[np.number])
        z_scores = np.abs(stats.zscore(df_numerical))
        threshold = 3
        inliers = (z_scores < threshold).all(axis=1)
        df = df[inliers]

        return df

    def plot_pairplot(self, df):
        """Plot the scatterplot matrix"""
        sns.pairplot(df)
        plt.suptitle('Scatterplot Matrix', y=1.02)  # Add a title
        plt.show()

    def plot_correlation_matrix(self, df, alpha=0.05):
        """Plot the correlation matrix"""

        correlation_matrix = df.corr()

        # Initialize a DataFrame to store p-values
        p_values = pd.DataFrame(np.zeros_like(
            correlation_matrix), columns=correlation_matrix.columns, index=correlation_matrix.index)

        # Calculate p-values
        for row in correlation_matrix.index:
            for col in correlation_matrix.index:
                if row == col:
                    # p-value is 0 for diagonal elements
                    p_values.loc[row, col] = 0.0
                else:
                    _, p_val = stats.pearsonr(df[row], df[col])

                    p_values.loc[row, col] = p_val

        # Create a mask for significant correlations
        significant_mask = p_values < alpha

        # Plot the correlation matrix with a mask
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, mask=~significant_mask,
                    cbar_kws={"label": "Correlation Coefficient"}, annot_kws={"size": 10})

        plt.title('Significant Correlations')
        plt.show()

        return correlation_matrix, p_values

    def hypothesis1_price_likes(self, p_values):

        p_value = p_values['current_price']['likes_count']
        print("P-value:", p_value)

        alpha = 0.05
        if p_value < alpha:
            print("Reject the null hypothesis: There is a significant correlation between current_price and likes_count.")
        else:
            print("Fail to reject the null hypothesis: There is no significant correlation between current_price and likes_count.")

    def hypothesis2_discount_likes(self, p_values):
        p_value = p_values['discount']['likes_count']
        print("P-value:", p_value)

        alpha = 0.05
        if p_value < alpha:
            print("Reject the null hypothesis: There is a significant correlation between discount and likes_count.")
        else:
            print("Fail to reject the null hypothesis: There is no significant correlation between discount and likes_count.")

    def hypothesis3_likes_category(self, df):
        likes_count_by_category = [
            group['likes_count'].values for name, group in self.df.groupby('category')]
        f_statistic, p_value = stats.f_oneway(*likes_count_by_category)

        print("ANOVA F-statistic:", f_statistic)
        print("P-value:", p_value)

        alpha = 0.05
        if p_value < alpha:
            print("Reject the null hypothesis: The average likes_count differs across at least one product category.")
        else:
            print("Fail to reject the null hypothesis: The average likes_count is the same across all product categories.")

        df['category'] = self.le_category.inverse_transform(
            df['category'].to_numpy())

        plt.figure(figsize=(12, 8))
        sns.boxplot(x='category', y='likes_count', data=self.df)
        plt.title('Box Plot of Likes Count Across Product Categories')
        plt.xlabel('Product Category')
        plt.ylabel('Likes Count')
        plt.grid(True)
        plt.show()

    def analyze_top_categories_and_products(self, df, top_n=7):
        """Analyze top categories and products based on likes_count"""

        def get_top_categories_by_likes(df, top_n):
            avg_likes_per_category = df.groupby(
                'category')['likes_count'].mean().sort_values(ascending=False)
            top_categories = avg_likes_per_category.head(top_n).index.tolist()
            return top_categories

        def filter_top_categories(df, top_categories):
            return df[df['category'].isin(top_categories)]

        def get_top_products(df, top_n):
            return df.sort_values(
                by=['likes_count', 'current_price', 'discount'],
                ascending=[False, False, False]
            ).head(top_n)

        top_categories = get_top_categories_by_likes(df, top_n=top_n)
        filtered_df = filter_top_categories(df, top_categories)
        top_products = get_top_products(filtered_df, top_n=10)

        print("Top 7 Categories (based on average likes count):",
              self.le_category.inverse_transform(top_categories))
        print()
        print("Top 10 Products in the Top 7 Categories:\n",
              self.le_name.inverse_transform(top_products['name'].to_list()))

        top_products['category'] = self.le_category.inverse_transform(
            top_products['category'].to_numpy())
        return top_products


class Task2:
    """Clustering"""

    def __init__(self, df, top_products):
        # Drop non-required columns
        self.df_cluster = df.drop(['raw_price', 'likes_count'], axis=1)
        self.X = self.df_cluster[['current_price', 'discount']]
        self.y = self.df_cluster.drop(columns=['current_price', 'discount'])
        self.pnames = self.y['name'].to_numpy().ravel()
        self.categories = self.y['category'].to_numpy().ravel()
        self.top_products = top_products
        self.metrics = {
            "Silhouette Score": {},
            "Calinski-Harabasz Index": {},
            "Davies-Bouldin Index": {}
        }
        self.kmeans_model = None
        self.agglomerative_labels = None
        self.centroids = None

    def elbow_method(self):
        wss = []
        for i in range(1, 16):
            kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
            kmeans.fit(self.X)
            wss.append(kmeans.inertia_)

        plt.plot(range(1, 16), wss)
        plt.title('The Elbow Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('WSS')
        plt.show()

    def perform_clustering(self):
        algorithms = {
            "KMeans": KMeans(n_clusters=7, max_iter=1000),
            "Agglomerative": AgglomerativeClustering(n_clusters=7, linkage='ward')
        }

        for name, algorithm in algorithms.items():
            labels = algorithm.fit_predict(self.X)

            if name == "KMeans":
                self.kmeans_model = algorithm
            elif name == "Agglomerative":
                self.agglomerative_labels = labels
                self.compute_centroids(labels)

            # Store the calculated metrics
            self.metrics["Silhouette Score"][name] = silhouette_score(
                self.X, labels)
            self.metrics["Calinski-Harabasz Index"][name] = calinski_harabasz_score(
                self.X, labels)
            self.metrics["Davies-Bouldin Index"][name] = davies_bouldin_score(
                self.X, labels)

            # Create a copy of the data with the new labels
            df_copy = self.X.copy()
            df_copy['label'] = labels
            df_copy['name'] = self.pnames

            # Select only numeric columns for mean calculation
            numeric_cols = df_copy.select_dtypes(include=['number']).columns

            # Compute mean values for each cluster, only for numeric columns
            df_mean = df_copy.groupby('label')[numeric_cols].agg('mean')

            # Visualization
            self.visualize_clusters(df_copy, df_mean, labels, name)

    def compute_centroids(self, labels):
        """Compute centroids for Agglomerative Clustering"""
        centroids = []
        for label in np.unique(labels):
            # Compute the mean of all points in the cluster
            centroid = self.X[labels == label].mean(axis=0)
            centroids.append(centroid)
        self.centroids = np.array(centroids)

    def predict_cluster_kmeans(self, current_price, discount):
        """Predict the cluster using KMeans"""
        if self.kmeans_model is None:
            raise ValueError(
                "KMeans model has not been trained. Call perform_clustering() first.")
        input_data = pd.DataFrame(
            {'current_price': [current_price], 'discount': [discount]})
        predicted_cluster = self.kmeans_model.predict(input_data)
        return predicted_cluster[0]

    def predict_cluster_agglomerative(self, current_price, discount):
        """Predict the cluster using Agglomerative Clustering"""
        if self.centroids is None:
            raise ValueError(
                "Centroids have not been computed. Call perform_clustering() first.")
        input_data = np.array([[current_price, discount]])
        closest, _ = pairwise_distances_argmin_min(input_data, self.centroids)
        return closest[0]

    def map_top_products_to_clusters(self, le_name, le_category):
        if self.kmeans_model is None and self.agglomerative_labels is None:
            raise ValueError(
                "Clustering has not been performed. Call perform_clustering() first.")

        top_products_df = self.X.copy()
        top_products_df['name'] = self.pnames
        top_products_df['category'] = self.categories

        top_products_df['kmeans_cluster'] = self.kmeans_model.predict(self.X)

        top_products_df['agglomerative_cluster'] = self.agglomerative_labels

        top_products_clusters = top_products_df[top_products_df['name'].isin(
            self.top_products['name'].to_list())]

        top_products_clusters['name'] = le_name.inverse_transform(
            top_products_clusters['name'].to_numpy())
        top_products_clusters['category'] = le_category.inverse_transform(
            top_products_clusters['category'].to_numpy())

        return top_products_clusters

    def visualize_clusters(self, df_copy, df_mean, labels, algorithm_name):
        fig, ax = plt.subplots(figsize=(20, 8))
        columns = list(df_mean.columns)
        palette = sns.color_palette("husl", n_colors=len(df_mean))

        sns.scatterplot(
            x=df_copy[columns[0]],
            y=df_copy[columns[1]],
            hue=labels,
            palette=palette,
            ax=ax,
            s=100,
            edgecolor='k'
        )

        # Mark top products with a large red dot
        top_products_df = df_copy[df_copy['name'].isin(
            self.top_products['name'].to_list())]
        ax.scatter(
            x=top_products_df[columns[0]],
            y=top_products_df[columns[1]],
            color='red',
            s=400,  # Size of the red dot
            label='Top Products',
            edgecolor='black'
        )

        ax.set_title(f'{columns[0]} vs {columns[1]}')
        ax.legend(title='Cluster', loc='upper right')
        ax.set_xlabel(columns[0])
        ax.set_ylabel(columns[1])
        plt.suptitle(f"{algorithm_name} Clustering Results", fontsize=16)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    def print_metrics(self):
        for metric_name, metric_values in self.metrics.items():
            print(f"\n{metric_name}:")
            for algo_name, value in metric_values.items():
                print(f"{algo_name}: {value:.4f}")

        # Determine the best algorithm based on the Silhouette Score
        best_algorithm = max(
            self.metrics["Silhouette Score"], key=self.metrics["Silhouette Score"].get)
        print(f"\nBest Algorithm based on Silhouette Score: {best_algorithm}")


class Task3:
    """Classification"""

    def __init__(self, df):
        self.df = df.copy()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.knn_model = None
        self.dt_model = None
        self.skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    def preprocess_data(self, le_category):
        print("Creating popularity column based on likes_count")
        self.df['popularity'] = self.df['likes_count'].apply(
            lambda x: 1 if x > self.df['likes_count'].mean() else 0)

        # Selecting features and target
        print(
            "Selecting features and target ['category', 'current_price', 'discount']")
        X = self.df[['category', 'current_price', 'discount']]
        X['category'] = le_category.fit_transform(X['category'])
        y = self.df['popularity']

        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

    def train_knn(self):
        print("KNN")
        knn = KNeighborsClassifier()
        param_grid = {
            'n_neighbors': [i for i in range(13, 20, 2)],
            'weights': ['uniform'],
            'metric': ['manhattan']
        }
        grid_search = GridSearchCV(
            knn, param_grid, cv=self.skf, scoring='accuracy')
        grid_search.fit(self.X_train, self.y_train)

        # Best KNN model
        self.knn_model = grid_search.best_estimator_
        print(f"Best KNN Parameters: {grid_search.best_params_}")

        # Cross-validation scores
        cv_scores = cross_val_score(
            self.knn_model, self.X_train, self.y_train, cv=self.skf, scoring='accuracy')
        print(f'KNN Average Accuracy: {cv_scores.mean() * 100:.2f}%')
        print(f'KNN Standard Deviation: {cv_scores.std() * 100:.2f}%')

    def evaluate_knn(self):
        self.knn_model.fit(self.X_train, self.y_train)
        y_pred = self.knn_model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        cm = confusion_matrix(self.y_test, y_pred)

        print(f'KNN Accuracy: {accuracy * 100:.2f}%')
        print(f'KNN Precision: {precision:.2f}')
        print(f'KNN Recall: {recall:.2f}')
        print(f'KNN F1-Score: {f1:.2f}')
        print('KNN Confusion Matrix:')
        print(cm)

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix for KNN Popularity Prediction')
        plt.show()

        # Plot precision, recall, and f1-score
        metrics = pd.DataFrame(
            {'Metric': ['Precision', 'Recall', 'F1-Score'], 'Score': [precision, recall, f1]})
        plt.figure(figsize=(8, 6))
        sns.barplot(x='Metric', y='Score', data=metrics, palette='viridis')
        plt.ylim(0, 1)
        plt.title('KNN Precision, Recall, and F1-Score')
        plt.show()

    def train_decision_tree(self):
        print("Decision Tree")
        dt = DecisionTreeClassifier(random_state=42)
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 20, 30, 40, 50],
            'min_samples_split': [2, 5, 10, 15],
            'min_samples_leaf': [1, 2, 4, 6]
        }
        grid_search = GridSearchCV(
            dt, param_grid, cv=self.skf, n_jobs=-1, verbose=2)
        grid_search.fit(self.X_train, self.y_train)

        # Best Decision Tree model
        self.dt_model = grid_search.best_estimator_
        print(f"Best Decision Tree Parameters: {grid_search.best_params_}")

        # Cross-validation scores
        cv_scores = cross_val_score(
            self.dt_model, self.X_train, self.y_train, cv=self.skf, scoring='accuracy')
        print(f'Decision Tree Average Accuracy: {cv_scores.mean() * 100:.2f}%')
        print(f'Decision Tree Standard Deviation: {cv_scores.std() * 100:.2f}%')

    def evaluate_decision_tree(self):
        self.dt_model.fit(self.X_train, self.y_train)
        y_pred = self.dt_model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        cm = confusion_matrix(self.y_test, y_pred)

        print(f'Decision Tree Accuracy: {accuracy * 100:.2f}%')
        print(f'Decision Tree Precision: {precision:.2f}')
        print(f'Decision Tree Recall: {recall:.2f}')
        print(f'Decision Tree F1-Score: {f1:.2f}')
        print('Decision Tree Confusion Matrix:')
        print(cm)

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix for Decision Tree Popularity Prediction')
        plt.show()

        # Plot precision, recall, and f1-score
        metrics = pd.DataFrame(
            {'Metric': ['Precision', 'Recall', 'F1-Score'], 'Score': [precision, recall, f1]})
        plt.figure(figsize=(8, 6))
        sns.barplot(x='Metric', y='Score', data=metrics, palette='viridis')
        plt.ylim(0, 1)
        plt.title('Decision Tree Precision, Recall, and F1-Score')
        plt.show()


def main():
    """main function"""
    Setup.section("Assignment 1 - Group 17")

    Setup.section("Task 1 - Problem Analysis and Data Preprocess")

    base_path = r'.'
    files = ['accessories.csv', 'bags.csv', 'beauty.csv', 'house.csv',
             'jewelry.csv', 'kids.csv', 'men.csv', 'shoes.csv', 'women.csv']
    task1 = Task1(base_path, files)

    Setup.section("Processing Data")
    df = task1.preprocess_data_frame()
    print(df)

    _, p_values = task1.plot_correlation_matrix(df)

    Setup.section("Analyzing Data")

    Setup.section("Hypothesis 1")
    task1.hypothesis1_price_likes(p_values)
    Setup.section("Hypothesis 2")
    task1.hypothesis2_discount_likes(p_values)
    Setup.section("Hypothesis 3")
    task1.hypothesis3_likes_category(df.copy())

    task1.plot_pairplot(df)

    Setup.section("Top categories and products")
    top_products = task1.analyze_top_categories_and_products(df)

    Setup.section("Clustering")

    task2 = Task2(df, top_products)
    task2.elbow_method()
    task2.perform_clustering()
    task2.print_metrics()
    top_products_clusters = task2.map_top_products_to_clusters(
        task1.le_name, task1.le_category)
    print(top_products_clusters[['name', 'category',
          'kmeans_cluster', 'agglomerative_cluster']])

    Setup.section("Classification")

    task3 = Task3(task1.df)
    task3.preprocess_data(task1.le_category)
    task3.train_knn()
    task3.evaluate_knn()
    task3.train_decision_tree()
    task3.evaluate_decision_tree()


if __name__ == "__main__":
    main()
