from clustering.dbscan import DBSCANModel
from clustering.kmeans import KMeansModel
from clustering.som import SOMModel

from classification.knn import KNNClassifier
from classification.svm import SVMClassifier
from classification.dtree import DecisionTreeClassifierModel, RandomForestClassifierModel

from regression.regression import Regression

from neural.neural import NeuralNetworkClassifier

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class Task2_3:
    """Applying core Models"""

    def clustering(self, df_preprocessed):
        X = np.hstack([np.array(df_preprocessed['bow_feature'].tolist()),
                    np.array(df_preprocessed['word2vec_embeddings'].tolist())])
        # Split the data into training and testing sets for visualization purposes
        X_train, _ = train_test_split(X, test_size=0.2, random_state=42)

        # Initialize and evaluate clustering models

        # K-Means Model

        kmeans_model = KMeansModel(n_clusters=2)  # k=2 clusters
        kmeans_model.train(X_train)
        kmeans_model.evaluate(X_train)  # K-Means does not use a test set, but evaluation happens on the training data
        kmeans_model.visualize(X_train)

        # DBSCAN Model
        dbscan_model = DBSCANModel(eps=0.5, min_samples=5)
        dbscan_model.train(X_train)
        dbscan_model.evaluate(X_train)  # DBSCAN does not use a test set, but evaluation happens on the training data
        dbscan_model.visualize(X_train)

        # SOM Model
        som_model = SOMModel(x=5, y=5, input_len=X_train.shape[1])
        som_model.train(X_train, num_iterations=200)
        som_model.evaluate(X_train) #SOM does not use a test set, but evaluation happens on the training data
        som_model.visualize(X_train)

    def classification(self, df_preprocessed):
        knn_classifier = KNNClassifier(df_preprocessed)
        best_knn_model, knn_params = knn_classifier.train()

        # svm_classifier = SVMClassifier(df_preprocessed)
        # best_svm_model, svm_params = svm_classifier.train()

        dt_classifier = DecisionTreeClassifierModel(df_preprocessed)
        best_dt_model, dt_params = dt_classifier.train()

        rf_classifier = RandomForestClassifierModel(df_preprocessed)
        best_rf_model, rf_params = rf_classifier.train()

    def regression(self, df_preprocessed):
        pipeline = Regression(df_preprocessed, features=['retweet_count', 'tweet_count'], target='gender', embedding_col='word2vec_embeddings')
        pipeline.logistic_regression()
        pipeline.linear_regression()

    def neural_network(self, df_preprocessed):
        text_embeddings = np.vstack(df_preprocessed['word2vec_embeddings'])
        numeric_features = df_preprocessed[['fav_number', 'retweet_count', 'tweet_count']].values
        X = np.hstack((text_embeddings, numeric_features))  # Combine embeddings and numeric features
        y = df_preprocessed['gender'].values  # Labels

        # Step 5: Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Step 6: Build and train the neural network
        nn_classifier = NeuralNetworkClassifier(input_dim=X_train.shape[1], output_dim=len(np.unique(y)))
        nn_classifier.build_model()
        history = nn_classifier.train_model(X_train, y_train, epochs=50)

        # Step 7: Evaluate the model
        nn_classifier.evaluate_model(X_test, y_test)

        # Step 8: Plot training history
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        # Step 9: Confusion matrix
        y_pred = np.argmax(nn_classifier.model.predict(X_test), axis=1)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['human', 'non-human'])
        disp.plot(cmap=plt.cm.Blues)
        plt.show()

