# Twitter Dataset Analysis Group 17
# 
#
#    Name & Student Number & Contribution \\
#    Karan Goel & 7836685 & 14.28\% \\
#    Alvin Jose & 8066358 & 14.28\% \\
#    Ashutosh Bhosale & 7795786 & 14.28\% \\
#    Banin Sensha Shreshta & 8447196 & 14.28\% \\
#    Gaurav Adarsh Santosh & 7032663 & 14.28\% \\
#    Lino Thankachan & 7926017 & 14.28\% \\
#    Rishab Manokaran & 7863974 & 14.28\% \\
# 


# System and utilities
import os
import logging
import warnings

# Machine Learning and NLP
import tensorflow as tf

# NLP Libraries
import nltk

import numpy as np

# Transformers
from transformers import logging as transformers_logging

from textProcessor import TextProcessor

from task2_P1 import Task2_P1
from task2_3 import Task2_3

# Configure warnings and logging
warnings.filterwarnings("ignore")
transformers_logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
tf.get_logger().setLevel(logging.ERROR)

class Setup:
    """Class to Setup"""
    def __init__(self):
        self.setup_nltk()
        self.setup_gpu()

    @staticmethod
    def setup_nltk():
        """Download necessary NLTK datasets"""
        nltk.download('averaged_perceptron_tagger')
        nltk.download('punkt')
        nltk.download('punkt_tab')
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('brown')

    @staticmethod
    def setup_gpu():
        """Set up GPU configuration."""
        pass

    @staticmethod
    def section(text,n=20):
        """section text"""
        print()
        print("-"*n)
        print(text)
        print("-"*n)
        print()

if __name__ == "__main__":
    Setup.section("Assignment 2 - Group 17")
    Setup.setup_nltk()

    Setup.section("Task 2 - Data Preparation")

    base_path = r'.'
    file = "twitter_user_data.csv"
    task2_P1 = Task2_P1(base_path, file)
    task2_P1.load_data()
    df_preprocessed = task2_P1.preprocess_data_frame()[:1000]

    _, p_values = task2_P1.plot_correlation_matrix(df_preprocessed)

    print("P values for twitter data")
    print(p_values)

    task2_P1.plot_pairplot(df_preprocessed)

    text_processor = TextProcessor()
    df_preprocessed['text'] = df_preprocessed['text'].apply(text_processor.denoise_and_standardize_and_lemmatize_text)

    text_processor = TextProcessor()
    df_preprocessed['text'] = df_preprocessed['text'].apply(text_processor.denoise_and_standardize_and_lemmatize_text)

    print("Word cloud for tweet")
    text_processor.word_cloud(df_preprocessed['text'].ravel())

    ## BAG OF WORDS MODEL
    bow_dense_array, bow_feature_names = text_processor.vectorize(df_preprocessed['text'])
    df_preprocessed['bow_feature'] = bow_dense_array.tolist()
    
    ### Word2Vec
    model = text_processor.embeddings(df_preprocessed['text'])
    word2vec_embeddings = [text_processor.get_average_embedding(text, model) for text in df_preprocessed['text']]

    df_preprocessed['word2vec_embeddings'] = word2vec_embeddings

    task2_P2 = Task2_3()

    Setup.section("Task 2 - Applying Clustering")

    task2_P2.clustering(df_preprocessed)

    Setup.section("Task 2 - Applying Classification")

    task2_P2.classification(df_preprocessed)

    Setup.section("Task 2 - Applying Regression")

    task2_P2.regression(df_preprocessed)

    Setup.section("Task 2 - Applying NN")

    task2_P2.neural_network(df_preprocessed)