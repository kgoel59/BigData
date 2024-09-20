# System and utilities
import os

# Data manipulation and visualization
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

# Machine Learning and NLP
from sklearn.preprocessing import StandardScaler, LabelEncoder


class Task2_P1:
    """Preprocessing and Analysisng Data"""
    def __init__(self, base_path, file):
        self.base_path = base_path
        self.file = file
        self.df = None
        self.scaler = StandardScaler()
        self.le_gender = LabelEncoder()

    def load_data(self):
      """load data"""
      self.df = pd.read_csv(os.path.join(self.base_path, self.file), encoding='latin1')

    def encode_labels(self, df):
        """Encode labels"""
        print("Encoding Gender and Brand")
        df['gender'] = df['gender'].replace({'male': 'human', 'female': 'human'})
        df['gender'] = df['gender'].replace({'brand': 'non-human'})
        df['gender'] = self.le_gender.fit_transform(df['gender'])
        return df

    def scale_features(self, df):
        """Scale numeric features using StandardScaler"""
        print("Applying Standard Scaler")
        columns_to_scale = ['fav_number','retweet_count','tweet_count']
        numerical = df[columns_to_scale]
        scaled_features = self.scaler.fit_transform(numerical)
        scaled_features_df = pd.DataFrame(
            scaled_features, columns=columns_to_scale, index=df.index)
        df[columns_to_scale] = scaled_features_df

        return df

    def preprocess_data_frame(self):
        """Select features, fill missing values, scale features, encode labels"""
        print("Selecting rows with high gender confidence")
        df = self.df.copy()
        df = df[df['gender'].isin(['female','male','brand'])]
        df = df[df['gender:confidence'] > 0.9]

        chosen_columns = {
            'gender', 'fav_number', 'retweet_count','text','tweet_count'
        }

        print(f"Selecting features: {chosen_columns}")
        df = df.loc[:, self.df.columns.intersection(chosen_columns)]

        print("Filling missing values")
        df.fillna('', inplace=True)


        print("Scaling features")
        df = self.scale_features(df)

        print("Encoding Labels")
        df = self.encode_labels(df)

        return df

    def plot_correlation_matrix(self, df, alpha=0.05):
        """Plot the correlation matrix"""

        df = df.select_dtypes(include=['int64', 'float64'])
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

    def plot_pairplot(self, df):
        """Plot the scatterplot matrix"""
        df = df.select_dtypes(include=['int64', 'float64'])
        sns.pairplot(df)
        plt.suptitle('Scatterplot Matrix', y=1.02)  # Add a title
        plt.show()