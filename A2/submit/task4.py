import pandas as pd
from collections import Counter

class Task4:
    def __init__(self):
        pass

    @staticmethod
    def word_analysis(df_processed):
        if 'text' not in df_processed.columns:
            raise ValueError("DataFrame must contain a 'text' column.")

        # Combine all text into a single string
        all_text = ' '.join(df_processed['text'].astype(str))

        # Split text into words
        words = all_text.split()

        # Count word frequencies
        word_counts = Counter(words)

        # Convert the word counts to a DataFrame
        word_freq_df = pd.DataFrame(word_counts.items(), columns=['word', 'frequency'])

        # Sort by frequency in descending order
        word_freq_df = word_freq_df.sort_values(by='frequency', ascending=False).reset_index(drop=True)

        return word_freq_df
