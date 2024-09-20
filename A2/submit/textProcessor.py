
# System and utilities
import re
import string

# Data manipulation and visualization
import numpy as np
import matplotlib.pyplot as plt

# Machine Learning and NLP

from sklearn.feature_extraction.text import CountVectorizer

# NLP Libraries
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import spacy
from bs4 import BeautifulSoup
from gensim.models import Word2Vec

# Transformers

# Visualization Libraries
from wordcloud import WordCloud



class TextProcessor:
    """Class to process text"""
    def __init__(self):
        self.remove_words = set([
            "im", "tweet", "like", "follow", "rt", "dm", "pm",
            "#", "@", "followfriday", "ff", "l4l", "boost", "promo", "deal", "win",
            "now", "today", "tomorrow", "free", "best"
        ])
        self.stopwords_set = set(stopwords.words('english')).union(self.remove_words)
        self.lemmatizer = WordNetLemmatizer()
        self.nlp = spacy.load("en_core_web_sm")
        self.vectorizer = CountVectorizer()

    def word_cloud(self, texts):
      """Plot word cloud for text"""
      text = ' '.join(texts)
      wordcloud = WordCloud(width=800, height=800, background_color='white', min_font_size=10).generate(text)
      plt.figure(figsize=(8, 8), facecolor=None)
      plt.imshow(wordcloud)
      plt.axis("off")
      plt.tight_layout(pad=0)
      plt.show()

    def denoise_text(self, text):
        """Remove unwanted characters and format text."""
        text = self.remove_whitespace(text)
        text = self.remove_html(text)
        text = self.remove_between_square_brackets(text)
        text = self.remove_url(text)
        text = self.remove_special_characters(text)
        text = self.remove_punctuation(text)
        text = self.remove_hashtags(text)
        return text

    def standardize_text(self, text):
        """Convert text to lowercase."""
        return text.lower()

    def denoise_and_standardize_and_lemmatize_text(self, text):
        """Apply denoise and standardize functions"""
        return self.lemmatize(self.standardize_text(self.denoise_text(text)))

    def lemmatize(self, text):
        """Lemmatize text, excluding all stopwords."""
        tokens = word_tokenize(text)
        return ' '.join([self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stopwords_set])


    # Bag of Words Models
    def vectorize(self, texts):
        """Vectorize text"""
        X = self.vectorizer.fit_transform(texts)
        feature_names = self.vectorizer.get_feature_names_out()
        dense_array = X.toarray()
        return dense_array, feature_names

    # Word2Vec
    def embeddings(self, texts):
        """Embeddings"""
        sentences = [text.split() for text in texts]
        model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
        return model

    def get_average_embedding(self, sentence, model):
        """Get average embedding"""
        embeddings = [model.wv[word] for word in sentence.split()]
        if len(embeddings) == 0:
            return np.zeros(model.vector_size)
        return np.mean(embeddings, axis=0)

    @staticmethod
    def remove_whitespace(text):
        """Remove Whitespace"""
        return re.sub(r'\s+', ' ', text).strip()

    @staticmethod
    def remove_html(text):
        """Remove HTML"""
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text()

    @staticmethod
    def remove_between_square_brackets(text):
        """Remove Brackets"""
        return re.sub('\[[^]]*\]', '', text)

    @staticmethod
    def remove_url(text):
        """Remove Url"""
        return re.sub(r"(?:\@|https?\://)\S+", '', text)

    @staticmethod
    def remove_special_characters(text):
        """Remove Special Characters"""
        pattern = r'[^a-zA-Z0-9\s]'
        return re.sub(pattern, '', text)

    @staticmethod
    def remove_punctuation(text):
        """Remove Punctuation"""
        translator = str.maketrans('', '', string.punctuation + 'Ã'+'±'+'ã'+'¼'+'â'+'»'+'§')
        return text.translate(translator)

    @staticmethod
    def remove_hashtags(text):
        """Remove Hashtags"""
        text = " ".join(word.strip() for word in re.split('#(?!(?:hashtag)\b)[\w-]+(?=(?:\s+#[\w-]+)*\s*$)', text))
        text = " ".join(word.strip() for word in re.split('#|_', text))
        return text