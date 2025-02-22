import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from langlens.configuration.config import get_logger
from langlens.data import Dataset

log = get_logger(__name__)


def vectorize_data(train_data: Dataset, val_data: Dataset, test_data: Dataset, n_gram_type: str, vocab_size: int) \
        -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorizes the training, validation, and test data using n-grams.

    Args:
        train_data (Dataset): The training dataset.
        val_data (Dataset): The validation dataset.
        test_data (Dataset): The test dataset.
        n_gram_type (str): The type of n-gram analyzer ('word' or 'char').
        vocab_size (int): The maximum number of features to extract.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: The vectorized training, validation, and test data.
    """
    unigram_vectorizer = NGramVectorizer(1, n_gram_type, vocab_size)
    unigram_vectorizer.fit(train_data.x)
    train_features, val_features, test_features \
        = map(unigram_vectorizer.transform, [train_data.x, val_data.x, test_data.x])
    coverages = map(unigram_vectorizer.compute_coverage, [train_data.x, val_data.x, test_data.x])
    log.info(
        "The vocabulary covers %.2f%% of the training data, %.2f%% of the validation data, and %.2f%% of the test data.",
        *map(lambda x: x * 100, coverages))

    return train_features, val_features, test_features


class NGramVectorizer:
    def __init__(self, n: int, n_gram_type: str, max_features: int = None) -> None:
        """
        Initialize the NGramVectorizer.

        :param n: The number of n-grams.
        :param n_gram_type: The type of n-gram analyzer ('word' or 'char').
        :param max_features: The maximum number of features to extract.
        """
        self.vectorizer = CountVectorizer(analyzer=n_gram_type, ngram_range=(n, n), max_features=max_features)

    def fit(self, x_train: np.ndarray) -> None:
        """
        Fit the vectorizer to the training data.

        :param x_train: The training data as a NumPy array.
        """
        self.vectorizer.fit(x_train.tolist())

    def transform(self, x_data: np.ndarray) -> np.ndarray:
        """
        Transform the data using the fitted vectorizer.

        :param x_data: The data to transform as a NumPy array.
        :return: The transformed data as a NumPy array.
        """
        return self.vectorizer.transform(x_data.tolist()).toarray()

    def compute_coverage(self, corpus: np.ndarray) -> float:
        """
        Compute the proportion of tokens in the corpus that are covered by the vocabulary.

        :param corpus: NumPy array of documents.
        :return: Coverage ratio (float).
        """
        vocab = set(self.vectorizer.get_feature_names_out())
        tokenize = (lambda s: s.split()) if self.vectorizer.analyzer == 'word' else list
        tokens = np.concatenate([np.array(tokenize(doc)) for doc in corpus])
        return float(np.mean(np.isin(tokens, list(vocab))))
