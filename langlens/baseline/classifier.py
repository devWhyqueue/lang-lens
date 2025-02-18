import numpy as np
from sklearn.naive_bayes import MultinomialNB


class NaiveBayes:
    def __init__(self) -> None:
        self.model = MultinomialNB()

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Fit the NaiveBayes model to the training data.

        :param x_train: Training features as a numpy array.
        :param y_train: Training labels as a numpy array.
        """
        self.model.fit(x_train, y_train)

    def predict(self, x_data: np.ndarray) -> np.ndarray:
        """
        Predict the labels for the given data using the trained model.

        :param x_data: Data to predict as a numpy array.
        :return: Predicted labels as a numpy array.
        """
        return self.model.predict(x_data)
