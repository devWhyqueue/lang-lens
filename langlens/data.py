from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from langlens.configuration.config import settings


@dataclass
class Dataset:
    x: np.ndarray
    y: np.ndarray

    def __len__(self) -> int:
        return len(self.y)


def load_and_split(dataset_path: str) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load a dataset from a CSV file and split it into training, validation, and test sets.

    Args:
        dataset_path (str): The path to the CSV file containing the dataset.

    Returns:
        Tuple[Dataset, Dataset, Dataset]: A tuple containing the training, validation, and test datasets.
    """
    df = pd.read_csv(dataset_path)
    x, y = df['text'], df['language']
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.2, random_state=settings['seed'])
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=settings['seed'])

    return (
        Dataset(x_train.values, y_train.values),
        Dataset(x_val.values, y_val.values),
        Dataset(x_test.values, y_test.values)
    )
