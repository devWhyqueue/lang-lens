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
    df = _clean_data(df)

    x, y = df['text'], df['language']
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.2, random_state=settings['seed'])
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=settings['seed'])

    return (
        Dataset(x_train.values, y_train.values),
        Dataset(x_val.values, y_val.values),
        Dataset(x_test.values, y_test.values)
    )


def _clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by performing the following operations:
    1. Remove rows where the 'text' column contains only whitespace.
    2. Correct the typo in the 'language' column from 'Portugese' to 'Portuguese'.
    3. Map language names to their corresponding ISO 639-3 codes using a mapping file.

    Args:
        data (pd.DataFrame): The input dataset containing 'text' and 'language' columns.

    Returns:
        pd.DataFrame: The cleaned dataset with an additional 'lang_code' column.
    """
    data = data.copy()
    # Clean the dataset by removing whitespace-only texts
    data = data[data['text'].str.strip() != ""]
    # Correct the typo in 'Portuguese'
    data['language'] = data['language'].replace('Portugese', 'Portuguese')
    # Map language names to ISO 639-3 codes
    mapping_df = pd.read_csv("../data/lang_codes.csv")
    iso_639_3_codes = dict(zip(mapping_df['language'], mapping_df['lang_code']))
    data['lang_code'] = data['language'].map(iso_639_3_codes)

    return data
