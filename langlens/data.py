import re
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


def load(dataset_path: str) -> pd.DataFrame:
    """
    Load a dataset from a CSV file and clean it.

    Args:
        dataset_path (str): The path to the CSV file containing the dataset.

    Returns:
        pd.DataFrame: The cleaned dataset.
    """
    df = pd.read_csv(dataset_path)
    df = _clean_data(df)
    return df

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


def split(df: pd.DataFrame) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Split the dataset into training, validation, and test sets.

    Args:
        df (pd.DataFrame): The cleaned dataset.

    Returns:
        Tuple[Dataset, Dataset, Dataset]: A tuple containing the training, validation, and test datasets.
    """
    x, y = df['text'], df['language']
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.2, random_state=settings['seed'])
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=settings['seed'])

    return (
        Dataset(x_train.values, y_train.values),
        Dataset(x_val.values, y_val.values),
        Dataset(x_test.values, y_test.values)
    )


def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the dataset by removing noise from the text data.

    Args:
        data (pd.DataFrame): The input dataset containing a 'text' column.

    Returns:
        pd.DataFrame: The preprocessed dataset with cleaned text data.
    """

    def remove_noise(text):
        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)
        # Remove emails
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', '', text)
        # Remove digits
        text = re.sub(r'\d+', '', text)
        # Remove punctuation and special characters (excluding spaces)
        text = re.sub(r'[^\w\s]', '', text)
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    data['text'] = data['text'].apply(remove_noise)
    return data
