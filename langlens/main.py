import click
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC

from langlens.configuration.config import get_logger
from langlens.data import load, split, Dataset
from langlens.data import preprocess as preprocess_data
from langlens.evaluation import evaluate_model_performance
from langlens.vectorize import vectorize_data

log = get_logger(__name__)


@click.group()
def cli():
    """langlens -- A language identification tool."""
    pass


@cli.command(help="Run the model on the provided dataset.")
@click.option('--dataset-path', required=True, type=click.Path(exists=True), help="Path to the dataset CSV file.")
@click.option("--preprocess", is_flag=True, help="Flag to indicate whether to preprocess the text data.")
@click.option("--val-for-training", is_flag=True, help="Flag to include the validation set for training.")
@click.option('--n-gram-type', type=click.Choice(['word', 'char']), help="Type of n-gram to use for vectorization.")
@click.option('--vocab-size', type=int, help="Size of the vocabulary to use for vectorization.")
def train(dataset_path: str, preprocess: bool, val_for_training: bool, n_gram_type: str, vocab_size: int):
    """
    Run the baseline model on the provided dataset.

    Args:
        dataset_path (str): Path to the dataset CSV file.
        preprocess (bool): Flag to indicate whether to preprocess the text data.
        val_for_training (bool): Flag to include the validation set for training.
        n_gram_type (str): Type of n-gram to use for vectorization.
        vocab_size (int): Size of the vocabulary to use for vectorization.
    """
    log.info("Loading dataset from %s", dataset_path)
    data_frame = load(dataset_path)

    if preprocess:
        log.info("Preprocessing data...")
        data_frame = preprocess_data(data_frame)

    train_data, val_data, test_data = split(data_frame)
    if val_for_training:
        log.info("Including validation set for training.")
        train_data = Dataset(np.concatenate([train_data.x, val_data.x]), np.concatenate([train_data.y, val_data.y]))
        val_data = test_data
    log.info("Training dataset size: %d, Val/Test dataset size: %d", len(train_data), len(val_data))

    log.info("Vectorizing and normalizing data...")
    train_data.x, val_data.x, test_data.x = vectorize_data(train_data, val_data, test_data, n_gram_type, vocab_size)
    train_data.x, val_data.x, test_data.x = map(normalize, [train_data.x, val_data.x, test_data.x])

    log.info("Training classifier...")
    model = LinearSVC()
    model.fit(train_data.x, train_data.y)

    evaluate_model_performance(model, train_data, val_data, test_data)


if __name__ == "__main__":
    cli()
