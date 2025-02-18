import click
from sklearn.preprocessing import normalize

from langlens.baseline.classifier import NaiveBayes
from langlens.baseline.vectorize import vectorize_data
from langlens.configuration.config import get_logger
from langlens.data import load_and_split
from langlens.evaluation import evaluate_model

log = get_logger(__name__)


@click.group()
def cli():
    """langlens -- A language identification tool."""
    pass


@cli.command(help="Run the baseline model on the provided dataset.")
@click.option('--dataset-path', required=True, type=click.Path(exists=True), help="Path to the dataset CSV file.")
@click.option("--preprocess", is_flag=True, help="Flag to indicate whether to preprocess the text data.")
@click.option('--n-gram-type', type=click.Choice(['word', 'char']), help="Type of n-gram to use for vectorization.")
@click.option('--vocab-size', type=int, help="Size of the vocabulary to use for vectorization.")
def baseline(dataset_path: str, preprocess: bool, n_gram_type: str, vocab_size: int):
    """
    Run the baseline model on the provided dataset.

    Args:
        dataset_path (str): Path to the dataset CSV file.
        preprocess (bool): Flag to indicate whether to preprocess the text data.
        n_gram_type (str): Type of n-gram to use for vectorization.
        vocab_size (int): Size of the vocabulary to use for vectorization.
    """
    log.info(f"Loading dataset from {dataset_path}")
    train_data, val_data, test_data = load_and_split(dataset_path)
    log.info(f"Training dataset size: {len(train_data)}, Val/Test dataset size: {len(val_data)}")

    if preprocess:
        log.info("Preprocessing data...")
        train_data.x, val_data.x, test_data.x = map(preprocess, [train_data.x, val_data.x, test_data.x])

    log.info("Vectorizing and normalizing data...")
    train_data.x, val_data.x, test_data.x = vectorize_data(train_data, val_data, test_data, n_gram_type, vocab_size)
    train_data.x, val_data.x, test_data.x = map(normalize, [train_data.x, val_data.x, test_data.x])

    log.info("Training classifier...")
    model = NaiveBayes()
    model.fit(train_data.x, train_data.y)

    language_set = set(train_data.y) | set(val_data.y) | set(test_data.y)
    log.info("Validation set performance:")
    y_val_pred = model.predict(val_data.x)
    evaluate_model(y_val_pred, train_data, val_data, language_set)
    log.info("Test set performance:")
    y_test_pred = model.predict(test_data.x)
    evaluate_model(y_test_pred, train_data, test_data, language_set)


if __name__ == "__main__":
    cli()
