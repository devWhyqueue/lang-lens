import numpy as np
import pandas as pd
import seaborn as sn
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix

from langlens.configuration.config import get_logger, settings
from langlens.data import Dataset

log = get_logger(__name__)


def evaluate_model_performance(model: object, train_data: Dataset, val_data: Dataset, test_data: Dataset) -> None:
    """
    Evaluates the model performance on training, validation, and test datasets.

    Args:
        model (object): The trained model.
        train_data (Dataset): Training dataset.
        val_data (Dataset): Validation dataset.
        test_data (Dataset): Test dataset.
    """
    language_set = set(train_data.y) | set(val_data.y) | set(test_data.y)

    log.info("Training set performance:")
    y_train_pred = model.predict(train_data.x)
    _evaluate(y_train_pred, train_data, train_data, language_set)

    log.info("Validation set performance:")
    y_val_pred = model.predict(val_data.x)
    _evaluate(y_val_pred, train_data, val_data, language_set)

    log.info("Test set performance:")
    y_test_pred = model.predict(test_data.x)
    _evaluate(y_test_pred, train_data, test_data, language_set)


def _evaluate(y_pred: np.ndarray, train_data: Dataset, test_data: Dataset, language_set: set) -> None:
    """
    Evaluates the model by logging the classification report, plotting the confusion matrix,
    and plotting the PCA projection of the test data.

    Args:
        y_pred (np.ndarray): Predicted labels.
        train_data (Dataset): Training dataset.
        test_data (Dataset): Test dataset.
        language_set (set): Set of language labels.
    """
    log_classification_report(y_pred, test_data.y)
    plot_confusion_matrix(y_pred, test_data.y)
    plot_pca(train_data.x, test_data.x, test_data.y, language_set)


def log_classification_report(y_pred: np.ndarray, y_true: np.ndarray) -> None:
    """
    Logs the classification report.

    Args:
        y_pred (np.ndarray): Predicted labels.
        y_true (np.ndarray): True labels.
    """
    report = classification_report(y_true, y_pred)
    log.info(report)


def plot_confusion_matrix(y_pred: np.ndarray, y_true: np.ndarray) -> None:
    """
    Plots the confusion matrix.

    Args:
        y_pred (np.ndarray): Predicted labels.
        y_true (np.ndarray): True labels.
    """
    unique_labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)

    df_cm = pd.DataFrame(cm, index=unique_labels, columns=unique_labels)
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'

    sn.set_theme(style="whitegrid", font_scale=0.8)
    plt.figure(figsize=(10, 8))
    sn.heatmap(df_cm, annot=True, fmt='g', cmap="Blues", annot_kws={"size": 12})
    plt.show()


def plot_pca(x_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, language_set: set) -> None:
    """
    Plots the PCA projection of the test data.

    Args:
        x_train (np.ndarray): Training features.
        x_test (np.ndarray): Test features.
        y_test (np.ndarray): Test labels.
        language_set (set): Set of language labels.
    """
    pca = PCA(n_components=2, random_state=settings['seed'])
    pca.fit(x_train)
    pca_test = pca.transform(x_test)

    log.info('Variance explained by PCA: %s', pca.explained_variance_ratio_)

    y_test_array = np.asarray(y_test)
    plt.figure(figsize=(8, 6))
    for lang in language_set:
        mask = (y_test_array == lang)
        plt.scatter(pca_test[mask, 0], pca_test[mask, 1], label=lang, alpha=0.7)

    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
    plt.grid(True)
    plt.tight_layout()
    plt.show()
