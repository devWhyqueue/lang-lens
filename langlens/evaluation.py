import numpy as np
import pandas as pd
import seaborn as sn
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix

from langlens.configuration.config import get_logger, settings
from langlens.data import Dataset

log = get_logger(__name__)


def evaluate_model(y_pred: np.ndarray, train_data: Dataset, test_data: Dataset, language_set: set) -> None:
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

    plt.legend(loc="upper left")
    plt.grid(True)
    plt.show()
