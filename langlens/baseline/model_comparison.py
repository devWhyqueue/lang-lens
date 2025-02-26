import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from langlens.data import preprocess, load, split
from vectorize import vectorize_data


def load_and_prepare_data():
    """
    Load, preprocess, and split the dataset into training, validation, and test sets.

    :return: A tuple (train_data, val_data, test_data) containing the dataset splits.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    # Change working directory so that _clean_data's relative paths work
    os.chdir(os.path.join(project_root, "langlens"))
    dataset_path = os.path.join(project_root, "data", "wili_subset_khan.csv")
    df = load(dataset_path)
    df = preprocess(df)
    train_data, val_data, test_data = split(df)
    return train_data, val_data, test_data


def run_experiment(model, model_name, train_data, val_data, test_data, ngram, vocab):
    """
    Run a single experiment for a given model, feature setting, and vocabulary size.

    :param model: The scikit-learn model instance to train.
    :param model_name: A string identifier for the model (e.g., "SVM", "Naive Bayes").
    :param train_data: The training dataset.
    :param val_data: The validation dataset.
    :param test_data: The test dataset.
    :param ngram: The n-gram type to use (e.g., "word" or "char").
    :param vocab: The maximum number of features (vocabulary size) to extract.
    """
    # Vectorize data
    X_train, _, X_test = vectorize_data(train_data, val_data, test_data, ngram, vocab)

    # Train and predict
    model.fit(X_train, train_data.y)
    y_pred = model.predict(X_test)

    print(f"\n{'-' * 40}")
    print(f"Model: {model_name} | N-gram: {ngram} | Vocab Size: {vocab}")
    print(classification_report(test_data.y, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(test_data.y, y_pred))

    # PCA visualization of decision boundaries
    pca = PCA(n_components=2)
    X_test_pca = pca.fit_transform(X_test)

    unique_labels = sorted(set(test_data.y))
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
    colors = np.array([label_to_int[label] for label in test_data.y])

    cmap = plt.get_cmap('viridis', len(unique_labels))

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=colors, cmap=cmap, alpha=0.7)
    plt.title(f"PCA: {model_name} ({ngram}, Vocab: {vocab})")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")

    # Define colorbar with language labels
    cbar = plt.colorbar(scatter, ticks=np.arange(len(unique_labels)))
    cbar.set_label('Language Labels')
    cbar.set_ticklabels(unique_labels)

    plt.show(block=False)
    plt.pause(2)
    plt.close('all')

    # Error Analysis: Display a few misclassified examples
    misclassified_indices = [i for i, (true, pred) in enumerate(zip(test_data.y, y_pred)) if true != pred]
    misclassified_texts = test_data.x[misclassified_indices]
    misclassified_true = test_data.y[misclassified_indices]
    misclassified_pred = y_pred[misclassified_indices]

    print("Sample Misclassified Examples:")
    for text, true_label, pred_label in zip(misclassified_texts[:5], misclassified_true[:5], misclassified_pred[:5]):
        print(f"Text: {text}\nTrue: {true_label} | Predicted: {pred_label}\n")


def run_naive_bayes(ngram_types=None):
    """
    Run experiments for the Naive Bayes classifier over various n-gram types and vocabulary sizes.

    For word n-grams, test with vocab sizes [250, 500, 1000, 64000].
    For char n-grams, test with vocab sizes [250, 500, 1000, 6838].

    :param ngram_types: A list of n-gram types to experiment with (default: ["word", "char"]).
    """
    if ngram_types is None:
        ngram_types = ["word", "char"]

    train_data, val_data, test_data = load_and_prepare_data()
    for ngram in ngram_types:
        if ngram == "word":
            vocab_sizes = [250, 500, 1000, 64000]
        elif ngram == "char":
            vocab_sizes = [250, 500, 1000, 6838]
        else:
            raise ValueError(f"Unexpected ngram type: {ngram}")

        for vocab in vocab_sizes:
            model = MultinomialNB()
            run_experiment(model, "Naive Bayes", train_data, val_data, test_data, ngram, vocab)


def run_svm(ngram_types=None):
    """
    Run experiments for the SVM classifier over various n-gram types and vocabulary sizes.
    For word n-grams, test with vocab sizes [250, 500, 1000, 64000].
    For char n-grams, test with vocab sizes [250, 500, 1000, 6838].

    :param ngram_types: A list of n-gram types to experiment with (default: ["word", "char"]).
    """
    if ngram_types is None:
        ngram_types = ["word", "char"]

    train_data, val_data, test_data = load_and_prepare_data()

    for ngram in ngram_types:
        # Select vocabulary sizes based on the ngram type
        if ngram == "word":
            vocab_sizes = [250, 500, 1000, 64000]
        elif ngram == "char":
            vocab_sizes = [250, 500, 1000, 6838]
        else:
            raise ValueError(f"Unexpected ngram type: {ngram}")

        for vocab in vocab_sizes:
            model = SVC(probability=True)
            run_experiment(model, "SVM", train_data, val_data, test_data, ngram, vocab)


def run_logistic_regression(ngram_types=None):
    """
    Run experiments for the Logistic Regression classifier over various n-gram types and vocabulary sizes.
    For word n-grams, test with vocab sizes [250, 500, 1000, 64000].
    For char n-grams, test with vocab sizes [250, 500, 1000, 6838].

    :param ngram_types: A list of n-gram types to experiment with (default: ["word", "char"]).
    """
    if ngram_types is None:
        ngram_types = ["word", "char"]

    train_data, val_data, test_data = load_and_prepare_data()

    for ngram in ngram_types:
        if ngram == "word":
            vocab_sizes = [250, 500, 1000, 64000]
        elif ngram == "char":
            vocab_sizes = [250, 500, 1000, 6838]
        else:
            raise ValueError(f"Unexpected ngram type: {ngram}")

        for vocab in vocab_sizes:
            model = LogisticRegression(max_iter=2000)
            run_experiment(model, "Logistic Regression", train_data, val_data, test_data, ngram, vocab)



if __name__ == '__main__':
    # To test only Naive Bayes:
    #run_naive_bayes()

    # To test only SVM:
    #run_svm()

    # To test only Logistic Regression:
    run_logistic_regression()
