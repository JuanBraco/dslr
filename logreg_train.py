import numpy as np
import pandas as pd
from load_csv import load
from ft_stat import normalize
import argparse
import sys


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class LogisticRegression():
    def __init__(self, lr=0.05, n_iters=5000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, features_matrix, training_results):
        n_samples, n_features = features_matrix.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_pred = np.dot(features_matrix, self.weights) + self.bias
            logistic_pred = sigmoid(linear_pred)

            dw = (1 / n_samples) * np.dot(features_matrix.T,
                                          (logistic_pred - training_results))
            db = (1 / n_samples) * np.sum(logistic_pred - training_results)

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db
        return self.weights, self.bias


def parse_arguments():
    parser = argparse.ArgumentParser(prog="logreg",
                                     description="A program that trains\
                                a logistic regression model on provided data")
    parser.add_argument('file', metavar="FILE", type=str,
                        help='enter the chosen file for training')
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    dataset = load(args.file)
    if dataset is None:
        print("Error when loading dataset", file=sys.stderr)
        return
    numerical_dataset = dataset.select_dtypes(include=['int', 'float'])
    normalized_df = numerical_dataset.transform(normalize)
    normalized_df["House"] = dataset["Hogwarts House"]
    # remove columns not useful for house predictions
    normalized_df.drop(columns=["Arithmancy", "Care of Magical Creatures",
                                "Defense Against the Dark Arts"], inplace=True)
    # drop rows with nan values
    cleaned_df = normalized_df.dropna().reset_index(drop=True)
    results = {}
    X_train = cleaned_df.drop(columns=['House'])
    for house in cleaned_df["House"].unique():
        y_train = cleaned_df["House"].map(lambda x: x == house)
        clf = LogisticRegression()
        weights, bias = clf.fit(X_train, y_train)
        results[house] = np.append(weights, bias)
    index = list(X_train.columns)
    index.append("Bias")
    res = pd.DataFrame(results, index=index)
    res.index.name = "Subject"
    try:
        res.to_csv("weights.csv")
    except Exception as e:
        print(f"Error when exporting res: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
