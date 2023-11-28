import numpy as np
import pandas as pd
from load_csv import load
from ft_stat import normalize
import argparse
import sys
from tqdm import tqdm


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class LogisticRegression():
    def __init__(self, lr=0.05, n_iters=5000, pbar=None):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.pbar = pbar

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
            if self.pbar is not None:
                self.pbar.update(1)
        return self.weights, self.bias

    def predict(self, matrice_features):
        linear_pred = np.dot(matrice_features, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)
        return y_pred


def parse_arguments():
    parser = argparse.ArgumentParser(prog="logreg",
                                    description="A program that trains\
                                a logistic regression model on provided data")
    parser.add_argument('file', metavar="FILE", type=str,
                        help='enter the chosen file for training')
    args = parser.parse_args()
    return args


def accuracy(y_pred, y_test):
    return np.sum(y_pred==y_test)/len(y_test)

    
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
    pred = {}
    X_train = cleaned_df.drop(columns=['House'])
    for lr in [0.005, 0.01, 0.02, 0.05, 0.1, 0.5]:
        for nb_iters in [1000, 2000, 5000, 10000, 20000]:
            with tqdm(total=len(cleaned_df["House"].unique()) * nb_iters,
                    desc="Model training") as pbar:
                for house in cleaned_df["House"].unique():
                    y_train = cleaned_df["House"].map(lambda x: x == house)
                    clf = LogisticRegression(lr=lr, n_iters=nb_iters, pbar=pbar)
                    weights, bias = clf.fit(X_train, y_train)
                    results[house] = np.append(weights, bias)
                    pred[house] = clf.predict(X_train)
            index = list(X_train.columns)
            index.append("Bias")
            res = pd.DataFrame(results, index=index)
            res.index.name = "Subject"
            max_values = np.amax(np.array([pred["Gryffindor"], pred["Slytherin"], pred["Ravenclaw"], pred["Hufflepuff"]]), axis=0)
            res = []
            for i, x in enumerate(max_values):
                if x == pred["Gryffindor"][i]:
                    res.append("Gryffindor")
                elif x == pred["Ravenclaw"][i]:
                    res.append("Ravenclaw")
                elif x == pred["Slytherin"][i]:
                    res.append("Slytherin")
                elif x == pred["Hufflepuff"][i]:
                    res.append("Hufflepuff")
            print(f"accuracy with lr={lr} and nb_iters={nb_iters}: {accuracy(res, cleaned_df['House'])}")
    # try:
    #     res.to_csv("weights.csv")
    # except Exception as e:
    #     print(f"Error when exporting res: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
