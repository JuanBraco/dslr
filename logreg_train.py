import numpy as np
import pandas as pd
from load_csv import load
from ft_stat import standardization
import argparse
import sys
from tqdm import tqdm


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class LogisticRegression():
    def __init__(self, house, lr=0.05, n_iters=5000, pbar=None, testing=False):
        self.house = house
        self.lr = lr
        self.n_iters = n_iters
        self.testing = testing
        self.weights = None
        self.bias = None
        self.pbar = pbar
        self.accuracy = [0, 0]
        self.results = {}

    def predict_binary(self, X_test):
        linear_pred = np.dot(X_test, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)
        y_pred = [x >= 0.5 for x in y_pred]
        return y_pred

    def predict(self, X_test):
        linear_pred = np.dot(X_test, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)
        return y_pred

    @staticmethod
    def compute_accuracy(y_pred, y_test):
        return np.sum(y_pred == y_test) / len(y_test)

    def fit(self, X_train, y_train, X_test, y_test):
        n_samples, n_features = X_train.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for i in range(self.n_iters):
            linear_pred = np.dot(X_train, self.weights) + self.bias
            logistic_pred = sigmoid(linear_pred)

            test_pred = self.predict_binary(X_test)
            accuracy = LogisticRegression.compute_accuracy(test_pred, y_test)
            if accuracy > self.accuracy[0]:
                self.results["weights"] = self.weights
                self.results["bias"] = self.bias
                self.accuracy[0], self.accuracy[1] = accuracy, i
            dw = (1 / n_samples) * np.dot(X_train.T,
                                          (logistic_pred - y_train))
            db = (1 / n_samples) * np.sum(logistic_pred - y_train)

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db
            if self.pbar is not None:
                self.pbar.update(1)
        if self.testing:
            print(f"max accuracy for {self.house}: {self.accuracy[0]} "
                  f"reached in {self.accuracy[1]} iterations")
        # reset weights and bias to optimal values
        self.weights = self.results["weights"]
        self.bias = self.results["bias"]
        return self.results


def parse_arguments():
    parser = argparse.ArgumentParser(prog="logreg",
                                     description="A program that trains\
                                a logistic regression model on provided data")
    parser.add_argument('file', metavar="FILE", type=str,
                        help='enter the chosen file for training')
    parser.add_argument('-t', action='store_true', help="testing mode")
    args = parser.parse_args()
    return args


def test_model_accuracy(pred, y_test):
    max_values = np.amax(np.array([pred["Gryffindor"], pred["Slytherin"],
                                   pred["Ravenclaw"], pred["Hufflepuff"]]),
                         axis=0)
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
    print(f"model accuracy: "
          f"{LogisticRegression.compute_accuracy(res, y_test)}")


def main():
    args = parse_arguments()
    dataset = load(args.file)
    if dataset is None:
        print("Error when loading dataset", file=sys.stderr)
        return
    try:
        numerical_dataset = dataset.select_dtypes(include=['int', 'float'])
        normalized_df = numerical_dataset.transform(standardization)
        normalized_df["House"] = dataset["Hogwarts House"]
        # remove columns not useful for house predictions
        normalized_df.drop(columns=["Arithmancy", "Care of Magical Creatures",
                                    "Defense Against the Dark Arts"],
                           inplace=True)
        # drop rows with nan values
        cleaned_df = normalized_df.dropna().reset_index(drop=True)
        nb_iters = 5000
        lr = 0.05
        results = {}
        preds = {}
        train_df = cleaned_df.sample(frac=0.8, random_state=100)
        test_df = cleaned_df.drop(train_df.index)
        X_train = train_df.drop(columns=['House'])
        X_test = test_df.drop(columns=['House'])
        with tqdm(total=len(cleaned_df["House"].unique()) * nb_iters,
                  desc="Model training") as pbar:
            for house in cleaned_df["House"].unique():
                y_train = train_df["House"].map(lambda x: x == house)
                y_test = test_df["House"].map(lambda x: x == house)
                clf = LogisticRegression(house=house, n_iters=nb_iters,
                                         lr=lr, pbar=pbar, testing=args.t)
                res = clf.fit(X_train, y_train, X_test, y_test)
                results[house] = np.append(res["weights"], res["bias"])
                preds[house] = clf.predict(X_test)
        if args.t:
            print(f"lr: {lr}, nb_iters: {nb_iters}")
            test_model_accuracy(preds, test_df["House"])
        index = list(X_train.columns)
        index.append("Bias")
        res = pd.DataFrame(results, index=index)
        res.index.name = "Subject"
        res.to_csv("weights.csv")
    except Exception as e:
        print(f"{Exception.__name__}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
