import numpy as np
import pandas as pd
from load_csv import load
from ft_stat import standardization
import argparse
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class LogisticRegression():
    def __init__(self, house, lr=0.001, n_iters=1000, pbar=None,
                 testing=False):
        self.house = house
        self.lr = lr
        self.n_iters = n_iters
        self.testing = testing
        self.weights = None
        self.bias = None
        self.pbar = pbar
        self.loss = [float('inf'), 0]
        self.loss_evolution = []
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

    @staticmethod
    def compute_loss(y_pred, y_test):
        y_test, y_pred = np.array(y_test), np.array(y_pred)
        loss_vector = (y_test * np.log(y_pred) +
                       (1 - y_test) * np.log(1 - y_pred))
        return -1 / len(y_test) * sum(loss_vector)

    def gradient_descent(self, X_train, n_samples, y_train):
        linear_pred = np.dot(X_train, self.weights) + self.bias
        logistic_pred = sigmoid(linear_pred)
        dw = (1 / n_samples) * np.dot(X_train.T,
                                      (logistic_pred - y_train))
        db = (1 / n_samples) * np.sum(logistic_pred - y_train)

        self.weights = self.weights - self.lr * dw
        self.bias = self.bias - self.lr * db

    def stochastic_gradient_descent(self, X_train, n_samples, y_train):
        sample = random.randrange(n_samples)
        sample_features = X_train.iloc[sample]
        sample_result = y_train.iloc[sample]
        linear_pred = np.dot(sample_features, self.weights) + self.bias
        logistic_pred = sigmoid(linear_pred)
        dw = np.dot(sample_features, (logistic_pred - sample_result))
        db = logistic_pred - sample_result

        self.weights = self.weights - self.lr * dw
        self.bias = self.bias - self.lr * db

    def minibatch_gradient_descent(self, X_train, y_train):
        # minibatch are often a power of 2 (most used are 32, 64, 128)
        minibatch_size = 32
        sample_features = X_train.sample(n=minibatch_size).sort_index()
        sample_result = y_train[y_train.index.isin(sample_features.index)]
        linear_pred = np.dot(sample_features, self.weights) + self.bias
        logistic_pred = sigmoid(linear_pred)
        dw = (1 / minibatch_size) * np.dot(sample_features.T,
                                           (logistic_pred - sample_result))
        db = (1 / minibatch_size) * np.sum(logistic_pred - sample_result)

        self.weights = self.weights - self.lr * dw
        self.bias = self.bias - self.lr * db

    def fit(self, X_train, y_train, X_test, y_test, type="normal"):
        n_samples, n_features = X_train.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for i in range(self.n_iters):
            test_pred = self.predict(X_test)
            loss = LogisticRegression.compute_loss(test_pred, y_test)
            self.loss_evolution.append(loss)
            if loss < self.loss[0]:
                self.results["weights"] = self.weights
                self.results["bias"] = self.bias
                self.loss[0], self.loss[1] = loss, i

            if type == "normal":
                self.gradient_descent(X_train, n_samples, y_train)
            elif type == "stochastic":
                self.stochastic_gradient_descent(X_train, n_samples, y_train)
            elif type == "minibatch":
                self.minibatch_gradient_descent(X_train, y_train)
            if self.pbar is not None:
                self.pbar.update(1)
        if self.testing:
            print(f"min loss for {self.house}: {self.loss[0]} "
                  f"reached in {self.loss[1]} iterations")
        # reset weights and bias to optimal values
        self.weights = self.results["weights"]
        self.bias = self.results["bias"]
        return self.results

    def get_loss_evolution(self):
        return self.loss_evolution


def parse_arguments():
    parser = argparse.ArgumentParser(prog="logreg",
                                     description="A program that trains\
                                a logistic regression model on provided data")
    parser.add_argument('file', metavar="FILE", type=str,
                        help='enter the chosen file for training')
    parser.add_argument('-t', action='store_true', help="testing mode")
    parser.add_argument('--gradient', metavar="GRADIENT_TYPE", type=str,
                        choices=["normal", "stochastic", "minibatch"],
                        default="normal",
                        help="select the chosen mode for gradient descent:\
                              normal or stochastic")
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


def plot_loss(losses, type):
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    fig.suptitle(f"Loss evolution over iterations for each house classifier,"
                 f"using {type} gradient descent")
    i, j = 0, 0
    for house in losses.keys():
        sns.lineplot(ax=axes[i][j], data=losses[house])
        axes[i][j].set(xlabel="Iterations number", ylabel="Loss in %",
                       title=house)
        if j == 1:
            j = 0
            i += 1
        else:
            j += 1
    plt.tight_layout()
    if not os.path.exists("graphs/"):
        os.mkdir("graphs")
    plt.savefig(f'graphs/loss_{type}.png', dpi=300, bbox_inches='tight')


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
        losses = {}
        train_df = cleaned_df.sample(frac=0.8, random_state=100).sort_index()
        test_df = cleaned_df.drop(train_df.index)
        X_train = train_df.drop(columns=['House'])
        X_test = test_df.drop(columns=['House'])
        with tqdm(total=len(cleaned_df["House"].unique()) * nb_iters,
                  desc="Model training") as pbar:
            for house in cleaned_df["House"].unique():
                y_train = train_df["House"].map(lambda x: 1
                                                if x == house else 0)
                y_test = test_df["House"].map(lambda x: 1 if x == house else 0)
                clf = LogisticRegression(house=house, n_iters=nb_iters,
                                         lr=lr, pbar=pbar, testing=args.t)
                res = clf.fit(X_train, y_train, X_test, y_test, args.gradient)
                results[house] = np.append(res["weights"], res["bias"])
                preds[house] = clf.predict(X_test)
                losses[house] = clf.get_loss_evolution()
        if args.t:
            print(f"lr: {lr}, nb_iters: {nb_iters}")
            test_model_accuracy(preds, test_df["House"])
            plot_loss(losses, args.gradient)
        index = list(X_train.columns)
        index.append("Bias")
        res = pd.DataFrame(results, index=index)
        res.index.name = "Subject"
        res.to_csv("weights.csv")
    except Exception as e:
        print(f"{Exception.__name__}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
