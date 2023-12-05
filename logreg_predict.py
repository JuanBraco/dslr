from load_csv import load
import numpy as np
import sys
import csv
from ft_stat import standardization
import seaborn as sns
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def predict(matrice_features, weights, bias):
    linear_pred = np.dot(matrice_features, weights) + bias
    y_pred = sigmoid(linear_pred)
    return y_pred


def main():
    try:
        if len(sys.argv) != 3:
            raise AssertionError("Incorrect number of arguments")
        dataset = load(sys.argv[1])
        weight = load(sys.argv[2])
        if dataset is None:
            print("Error when loading dataset or weight file", file=sys.stderr)
            return
    except AssertionError as error:
        print(AssertionError.__name__ + ":", error)
        return

    try:
        numerical_dataset = dataset.select_dtypes(include=['int', 'float'])

        stdized_df = numerical_dataset.transform(standardization).fillna(0)

        X_train = stdized_df.drop(columns=["Arithmancy",
                                           "Care of Magical Creatures",
                                           "Defense Against the Dark Arts",
                                           "Hogwarts House"])

        bias = weight.iloc[10]

        weight.drop(index=weight.index[-1], axis=0, inplace=True)

        pred_Gryffindor = predict(X_train, weight['Gryffindor'].values,
                                  bias['Gryffindor'])
        pred_Ravenclaw = predict(X_train, weight['Ravenclaw'].values,
                                 bias['Ravenclaw'])
        pred_Slytherin = predict(X_train, weight['Slytherin'].values,
                                 bias['Slytherin'])
        pred_Hufflepuff = predict(X_train, weight['Hufflepuff'].values,
                                  bias['Hufflepuff'])

        max_values = np.amax(np.array([pred_Gryffindor,
                                      pred_Ravenclaw,
                                      pred_Slytherin,
                                      pred_Hufflepuff]), axis=0)
        res = []
        for i, x in enumerate(max_values):
            if x == pred_Gryffindor[i]:
                res.append([i, "Gryffindor"])
            elif x == pred_Ravenclaw[i]:
                res.append([i, "Ravenclaw"])
            elif x == pred_Slytherin[i]:
                res.append([i, "Slytherin"])
            elif x == pred_Hufflepuff[i]:
                res.append([i, "Hufflepuff"])

        with open('houses.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Index', 'Hogwarts House'])
            writer.writerows(res)
    except Exception as e:
        print(f"{Exception.__name__}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
