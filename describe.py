from load_csv import load
import ft_stat
import pandas as pd
import sys


def main():
    try:
        if len(sys.argv) > 2:
            raise AssertionError("Incorrect number of arguments")
        dataset = load(f'data/{sys.argv[1]}')
        if dataset is None:
            return
    except AssertionError as error:
        print(AssertionError.__name__ + ":", error)
        return

    numerical_dataset = dataset.select_dtypes(include=['int', 'float'])

    d = {}
    for column in numerical_dataset:
        if column == "Index" or column == "Best Hand":
            continue
        args = ft_stat.clean_list(list(numerical_dataset[column]))
        count = len(args)
        mean = ft_stat.ft_mean(args)
        std = (ft_stat.ft_var(args)) ** 0.5
        min, max = ft_stat.ft_min_max(args)
        first_q, third_q = ft_stat.ft_quartile(args)
        median = ft_stat.ft_median(args)
        d[column] = [count, mean, std, min, first_q, median, third_q, max]
    index = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    df = pd.DataFrame(d, index)
    print(df)
    print(numerical_dataset.describe())


if __name__ == "__main__":
    main()
