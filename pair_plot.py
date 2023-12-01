from load_csv import load
from ft_stat import standardization
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import argparse
import os


def parse_arguments():
    parser = argparse.ArgumentParser(prog="pairplot",
                                     description="A program that analyzes\
                                       relations between scores in different\
                                        courses for hogwart students")
    parser.add_argument('--features', metavar="GRAPH_TYPE", type=int,
                        choices=range(2), default=0, required=False,
                        help='enter the chosen filter for values:\
    0: all, 1: only relevant')
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    dataset = load("data/dataset_train.csv")
    if dataset is None:
        print("Error when loading dataset", file=sys.stderr)
        return
    numerical_dataset = dataset.select_dtypes(include=['int', 'float'])
    normalized_df = numerical_dataset.transform(standardization)
    normalized_df["House"] = dataset["Hogwarts House"]
    if args.features == 1:
        '''we can drop Arithmancy and Care of Magical Creatures as we saw\
            in histogram that they have an homogeneous distribution among\
            houses and are therefore not relevant to predict the house.\
            We can also drop one of Astronomy or Defense against the Dark Arts\
            as they are perfectly correlated'''
        normalized_df.drop(["Arithmancy", "Care of Magical Creatures",
                            "Defense Against the Dark Arts"], axis=1,
                           inplace=True)
    sns.pairplot(normalized_df, hue="House",
                 diag_kind="hist",
                 diag_kws={'multiple': 'stack',
                           'stat': 'count',
                           'common_norm': True})
    plt.tight_layout()
    try:
        if not os.path.exists("graphs/"):
            os.mkdir("graphs")
        plt.savefig('graphs/pairplot.png', dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"Error saving the image: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
