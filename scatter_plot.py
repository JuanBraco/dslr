import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from load_csv import load
import argparse
from ft_stat import normalize
import sys

graph_names = ["all_features", "similar_features"]
graph_types = {
    "all_features": {},
    "similar_features": {},
}


def parse_arguments():
    parser = argparse.ArgumentParser(prog="scatter plot",
                                     description="A program that plots\
                                        score of different school subject\
                                        with each other")
    parser.add_argument('--graph_type', metavar="GRAPH_TYPE", type=int,
                        choices=range(2), default=0, required=False,
                        help='enter the chosen type of graph:\
    0: all features, 1: the two similar features')
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    dataset = load("data/dataset_train.csv")
    if dataset is None:
        print("Error when loading dataset", file=sys.stderr)
        return
    numerical_dataset = dataset.select_dtypes(include=['int', 'float'])
    normalized_df = numerical_dataset.transform(normalize)

    normalized_df.rename(
        columns={'Defense Against the Dark Arts': 'Defense',
                 'Care of Magical Creatures': 'Care'}, inplace=True)

    column = normalized_df.columns
    n_features = len(column)
    total_plots = n_features * (n_features - 1) // 2
    width = 8
    height = total_plots // width + (total_plots % width != 0)

    graph_type = graph_names[args.graph_type]

    if graph_type == "all_features":
        fig, axes = plt.subplots(height, width, figsize=(40, 20))
        fig.suptitle("Scatter plot between features", fontsize=10, y=1)

        feature_pairs = list(itertools.combinations(range(n_features), 2))
        for idx, (i, j) in enumerate(feature_pairs):
            row = idx // width
            col = idx % width
            ax = axes[row, col]
            sns.scatterplot(ax=ax, data=normalized_df, x=column[i],
                            y=column[j], s=8)
            ax.set_title(f'{column[i]} vs {column[j]}', fontsize=6, pad=1)
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.tick_params(axis='both', which='major', labelsize=6, pad=1)
            # ax.set_aspect('equal', adjustable='box')

        for idx in range(len(feature_pairs), height * width):
            row = idx // width
            col = idx % width
            axes[row, col].axis('off')
    elif graph_type == "similar_features":
        normalized_df["House"] = dataset["Hogwarts House"]
        fig, ax = plt.subplots(figsize=(8, 8))
        fig.suptitle("Scatter plot of Astronomy vs Defense", fontsize=10)
        sns.scatterplot(ax=ax, data=normalized_df, x='Astronomy', y='Defense',
                        hue=normalized_df['House'], palette='tab10')
        ax.set_title('Astronomy vs Defense', fontsize=6, pad=1)
        ax.set_xlabel('Astronomy')
        ax.set_ylabel('Defense')
        ax.legend(title='House', fontsize='small', loc='upper right')

    plt.tight_layout(pad=2.0)
    plt.show()


if __name__ == "__main__":
    main()
