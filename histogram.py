from load_csv import load
from ft_stat import normalize
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import sys

graph_names = ["stacked_bars", "overlapped_bars", "overlapped_steps",
               "overlapped_density", "separated_density"]
graph_types = {
    "stacked_bars": {"element": "bars", "multiple": "stack",
                     "stat": "count", "common_norm": True},
    "overlapped_bars": {"element": "bars", "multiple": "layer",
                        "stat": "count", "common_norm": True},
    "overlapped_steps": {"element": "step", "multiple": "layer",
                         "stat": "count", "common_norm": True},
    "overlapped_density": {"element": "step", "multiple": "layer",
                           "stat": "density", "common_norm": False},
    "separated_density":  {"element": "bars", "multiple": "dodge",
                           "stat": "density", "common_norm": False}
}


def parse_arguments():
    parser = argparse.ArgumentParser(prog="histogram",
                                     description="A program that analyzes\
                                        score distribution between houses in\
                                        different school subject")
    parser.add_argument('--graph_type', metavar="GRAPH_TYPE", type=int,
                        choices=range(5), default=0, required=False,
                        help='enter the chosen type of graph:\
    0: stacked bars, 1: overlapped bars, 2: overlapped steps,\
    3: overlapped density, 4: separated density')
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
    normalized_df["House"] = dataset["Hogwarts House"]

    length = len(normalized_df.columns) - 1

    width = 4
    if length % width == 0:
        height = length // width
    else:
        height = length // width + 1

    fig, axes = plt.subplots(height, width, figsize=(18, 10))
    fig.suptitle("Score distribution per house for each subject")

    graph_type = graph_names[args.graph_type]
    element = graph_types[graph_type]["element"]
    multiple = graph_types[graph_type]["multiple"]
    stat = graph_types[graph_type]["stat"]
    norm = graph_types[graph_type]["common_norm"]

    for i in range(height):
        for j in range(width):
            cur = i * width + j
            if cur >= length:
                fig.delaxes(axes[i, j])
            else:
                ax = axes[i, j]
                col = normalized_df.columns[cur]
            if cur == 0:
                plot = sns.histplot(ax=ax, data=normalized_df, hue="House",
                                    x=col, bins=10, element=element,
                                    multiple=multiple, stat=stat,
                                    common_norm=norm)
                sns.move_legend(plot, loc="upper left")
            else:
                plot = sns.histplot(ax=ax, data=normalized_df, hue="House",
                                    x=col, bins=10, element=element,
                                    multiple=multiple, stat=stat,
                                    common_norm=norm, legend=False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
