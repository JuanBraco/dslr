import pandas as pd
import os


def load(path: str) -> pd.DataFrame:
    """
    Load a CSV dataset from the specified path and return it as a pandas
    DataFrame.

    Parameters:
    path (str): The path to the CSV file to be loaded.

    Returns:
    pd.DataFrame or None: The loaded dataset as a pandas DataFrame, or
    None if there was an error.
    """
    try:
        if not os.path.exists(path):
            raise AssertionError("The file doesnt exist")
        if not path.lower().endswith('.csv'):
            raise AssertionError("The file format is not .csv")
        dataset = pd.read_csv(path)
        print(f"Loading dataset of dimensions {dataset.shape}")
        return dataset
    except AssertionError as error:
        print(AssertionError.__name__ + ":", error)
        return None
