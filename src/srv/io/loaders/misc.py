import logging
import pandas as pd


def load_csv(filepath, load_as=None, return_header=False) -> pd.DataFrame:

    og_csv_file = pd.read_csv(filepath)
    if load_as == 'numpy':
        csv_file = og_csv_file.to_numpy()
    elif load_as == 'dict':
        csv_file = og_csv_file.to_dict()
    else:
        csv_file = og_csv_file
    if return_header:
        return csv_file, list(og_csv_file.columns)
    return csv_file
