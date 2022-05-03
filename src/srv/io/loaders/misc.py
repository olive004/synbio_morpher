import pandas as pd


def load_csv(filepath, load_as=None) -> pd.DataFrame:

    csv_file = pd.read_csv(filepath)
    if load_as == 'numpy':
        csv_file = csv_file.to_numpy()
    elif load_as == 'dict':
        csv_file = csv_file.to_dict()
    return csv_file
