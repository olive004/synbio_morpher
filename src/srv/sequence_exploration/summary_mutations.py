

import pandas as pd


def summarise_mutation_groups(data: pd.DataFrame):

    data.groupby('circuit_name').agg()


if __name__ == "main":
    path = 'data/ensemble_mutation_effect_analysis/2022_12_14_183947/summarise_simulation/tabulated_mutation_info.json'
    data = pd.read_json(path)
    summarise_mutation_groups
    pass