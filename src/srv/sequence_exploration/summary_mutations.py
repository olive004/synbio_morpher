

import pandas as pd
from src.utils.results.analytics.timeseries import get_analytics_types_base, get_diffs, get_ratios


def summarise_mutation_groups(data: pd.DataFrame):

    path = 'data/ensemble_mutation_effect_analysis/2022_12_15_012941/summarise_simulation/tabulated_mutation_info.json'
    data = pd.read_json(path)
    summarise_mutation_groups
 
    analytics = get_analytics_types_base() + get_diffs(get_analytics_types_base) + get_ratios(get_analytics_types_base)
    data.groupby(['circuit_name', 'sample_name']).agg({k: ['mean', 'std'] for k in analytics})


if __name__ == "__main__":
    # local: 2022_12_14_183947
    # remote: 2022_12_15_012941
    pass