
# Copyright (c) 2023, Olivia Gallup
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree. 
    


import pandas as pd
from synbio_morpher.utils.results.analytics.naming import get_true_names_analytics
from synbio_morpher.utils.results.writer import DataWriter


def summarise_mutation_groups(data: pd.DataFrame, data_writer: DataWriter = None):

    # path = 'data/ensemble_mutation_effect_analysis/2022_12_15_012941/summarise_simulation/tabulated_mutation_info.json'
    # data = pd.read_json(path)
 
    analytics = get_true_names_analytics(data.columns)
    summary = data.groupby(['circuit_name', 'sample_name', 'mutation_num']).agg({k: ['mean', 'std'] for k in analytics})

    if data_writer:
        data_writer.output(
            out_type='csv', out_name='mutation_means_stds', **{'data': summary})
    return summary.reset_index()


if __name__ == "__main__":
    # local: 2022_12_14_183947
    # remote: 2022_12_15_012941
    pass