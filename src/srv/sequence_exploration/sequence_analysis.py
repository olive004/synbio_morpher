

import logging
import os
import pandas as pd


from src.srv.io.loaders.data_loader import DataLoader
from src.srv.io.results.writer import DataWriter
from src.srv.parameter_prediction.interactions import InteractionMatrix
from src.utils.misc.io import get_pathnames


def generate_interaction_stats(path_name, writer: DataWriter, **stat_addons):

    interactions = InteractionMatrix(matrix_path=path_name)

    stats = interactions.get_stats()
    add_stats = pd.DataFrame.from_dict({'path': [path_name]})
    stats = pd.concat([stats, add_stats], axis=1)

    writer.output(out_type='csv', out_name='circuit_stats', data=stats, write_master=False)

    return stats

    # note sequence mutation method


def pull_circuits_from_stats(stats_pathname, filters: dict, write_key='data_path') -> list:

    stats = DataLoader().load_data(stats_pathname).data

    filt_stats = stats[stats['num_interacting']
                       >= filters.get("min_num_interacting")]
    filt_stats = filt_stats[filt_stats['num_self_interacting'] < filters.get(
        "max_self_interacting")]

    circuit_names = sorted(filt_stats["name"].tolist())
    circuit_folder = os.path.dirname(filt_stats['path'].to_list()[0])

    circuit_paths = []
    for name in circuit_names:
        circuit = {"data_path": get_pathnames(circuit_folder, name, first_only=True)}
        circuit_paths.append(circuit)
    return circuit_paths
