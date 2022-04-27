

from copy import deepcopy
import logging
import os
import pandas as pd


from src.srv.io.loaders.data_loader import DataLoader
from src.srv.io.results.writer import DataWriter
from src.srv.parameter_prediction.interactions import InteractionMatrix
from src.utils.misc.io import get_pathname_by_search_str


def generate_interaction_stats(path_name, writer: DataWriter, **stat_addons):

    interactions = InteractionMatrix(matrix_path=path_name)

    stats = interactions.get_stats()
    add_stats = pd.DataFrame.from_dict({'path': [path_name]})
    stats = pd.concat([stats, add_stats], axis=1)

    writer.output(out_type='csv', out_name='circuit_stats', data=stats, write_master=False)

    return stats

    # note sequence mutation method


def pull_circuits_from_stats(stats_pathname, filters: dict, write_to: dict = None) -> list:
    stats = DataLoader().load_data(stats_pathname).data

    filt_stats = stats[stats['num_interacting']
                       >= filters.get("min_num_interacting")]
    filt_stats = filt_stats[filt_stats['num_self_interacting'] < filters.get(
        "max_self_interacting")]

    circuit_names = sorted(filt_stats["name"].tolist())

    base_folder = os.path.dirname(stats_pathname)
    circuits = []
    for name in circuit_names:
        circuit = deepcopy(write_to)
        circuit["data_path"] = get_pathname_by_search_str(base_folder, name)
        # logging.info(circuit["data_path"])
        circuits.append(circuit)
