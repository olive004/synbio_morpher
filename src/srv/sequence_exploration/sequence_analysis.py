

import pandas as pd


from src.srv.io.loaders.data_loader import DataLoader
from src.srv.io.results.writer import DataWriter
from src.srv.parameter_prediction.interactions import InteractionMatrix


def generate_interaction_stats(pathname, writer: DataWriter):

    interactions = InteractionMatrix(matrix_path=pathname)

    stats = interactions.get_stats()

    writer.output(out_type='csv', out_name='circuit_stats', data=stats)

    return stats

    # note sequence mutation method


def pull_circuits_from_stats(stats_pathname, filters: dict, write_to: dict = None):
    stats = DataLoader().load_data(stats_pathname)

    filt_stats = stats[stats['num_interacting']
                       >= filters.get("min_num_interacting")]
    # filt_stats = filt_stats[filt_stats['num_interacting'] <= filters.get(
    #     "max_num_interacting")]
    filt_stats = filt_stats[filt_stats['num_self_interacting'] < filters.get(
        "max_self_interacting")]

    circuits = filt_stats["name"].tolist()
    
