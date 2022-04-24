

import pandas as pd


from src.srv.io.results.writer import DataWriter
from src.srv.parameter_prediction.interactions import InteractionMatrix
from src.utils.data.loaders.generic import DataLoader


def generate_interaction_stats(pathname, writer: DataWriter):

    interactions = InteractionMatrix(matrix_path=pathname)

    stats = interactions.get_stats()

    writer.output(out_type='csv', out_name='circuit_stats', data=stats)

    return stats

    # note sequence mutation method


def pull_circuit_from_stats(stats_pathname):
    stats = DataLoader.load_data(stats_pathname)
